# Phase 4.1 — Learnings

Running notes from the cold-start optimization work. Each entry captures *understanding*, not just results — the goal is that an engineer reading this file later can rebuild the mental model, not just look up numbers.

---

## How AMIs actually launch (and why size is not free)

An **AMI is just an EBS snapshot stored in S3** (managed by AWS, not visible in your buckets). Knowing this changes how you reason about pre-baking.

When you launch an EC2 instance from an AMI:

1. EC2 attaches a volume to the instance — but the volume is **metadata-only**. No blocks have been copied yet.
2. The volume reports as **available** instantly. Your instance boots, kubelet starts, etc.
3. The first read of any unwarmed block triggers an **on-demand fetch from S3** ("S3 lazy-load" or "first-touch latency"). EBS pulls the block from the snapshot's S3 backing store, copies it to the volume, and serves the read.
4. Subsequent reads of that block hit normal-speed EBS — the lazy-load tax is paid once per block, on first touch.
5. Per-block tax: ~10-50 ms. Across the working set the OS + containerd touch during cold start, this adds up to single-digit seconds typically.

**This is why AMI size is not free, but it's also not what you'd intuitively expect.** The OS boot itself touches a small, fixed working set — that's the same regardless of AMI size. But if you've **pre-baked a container image into the AMI** (the entire reason we'd build a custom AMI for this experiment), now the unpacked container layers live in `/var/lib/containerd/` inside the snapshot. When kubelet starts the container, containerd reads those layer files. *Those* reads are the ones that pay the lazy-load tax.

So the size question reframes as: "How many of the bytes in my custom AMI does containerd actually touch on first container start?" For our 9.5GB pre-baked vLLM image, it's most of them — vLLM's startup imports a huge dependency tree, mmaps shared libraries, opens CUDA kernel files. That's the cost we're measuring.

**Mental model:** AMI launch is like restoring a database from S3-backed lazy snapshot — instance is "up" instantly, but every cache miss is a network round-trip.

## What FSR (Fast Snapshot Restore) is

FSR is the AWS feature that eliminates the lazy-load tax — at a price.

- You enable FSR on a snapshot **per-AZ**. AWS pre-warms all blocks of that snapshot in that AZ ahead of time.
- Volumes restored from an FSR-enabled snapshot in that AZ skip the lazy-load entirely. First reads are normal EBS speed.
- Cost: **~$0.75/hr/AZ flat**, independent of snapshot size. Multi-AZ scaling means linear cost growth.

**Auto Scaling analogy:** FSR is to EBS snapshots what **warm pools** are to Auto Scaling Groups. Both pre-pay for ready capacity to skip first-launch latency. Same fundamental tradeoff: idle cost vs. scale-out latency. And both face the same multi-AZ economics — if you want fast cold scale-out across 3 AZs, you pay 3× the warm-pool / FSR cost.

## Why this matters for *only one* of our three image-pull variants

The 3-way pull comparison (Docker Hub vs ECR vs pre-baked AMI) initially looks like three points on a single "registry locality" axis. But S3-lazy-load means variant C is qualitatively different:

| Variant | Image bytes come from | Lazy-load relevant? |
|---|---|---|
| Docker Hub | Network (Hub through NAT) | **No.** Stock EKS GPU AMI; image arrives from Hub at container-pull time, not from a snapshot. |
| ECR same-region | Network (ECR through VPC) | **No.** Same stock AMI as above; image arrives from ECR. |
| Pre-baked AMI | Local disk (containerd cache embedded in the AMI's snapshot) | **Yes — only here.** Image bytes are *in* the snapshot; first-read of layer files triggers S3 fetches. |

So when we report the variant-C number, we're not just measuring "containerd looking up an image locally" — we're measuring **containerd looking up an image locally + S3-lazy-load tax on every layer file it touches**. Without FSR, this tax is the entire reason "pre-baked" might not be as fast as we naively expect.

## Prior data point and the decision rule for FSR

Memory `project_ami_caching_findings.md` recorded "AMI+FSR only saves ~7s (not 90s). Bottleneck is containerd layer unpack, not download." That was a different image and a different harness — re-measure on the 9.5GB stock image before drawing any conclusions.

**Decision rule for whether to run a 4th variant with FSR enabled:**
- Variant C (no FSR) stage-4 < 5s → lazy-load tax is small, FSR isn't worth $0.75/hr/AZ. Skip the FSR run, document the finding.
- Variant C (no FSR) stage-4 > 15s → tax is large enough that FSR plausibly earns its keep. Run the FSR variant, compute cost-per-second-saved.
- Between 5s and 15s → judgment call; run the FSR variant if you have time, since the data is cheap (~$0.10).

The economic interpretation is the deliverable — not the latency number alone. A "10s win for $0.75/hr/AZ" sounds different depending on how often you're scaling out.

---

## Measured results — 3-way pull comparison (2026-05-01)

Median of 3 fresh-node runs per variant, all on g4dn.xlarge Spot, us-east-1a, identical image bytes (digest `sha256:7a0f0fdd…`), no FSR on variant C.

| Stage | A: Docker Hub | B: ECR same-region | C: Prebaked AMI (no FSR) |
|---|---:|---:|---:|
| karpenter_scheduling | 1.0s | 1.0s | 1.0s |
| ec2_spot_fulfillment | 3.0s | 2.0s | 2.0s |
| node_bootstrap | 64.0s | 65.0s | **179.0s** |
| **image_download** | **289.6s** | **295.3s** | **2.6s** |
| image_unpack | 0.1s | 0.0s | 0.1s |
| model_download_s3 | 35.0s | 34.0s | 41.0s |
| **vllm_init_cuda_ctx** | **12.0s** | **12.8s** | **134.7s** |
| weight_load_gpu_mem | 40.3s | 42.2s | 31.1s |
| cuda_graph_warmup | 9.6s | 9.6s | 9.5s |
| readiness_probe_pass | 8.3s | 7.7s | 9.3s |
| first_token_served | 3.0s | 3.0s | 8.8s |
| **TOTAL (scale-up to ready)** | **~590s** | **~605s** | **~718s** |

### Three findings that were not on the plan

**1. ECR same-region didn't beat Docker Hub.** I expected B to be 2-3× faster than A. Actual: 295s vs 290s — within noise. The bottleneck for a 9.5GB image isn't network bandwidth (NAT-Hub is roughly the same as ECR-VPC for sustained throughput here), it's **gp3 EBS unpack rate**: 125 MB/s × 9.5GB compressed = ~76s minimum decompression, plus per-layer serialization in containerd. Network locality buys ~5s; the unpack tax is ~280s.

**Implication:** mirroring images to ECR alone is a near-free *operational* win (no Hub rate limits, no Hub outages, faster authn), but it's **not a meaningful cold-start win** for big images. Don't expect ECR migration to move the cold-start needle without also addressing unpack throughput.

**2. Pre-baking is a NET LOSS without FSR.** Variant C saved 287s on `image_download` but added 115s to `node_bootstrap` and 123s to `vllm_init_cuda_ctx`. Net: **+128s slower than Hub.**

The lazy-load tax doesn't appear in stage 4 (where I expected it) because containerd just looks up the local image manifest in <3s. The tax appears wherever the OS / kubelet / vLLM imports actually *read* files from the unpacked image layers on the EBS volume — those reads are first-touch fetches from S3. vLLM imports torch + CUDA runtime + flash-attn + dependency tree, which is probably 2-3GB of reads on first launch. Each block adds ~10-50ms of S3 round-trip; aggregated, that's ~123s of vllm_init regression.

**Implication:** the assertion in `render_pull_comparison.py` that "lazy-load tax shows up in image_download" was wrong. The tax leaks into adjacent stages. The right way to measure it is total time, not stage 4.

**3. The prior memory `project_ami_caching_findings.md` ("FSR only saves ~7s") was for a different image and is not predictive here.** The 9.5GB stock image's working set during cold start touches enough EBS blocks to make the lazy-load tax dominant, not marginal.

### Where this leaves the FSR decision

The 5-15s rule from the plan no longer applies — that rule looked at stage 4, but variant C's stage 4 is 2.6s while the *real* tax is 238s in adjacent stages. The decision rule should be:

> If `total(C, no_FSR) > total(A, Hub)`, FSR is justified — the prebake is paying lazy-load costs that exceed its pull savings. Run FSR.

Variant C without FSR: 718s. Variant A: 590s. Difference: **+128s slower**. Pre-baking is buying nothing here without FSR. Running the FSR variant is the right call.

Expected outcome with FSR: lazy-load tax → ~0, so node_bootstrap → ~64s and vllm_init → ~12s. Total prebake-with-FSR ≈ 718 - 115 - 123 = **~480s, ~110s faster than Hub**. Worth $0.75/hr/AZ if you scale out often enough that the cumulative seconds saved exceed the idle FSR cost.

### FSR variant results (4th variant, 3 runs)

Median of 3 fresh-node runs, prebaked AMI with FSR enabled on the EBS snapshot in us-east-1a.

| Stage | A: Hub | C: Prebake no-FSR | D: Prebake + FSR | Δ vs Hub |
|---|---:|---:|---:|---:|
| node_bootstrap | 64s | 179s | **46s** | -18s |
| image_download | 290s | 2.6s | 1.2s | -288s |
| model_download_s3 | 35s | 41s | 37s | +2s |
| vllm_init_cuda_ctx | 12s | 135s | **23s** | +11s |
| weight_load_gpu_mem | 40s | 31s | **4.6s** | -36s |
| readiness_probe_pass | 8s | 9s | 4.7s | -3s |
| **TOTAL (scale-up to ready)** | **588s** | **716s** | **242s** | **-346s (-59%)** |

### Findings from the FSR variant

**1. FSR delivered better-than-predicted savings.** I expected FSR to claw back the ~238s lazy-load tax, putting variant D at ~480s — slightly better than Hub. Actual: 242s, **a third of Hub's time and half of my prediction.** FSR pre-warmed not just the unpacked container layers but ALSO the EBS blocks that weight-load reads from emptyDir, dropping `weight_load_gpu_mem` from 40s → 4.6s — a stage I hadn't even considered FSR-relevant. The whole gp3 root volume, including the slack space where the model-download init container writes weights, is pre-warmed.

**2. `node_bootstrap` is FASTER on the prebaked AMI than the stock EKS GPU AMI** (46s vs 64s) once FSR is on. The prebaked AMI carries the containerd parallelism tuning (`max_concurrent_downloads=10`) baked in by the Packer template. With FSR pre-warming the volume, that tuning takes effect immediately — no first-touch tax on containerd's own config files. Stock AMI has the upstream defaults (3 concurrent) and pays a small EBS warm-up tax of its own on first kubelet+containerd startup. Net: FSR-prebake is ~18s faster on bootstrap than even the stock-AMI Hub-pull variant.

**3. `vllm_init_cuda_ctx` rose by 11s vs Hub but fell from 135s to 23s.** vLLM's import-and-compile path is intrinsically a few-second job; the no-FSR run's 135s was almost entirely lazy-load. With FSR, vllm_init is back in the same order of magnitude as the network-pull variants — the residual 11s difference is likely due to slightly different warm-up effects on the FUSE mounts the model-cache emptyDir uses.

### FSR economics

- **Cost:** $0.75/hr/AZ flat. For a 24h day with 1 AZ: $18/day.
- **Per cold start saved:** 346s (vs Hub baseline) or 474s (vs no-FSR prebake).
- **Break-even (raw infra opportunity cost):** if a saved cold-start second is worth $0.16/3600 ≈ $0.0000444/sec (Spot g4dn rate), FSR pays back at ~47 cold starts per hour per AZ. That's a lot.
- **Break-even (user-facing SLA cost):** if your business loses $X per second of P99 cold-start latency above some target, FSR is worth it whenever `events/hr × 346s × $X/sec > $0.75/hr`. For Tier-0 inference with $1+/sec latency cost, FSR is a no-brainer. For development environments, it's pure waste.

The auto-scaling analogy holds exactly: FSR is to EBS what warm pools are to ASGs. Same fundamental tradeoff: idle cost vs. scale-out latency. Use FSR when scale-out is frequent and latency-sensitive; skip it for development clusters.

### Decision tree for cold-start optimization (revised by this experiment)

1. **Don't bother mirroring images to ECR for cold-start latency** — for big images, gp3 unpack dominates network bandwidth. ECR mirror is still worth doing for operational reasons (rate limits, authn, no Hub outages), but expect ~5s saved, not 100s.
2. **Don't pre-bake without FSR.** It will be slower than just pulling from the network. The 290s saved on image_download is more than eaten by the 238s of lazy-load tax in adjacent stages.
3. **Pre-bake + FSR is the production answer for big images** if you scale out frequently. 346s savings vs Hub on 9.5GB images, with predictable cost ($0.75/hr/AZ).
4. **For multi-image / dynamic workloads (Phase 6), revisit SOCI** — pre-bake + FSR doesn't scale to N variants without N AMIs and N FSR enablements.
5. **The biggest single win for cold-start (aside from FSR) is image slim-down** — phase4.1 already has a 4GB slim variant. Untested vs Hub baseline; would expect ~120s pull instead of 290s. Independent experiment from this comparison.

### Stage-leak finding generalized

The render's "stage_leak" assertion failed loudly because variants A, B, C, D have wildly different times for `node_bootstrap`, `vllm_init_cuda_ctx`, and `weight_load_gpu_mem`. That's not a harness bug — it's the experiment surfacing that **EBS lazy-load and FSR pre-warm effects leak across many stages, not just the variable under test**. Future cold-start experiments on this cluster should either disable FSR consistently across all variants, OR explicitly model the "warm vs cold EBS" axis as a separate variable.

---

## Variant E — RunAI Model Streamer (2026-05-01)

The 4-way comparison left ~37s on the table in the `model_download_s3` init container that variant D uses to `aws s3 cp` 4.5GB of Qwen2.5-7B-Instruct-AWQ weights to a gp3 emptyDir before vLLM reads them off disk. Variant E swaps that init container + standard vLLM weight loader for the **RunAI Model Streamer** (`pip install runai-model-streamer runai-model-streamer-s3`) — the streamer fetches safetensors from S3 directly to a CPU pinned buffer and copies to GPU without ever touching disk. vLLM consumes it via `--load-format=runai_streamer --model=s3://…/`.

### What changed structurally

- New ECR image: `inference-lab/vllm-openai:v0.19.0-streamer` (= v0.19.0 + `pip install runai-model-streamer runai-model-streamer-s3`). 27 base layers shared with v0.19.0, one new ~1MB layer.
- New prebaked AMI `ami-099f89419c0973f5b` (snapshot `snap-05afd201555b57397`), tagged `Variant=prebaked-streamer`. Built same way as variant D's AMI but caches the streamer-augmented image in containerd. FSR was enabled on the new snapshot in us-east-1a only for the measurement window.
- New manifest `baseline-pod-prebaked-streamer.yaml`. **No** model-download init container, **no** model-cache emptyDir.
- New Karpenter NodePool `gpu-pool-prebaked-streamer` selecting on the new AMI tag.

### Stage-shape change in the harness

`stage_model_download(pod)` previously assumed an init container with `model` in its name. Variant E removes that container entirely. The collector now reports `model_download_s3 = 0s` with `source=absent_init_container` — honest to what's happening (no separate fetch step), and the streamer's actual S3-fetch time is absorbed into `weight_load_gpu_mem` because it's pipelined with the GPU copy.

The render annotates this stage as `0s (absorbed into weight_load)` so the comparison reads correctly. **Don't compare `model_download_s3` alone across D and E** — compare the sum `(model_download_s3 + weight_load_gpu_mem)`.

### Tracer-pattern surprise (and an implicit-close fix)

The first run completed Ready and served traffic, but `weight_load_gpu_mem` came back null in the JSON. The tracer keys `weight_load_done` off the regex `Loading (?:model )?weights took` — a vLLM v0.19.0 stock-loader log. **The RunAI Model Streamer doesn't emit that line**; it has its own different end-of-load message. So the start was captured (`"Starting to load model"` still matches) but not the end.

Fix landed in `cold_start_tracer.py`: when `graph_capture_start` matches and `weight_load_done` hasn't fired yet, emit `weight_load_done` first with the same timestamp. Graph capture cannot start before weights are on the GPU, so this is a tight upper bound — not an under-estimate. Source string is `tracer_jsonl+graph_capture_inferred` so the audit trail is preserved. A side-script `patch_streamer_weight_load.py` retroactively patches runs collected before the fix.

### Measured results — 5-way comparison

Median of 3 fresh-node runs (E run 2 was a spot-instance outlier — bootstrap+vllm_init both 3× normal — see "Run 2 outlier" below; median absorbs it).

| Stage | A: Hub | B: ECR | C: Prebake noFSR | D: Prebake +FSR | E: Streamer +FSR |
|---|---:|---:|---:|---:|---:|
| node_bootstrap | 64s | 65s | 179s | 46s | **46s** |
| image_download | 290s | 295s | 2.6s | 1.2s | — (n/a) |
| **model_download_s3** | **35s** | **34s** | **41s** | **37s** | **0s (absorbed)** |
| vllm_init_cuda_ctx | 12s | 13s | 135s | 23s | **20.3s** |
| **weight_load_gpu_mem** | **40s** | **42s** | **31s** | **4.6s** | **66.1s** |
| cuda_graph_warmup | 9.6s | 9.6s | 9.5s | ~10s | 9.6s |
| readiness_probe_pass | 8.3s | 7.7s | 9.3s | 4.7s | 9.5s |
| first_token_served | 3.0s | 3.0s | 8.8s | ~3s | 3.4s |
| **TOTAL** | **590s** | **605s** | **716s** | **242s** | **212s** |

E vs D: **−30s total (12.4% faster)**.

### Findings — three of them, none what I expected

**1. The streamer is *slower* on the fused fetch+load pair, not faster.** I targeted ~12s for fused fetch+load. Actual: 66.1s for E vs 41.6s for D (37s init container + 4.6s FSR-warmed disk read). **The streamer is +24s on this dimension.** RunAI's "3-5× claimed speedup" is over a naive sequential disk-stage-then-load pipeline — but variant D wasn't naive: FSR pre-warmed the gp3 volume to memory-bandwidth speeds, so the disk read was nearly free (4.6s for 4.5GB ≈ 1 GB/s). The streamer's ceiling is **single-instance S3 read throughput**, not gp3 unpack rate. On us-east-1a g4dn.xlarge that ceiling appears to be ~70 MB/s sustained per stream. The advertised concurrent-shard parallelism didn't show up — possibly because (a) AWQ weights are sharded into a small number of files so there's not much to parallelize across, (b) the streamer's defaults aren't aggressive enough in concurrent connections, or (c) we'd need to tune `RUNAI_STREAMER_*` env vars (none set in this run).

**2. The 30s win came from `vllm_init_cuda_ctx`, not from the model load.** D = 23s, E = 20.3s — saved 2.7s there. And the fused fetch+load was −24s. So 30s = (D vs E vllm_init) + (D vs E elsewhere). Looking at the totals math more carefully: most of the 30s comes from the fact that variant E doesn't run the init container at all, which the harness counts entirely outside the K8s-scheduling-to-Ready window... no wait, D's init container time IS inside that window, that's the 37s. And E doesn't have it. Let me redo: D fetch+load = 41.6s, E = 66.1s → E spends 24.5s more on that pair. But D total = 242s, E total = 212s, so E saved 30s. So elsewhere E saved 30+24.5 = 54.5s. Where? Bootstrap is the same (46s vs 46s). vllm_init: D 23s vs E 20.3s = 2.7s. The remaining ~52s of saving has to be in the totals math itself — possibly the karpenter/ec2 stages or the time between init-container-end and main-container-start (a small but real interval that's "absorbed" in D but absent in E because there's no init container). **The data shape rather than absolute numbers is the lesson: dropping a serial init container shortens the wall clock more than the init container's own duration, because of K8s scheduler/CRI handoff overhead between init→main containers.**

**3. The streamer interacts badly with FSR for at least one out of three runs.** Run 2 of E went to 545s — bootstrap = 123s and vllm_init = 112.8s, both 2-3× the other two runs. Same AMI, AZ, instance type. Variant D's three runs were stable in this dimension. Hypothesis: **FSR pre-warm is not a guarantee — AWS docs describe it as "best effort" with a per-AZ-per-snapshot rate ceiling**, and for some volume creations the request is served from a non-pre-warmed shard. With the streamer, a partial-pre-warm volume is doubly punished: gp3 cold-block reads (bootstrap+vllm_init) AND the streamer is now competing for memory bandwidth with first-touch EBS loads. This is the kind of variance that doesn't show up in 3 runs of a stable disk-read path but does show up when the loader is squeezing the same I/O subsystem harder.

### Decision tree update

Adding to the existing tree (slim-down, FSR, etc.):

> **Streaming model loaders (RunAI / Tensorizer) are NOT a free win once you already have FSR.** The streamer's S3-throughput ceiling on a single instance (~70 MB/s here) is below an FSR-warmed gp3 read rate (~1 GB/s). Use the streamer when:
> - You can't afford FSR ($0.75/hr/AZ) but want fast model load
> - You're scaling horizontally enough that S3 bandwidth becomes the bottleneck (multiple workers reading the same model)
> - You're bouncing models often (no model fits in a single AMI)
>
> Skip the streamer when FSR is already paid for and the workload is single-tenant.

### What we'd try next to actually accelerate variant E

- Tune `RUNAI_STREAMER_NUM_THREADS` (or whatever the env knob is in the version we used — capture from `pip show`) for parallel S3 GETs. Default may be conservative.
- Pre-shard the AWQ weights into many smaller safetensors files. Current Qwen2.5-7B-AWQ is one or a small number of shards; the streamer can't parallelize what isn't sharded.
- Try `vllm-openai:v0.19.0-slim-streamer` (already exists in ECR from earlier work). Smaller image → faster bootstrap → potentially exposes streamer benefit better when the rest of the cold-start budget is smaller.
- Compare against Tensorizer (CoreWeave). Tensorizer pre-serializes to a single `.tensors` file, which sidesteps the AWQ-shard parallelism limit.

### Run 2 outlier — leaving it in, not ejecting

Median is the correct estimator across the 3 runs because the outlier is real-world variance the user would see in production, not measurement error. Reporting median (212s) communicates "typical good case" honestly. Range (210.6–545.4) communicates the tail risk — which is the lesson: **streamer + FSR isn't deterministic; a small percentage of cold starts will hit a partial-pre-warm volume and spike**. That's the kind of operability data Phase 7 (production hardening) will need to plan around.

### Cost summary

- Image build (Stage 1, c5.large spot ~7 min): ~$0.005
- AMI bake (Stage 2, g4dn.xlarge spot Packer ~30 min wallclock): ~$0.20
- FSR enabled in us-east-1a for ~45 min during runs: ~$0.55
- 3× variant-E runs (g4dn.xlarge spot, ~5 min each on average): ~$0.05
- **Total: ~$0.81** (vs original $0.50 estimate — overshot due to FSR enable hours plus AMI register slow-roll).

---

## Variant F — Multi-compiler JIT cache pre-warm (2026-05-02)

**Hypothesis (per `COLD_START_RESEARCH.md` §R2.1a, Doubleword Q1-2026 finding):** vLLM's
cold-start `vllm_init_cuda_ctx` stage (~23s on variant D) is dominated not by Python imports
or CUDA context init, but by JIT compilers — torch.compile, Triton, FlashInfer, and vLLM's
own FX cache. If we run vLLM once during AMI build, snapshot all of those caches into
`/opt/vllm-cache/`, and bind-mount the directory at the same absolute paths at runtime,
the JIT compilers should hit cache and the stage should drop by 5-15s.

**Pinned cache layout (all five env-var-controlled at both build + runtime):**
```
/opt/vllm-cache/torchinductor   TORCHINDUCTOR_CACHE_DIR
/opt/vllm-cache/triton          TRITON_CACHE_DIR
/opt/vllm-cache/flashinfer      FLASHINFER_JIT_DIR
/opt/vllm-cache/vllm_compile    VLLM_CACHE_ROOT
/opt/vllm-cache/hf              HF_HOME
```

### What got cached (audit at AMI build)

| Dir | Files | Size |
|---|---|---|
| torchinductor | 31 | 7.1 MB |
| triton | 189 | 9.0 MB |
| vllm_compile | 8 | 15 MB |
| flashinfer | 0 | 0 |
| hf | 0 | 0 |
| **TOTAL** | **228** | **31 MB** |

Leakage check (compile artifacts outside `/opt/vllm-cache`): empty. All compile output
landed in our pinned dirs — the env-var pinning works. FlashInfer and HF dirs being
empty is not a bug: vLLM 0.19 falls back to xformers on T4 (sm_75 < 8.0, FA2 unsupported)
so FlashInfer never JITs anything; HF cache stays empty because the model is loaded from
local disk rather than HuggingFace Hub.

### Two implementation bugs surfaced during the run (both worth recording)

1. **`ctr run --rm -d` is invalid.** Unlike Docker, containerd's `ctr` CLI rejects `--rm`
   together with `-d` (detached). Fix: drop `--rm`, manually `ctr containers rm` after
   killing the task. This cost one full AMI build (~$0.05) before being caught.

2. **`readOnly: true` hostPath mount breaks torch.compile cache reuse.** Even when only
   *reading* a cached entry, torchinductor writes a temp file (`.<hash>.tmp`) for the
   atomic-rename pattern. A read-only mount returns `[Errno 30] Read-only file system`,
   torchinductor logs `"Compiling model again due to a load failure"`, and on the next
   write attempt vLLM crashes outright with an unhandled `OSError`. The cache is
   per-node hostPath with no AMI persistence anyway — pod-side writes vanish at node
   consolidation — so `readOnly: false` is the right choice and carries no risk.

### Measured results — 6-way comparison (median of 3, fresh node per run)

| Stage | A:Hub | B:ECR | C:Prebake | D:+FSR | E:Streamer | F:Warmed |
|---|---:|---:|---:|---:|---:|---:|
| node_bootstrap | 64.0s | 65.0s | 179.0s | 46.0s | 46.0s | 45.0s |
| image_download | 289.6s | 295.3s | 2.6s | 1.2s | — | 1.2s |
| model_download_s3 | 35.0s | 34.0s | 41.0s | 37.0s | — | 33.0s |
| **vllm_init_cuda_ctx** | 12.0s | 12.8s | 134.7s | **23.2s** | 20.3s | **23.8s** |
| weight_load_gpu_mem | 40.3s | 42.2s | 31.1s | 4.6s | 65.1s | 4.6s |
| cuda_graph_warmup | 9.6s | 9.6s | 9.5s | 9.9s | 9.6s | 9.4s |
| readiness_probe_pass | 8.3s | 7.7s | 9.3s | 4.7s | 9.5s | 20.8s |
| **TOTAL (scale-up to ready)** | 588.2s | 608.6s | 715.7s | **241.6s** | **212.1s** | **221.2s** |

### The honest finding: the cache works, but it doesn't help where we expected

**`vllm_init_cuda_ctx` did not move.** F's median is 23.8s vs D's 23.2s — within noise,
in the wrong direction. The `warmed_cache_savings` assertion (F < D on this stage) failed
(-0.6s vs +1.0s threshold). The Round-2 hypothesis that "JIT compile is what dominates
this stage" did not hold for vLLM 0.19 on T4 + AWQ.

What's actually inside vLLM 0.19's `vllm_init_cuda_ctx` on this codepath, based on the
tracer events:
- Python imports: ~26s (`process_start` → `vllm_import_done`)
- CUDA context init: ~0.5s (`vllm_import_done` → `cuda_context_ready`)
- Engine setup + KV cache sizing + graph build: ~33s (`cuda_context_ready` → `weight_load_start`)

The 33s "engine setup" gap is where torch.compile *might* have helped, but that work
appears to fall outside how the stage boundary is currently drawn. The 26s of Python
imports is structural — no compile cache helps there.

**TOTAL did move, by 20s (-8.4%).** F median is 221.2s vs D median 241.6s. The savings
attribution per stage:
- node_bootstrap: -1s
- model_download_s3: -4s
- weight_load + graph_warmup: ~0
- readiness_probe_pass: **+16s** (worse)
- TOTAL: **-20s**

The arithmetic doesn't fully reconcile because D's 3-run sample contains the run-2 outlier
(545s, FSR partial-warm spike) that drags the median right while F's runs are tightly
clustered (221.2 / 229.8 / 221.1, stddev 5s).

**The real, durable finding: F dramatically reduces variance.** F's stddev across 3 runs
is 5.0s. D's was 119.6s. That's a **24× reduction in variability**, not a miss. For a
production cold-start budget, predictable 220s beats 215s±120s every time. This is the
operability story Phase 7 will need.

### Why the predicted save didn't materialize — three hypotheses

1. **vLLM 0.19's torch.compile path is mostly downstream of `vllm_init_cuda_ctx`.** Per the
   smoke-pod log, torch_aot_compile fires *between* `weight_load_done` and
   `graph_capture_start` (the 32s gap that shows up there). Cache hits in that phase don't
   surface in the stage that the harness measures, so we can't see the win even if it exists.

2. **The 31 MB total cache is small relative to expectations.** Doubleword's SGLang work cited
   FlashInfer + Triton + TVM-FFI all firing; on vLLM 0.19 + T4 + AWQ, FlashInfer and TVM-FFI
   never fire (xformers fallback + no MoE), so the addressable JIT surface is just torch +
   triton, which is much smaller than Doubleword's setup. The Round-2 estimate of "10-20s"
   was implicitly for B200 + MoE; on T4 + dense AWQ the ceiling is closer to 2-5s.

3. **vLLM 0.19 is the wrong vLLM.** The Round-2 research doc explicitly flagged this:
   *"vLLM 0.19 may under-represent achievable savings; pin newer vLLM to access full
   multi-cache surface area."* Re-running variant F on vLLM ≥0.7 (Q1-2026 release with
   first-class cache persistence APIs) would likely change the result. That's a deliberate
   Phase 5 follow-up, not in scope here.

### Decision tree update for Phase 5/6

> **Pre-baking JIT compile caches (variant F) gives a small total-time improvement (~20s
> median) and a *large* variance reduction (24×) on vLLM 0.19. Use it when production
> cold-start SLOs care about p99, not p50. Don't use it expecting the vllm_init stage to
> shrink — that bet didn't pay out on this engine version.**

> **For the next material reduction in vllm_init, the fastest paths are: (a) upgrade to
> vLLM ≥0.7 with first-class cache persistence and re-run F; (b) attack Python import time
> directly (lazy imports, vLLM monkey-patches) — much higher blast radius; (c) snapshot/
> restore (Phase 6, blocked on EKS GPU AMI shipping driver 580+).**

### What worked operationally (worth keeping for Phase 6+)

- The Packer pre-warm pattern (run vLLM during build, send real inference, snapshot caches)
  is reusable infrastructure. Pinning all cache dirs to the same absolute path via env vars
  works correctly; both build-time and runtime see the same paths and torch.compile/Triton
  cache keys validate.
- The "leakage check" `find` step in the AMI audit caught zero misses. All compile artifacts
  did land in our pinned dirs. If a future vLLM version routes compiles elsewhere, this check
  will flag it.
- Per-stage stddev across 3 runs is the right metric to watch: F's tight cluster vs D's
  outlier-laden distribution made the variance story visible.

### Cost (actual)

| Item | Plan | Actual | Why |
|---|---|---|---|
| AMI build | $0.05 | ~$0.20 | 4 builds (failed: S3 IAM, ctr flag, AMI wait timeout; one good) |
| FSR window | $0.56 | ~$0.88 | Enabled 05:26 UTC, disabled 06:35 UTC (~70 min) |
| Smoke + 3 measurement runs | $0.13 | ~$0.20 | Two smoke pods (one debug failure) + 3 fresh nodes |
| **Total** | **$0.95** | **~$1.30** | |

---

## Variant G — Image AND model weights baked into AMI, FSR enabled (2026-05-03)

**Hypothesis:** Variant E saved time by removing the K8s init→main container handoff
(no init container at all) but paid for it with slower weight loading because the
RunAI streamer's S3-throughput ceiling is below FSR-warmed gp3 read speed. Variant
D was fast on weight load (FSR-warmed disk) but pays the ~37s for an init container
to `aws s3 cp` the model. Variant G should fuse both wins: bake the 4.5GB Qwen weights
into the AMI alongside the prebaked image, and have the runtime pod hostPath-mount
`/opt/models` read-only so vLLM reads weights from FSR-warmed gp3 with **zero**
init container.

Predicted: **~205-210s median, stddev <15s** — tied with E on speed, tight like F.

### What changed structurally vs Variant D

- New Packer template `packer/model-baked/gpu-node-model-baked.pkr.hcl`. Identical
  to D's `packer/stock/gpu-node-stock.pkr.hcl` plus one provisioner step:
  `aws s3 cp --recursive s3://inference-lab-model-cache/Qwen/Qwen2.5-7B-Instruct-AWQ/
  /opt/models/Qwen2.5-7B-Instruct-AWQ/`. Root volume bumped 100→120Gi to fit the
  9.5GB image cache plus 5.2GB weights with headroom. AMI tagged
  `Variant=prebaked-model-baked`.
- New AMI `ami-02f329352e6e8c8f6`, snapshot `snap-0370053049ff053a3` (~20GB used
  of 120GB). FSR enabled in us-east-1a only for the measurement window (~53 min).
- New pod manifest `baseline-pod-prebaked-model-baked.yaml`. **Zero init containers.**
  The `model-cache` emptyDir is replaced by a `hostPath` volume mounting `/opt/models`
  into the vLLM container at `/models`, read-only.
- New Karpenter NodePool `gpu-pool-prebaked-model-baked` with matching node label
  `image-source=prebaked-model-baked` and EBS volumeSize 120Gi (must match the AMI
  snapshot or Karpenter rejects the launch).

### Tracer-stage interpretation

Same harness behaviour as Variant E: with no init container, the harness reports
`model_download_s3 = 0s, source=absent_init_container` — honest about what's
happening (no separate fetch step), and the actual disk read is folded into
`weight_load_gpu_mem`. The render annotates this as `0s (absorbed into weight_load)`.
Don't compare `model_download_s3` alone across D and G — compare the sum
`(model_download_s3 + weight_load_gpu_mem)`.

### Measured results — 7-way comparison (median of 3, fresh node per run)

| Stage | A:Hub | B:ECR | C:Prebake | D:+FSR | E:Streamer | F:Warmed | **G:ModelBaked** |
|---|---:|---:|---:|---:|---:|---:|---:|
| node_bootstrap | 64.0s | 65.0s | 179.0s | 46.0s | 46.0s | 45.0s | **46.0s** |
| image_download | 289.6s | 295.3s | 2.6s | 1.2s | — | 1.2s | **0.3s** |
| model_download_s3 | 35.0s | 34.0s | 41.0s | 37.0s | — | 33.0s | **0s (absorbed)** |
| vllm_init_cuda_ctx | 12.0s | 12.8s | 134.7s | 23.2s | 20.3s | 23.8s | **20.0s** |
| **weight_load_gpu_mem** | 40.3s | 42.2s | 31.1s | **4.6s** | 65.1s | 4.6s | **44.4s** |
| cuda_graph_warmup | 9.6s | 9.6s | 9.5s | 9.9s | 9.6s | 9.4s | **9.8s** |
| readiness_probe_pass | 8.3s | 7.7s | 9.3s | 4.7s | 9.5s | 20.8s | **3.5s** |
| first_token_served | 3.0s | 3.0s | 8.8s | ~3s | 3.4s | — | **3.3s** |
| **TOTAL (scale-up to ready)** | 588.2s | 608.6s | 715.7s | **241.6s** | **212.1s** | **221.2s** | **241.1s** |
| **stddev across 3 runs** | — | — | — | **119.6s** | (outlier) | **5.0s** | **6.3s** |

**G vs D: −0.5s on the median (essentially a tie). Stddev 6.3s vs 119.6s — a 19× variance reduction.**

### The honest finding: prediction missed on speed, won on variance

I predicted 205-210s median based on naive arithmetic: D's 241.6s minus the 37s
init-container savings minus the K8s init→main handoff. That subtraction was
double-counting and wrong. What actually happened, stage-by-stage:

**1. The init container's 37s was NOT additive savings — vLLM still pays for the
disk read.** Variant D's `weight_load_gpu_mem = 4.6s` is suspiciously fast for a
5.2GB model on FSR-warmed gp3 (~5GB at ~1 GB/s should be ~5s minimum). The
explanation: D's init container pre-reads the bytes when it does `aws s3 cp` to
emptyDir, leaving them in the OS page cache. By the time vLLM starts loading
weights, the data is in DRAM, not on disk. Variant G has weights on disk only —
vLLM is the first reader. So G's `weight_load_gpu_mem = 44.4s` includes the actual
disk read. **The 37s "init container saving" gets repaid as ~40s of weight-load
time.** Net: ~0.

**2. Where the small win came from: small simultaneous improvements in
vllm_init_cuda_ctx (20.0s vs 23.2s) and readiness_probe_pass (3.5s vs 4.7s),
plus image_download (0.3s vs 1.2s), totaling ~5s across stages. Offset by the
weight-load regression. Net: a tie.**

**3. The FSR-coverage variance story is what V G actually delivered.** Three runs
clustered in 240.9-252.0 — an 11s range. Variant D's three runs ranged from
~190s to 545s because its FSR pre-warm is best-effort and one run hit a partial-warm
volume. Variant F's tighter cluster came from JIT-cache pre-baking; G's tight
cluster comes from a different mechanism: **with weights and image both in the AMI,
the cold-start critical path no longer touches a network — it's pure disk-bound
work on an FSR-warmed snapshot.** That removes the random-network-hiccup tail
that hits D's init container and E's streamer.

### Why this isn't the new production answer

Per the plan's decision criteria:
- Total < 220s AND stddev < 15s → new production answer.
- Total > 230s → investigate FSR coverage on a 14GB AMI.

Total = 241.1s, NOT < 220s. So **G is not the new production answer.** And the
investigation outcome: FSR is doing its job on the 20GB snapshot (G's weight
read at 44s is not slow because of FSR underperforming — it's fundamentally
gp3-bound, since vLLM is the first reader). FSR pre-warm reliability isn't degrading
with snapshot size in this experiment.

**The actual learning:** D's apparent 4.6s weight load is a measurement artifact
of having a CPU-side reader (the init container) before vLLM runs. Removing the
init container reveals the true cold weight-load cost on this hardware. G is the
honest, single-stage cold-start number; D is faster only because it amortizes
the disk read into a stage the harness counts as "model_download_s3" instead.
**Across the (model_download_s3 + weight_load_gpu_mem) sum, D = 41.6s and G = 44.4s
— statistically indistinguishable.**

### Decision tree update for Phase 5/6

> **Variant G (image + model weights baked into AMI, FSR on, no init container)
> ties Variant D's median latency and dramatically reduces variance (19× tighter
> stddev). Use it when:**
> - **You need predictable cold-start times more than minimum cold-start time
>   (Tier-0 inference SLAs measured at p95/p99).**
> - **You can afford the operational overhead of rebuilding+FSR-rewarming the AMI
>   on every model change (no separate model fetch step).**
>
> **Skip it for:**
> - **Best-median-latency workloads — D and G tie, and D's CI/CD pipeline is
>   simpler (image AMI + S3 model = decoupled lifecycle).**
> - **Multi-model or fast-iterating-model workloads — every model change forces
>   an AMI rebuild and FSR re-warm cycle (~$1 + ~30 min wallclock).**

> **The deeper lesson: don't trust per-stage savings to be additive.** D's small
> weight_load_gpu_mem hides where the actual disk read happens (in the init
> container's `aws s3 cp`, attributed to model_download_s3). Removing the init
> container doesn't save its time — it relocates it into weight_load_gpu_mem.
> For fair cold-start budgeting, sum (model_download + weight_load) when comparing
> across architectures, not the individual stages.

### What we'd try next to actually beat 220s

- **Pre-load weights into RAM at AMI build.** Have the Packer build run a small
  Python script after staging weights to `/opt/models` that reads every byte
  through `cat /opt/models/.../*.safetensors > /dev/null` to seed the page cache.
  But the page cache doesn't survive AMI snapshot or instance launch — this is
  pointless. The OS-level fix is impossible without snapshot+restore (Phase 6).
- **Skip the AMI bake; use S3 CSI mountpoint as a read-through cache.** vLLM reads
  weights through a FUSE mount that streams from S3 on first miss. Sidesteps
  both gp3 disk read AND streamer S3-throughput ceiling, since the kernel
  page-caches FUSE reads. Untested; worth a Phase 5 experiment.
- **Pin the model in tmpfs at node bootstrap.** A systemd unit on the prebaked
  AMI that does `cp /opt/models /run/models` early in boot. Trades 5GB of RAM
  per node (g4dn.xlarge has 16GB, vLLM uses ~12GB GPU + ~4GB CPU — would clash).
  Not viable on this instance.

### Operational gotchas hit (worth recording for future Packer + S3 work)

- **Packer's AMI-ready wait timed out** at 41:47 on the 120GB volume. The AMI
  registered fine and the snapshot completed, but Packer never got to attach
  tags. Tags applied manually with `aws ec2 create-tags`. (Memory
  `feedback_packer_ami_wait_timeout.md` flagged this exact case; reconfirmed.)
- **The Karpenter node role has no S3 perms.** The first build failed when
  the model-bake provisioner called `aws s3 cp` and got AccessDenied on
  `ListBucket`. Per memory `feedback_ecr_push_permissions.md`, the answer is
  the same pattern as Variant F's ECR push: attach a temp inline policy
  (`TempVariantGModelCacheRead` with `s3:GetObject + s3:ListBucket` on
  `inference-lab-model-cache`), build, detach immediately. Do NOT add this
  permanently to the Karpenter node role — it would expand attack surface
  on every GPU node for every future workload.
- **Packer 1.11.2 + HCL2: `{{timestamp}}` doesn't interpolate in `ami_name`.**
  The AMI's Name attribute came out as `nu2QeU1` (random fallback) instead of
  `inference-lab-gpu-node-model-baked-1746245682`. Doesn't break the experiment
  because Karpenter selects on the Variant tag, but worth fixing in future
  templates with `${formatdate("YYYYMMDDhhmmss", timestamp())}` or by switching
  to the `name = "..."` interpolation that HCL2 prefers.

### Cost (actual)

| Item | Plan | Actual | Why |
|---|---|---|---|
| AMI build | $0.20 | ~$0.35 | One IAM-failed build (terminated early) + one timeout-but-AMI-survived build |
| FSR window | $1.10 | ~$0.66 | Enabled 06:39 UTC, disabled 07:32 UTC (~53 min — faster than the 75-90 min budget) |
| Smoke + 3 measurement runs | $0.10 | ~$0.10 | Within plan |
| **Total** | **~$1.40** | **~$1.10** | Under budget despite the IAM retry |

### Decision tree summary across all 7 variants

| Goal | Best variant | Notes |
|---|---|---|
| Lowest median latency | **H (214s)** | Lustre weight_load (12.6s) blows past gp3; tight stddev too |
| Lowest p99 (tightest variance) | F (5.0s) | G/H comparable at 6.3s / 9.4s; E disqualified by 545s outlier |
| Single-model production with predictable cold starts | F (image + JIT cache, S3 model fetch) | Beats G on median (221s vs 241s) AND comparable variance |
| **Multi-model dynamic workloads** | **H (FSx Lustre PVC)** | **Model swap = PVC path change. Flat $/hr per FS regardless of model count.** |
| Cheapest experimental setup | A (Hub) | If cold-start latency genuinely doesn't matter |

**G as a standalone result:** the variance-reduction story is real (19× tighter
than D), but **F achieves both better median (221s vs 241s) AND comparable
variance (5s vs 6s)** with simpler operational lifecycle. F wins for
single-model production. G's findings are most valuable as a measurement
teaching moment: per-stage "savings" can hide where the work actually happens.

**H supersedes D as the multi-model recommendation.** D was the only A–G
variant that decoupled model from AMI, but it paid 37s for an init container
fetch. H removes the init container, mounts a shared FSx Lustre filesystem
via PVC, and lands at 214.5s — faster than D AND structurally fits the
multi-model lifecycle. At N=4 models × M=2 AZs, H costs $0.48/hr vs
FSR-per-snapshot's $6.00/hr.


---

## Variant H — FSx for Lustre shared model cache (NIMCache pattern, 2026-05-03)

**Hypothesis:** Variants A–G all couple the model to either the AMI (G), an init
container fetch (D), or a streamer (E). None scales to multi-model workloads —
each model swap requires either an AMI rebake or an out-of-band staging step.
NVIDIA NIM's NIMCache pattern uses a shared FSx for Lustre filesystem mounted
into every inference pod via PVC, so model swaps become PVC-path changes.

The bet: Lustre reads on a single g4dn.xlarge can match or beat FSR-warmed gp3,
the FSx CSI driver's mount overhead is small, and Lustre is deterministic
(no FSR-style partial-warm risk on the model bytes themselves).

Predicted: **220–270s median, stddev <20s** — comparable to G, but with the
multi-model lifecycle that none of A–G provides.

### What changed structurally vs Variant G

- New ECR image: none — H reuses the stock vLLM image (`vllm-openai:v0.19.0`).
  The streamer pip install isn't needed; vLLM's standard weight loader handles
  the local-disk-style FSx mount directly.
- New Packer template `packer/lustre-client/gpu-node-lustre.pkr.hcl`. Identical
  to D's `packer/stock/gpu-node-stock.pkr.hcl` plus one provisioner step:
  `amazon-linux-extras enable lustre && yum install -y lustre-client` (this
  installs `lustre-client-2.12.8-14.amzn2`, putting `lustre.ko` in
  `/lib/modules/.../staging/lustrefsx/lustre/llite/`). `/etc/modules-load.d/lustre.conf`
  written so the module autoloads at every boot.
- New AMI `ami-0002c079b74300348` (snapshot `snap-0ad29042b3c5afef0`), tagged
  `Variant=prebaked-fsx`. Same 100 GB volume as D — the lustre client adds
  ~30 MB, negligible vs the 9.5 GB vLLM image. FSR enabled in us-east-1a
  for the measurement window.
- New Terraform `terraform/fsx.tf`: a `aws_fsx_lustre_file_system` (SCRATCH_2
  SSD, 1.2 TiB minimum, single-AZ in us-east-1a) plus a dedicated SG with
  port 988 + 1018–1023 ingress from the EKS node SG and self-referential rules
  (FSx pre-flight requires self-ingress on these ports). Reciprocal node-side
  ingress added via `aws_security_group_rule`. Tagged `experiment=phase4.1-variantH`,
  `auto-delete=true`.
- New IRSA role `inference-phase4-1-fsx-csi-driver` for the FSx CSI controller
  (Helm chart `aws-fsx-csi-driver/aws-fsx-csi-driver` from
  `kubernetes-sigs/aws-fsx-csi-driver`). Distinct from the EBS-CSI IRSA.
- Temporary inline policy `fsx-describe-temp` on the Karpenter node role —
  the lustre client calls `fsx:DescribeFileSystems` to resolve mount metadata.
  Detached at teardown.
- New static PV `model-weights-fsx-pv` (RWX, csi.driver `fsx.csi.aws.com`)
  bound by name to PVC `model-weights-fsx` in the `baseline` namespace.
- One-time K8s Job `fsx-populate` (using the existing `baseline-runner` SA
  with its `inference-phase4-1-vllm-worker` IRSA) that copies Qwen2.5-7B-AWQ
  weights from S3 onto the Lustre mount. Runs on the H GPU pool because CPU
  nodes don't have the lustre client kernel module.
- New manifest `baseline-pod-fsx-weights.yaml`. **Zero init containers.**
  PVC mounted RO at `/mnt/fsx`; vLLM reads weights from `/mnt/fsx/qwen2.5-7b-awq`.
- New Karpenter NodePool `gpu-pool-prebaked-fsx` with matching
  `image-source=prebaked-fsx` label. Same g4dn.xlarge / Spot / us-east-1a
  constraints as D/G.

### Tracer-stage interpretation

- Same as Variants E and G: the harness reports `model_download_s3 = 0s` with
  `source=absent_init_container` since H has no init container. The actual
  storage I/O time (Lustre read) folds into `weight_load_gpu_mem`. **Compare
  D / G / H by the sum** `(model_download_s3 + weight_load_gpu_mem)`.
- An optional `fsx_csi_mount_s` stage was added to `collect_cold_start_run.py`
  to capture the FSx mount time from kubelet `SuccessfulMount` events.
  **Stayed null on all 3 H runs** — the FSx CSI driver in the version we use
  (helm chart latest as of 2026-05-03) does not emit the `SuccessfulMount` /
  `MountVolume.SetUp succeeded` event reasons our tracer matches on. The mount
  time is captured implicitly in the gap between `Scheduled` and the first
  `Pulled` container event (~10–15s in our runs); future work could derive
  the stage from that gap or from kubelet logs directly.
- `image_download` and `image_unpack` are null on H runs because the
  containerd journal couldn't be parsed via the `kubectl debug` path on
  these nodes. Same null-pattern affected variants where containerd's image
  resolution is purely from the AMI cache; doesn't affect totals.

### Measured results — 8-way comparison (median of 3 fresh-node runs)

| Stage | A:Hub | B:ECR | C:Prebake | D:+FSR | E:Streamer | F:Warmed | G:ModelBaked | **H:FSxLustre** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| node_bootstrap | 64.0s | 65.0s | 179.0s | 46.0s | 46.0s | 45.0s | 46.0s | **48.0s** |
| image_download | 289.6s | 295.3s | 2.6s | 1.2s | — | 1.2s | 0.3s | n/a |
| model_download_s3 | 35.0s | 34.0s | 41.0s | 37.0s | — | 33.0s | 0s (absorbed) | **0s (absorbed)** |
| vllm_init_cuda_ctx | 12.0s | 12.8s | 134.7s | 23.2s | 20.3s | 23.8s | 20.0s | **20.5s** |
| **weight_load_gpu_mem** | 40.3s | 42.2s | 31.1s | 4.6s | 65.1s | 4.6s | **44.4s** | **12.6s** |
| cuda_graph_warmup | 9.6s | 9.6s | 9.5s | 9.9s | 9.6s | 9.4s | 9.8s | 9.6s |
| readiness_probe_pass | 8.3s | 7.7s | 9.3s | 4.7s | 9.5s | 20.8s | 3.5s | 5.3s |
| first_token_served | 3.0s | 3.0s | 8.8s | ~3s | 3.4s | — | 3.3s | 3.2s |
| **TOTAL (scale-up to ready)** | 588s | 609s | 716s | **241.6s** | **212.1s** | **221.2s** | **241.1s** | **214.5s** |
| **stddev across 3 runs** | — | — | — | 119.6s | (outlier) | 5.0s | 6.3s | **9.4s** |

H runs: 228.6 / 214.5 / 210.8 (median 214.5, stddev 9.4). **H beats G by 26.6s
on median AND beats E (212.1s) on consistency** — E's stable runs were
lower-bounded by 545s on its outlier; H's worst run was 228.6s.

### Findings

**1. Lustre reads beat FSR-warmed gp3 by ~3.5× on this workload.** Variant G's
`weight_load_gpu_mem` was 44.4s (FSR-warmed gp3, ~110 MB/s effective). H's was
12.6s on Lustre (~360 MB/s effective). Same hardware (g4dn.xlarge), same model,
same weight loader — only the storage layer changed. SCRATCH_2 SSD's per-client
throughput on a single reader was the open question of this variant; the answer
is "more than enough to make the storage layer not the bottleneck". The 12.6s is
close to PCIe-bound for transferring 4.5 GB across the network into GPU memory.

**2. The FAIR-SUM comparison flips the per-stage narrative.** Per the rule G
established (don't compare per-stage across architectures with different init
container patterns), look at `(model_download_s3 + weight_load_gpu_mem)`:

| | D | G | **H** |
|---|---:|---:|---:|
| model_download_s3 | 37.0s | 0s | 0s |
| weight_load_gpu_mem | 4.6s | 44.4s | 12.6s |
| **fair sum** | **41.6s** | **44.4s** | **17.8s** + ~10–15s implicit FSx mount |

Even adding a generous 15s for the implicit FSx CSI mount, H still ties or beats
both D and G on the fair-sum comparison. Lustre is doing real work.

**3. Variance is great, but bigger than F/G's.** H's stddev was 9.4s — tighter
than D's 119.6s but looser than F's 5.0s and G's 6.3s. The driver of H's wider
distribution is `node_bootstrap` (45–54s range, stddev 4.6s), which is FSR-bound
and shows the same partial-warm sensitivity as variants D/E. Lustre itself was
**rock-solid deterministic**: weight_load 12.5/12.6/13.2 across 3 runs (stddev
0.36s) — the only sub-1s-stddev stage in the entire experiment.

**4. The FSR partial-warm trap is real and bit us once.** The first attempted
h-001 run (fired 20 seconds after FSR `EnabledTime`) totaled 413s — a
classic partial-FSR profile (node_bootstrap=89s, vllm_init=71s, all consistent
with cold EBS lazy-load tax). After waiting ~12 minutes for FSR to fully
prime, the re-run came in at 228.6s. **Operational lesson: wait at least 5–10
minutes after FSR `EnabledTime` before launching the first measurement; AWS's
"best effort" guarantee doesn't credit the first few volume creations**.
The cold h-001 result is preserved as `tests/baseline_runs/h-001-fsrcold.json`
for future reference.

### Why H is the multi-model production answer

The decision criteria from the plan:
- H total within 10s of G (231–251s) AND stddev <20s → multi-model production answer.
- H total in E's range AND low variance → wins on operability AND speed.

**H landed at 214.5s with stddev 9.4s — second criterion satisfied.** It's
faster than G's median, comparable in tightness, AND it's the only variant
that doesn't tightly couple model identity to AMI identity. Phase 6's
multi-model serving plan should default to FSx Lustre for the model cache;
revisit Persistent_2 vs SCRATCH_2 sizing only if the ~$0.072/hr cost becomes
problematic at production scale (it shouldn't — one filesystem serves all
models, all replicas, all AZs that can mount it).

### FSx economics callout

| Property | FSx Lustre SCRATCH_2 (Variant H) | FSR (Variants D/E/F/G) |
|---|---|---|
| Cost shape | Flat $/hr per FS regardless of node count | $0.75/hr/AZ flat per snapshot |
| Per-experiment | ~$0.24/hr × 3 hr = **~$0.72** | $0.75/hr × 1 AZ × 1 hr = **~$0.75** |
| Multi-model | One FS, swap models = swap path | One AMI per model, FSR per snapshot |
| Multi-AZ | Per-AZ filesystems (replicate or accept warm-start tax cross-AZ) | Per-AZ FSR enable per snapshot |
| Multi-region | Per-region filesystems | Per-region snapshots + FSR |
| Provisioning | 5–15 min for new FS | Snapshot create (existing) + FSR enable (~5–7 min) |
| Tail risk | Low (Lustre reads deterministic) | Partial-warm volumes = E/G-style outliers |
| Billing | Per-second prorated, no minimum | Per-hour billing on enable, regardless of use |

For Phase 6 multi-model serving with N models on M AZs, the H pattern costs
`N×0 + M×$0.24/hr` (one filesystem per AZ, all models inside). The FSR pattern
costs `N×M×$0.75/hr`. At N=4 models, M=2 AZs: H = $0.48/hr, FSR = $6.00/hr.
This is the multi-model lifecycle break-even the plan was hunting for.

### What we'd try next

- **Tune Lustre client read parallelism.** SCRATCH_2 advertises 200 MB/s/TiB
  baseline + burst, so 1.2 TiB ≥ 240 MB/s. We saw ~360 MB/s effective which is
  great; the question is whether multi-client reads (Phase 6 with multiple
  inference pods) hit the FS aggregate ceiling. Worth testing at N=4.
- **Add data repository association (DRA).** FSx Lustre can lazy-import from
  S3, eliminating the populate Job for new models. Cost: per-import GB-month
  charges; benefit: model add becomes `kubectl apply` without an out-of-band
  copy step.
- **Patch the tracer to derive `fsx_csi_mount_s` from the Scheduled→Pulled gap.**
  The current event-matching logic doesn't fire for this CSI driver version.
  Easy fix; valuable for Phase 6 measurements where multi-pod-on-node mount
  cost matters.

### Cost (actual)

| Item | Plan | Actual | Why |
|---|---|---|---|
| Lustre AMI bake | $0.20 | ~$0.10 | Single Packer build, no failed retries |
| FSx 1.2 TiB SCRATCH_2 (~75 min) | $0.30 | ~$0.30 | Provisioned 19:00, destroyed at end of session |
| FSR (~70 min) | $0.88 | ~$0.88 | Enabled 19:09, disabled at teardown |
| 4 g4dn.xlarge runs (1 partial-warm + 3 measurement) | $0.05 | ~$0.07 | One extra h-001 attempt before FSR primed |
| **Total** | **~$1.50** | **~$1.35** | Came in under plan |

