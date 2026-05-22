# Why your LLM cold start is slower than you think — a 590s → 212s walkthrough

I spent a week pulling LLM inference cold-start latency from **590 seconds to 212 seconds** on a single g4dn.xlarge GPU node — a 64% reduction across six experimental variants. Along the way, three of my prior intuitions broke:

- Mirroring images from Docker Hub to ECR same-region didn't move the needle.
- Pre-baking the container image into the AMI made cold start *worse* — until I added one more thing.
- A purpose-built streaming model loader lost to a warm gp3 disk read.

The keeper insight isn't any single number — it's that cold-start optimizations don't compose cleanly by stage. They leak across stages, and which one wins depends entirely on your image size, your model, and your GPU. This post walks through what worked, what didn't, and the mental models I'd reach for first next time.

## The setup matters — read this before the results

Cold-start results don't generalize across environments. None of the rankings below would survive unchanged on, say, an H100 running a 70B model with FlashInfer. So before any numbers, here's exactly what I measured on:

| Component | Choice |
|---|---|
| Cluster | EKS 1.30, Karpenter v0.37, scale from zero |
| Instance | g4dn.xlarge Spot, us-east-1a, NVIDIA T4 (sm_75) |
| Model | Qwen2.5-7B-Instruct-AWQ — 4.5 GB, small number of safetensors shards |
| Engine | vLLM 0.19.0 |
| Image | 9.5 GB (vLLM + CUDA runtime + torch + flash-attn deps) |
| Storage | gp3 root volumes (125 MB/s default), EBS-backed AMIs |
| Measurement | wall clock from Karpenter scheduling decision to first token served, broken into ~10 stages by a tracer that parses kubelet/containerd/vLLM logs |

Two facts from this list end up doing a lot of work later. First, **T4 is sm_75**, which is below FlashAttention 2's 8.0 cutoff — vLLM falls back to xformers, and FlashInfer never JITs anything. Second, the **AWQ weights are sharded into a small number of files**, which limits how much parallelism a streaming loader can extract.

If your stack is different — H100s, larger model, newer vLLM — expect the rankings to shuffle. The method (measure stages, attack the dominant one, re-measure) is what transfers; the answers are environment-specific.

## The cold-start stage map

Cold start in LLM inference isn't one number. It's a budget spread across roughly ten stages, each with a different owner and a different speed limit:

1. **karpenter_scheduling** — Karpenter sees an unschedulable pod and decides what node to launch (~1s)
2. **ec2_spot_fulfillment** — AWS allocates the Spot instance (~2-3s)
3. **node_bootstrap** — instance boots, kubelet + containerd start, node joins the cluster
4. **image_download** — containerd pulls the container image
5. **image_unpack** — layers unpack to `/var/lib/containerd/`
6. **model_download_s3** — init container `aws s3 cp` of model weights to a gp3 emptyDir
7. **vllm_init_cuda_ctx** — vLLM Python imports, CUDA context init, engine setup
8. **weight_load_gpu_mem** — weights move from disk to GPU memory
9. **cuda_graph_warmup** — vLLM captures CUDA graphs for the steady-state path
10. **readiness_probe_pass** — kubelet's first successful HTTP probe
11. **first_token_served** — actual end-to-end inference completes

Plant this warning early: **stages leak**. An optimization labeled "image delivery" can show up as latency in `vllm_init`. The clean partitioning above is how you *report* time, not how optimizations *behave*. The most painful lesson in this whole experiment came from missing that.

## Optimizing image delivery

Most engineers' first instinct on "make cold start faster" is to attack image delivery. It's the biggest single chunk of the baseline (290s of a 590s total) and the most tactile — registries, networks, AMIs are all things you can swap out and remeasure. Two of my three broken intuitions live here.

### Network locality is a red herring

Variants A and B were the network-locality test: pull from Docker Hub vs pull from ECR same-region. I expected ECR to be 2-3× faster — same VPC, no NAT, no Hub rate limits.

| Variant | image_download | TOTAL |
|---|---:|---:|
| A: Docker Hub | 289.6s | 590s |
| B: ECR same-region | 295.3s | 605s |

Within noise. The bottleneck for a 9.5 GB image isn't network throughput; it's **gp3 unpack rate**. 125 MB/s × 9.5 GB compressed ≈ 76s of decompression alone, plus containerd's per-layer serialization, gets you to ~280s before the network even matters. Network locality buys ~5s.

This doesn't mean don't mirror to ECR — it means mirror to ECR for the *operational* reasons (no Hub rate limits, no Hub outages, faster authentication, in-VPC IAM auth) and don't expect cold-start latency to improve. If a vendor benchmark shows a big speedup from registry locality, ask what their image size and disk type are — for small images on faster disks, network can be the bottleneck. For large GPU container images on gp3, it isn't.

### Pre-baking the image into the AMI loses, by a lot

Variant C was the obvious next move: bake the container image into a custom AMI so it's already unpacked in `/var/lib/containerd/` when the instance boots. `image_download` should drop to nearly zero. Done.

It dropped to 2.6s. And the total cold start got **128 seconds slower** than just pulling from Docker Hub.

| Variant | image_download | node_bootstrap | vllm_init_cuda_ctx | TOTAL |
|---|---:|---:|---:|---:|
| A: Hub | 290s | 64s | 12s | 590s |
| C: Prebaked AMI (no FSR) | **2.6s** | **179s** | **135s** | **716s** |

The 287s I "saved" on `image_download` reappeared as 115s on `node_bootstrap` and 123s on `vllm_init`. Where did it go?

**Mental model: an AMI is a lazy S3 snapshot.** When you launch from an AMI, EC2 attaches a metadata-only volume — instantly available, but no blocks have been copied yet. The first read of any unwarmed block triggers an on-demand fetch from the snapshot's S3 backing store. Each first-touch costs ~10-50 ms; once paid, that block is normal-speed EBS forever.

For a stock EKS GPU AMI booting kubelet, the working set is small and the lazy-load tax is single-digit seconds. For a 9.5 GB pre-baked AMI where containerd, the kernel, and vLLM's import path *all* read from blocks that live inside the snapshot, the working set explodes. vLLM imports torch + CUDA runtime + flash-attn + a long dependency tree — probably 2-3 GB of first-touch reads. Aggregate across blocks, and you get ~238s of lazy-load tax spread across `node_bootstrap` and `vllm_init`.

The optimization didn't fail. The bytes still got delivered — just on the read path instead of the pull path. And the read path is slower per-byte, because each block round-trips to S3 individually instead of streaming from a registry.

The generalizable lesson: **when you change where bytes live, re-measure every stage, not just the one you targeted.** I would have caught this immediately if I'd watched the full pipeline; I missed it because I was anchored on `image_download`.

### FSR is the warm-pool analog for EBS

EBS Fast Snapshot Restore (FSR) is the AWS feature that pre-warms a snapshot's blocks per-AZ ahead of time. Volumes restored from an FSR-enabled snapshot in that AZ skip the lazy-load tax entirely — first reads hit at normal EBS speed.

The cost is **$0.75/hr/AZ flat**, independent of snapshot size. The auto-scaling analogy is exact: FSR is to EBS what warm pools are to ASGs. Both pre-pay idle capacity cost to skip first-launch latency. Both face linear multi-AZ economics — fast cold scale-out across 3 AZs costs 3× the FSR bill.

Variant D enabled FSR on the prebaked snapshot in us-east-1a:

| Stage | A: Hub | C: Prebake no-FSR | D: Prebake + FSR |
|---|---:|---:|---:|
| node_bootstrap | 64s | 179s | **46s** |
| image_download | 290s | 2.6s | 1.2s |
| vllm_init_cuda_ctx | 12s | 135s | **23s** |
| weight_load_gpu_mem | 40s | 31s | **4.6s** |
| **TOTAL** | **588s** | **716s** | **242s** |

I expected ~480s — the lazy-load tax clawed back, leaving the prebake roughly the same as Hub plus its raw image-pull savings. I got 242s. **FSR pre-warmed more than I modeled.** The whole gp3 root volume — including the slack space where the model-download init container writes weights — was pre-warmed. `weight_load_gpu_mem` collapsed from 40s to 4.6s, a stage I hadn't even considered FSR-relevant.

Break-even math: a saved cold-start second is worth roughly $0.16/3600 ≈ $0.0000444 in raw Spot g4dn cost. FSR's $0.75/hr/AZ pays back at about 47 cold starts per hour per AZ on infra cost alone. For Tier-0 user-facing inference where P99 latency cost dominates, the math is much more permissive. For development clusters that scale once a day, FSR is pure waste.

## Optimizing model weight loading

After image delivery, the next big chunk is moving 4.5 GB of weights from S3 onto the GPU. Variant D handles this in two steps: an init container does `aws s3 cp` to a gp3 emptyDir (~37s), then vLLM reads from disk and copies to GPU (~4.6s with FSR-warmed disk). Total: ~42s.

The instinct: skip the disk hop. Stream from S3 directly to GPU memory.

### The RunAI Streamer didn't beat warm disk

Variant E used the [RunAI Model Streamer](https://github.com/run-ai/runai-model-streamer) — `pip install runai-model-streamer runai-model-streamer-s3`, set `--load-format=runai_streamer --model=s3://...` on vLLM, and the streamer fetches safetensors from S3 to a CPU pinned buffer and onward to GPU without touching disk.

Expected: ~12s on the fused fetch+load step. Actual: 66s.

| Stage | D: Prebake + FSR | E: Streamer + FSR |
|---|---:|---:|
| model_download_s3 | 37s | 0s (no init container) |
| weight_load_gpu_mem | 4.6s | **66.1s** |
| **fetch + load total** | **41.6s** | **66.1s** |
| **GRAND TOTAL** | **242s** | **212s** |

The streamer was **24s slower** on the dimension I was trying to optimize. RunAI's published "3-5× speedup" is over a *naive* disk-stage-then-load pipeline. Variant D wasn't naive — FSR pre-warmed the gp3 volume to memory-bandwidth speeds (~1 GB/s for sequential reads), so the disk hop was nearly free. The streamer's ceiling is **single-instance S3 read throughput**, which I observed at ~70 MB/s sustained. That's an order of magnitude below FSR-warmed gp3.

Two reasons the streamer's parallelism didn't show up: AWQ weights are sharded into a small number of files (the streamer can't parallelize what isn't sharded), and I didn't tune `RUNAI_STREAMER_*` env vars (defaults may be conservative).

But variant E still beat variant D on total cold start by 30s. The win didn't come from raw fetch speed — it came from **dropping the init container entirely**. Kubernetes' init→main container handoff has its own overhead (image starts, lifecycle hooks, scheduler accounting) that the streamer architecturally sidesteps. The architectural change mattered more than the throughput change.

When is a streaming loader the right call? When you can't pay for FSR, when you're scaling horizontally enough that aggregate S3 bandwidth from many workers exceeds what FSR delivers per node, or when you're rotating models too often to bake each one into an AMI. There's also an operability footnote: 1 of 3 streamer runs hit 545s — bootstrap and `vllm_init` both 3× normal — apparently a partial-FSR-pre-warm spike. Variant D's three runs were tightly clustered. **Streamer + FSR is non-deterministic in a way Variant D wasn't.**

## Optimizing engine initialization

The last chunk to attack is `vllm_init_cuda_ctx` — 23s on variant D. This stage covers Python imports, CUDA context creation, engine setup, KV cache sizing, and JIT compilation of CUDA kernels (Triton, torch.compile, FlashInfer). Recent industry research ([Doubleword on SGLang cold start](https://www.doubleword.ai/), Q1-2026) cited torch.compile + Triton + FlashInfer JIT as the dominant cost on H100-class hardware, with 10-20s recoverable by snapshotting compile caches into the AMI and mounting them at runtime.

Variant F implemented exactly that: pin all five compile-cache directories via env vars, run vLLM once during AMI build to populate them, and bind-mount the cache directory at the same absolute paths at runtime.

```
/opt/vllm-cache/torchinductor   TORCHINDUCTOR_CACHE_DIR
/opt/vllm-cache/triton          TRITON_CACHE_DIR
/opt/vllm-cache/flashinfer      FLASHINFER_JIT_DIR
/opt/vllm-cache/vllm_compile    VLLM_CACHE_ROOT
/opt/vllm-cache/hf              HF_HOME
```

The cache populated correctly (228 files, 31 MB), no compile artifacts leaked outside the pinned dirs, and the AMI built clean.

`vllm_init_cuda_ctx` did not move. F's median: 23.8s. D's: 23.2s. Within noise, in the wrong direction.

Three reasons the prediction missed:

1. **vLLM 0.19's torch.compile path is mostly downstream of `vllm_init_cuda_ctx` as the harness draws the boundary.** The torch_aot_compile work fires between `weight_load_done` and `graph_capture_start`, not inside the init stage. Cache hits there don't surface in the metric I was measuring.
2. **The addressable JIT surface on T4 + AWQ is small.** FlashInfer doesn't fire (sm_75 < 8.0), TVM-FFI doesn't fire (no MoE), so only torch.inductor + Triton are caching anything. The 31 MB total is way below Doubleword's H100 + MoE setup, which had hundreds of MB of compile artifacts to skip.
3. **vLLM 0.19 is the wrong vLLM.** Q1-2026 vLLM (≥0.7) ships first-class cache persistence APIs that exposed more of the JIT surface to caching. Re-running this on newer vLLM is the obvious next step.

But variant F still earned its keep — just not where I expected. Look at variance across the 3 runs:

| Variant | Median | Stddev | Range |
|---|---:|---:|---:|
| D: Prebake + FSR | 241.6s | **119.6s** | 215-545s |
| F: + JIT cache | 221.2s | **5.0s** | 221-230s |

**24× reduction in run-to-run variability.** Variant D had one run that spiked to 545s (the partial-FSR-pre-warm hypothesis again). Variant F's three runs landed within 9 seconds of each other. For a production cold-start SLO that cares about p99 — which Tier-0 inference always does — a tightly-clustered 220s beats a flickering 215s±120s every time.

This reframes the value of the optimization. JIT cache pre-warm wasn't a peak-perf win; it was an **operability win**. That's a different argument with a different audience (SREs, not perf engineers), but it's no less valuable.

For the next material reduction in `vllm_init`: upgrade to vLLM ≥0.7 and re-run F, attack Python import time directly (~26s of the stage is just imports — lazy import patterns or vLLM monkey-patches), or wait for CUDA snapshot/restore (blocked on EKS GPU AMI shipping driver 580+).

## The decision tree

Distilled from the experiments above, with the caveat that the rankings assume something close to my environment (single-node, big image, T4-class GPU, vLLM 0.19, AWQ weights):

1. **Don't mirror to ECR for cold-start latency.** Mirror for operational reasons. For big images on gp3, expect ~5s saved, not ~100s.
2. **Don't pre-bake without FSR.** The lazy-load tax in adjacent stages will exceed the image-download savings. You'll be slower than just pulling from the network.
3. **Pre-bake + FSR is the production answer for big single-model workloads** if you scale out frequently. ~350s savings on a 9.5 GB image, predictable cost ($0.75/hr/AZ).
4. **Streaming model loaders are not a free win once FSR is paid.** Per-instance S3 throughput is below FSR-warmed gp3. Use streamers when you can't afford FSR, when you're S3-bandwidth-bound across many workers, or when you're rotating models too often to bake them.
5. **JIT cache pre-warm is a p99 optimization, not a p50 optimization.** On vLLM 0.19, expect tight variance reduction, not a faster median. Revisit on vLLM ≥0.7.
6. **Image slim-down and SOCI lazy-loading remain on the table** for multi-image / dynamic workloads where pre-bake + FSR doesn't scale to N AMIs.

## Three meta-lessons

**Stages leak.** Optimization labels lie. "Image delivery" optimizations can surface as `vllm_init` time. Always measure the full pipeline before and after, never just the stage you targeted. The pre-bake-without-FSR result was a 128-second regression masquerading as a 287-second win.

**Median hides operability.** Variance reduction is a feature, not a consolation prize. A 24× tighter distribution at the same median is what makes a cold-start budget plannable. P99-driven SLOs reward optimizations that the perf-bench community would call "no-ops."

**Cited speedups assume a baseline you may not have.** RunAI's 3-5× streamer claim is true — over a naive disk-stage-then-load. Doubleword's 10-20s JIT-cache savings is true — on H100 + MoE on newer vLLM. Neither survived contact with my specific stack unmodified. Always re-derive against your own environment before committing to an optimization, especially one that costs a recurring infrastructure bill.

## What's next

The next phases of this work move past single-node single-model cold start. Multi-model serving (bin-packing, tiered fallback) breaks the pre-bake + FSR strategy because you can't bake N variants into N AMIs economically. CUDA checkpoint/restore — once EKS GPU AMIs ship driver 580+ — promises to skip `vllm_init` entirely by snapshotting a warm process. Both will need their own measurement passes, and I expect at least one of my current intuitions to break again.

If you're running a different stack — bigger model, H100s, vLLM 0.7+, FlashInfer-eligible GPU — the rankings here will shuffle. The harness, Packer templates, and decision tree are all on GitHub. Try the method, not the answers.
