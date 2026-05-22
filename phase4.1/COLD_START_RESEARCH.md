# Cold-Start Research — Phase 4.1 → Phase 5+

**Audience:** Engineering candidate brief. Owner is a Senior EM (AWS Auto Scaling) building an inference platform on EKS + Karpenter, vLLM 0.19, T4 (g4dn.xlarge), gp3, EKS 1.30. Scope is the *next* set of cold-start levers beyond what's been measured.

**Already measured (out of scope):** Hub vs ECR vs prebake-AMI vs FSR. See `LEARNINGS.md`. Headline: prebake+FSR = 242s vs Hub 588s (-59%). gp3 unpack dominates network; lazy-load tax leaks across stages.

**Recency convention:** [2025+] = current; [2024] = recent; [2023 or earlier] = stale, flag explicitly.

---

## 1. Snapshot / Checkpoint-Based Fast Restore

This is the single most active cold-start research area in 2025. Treat this section as the headline category.

### 1.1 Modal GPU Memory Snapshots (production proof)

**What it does.** Modal lets a vLLM server boot once, run warmup (torch.compile, CUDA-graph capture, weight load to VRAM), then snapshot the *entire CUDA process state* — VRAM contents, kernel modules, captured CUDA graphs, host process memory — to an optimized binary. On cold start, the snapshot is restored instead of re-running init. Uses NVIDIA's CUDA driver 570+ checkpoint API under the hood. Requires vLLM Sleep Mode toggled on first.

**Realistic savings.** Modal's published vLLM/Qwen2.5-0.5B example: **45s → 5s (9x).** A vLLM cold start in 5s is the public state-of-the-art [Aug 2025]. Mistral 3 case study: 10x. These are vendor-measured, but Modal published the technique and reproducer code, so the numbers are auditable.

**Cost / complexity / blast radius.** High complexity. Snapshot images are large (model weights × ~1.05 for snapshot overhead). Per-driver-version brittleness (cuda-checkpoint requires display driver 550+, snapshots are not portable across driver versions). NCCL communicators are destroyed on suspend — multi-GPU restore needs reinit. Single-GPU only is the safe path today.

**Compatibility with current setup.** **Partial.** g4dn.xlarge has T4 + driver typically 535/550. To use cuda-checkpoint you need **driver 570+** (CUDA 12.8+). EKS GPU AMI bumps lag NVIDIA releases by months. Single-GPU T4 is fine functionally; the question is who hosts the snapshot artifact and whether the EKS node's Linux kernel supports CRIU's PID-namespace tricks. Plan to test in Phase 6 once driver is available.

**Primary sources.**
- Modal blog: https://modal.com/blog/gpu-mem-snapshots
- Modal Mistral case study: https://modal.com/blog/mistral-3
- Modal docs: https://modal.com/docs/guide/memory-snapshots
- vLLM RFC #34303 (in-tree native support, in design): https://github.com/vllm-project/vllm/issues/34303

**Phase to consider.** **Phase 6** (multi-model swap) — the same primitive enables both fast cold-start and per-tenant model swap. Before vLLM ships native support, you'd be writing the orchestrator yourself.

---

### 1.2 cuda-checkpoint + CRIU (the underlying primitive)

**What it does.** NVIDIA's `cuda-checkpoint` userspace tool drives a state machine: RUNNING → LOCKED → CHECKPOINTED. It drains submitted CUDA work, evicts device memory to host pinned buffers, and releases GPU resources so plain CRIU can checkpoint the rest of the process. Restore reverses it: CRIU restores the process tree, cuda-checkpoint allocates a fresh CUDA context and memcpy's VRAM back.

**Realistic savings.** Same envelope as Modal (it *is* the engine inside Modal). For a 7B model, restore time is dominated by host→device VRAM copy (~25 GB/s on PCIe Gen3 T4, so ~14 GB weights = ~0.6s, plus a few seconds of context recreation). Sub-10s for a 14B is plausible.

**Cost / complexity / blast radius.** Open-source, but you own the orchestration. CRIUgpu (research → upstreamed in CRIU 4.0 [2025]) is the cleanest current path. Multi-GPU works but tensor-parallel vLLM (TP > 1) is fragile — there's an open NVIDIA cuda-checkpoint issue noting failures on 2-GPU vLLM servers.

**Compatibility.** Same driver caveat as Modal. Single-T4 g4dn fits. Kernel must be 5.15+ for CRIU full feature set; EKS AMI is fine.

**Primary sources.**
- https://github.com/NVIDIA/cuda-checkpoint
- CRIUgpu paper [Feb 2025]: https://arxiv.org/abs/2502.16631
- vLLM forum thread on CRIU experiments: https://discuss.vllm.ai/t/using-criu-to-reduce-cold-start-latency-for-llm-tasks/639
- 2-GPU regression: https://github.com/NVIDIA/cuda-checkpoint/issues/27

**Phase.** Phase 6 prerequisite knowledge. Don't build it yourself if Modal's path or vLLM's RFC ships.

---

### 1.3 Container Snapshotters Beyond SOCI — Nydus and eStargz

**What they do.** Lazy-pull container snapshotters that mount image layers as a FUSE filesystem and fetch chunks on first read, like SOCI. Differences:

- **Nydus** (Dragonfly project, ByteDance origin): chunk-based content-addressable RAFS format. Supports P2P distribution (peers in cluster pull from each other, not the registry). Supports converting OCI/eStargz images automatically.
- **eStargz** (Stargz Snapshotter, Google origin): seekable gzip layers. Older, simpler, less actively developed than Nydus in 2025.

**Realistic savings.** Reported 5-10x faster *time to first container ready* in published benchmarks for "wide-fanout, narrow-readset" images. **Caveat for our case:** vLLM imports a wide swath of its dependency tree at startup (torch, flash-attn, xformers, transformers — 2-3 GB of reads). Lazy snapshotters only beat eager-pull when most bytes are *not* read. Phase 4.1's measured lazy-load tax (FSR result: 238s of cross-stage tax) suggests vLLM reads enough that lazy is a net loss for this workload — same lesson as the prebake-no-FSR variant.

**Cost / complexity.** Medium. Nydus needs a snapshotter daemon on every node (DaemonSet) and image conversion (OCI → Nydus). EKS-supported via custom AMI or bootstrap script.

**Compatibility.** Works on EKS 1.30, containerd. Conflicts with FSR mental model (FSR pre-warms blocks; Nydus avoids touching blocks). Pick one approach.

**Primary sources.**
- https://github.com/containerd/nydus-snapshotter
- https://nydus.dev/
- https://github.com/containerd/stargz-snapshotter

**Phase.** Phase 6 (multi-image / multi-model). For single-image Phase 5, prebake+FSR (already proven) wins.

---

### 1.4 Knative scale-to-zero

**What it does.** Pod-level "scale to zero" with activator-based queueing during cold-start. When a request arrives at zero, the activator buffers, scales pod up, releases. Independent of any GPU magic — just plumbing.

**Realistic savings.** Doesn't reduce cold-start *latency*; it converts idle cost to first-request latency. With a 588s cold-start (Hub) or 242s (FSR), **scale-to-zero is unusable for Tier-0 user-facing inference unless paired with a fast-restore primitive (Modal-style snapshot or sleep mode).** Knative's autoscaler is fine; the cold-start engine underneath is the actual problem.

**Compatibility.** Works on EKS. Overlaps with KEDA (already deployed) — Knative's unique value is the activator/buffer; KEDA scales pods but doesn't buffer requests. Phase 4 already chose KEDA; not worth rewinding.

**Primary sources.**
- https://knative.dev/docs/serving/autoscaling/
- Alibaba's Fluid + Knative case study (90s → 20s for 17GB model via distributed cache): https://www.alibabacloud.com/blog/an-automatic-scaling-solution-for-llm-inference-services-based-on-knative_602223 [2025]

**Phase.** Skip unless migrating off KEDA. The Fluid distributed-cache idea (1.5) is the interesting transferable piece.

---

### 1.5 Fluid / distributed model-weight cache

**What it does.** Fluid (CNCF sandbox) deploys a distributed read-only cache of model weights across cluster nodes (typically with Alluxio or JuiceFS as the backend). New pod cold-starts read weights from the in-cluster cache instead of S3, parallelized across many cache nodes. Alibaba reported 17 GB model load 90s → 20s.

**Compatibility.** Works on EKS. Needs persistent cache pods (idle cost), Alluxio operator. **Tension with FSR:** if FSR already pre-warms the gp3 blocks the model is on, Fluid is redundant. Fluid wins when you can't FSR (e.g., dynamic model swap).

**Primary source.** https://www.alibabacloud.com/blog/601454

**Phase.** Phase 6.

---

## 2. Pre-loaded Weight Strategies (brief — deep dive next session)

### 2.1 RunAI Model Streamer [2024 GA, integrated into vLLM `--load-format runai_streamer`]

**What.** Concurrent multi-stream weight reader. Pulls safetensors shards from S3/SSD/local with N parallel HTTP/file streams directly into pinned host buffers, then async-copies to GPU. Bypasses the HF safetensors loader's serial path.

**Savings (NVIDIA/Run:AI bench, Llama-3-8B, A10G):** S3 → vLLM ready in **23.18s**, vs HF safetensors loader **~50s**. On gp3 SSD: 35.08s vs ~70s. Roughly **2x faster weight load**.

**Compatibility.** Native vLLM flag. T4/g4dn fine. **Highly compatible with current setup.** Drop-in for the 35-41s `model_download_s3` + 31-40s `weight_load_gpu_mem` stages — those are the next addressable cluster of seconds after FSR.

**Source.** https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer/

**Phase 5. High priority.**

### 2.2 CoreWeave Tensorizer [2023, still maintained]

**What.** Custom weight serialization format + streaming loader. Pre-2024 was the SOTA; Run:AI Model Streamer benches now beat it on most setups. Requires re-serialization step (your safetensors → tensorizer format).

**Compatibility.** vLLM supports `--load-format tensorizer`. Re-serialization is operational overhead; Run:AI Streamer is generally a better choice today.

### 2.3 vLLM Sleep Mode (multi-model swap, not cold-start)

**What.** Keep vLLM process alive, free GPU memory by offloading weights to CPU (Level 1) or discarding entirely (Level 2). Wake = reload from CPU/disk. **18-200x faster than full restart** [Oct 2025 vLLM blog]. Different problem from cold-start — solves *tenant switch on a warm GPU*, not *first pod up*. Critical for Phase 6 multi-model.

**Source.** https://vllm.ai/blog/sleep-mode

---

## 3. Engine-Level Startup Optimizations

### 3.1 vLLM `--enforce-eager` (skip CUDA graph capture)

**What.** Skips the 67-graph capture phase. Saves ~10s on T4 [vLLM docs]. **Cost:** 10-20% throughput regression at decode (kernel launch overhead). Tradeoff: spend 10s now to save 10-20% on every token forever — bad trade for steady-state, good trade for bursty-with-cold-start workloads.

**Compatibility.** One flag. Already on the team's radar (LEARNINGS shows `cuda_graph_warmup` ~9.5s).

**Recommendation.** Don't enable. The 9.5s is small; the throughput cost compounds.

**Source.** https://docs.vllm.ai/en/stable/design/cuda_graphs/

### 3.2 vLLM torch.compile cache pre-warm [next-next session, mentioned for completeness]

**What.** Persist `~/.cache/vllm/torch_compile_cache` across pods (PVC, S3, baked into image). vLLM saves FX graphs + Triton kernels here on first run; warm starts read instead of recompile. Tensorfuse measured 294s → 82s (-72%) by caching the compile dir + model weights together [2025].

**Compatibility.** Works today with vLLM 0.6+ (the V1 engine is the default in 0.19). Cache must match GPU + CUDA + PyTorch version exactly — invalidate on any of those changes.

**Source.** https://blog.vllm.ai/2025/08/20/torch-compile.html, https://tensorfuse.io/docs/blogs/reducing_gpu_cold_start

**Phase 5. High priority.** (Already on roadmap.)

### 3.3 vLLM `-O0` / `-O1` compilation levels [2025]

**What.** New CLI knob (`-O0` to `-O3`) trading compile time for runtime perf. `-O0` = no compile, fastest start, slowest decode. `-O1` = piecewise CUDA graphs only. Different from `--enforce-eager` (which is a coarser hammer).

**Source.** https://github.com/vllm-project/vllm/issues/20283

### 3.4 SGLang vs vLLM cold start

**Savings.** Reported 58s vs 62s for similar models — within noise. SGLang is *not* meaningfully faster to start. Its win is throughput on structured outputs.

### 3.5 TensorRT-LLM cold start

**Savings.** **Negative.** TRT-LLM compile path measured **~28 minutes** in one bench. AOT compile is offline; the shipped engine plan loads fast (~10s), but you pay weeks of operational overhead per model+config. Skip for Phase 5.

**Source.** https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/

---

## 4. GPU/Host-Level

### 4.1 GPU "hibernation" (H100/H200 Hopper)

**What.** Hopper-only feature where the GPU retains VRAM contents through a power-state transition. Doesn't help T4 — Turing has no equivalent. Skip until P5/Hopper.

### 4.2 NVIDIA MPS (Multi-Process Service) persistent daemon

**What.** Long-running daemon that lets multiple processes share one GPU context, avoiding context-creation overhead per pod. Saves ~1-3s of CUDA context init. Marginal vs the 242s we're at.

### 4.3 MIG (Multi-Instance GPU)

**Hopper/Ampere only.** N/A on T4.

---

## 5. Predictive / Proactive Scaling

### 5.1 QLM Request Wait-Time estimator [SoCC 2024]

**What.** Predicts queue wait via continuous-batching statistics. Used for SLO-aware scheduling, not cold-start. Improves SLO attainment 40-90% by reordering, not by reducing cold-start latency.

**Phase 7.** (Already on roadmap per CLAUDE.md.)

### 5.2 Mooncake early rejection [arXiv 2407.00079]

**What.** Predicts whether a request will meet SLO at admission; rejects early if no. Doesn't reduce cold-start latency — it prevents wasted cold-start work on requests that would miss anyway.

### 5.3 Traffic forecasting → pre-warm

**What.** Time-series forecast (Prophet, ARIMA, simple EWMA) on QPS to scale up *before* demand. Reduces *count* of cold starts, not their *latency*. **Honest answer:** this is just "Karpenter consolidation policy + a min-replicas knob driven by forecast." HPA's `behavior.scaleUp.policies` is enough for most cases. Don't over-engineer.

**Phase 5 / 7.**

---

## 6. Disaggregated Prefill/Decode (cold-start interaction)

### 6.1 DistServe / Mooncake / Sarathi-Serve

**Cold-start angle.** Smaller per-role pods (prefill-only or decode-only) carry less code path → faster init? **Not really.** Both pod types still load full model weights. The cold-start cost is dominated by weight load, not by the active code path. Disaggregation buys throughput/latency under load, not faster cold-start.

**One nuance.** Decode pods can have much smaller KV-cache budgets if you offload to CPU/SSD (Mooncake's KVCache pool). That can shrink GPU memory footprint, which lets prebake+FSR pre-warm a smaller working set. Indirect effect; small in practice.

**Phase 8 (already roadmapped).**

**Sources.**
- Mooncake: https://arxiv.org/abs/2407.00079
- Hao Lab DistServe retrospective [2025]: https://haoailab.com/blogs/distserve-retro/

---

## 7. Serverless Inference Patterns

### 7.1 SageMaker MME with GPU

**What.** Triton + dynamic model load/unload from S3 to a shared GPU. First-invocation cold-start = S3 download + load (10-60s for a 7B). Subsequent invocations on the same instance hit warm cache.

**Compatibility.** Different abstraction from EKS+vLLM; you'd be migrating, not adding. Worth evaluating if Phase 6 multi-tenant is high priority and you want managed fleet vs DIY.

**Source.** https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html

### 7.2 AWS Lambda + GPU?

**Status [late 2025].** Lambda has **no native GPU resource type.** "Lambda Managed Instances" announced late 2025 lets Lambda run functions on customer-selected EC2 (p4d/g5/etc.) — but at that point it's just EC2 with a Lambda-style API. No magic cold-start primitive. Skip.

### 7.3 Modal / Banana / Replicate "<1s cold start"

**Trick (decoded).** Modal's number is real — it's GPU memory snapshots (§1.1). Banana's pre-2025 numbers were from "warm pool of pre-loaded GPUs with custom weight format" — the marketing word "cold start" hides that they keep warm capacity. Replicate's "<1s" applies only to pre-hosted models on warm pool; custom deploys hit normal container cold-start. **The honest summary: snapshot-based restore is the only technique that's actually sub-second from a *truly cold* state for a 7B+ model.**

---

## 8. Container / Kubernetes Lifecycle

### 8.1 In-Place Pod Vertical Scaling [GA in K8s 1.35, Dec 2025]

**What.** Resize CPU/memory of running pod without restart. Combined with `kube-startup-cpu-boost`, you give the pod 4 CPU at startup (faster Python imports, faster torch.compile), then scale down to 1 CPU at steady state.

**Realistic savings.** Google measured ~9s → ~4.5s on JVM apps (~2x). For vLLM on g4dn.xlarge, the pod *already has all 4 vCPUs* (single-pod-per-node). **Probably zero savings on our setup** — boost only helps when the pod is contesting CPU with other pods.

**Compatibility.** EKS 1.30 has the feature gate at beta. Need 1.33 for full UX. **Not worth doing on g4dn.xlarge.** Reconsider on bin-packed multi-pod nodes (Phase 6).

**Sources.**
- https://kubernetes.io/blog/2025/12/19/kubernetes-v1-35-in-place-pod-resize-ga/
- https://github.com/google/kube-startup-cpu-boost

### 8.2 Karpenter "warm pool" / standby nodes

**Status.** **Not natively supported** [as of 2025]. Open issue #4354 since 2023. Workarounds: a NodePool with `minNodes: N` of small idle GPU instances, paying the always-on cost. EKS Auto Mode has node pools but the same cost applies.

**Compatibility.** Works as a min-replicas pattern. **Equivalent to FSR economics** — pre-pay capacity in exchange for skipping scale-out latency. The combination "1 always-on g4dn + KEDA scale-up to 4" is the cheap version of warm-pool.

**Source.** https://github.com/aws/karpenter-provider-aws/issues/3798

**Phase 5/7.** Already on the team's list.

### 8.3 Init container / sidecar startup ordering

**K8s 1.28+ sidecar containers** (initContainer with `restartPolicy: Always`) let model-download run *in parallel* with main container's bootstrap if you wire it right. Doesn't reduce critical-path time on Phase 4.1's measured stages (model download is already fast on FSR'd gp3) but worth knowing for Phase 6.

---

## 9. Hardware / Instance Choice

### 9.1 NVMe instance store vs gp3

**What.** g4dn.xlarge has 125 GB local NVMe SSD (instance store, not EBS). It's a separate device that:
- **Pro:** ~3-4 GB/s read throughput vs gp3's 125 MB/s default (gp3 maxes at 1 GB/s tuned). 8-32x faster for weight load.
- **Con:** Ephemeral. Disappears on stop/terminate. Empty at boot — you must populate it (extra step in init container).
- **Con:** No FSR equivalent. Pay the populate-from-S3 cost on every cold start.

**Realistic savings on phase 4.1's stages.** `weight_load_gpu_mem` 40s → ~5-10s if served from NVMe (matches RunAI Streamer's gp3-vs-IO2 delta of 35s vs 28s). Combined with Run:AI Streamer (§2.1), plausible **35-40s shaved off** the post-FSR baseline.

**Compatibility.** Native to g4dn. Need to mount it (initContainer or Bottlerocket bootstrap), copy model from S3 into it, point vLLM `--download-dir` at it. **Conflicts with FSR thinking** — FSR pre-warms gp3, NVMe is separate. Pick one.

**Phase 5. High priority.** This is one of the cheapest wins available.

### 9.2 g6 vs g4dn (instance refresh)

**g6 (L4 GPU):** 2x perf vs T4 for inference per AWS. Ada Lovelace arch supports better FP8. Cold-start envelope similar. Worth evaluating for Phase 5 throughput, not for cold-start specifically.

**Source.** https://aws.amazon.com/ec2/instance-types/g6/

### 9.3 AWS Capacity Blocks for ML

Reservation-based capacity for H100/A100 with predictable scheduling. Doesn't reduce *cold-start* latency, but eliminates Spot-interruption cold-start *frequency*. Phase 7 hardening.

---

## 10. Novel / Emerging 2025 Techniques

### 10.1 Engine-agnostic model hot-swap via checkpoint [arXiv 2511, late 2025]

Recent paper proposing checkpoint/restore as a mechanism to swap models on a single GPU without bringing the engine down. Same primitive as §1.1, framed for multi-tenant cost-efficiency.

### 10.2 vLLM `/suspend` `/resume` HTTP endpoints [in-flight RFC]

vLLM RFC #34303 proposes native suspend/resume endpoints distinct from sleep/wake. **Watch this RFC for Phase 6.** Once shipped, a lot of the cuda-checkpoint complexity becomes a vLLM API call.

### 10.3 InferX

Closed-source, claims similar Modal-style snapshot results for vLLM. No public benchmarks; mentioned in the vLLM RFC as evidence the technique works generally.

---

## Top 5 Not-Yet-Explored Levers Worth Testing (Phase 5)

Ranked by expected savings × ease, on top of the current 242s prebake+FSR baseline.

| # | Lever | Expected savings | Effort | Risk |
|---|---|---|---|---|
| 1 | **Run:AI Model Streamer** (`--load-format runai_streamer`) | **~30-40s** off `weight_load_gpu_mem` + `model_download_s3` (40s+35s → ~25s combined per NVIDIA bench) | Low — one vLLM flag + S3 IAM | Low; native vLLM support |
| 2 | **NVMe instance store for model cache** (g4dn has 125GB) | **~30-35s** off weight load if combined with #1; ~25s alone | Medium — initContainer to copy S3→NVMe, hostPath mount | Low; ephemeral disk so re-populate every cold start |
| 3 | **torch.compile cache persistence** (PVC or baked into AMI) | **~10-20s** off `vllm_init_cuda_ctx` (post-FSR baseline is 23s, much of which is import+compile) | Medium — version-pin everything (CUDA, torch, vLLM, GPU) | Medium; cache invalidation footguns |
| 4 | **Slim vLLM image (4GB instead of 9.5GB)** | **~5-8s** post-FSR (already small win there); **~150s pre-FSR**. Pairs well with Run:AI Streamer if you ever drop FSR for cost reasons. | Medium — multi-stage Dockerfile, drop dev deps | Low |
| 5 | **GPU memory snapshots (Modal-style or vLLM RFC #34303 when it lands)** | **Down to ~5-10s total cold start** for steady-state image; transformative | High — driver upgrade required, orchestrator code, single-GPU only initially | High; bleeding edge in 2026 |

**Stack-up estimate:** items 1+2+3 together could plausibly take 242s → ~160-170s without any snapshot tech. Item 5 alone could take it to <30s but is a Phase 6 effort.

---

## Sounded Promising But Probably Won't Help (For This Setup)

- **`--enforce-eager`.** Saves ~10s now in exchange for 10-20% steady-state token throughput loss forever. Bad trade except for one-shot batch jobs.
- **Knative scale-to-zero.** Doesn't reduce cold-start latency — it just defers it to the user. With 242s cold-start, this is unusable for interactive workloads. Pair with snapshot tech or skip.
- **kube-startup-cpu-boost.** Designed for CPU-contested pods. On 1-pod-per-node g4dn, the pod already owns all 4 vCPUs. Zero savings.
- **MPS persistent daemon.** Saves seconds, not tens of seconds. Marginal vs the budget we're working in.
- **TensorRT-LLM.** ~28-minute compile path. Strict no for any iterative dev flow. Reconsider only when shipping a frozen model to production at scale.
- **MIG slicing.** Hopper/Ampere only. T4 doesn't have it.
- **GPU hibernation.** Hopper only. T4 doesn't have it.
- **Lazy snapshotters (Nydus / eStargz / SOCI) for *single-image* workloads.** vLLM imports too much of its dependency tree at startup; lazy loaders win when most bytes are unread, which isn't this case. Already evidenced by the prebake-no-FSR result. Reconsider in Phase 6 for *multi-image* workloads.
- **SGLang migration.** ~4s cold-start delta vs vLLM is within noise. Migrate if you want SGLang's structured-output features, not for cold-start.
- **Disaggregated prefill/decode for cold-start.** Solves a different problem (steady-state goodput). Cold-start cost is ~the same per pod.
- **Predictive autoscaling (forecast → pre-warm).** Reduces *count* of cold starts, not their *latency*. The actual latency engineering is upstream of any forecaster.

---

## Open Questions / Ambiguities

1. **Does Run:AI Model Streamer interact well with FSR?** FSR pre-warms gp3 blocks; Streamer streams from S3 (or local). If you've already FSR'd the gp3 volume that holds the model, Streamer is reading already-warm bytes — savings should still apply (concurrent read paths), but unmeasured. **Worth a 1-day experiment in Phase 5.**

2. **Will EKS 1.30 / EKS GPU AMI support driver 570+ in time for Phase 6?** AWS's GPU AMI ships behind NVIDIA. If 570+ isn't available, custom AMI with packer (already have the toolchain from FSR work) is the path.

3. **Does NVMe instance store get FSR-equivalent zero-warmup behavior?** No — instance store is empty at boot. So NVMe + Run:AI Streamer is a *replacement* for prebake+FSR (skip EBS warmup entirely, populate NVMe from S3 fast), not a complement. Worth modeling the cost: (FSR @ $0.75/hr/AZ) vs (NVMe init time × cold-start frequency).

4. **vLLM RFC #34303 (CUDA checkpoint/restore) shipping date?** Watch the issue. If this lands by Phase 6, it eliminates the need to roll your own CRIU+cuda-checkpoint pipeline.

5. **Phase 4.1's prebake+FSR result vs the prior memory `project_ami_caching_findings.md` (FSR saves ~7s).** That memory was for a different image. The 9.5GB vLLM image's working set is what made FSR transformative. **Generalizable rule:** lazy-load tax scales with first-touch-byte working-set size. Worth instrumenting on every new image to know whether prebake+FSR earns its keep.

6. **Does FSR pre-warm extend across the gp3's slack space?** Phase 4.1 unexpectedly saw `weight_load_gpu_mem` drop 40s → 4.6s when FSR was on, even though weights live in an emptyDir written *after* boot. Hypothesis: FSR pre-warms the entire underlying volume, including not-yet-allocated blocks, so the emptyDir's writes go to already-warm extents. Confirm with AWS docs / a dedicated test.

---

## Sources Index

- Modal GPU Memory Snapshots: https://modal.com/blog/gpu-mem-snapshots
- Modal Mistral 3 case study: https://modal.com/blog/mistral-3
- vLLM RFC #34303 (CUDA checkpoint/restore): https://github.com/vllm-project/vllm/issues/34303
- vLLM Issue #33930 (GPU snapshot feature): https://github.com/vllm-project/vllm/issues/33930
- vLLM Sleep Mode blog: https://vllm.ai/blog/sleep-mode
- vLLM torch.compile blog: https://blog.vllm.ai/2025/08/20/torch-compile.html
- vLLM Q3 2025 roadmap: https://github.com/vllm-project/vllm/issues/20336
- NVIDIA cuda-checkpoint: https://github.com/NVIDIA/cuda-checkpoint
- CRIUgpu paper (arXiv 2502.16631): https://arxiv.org/abs/2502.16631
- NVIDIA Run:AI Model Streamer blog: https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer/
- Run:AI Model Streamer benchmarks PDF: https://pages.run.ai/hubfs/PDFs/White%20Papers/Model-Streamer-Performance-Benchmarks.pdf
- Tensorfuse cold-start blog: https://tensorfuse.io/docs/blogs/reducing_gpu_cold_start
- Mooncake paper (arXiv 2407.00079): https://arxiv.org/abs/2407.00079
- QLM SoCC '24 paper: https://dl.acm.org/doi/10.1145/3698038.3698523
- Nydus snapshotter: https://github.com/containerd/nydus-snapshotter
- Stargz/eStargz snapshotter: https://github.com/containerd/stargz-snapshotter
- Knative autoscaling docs: https://knative.dev/docs/serving/autoscaling/
- Alibaba Fluid + Knative case study: https://www.alibabacloud.com/blog/an-automatic-scaling-solution-for-llm-inference-services-based-on-knative_602223
- K8s 1.35 In-Place Pod Resize GA: https://kubernetes.io/blog/2025/12/19/kubernetes-v1-35-in-place-pod-resize-ga/
- kube-startup-cpu-boost: https://github.com/google/kube-startup-cpu-boost
- Karpenter warm-pool issue: https://github.com/aws/karpenter-provider-aws/issues/3798
- SageMaker MME GPU docs: https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html
- DistServe retrospective (Hao AI Lab, 2025): https://haoailab.com/blogs/distserve-retro/
- Lambda SnapStart cold start blog: https://aws.amazon.com/blogs/compute/optimizing-cold-start-performance-of-aws-lambda-using-advanced-priming-strategies-with-snapstart/
- Spheron vLLM/SGLang/TRT-LLM benchmarks 2026: https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/

---

# Round 2 Additions (May 2026)

Six months on from Round 1. Recency bias intentionally tilted toward late-2025 / early-2026 material. Round 1 leaned on Modal Aug 2025 + Run:AI 2024 + vLLM Sleep Mode (Oct 2025) as anchors; this round adds Doubleword/SGLang Q1 2026, Baseten BDN (Apr 2026), R-Fork (Dec 2025), CRIUgpu production status, and the vLLM CUDA checkpoint RFC actually opened (Feb 2026).

## R2.1 SGLang fast cold start (Doubleword, Q1 2026) — flagship new entry

**Source:** https://blog.doubleword.ai/fast-sglang-starts. SGLang v0.5.10 on B200 (192 GB), Qwen3.5 122B MoE.

**Headline:** **695s cold → 9.6s warm-snapshot restore** (≈70x). Theoretical floor they identify: 1.8s. This is the most thorough public dissection of cold-start anatomy published to date — more useful as a *taxonomy* than as numbers (B200/192GB/MoE-122B is nothing like our T4/16GB/7B world).

The post decomposes the journey into four stacked techniques. I'll list each as its own entry below since the structure of the optimisation is what transfers, not the numbers.

### R2.1a Kernel cache persistence (Doubleword, SGLang)

**What.** Persist *all* JIT/AOT compilation caches across pods, not just torch.compile's. Specifically: FlashInfer kernels, Triton kernels, torchinductor, **TVM FFI cache**, and **FlashAttention CUTE DSL cache**. Doubleword found that even with a "warm" torch.compile cache, three other compilers run on first request and cost real seconds.

**Savings:** 695s → 88s (cold→warm). The bulk of this is not torch.compile — it's the kernel-library JITs Round 1 didn't enumerate.

**Marketing flag.** Vendor-published, single-config, but they show per-cache breakdowns and the methodology is reproducible. Trust the *technique*; the magnitude is workload-specific.

**Compatibility (vLLM 0.19 / T4 / EKS 1.30 / gp3).** **Partial / requires newer vLLM.** vLLM 0.19's torch.compile cache work (covered in Round 1 §3.2) addresses torchinductor only. FlashInfer is shipped in vLLM but cache persistence around it isn't first-class. Worth checking whether vLLM ≥0.7 has equivalent "cache everything" knobs — if not, Round 1's torch.compile-cache-only estimate (~10-20s) is *understated* and there are more seconds available behind FlashInfer/Triton caches we weren't counting.

**Phase 5.** Promote priority — see "Top-5 reshuffle" below.

### R2.1b CRIU process-snapshot for serving engines (Doubleword)

**What.** Doubleword's path uses CRIU + cuda-checkpoint to snapshot the SGLang process *after* warmup. Strips weights and KV-cache out of the snapshot blob (those go to a separate fast path) leaving a 6.6 GB process image. This is the same primitive Round 1 §1.1/§1.2 covered (Modal/CRIUgpu) but published with full restore-time decomposition: 88s warm-start → 32.1s after CRIU restore.

**Marketing flag.** None — they show the breakdown.

**Compatibility.** **Driver 580+** (Doubleword's number — higher than Round 1's "570+"). Confirms that the GPU-driver bar keeps rising; EKS GPU AMI lag is a real Phase 6 blocker. Tracking "what driver does my AL2023-GPU AMI ship today" is now a recurring dependency.

### R2.1c Containerd + CRIU patches (zero-copy mmap restore)

**What [Round 2 NEW — not in R1].** Two patches to containerd that cut container-setup overhead 6s → 3s, plus a CRIU patch that mmap's checkpoint pages directly instead of copying. **32.1s → 24.5s.**

**Why this matters for us.** Phase 4.1 measurements show ~6s of containerd/sandbox overhead per cold start. Half of it appears to be addressable by upstream patches that may or may not have landed in the containerd version EKS 1.30 ships. Worth checking containerd release notes; not worth patching ourselves.

**Phase 6 watch-item, not Phase 5 work.**

### R2.1d Torch Memory Saver (TMS) daemon with hugepage staging

**What [Round 2 NEW — biggest novel technique].** Daemon that pre-allocates 2MB transparent hugepages, stages weight tensors into pinned host memory, then *pipelines* `cudaHostRegister` with the H2D transfer instead of doing register-then-copy serially. Reported effective GPU-load throughput **38 GB/s on PCIe Gen5** (theoretical 64 GB/s).

**Savings claim:** weight reload 31s → 19s → 3.1s (the two arrows are "naive TMS" vs "TMS + pipelined registration").

**Marketing flag.** Vendor numbers, single workload, B200/Gen5 PCIe (~64 GB/s) — translates poorly to our T4/Gen3 (~16 GB/s peak). At Gen3, 14 GB of weights ÷ 38/64 × 16 = ~2.4 GB/s ceiling, so a 7B model's host→device leg is 5-6s minimum even with TMS. **Useful but bounded by PCIe gen.**

**Compatibility.** Open-source (`torch_memory_saver` PyPI). SGLang-integrated; vLLM integration is via the same library but less polished. Hugepage allocation requires `vm.nr_hugepages` sysctl — node-level config, EKS user-data territory. **vLLM 0.19 likely too old** to have the integration; check 0.7+/main.

**Phase 5/6.** This is the lever Round 1 didn't have. Listed alongside Run:AI Streamer in the new Top-5.

### R2.1e Round 1 numbers reframed by R2.1

The 695s → 9.6s decomposition validates Round 1's intuition that **weight load and JIT-compile dominate post-image-pull cold start**, but it adds a new sub-bucket Round 1 missed: **kernel-library compilation caches** (FlashInfer/Triton/TVM-FFI), separate from torch.compile. On our 242s post-FSR baseline this could be 5-15s of *unattributed time* inside `vllm_init_cuda_ctx`.

---

## R2.2 Tensor R-Fork (LMSYS, Dec 2025)

**Source:** https://www.lmsys.org/blog/2025-12-10-rfork/. SGLang.

**What.** GPU-Direct RDMA between a *running* SGLang instance and a *new* one. New pod's GPU pulls weights peer-to-peer from the warm pod's GPU memory, bypassing CPU/PCIe-host entirely. Per-GPU-pair direct device-to-device.

**Savings claim.** DeepSeek-R1 load time "from several minutes to mere seconds" + ~600 GB DRAM/disk savings (no more local cache copy).

**Marketing flag.** Big-model showcase; small-model benefit unclear.

**Cost / complexity / blast radius.** **High.** Requires GPUDirect RDMA fabric — InfiniBand or AWS EFA. Single-instance EKS cold start has no peer to fork from. **Useless for first-pod-up cold start; only useful for "scale-out from N to N+1 when N≥1."**

**Compatibility (g4dn.xlarge / EKS).** **Incompatible.** g4dn has no EFA, no RDMA. p4d/p5 have EFA. Skip.

**Phase 8 (disaggregated) reading material only.**

---

## R2.3 Baseten Delivery Network — BDN (Apr 2026)

**Source:** https://www.baseten.co/blog/how-the-baseten-delivery-network-bdn-makes-cold-starts-fast/

**What.** Three-tier weight cache: node-local NVMe → in-cluster peer cache (consistent-hash ring) → mirrored origin. Plus image streaming (model load begins before image pull finishes).

**Savings claim.** >2 GB/s sustained to H100 nodes. 32 GB streaming-enabled image pulled in 15.851s. Snapshot path brings 20 GB models online in <10s (already noted in Round 1).

**Marketing flag.** Numbers are H100-class with node-local NVMe + RDMA peers. The architecture concept (peer-cache via consistent hashing) is the transferable insight; the absolute numbers are not portable to g4dn/gp3.

**Compatibility.** Conceptually overlaps with Fluid (R1 §1.5) and the AWS Mountpoint-S3 + cluster cache pattern. Implementation effort = building it ourselves. **Out of reach for Phase 5; useful Phase 6 reference architecture.** [Round 2] — confirms the multi-tier cache pattern is now the consensus production architecture across Baseten / Modal / Alibaba Fluid.

**Phase 6.**

---

## R2.4 vLLM Sleep Mode — promoted [Round 2]

Round 1 §2.3 covered Sleep Mode but tagged it "different problem from cold-start." Re-evaluating after the Oct-2025 blog and the Apr-2026 vLLM release notes:

**New numbers / framing [Round 2].** Sleep Mode quote-as-published: "61–88% faster *first inference* vs cold start" — that's a cold-start metric, not just a swap metric. If you keep one warm vLLM process per node and sleep/wake to switch *which model is hot*, you turn a Phase 6 multi-model problem into a sub-second problem *and* you eliminate cold-start for any model that's been served before since boot.

**Compatibility.** vLLM 0.19 has Sleep Mode Level 1 (CPU offload) and Level 2 (discard). **Drop-in for Phase 6.** Multi-model routing on the gateway side becomes the bigger lift.

**Phase 6 — strong candidate.** [Round 2] elevation: from "covered in passing" to "primary multi-model strategy."

---

## R2.5 vLLM RFC #34303 status [Round 2 update]

**Status as of May 2026:** **Opened Feb 11, 2026. Not merged.** Design proposes a `CheckpointOrchestrator` that wraps CRIU-CLI calls, handles POSIX-semaphore cleanup pre-dump, persists vLLM metadata (TP/PP topology, port, driver version), and exposes `POST /resume`. Modal cited in the RFC as proof point (10x), InferX named as another implementor.

**Round 1 said:** "Watch this RFC for Phase 6."
**Round 2 says:** Same. No movement toward merge in 3 months. Continue to assume DIY or Modal-managed if you want this primitive in 2026.

**Source:** https://github.com/vllm-project/vllm/issues/34303

---

## R2.6 CRIUgpu production maturity [Round 2 update]

**Round 1 said:** "research → upstreamed in CRIU 4.0 [2025]."
**Round 2 confirms:** CRIUgpu now in production at MemVerge and Modal. Podman ships CRIUgpu support natively. Benchmarked across GPT2, Llama3, Gemma 2, StableLM. **Status: production-ready for single-GPU; multi-GPU still rough.**

**Implication for us:** the open-source path (CRIU 4.0 + cuda-checkpoint) is now a credible *alternative* to paying Modal for the snapshot service. Cost trade-off is build-effort vs. monthly Modal bill. For a learning-lab / interview-portfolio context, building it ourselves on Phase 6 has educational ROI even if economics favor Modal.

**Driver bar** has crept up: Doubleword cites 580+, NVIDIA cuda-checkpoint docs cite 570+ minimum, 580+ for "device migration." EKS GPU AMI as of May 2026: confirm current shipping driver before betting Phase 6 on this.

**Source:** https://www.devzero.io/blog/gpu-container-checkpoint-restore

---

## R2.7 NVMe-instance-store vs FSR-warmed-EBS — head-to-head search result [Round 2]

**Open question from Round 1 #3:** is there a published head-to-head?

**Search verdict:** **No direct apples-to-apples public benchmark.** Closest evidence:
- WEKA blog (vendor, biased) claims a network FS at 3.5s model load vs ~5s on local NVMe — i.e., **NVMe local is *not* automatically the fastest tier when other caches exist.**
- Generic AWS NVMe-vs-EBS shows 10x+ raw IO advantage to instance store, but that's writes; reads on FSR-warmed gp3 hit page cache at near-DRAM speed once warm.
- The Round 1 thesis (NVMe + Run:AI Streamer ≈ FSR + Run:AI Streamer for our 7B/T4 case) **remains untested in public.** This is still a 1-2 day experiment worth running in Phase 5.

---

## R2.8 vLLM "Zero Reload" / cold-start work in Q1 2026 release notes

**What landed [Round 2].** vLLM Q1 2026 roadmap (#32455) and Q2 draft (#39749) emphasize torch.compile cache improvements, MoE cold-start fixes (`fast_moe_cold_start` PR #33735, undone in #35475 for torch≥2.11), and a perf dashboard that *separates* cold and warm start times. The official vLLM perf-tracking shifted to break out cold/warm as first-class metrics — an institutional signal that this is now a tracked priority, not a side issue.

**Implication.** Pin to vLLM **≥ Q1-2026 release** (whatever that is — likely 0.7.x+) before doing Phase 5 cold-start work. v0.19 will under-represent achievable savings.

---

## What changed since Round 1

| Item | Round 1 status | Round 2 status |
|---|---|---|
| vLLM RFC #34303 (CUDA checkpoint) | "in design" | **Opened Feb 2026, not merged. No movement.** |
| CRIUgpu | "research, upstreamed in CRIU 4.0" | **Production at Modal + MemVerge. Podman native support.** |
| Driver requirement for cuda-checkpoint | "550+ for snapshot, 570+ for full" | **580+ now cited for device migration. Bar keeps rising.** |
| torch.compile cache | "~10-20s saving" | **Understated — FlashInfer/Triton/TVM-FFI caches add another bucket. Pin newer vLLM to access.** |
| Sleep Mode | "different problem" | **Reframed as primary Phase-6 multi-model strategy. 61-88% cold-start improvement on warm-process boot.** |
| Modal "10x" snapshot claim | unverified vendor | **Cited in vLLM's own RFC as evidence; CRIUgpu paper validates the primitive.** |
| Run:AI Model Streamer | "2x weight load" | Unchanged. Still the cleanest Phase-5 lever. |
| NVMe vs FSR | open question | **Still open. No public head-to-head exists.** |

## New entries to the Top-5 levers list

Round 1 Top-5 was: (1) Run:AI Streamer, (2) NVMe instance store, (3) torch.compile cache, (4) slim image, (5) GPU snapshots.

**Round 2 reshuffle:**

| # | Lever | Change | Why |
|---|---|---|---|
| 1 | **Run:AI Model Streamer** | Unchanged | Still the cheapest, most-portable, vLLM-native win |
| 2 | **Multi-compiler cache persistence** (torch.compile + FlashInfer + Triton) | **Promoted from #3, expanded** | Doubleword evidence: there's more than torch.compile to cache. Probably ~15-25s on our setup, not 10. |
| 3 | **NVMe instance store** | Was #2 | Demoted slightly; the absence of a public head-to-head + the open question about FSR overlap make this less of a "definite win" and more of "needs measurement." |
| 4 | **vLLM Sleep Mode for multi-model** (Phase 6) | **NEW Top-5** | Replaces "slim image" — this is the highest-leverage architectural move once Phase 6 starts. |
| 5 | **GPU memory snapshots** (Modal-style or RFC #34303 if it ships) | Unchanged at #5 | RFC stalled, but CRIUgpu now production-ready as the open-source path |

**Slim image drops out of the Top-5** — it's now table-stakes hygiene rather than a top-tier lever.

**Torch Memory Saver (R2.1d)** is the most interesting *new* technique surfaced this round but at PCIe Gen3 the savings ceiling is bounded; it's a Phase-6 watch-item rather than a Phase-5 must-have on T4.

## Updated open questions

1. **What does vLLM ≥0.7 / Q1-2026 release expose for cache persistence beyond torch.compile?** Specifically: are FlashInfer, Triton, TVM-FFI caches first-class persisted? If yes, the Phase-5 budget for compile-cache work is bigger than Round 1 estimated.
2. **What CUDA driver does the current EKS-optimized GPU AMI ship?** With cuda-checkpoint demanding 580+, this gates Phase 6 directly. Check before designing.
3. **R-Fork-style P2P weight transfer on AWS:** does p4d/p5 EFA give us the substrate even though g4dn doesn't? Phase 8 question, not Phase 5.
4. **Round 1 Q1 (Run:AI Streamer × FSR interaction) and Q3 (NVMe-vs-FSR head-to-head)** remain unanswered in the public literature. Both are 1-day experiments and would be worth a short blog post if measured cleanly.
5. **Has containerd shipped the Doubleword-described patches** (zero-copy mmap restore) into a release EKS 1.30 might pick up? Cheap win if yes.
6. **Sleep Mode + Karpenter:** if you keep one warm vLLM-with-sleeping-models pod, what happens on Spot interruption? Sleep state is in-process, so interruption is full cold-start again. Pair with PDB + on-demand fallback — design question for Phase 6/7.

## Round 2 sources added

- Doubleword "Behind the Stack: Fast SGLang Starts": https://blog.doubleword.ai/fast-sglang-starts
- LMSYS Tensor R-Fork (Dec 2025): https://www.lmsys.org/blog/2025-12-10-rfork/
- Baseten Delivery Network (Apr 2026): https://www.baseten.co/blog/how-the-baseten-delivery-network-bdn-makes-cold-starts-fast/
- Baseten cold-start docs: https://docs.baseten.co/performance/cold-starts
- vLLM Sleep Mode blog (Oct 2025, re-read): https://vllm.ai/blog/sleep-mode
- vLLM RFC #34303 (opened Feb 2026): https://github.com/vllm-project/vllm/issues/34303
- vLLM Q1 2026 roadmap: https://github.com/vllm-project/vllm/issues/32455
- vLLM Q2 2026 roadmap (draft): https://github.com/vllm-project/vllm/issues/39749
- CRIUgpu production status (DevZero, 2026): https://www.devzero.io/blog/gpu-container-checkpoint-restore
- WEKA model loading vs NVMe (vendor): https://www.weka.io/blog/ai-ml/model-loading-that-is-faster-than-local-node-nvme-with-nvidia-runai/
- InferenceX (formerly InferenceMAX) v2: https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs
- SGLang 2026 Q1 roadmap: https://github.com/sgl-project/sglang/issues/12780
- SGLang 2026 Q2 roadmap: https://github.com/sgl-project/sglang/issues/22949

