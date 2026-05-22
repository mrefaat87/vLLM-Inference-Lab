# GPU Inference Cold Start Optimization Survey

**Date:** April 2026  
**Scope:** Comprehensive survey of techniques for reducing GPU inference cold start / scale-up latency in Kubernetes (EKS focus)

---

## Cold Start Pipeline Overview

A GPU inference cold start is a waterfall of five sequential stages. Each must complete before the next can begin. Total duration: **3-8 minutes** of billable compute serving zero requests.

| Stage | Typical Duration | What Happens |
|-------|-----------------|--------------|
| 1. Node Provisioning | 60-120s | Cloud API call, OS boot, GPU driver init, kubelet registration |
| 2. Container Image Pull/Unpack | 30-180s | Pull 5-15 GB image from registry, decompress layers |
| 3. Model Loading | 60-180s | Download weights from storage, transfer to GPU VRAM |
| 4. Runtime Initialization | 5-60s | CUDA context, torch.compile, CUDA graph capture |
| 5. Readiness | 5-30s | Health checks pass, pod receives traffic |

**Key insight:** You must instrument timestamps at each stage to know which one dominates YOUR deployment. Optimizing the wrong stage wastes effort.

---

## Stage 1: Node Provisioning

### 1.1 Karpenter (vs. Cluster Autoscaler)

| Aspect | Detail |
|--------|--------|
| **What** | Provisions nodes directly from EC2 fleet API based on pending pod requirements, bypassing node group abstraction |
| **Time savings** | 50-70s average GPU node provisioning (vs. 7-12 min with Cluster Autoscaler — 76% reduction) |
| **Maturity** | Production-ready. GA on EKS. |
| **EKS+vLLM applicable** | Yes, primary recommendation |

**Key configurations for GPU speed:**
- Diversify instance types across G and P families, multiple generations (`instance-category: [g, p]`, `instance-generation: Gt 3`) to avoid InsufficientCapacityErrors
- Pin AMI versions in EC2NodeClass to prevent drift-triggered fleet recycling
- Set `consolidationPolicy: WhenEmpty` with generous `consolidateAfter: 60m` for inference nodes to prevent premature GPU node termination
- Use disruption budgets (e.g., `nodes: 10%`) to prevent cascading replacements
- Enable `SpotToSpotConsolidation=true` for cost optimization

### 1.2 Bottlerocket OS

| Aspect | Detail |
|--------|--------|
| **What** | Minimal container-optimized Linux with pre-installed NVIDIA drivers, containerd, and kubelet |
| **Time savings** | ~6 seconds faster node readiness vs. Amazon Linux 2 |
| **Maturity** | Production-ready. GPU-optimized AMIs available (`aws-k8s-1.28-nvidia`) |
| **EKS+vLLM applicable** | Yes |

Bottlerocket's smaller footprint means faster boot, smaller attack surface, and pre-installed GPU device plugin. No SSH, no package manager, no shell by default.

### 1.3 On-Demand Capacity Reservations (ODCRs)

| Aspect | Detail |
|--------|--------|
| **What** | Pre-reserve GPU instance capacity in specific AZs to guarantee availability |
| **Time savings** | Eliminates InsufficientCapacityErrors entirely (not a latency reduction, but an availability guarantee) |
| **Maturity** | Production-ready. Karpenter supports `capacityReservationSelectorTerms` in NodeClass |
| **EKS+vLLM applicable** | Yes, especially for G-series instances |
| **Cost** | Pay On-Demand rates whether used or not |

### 1.4 Capacity Blocks for ML

| Aspect | Detail |
|--------|--------|
| **What** | Reserve P-series and Trainium instances for 24h to 182 days, colocated in EC2 UltraClusters |
| **Time savings** | Guaranteed capacity for multi-GPU setups |
| **Maturity** | Production-ready |
| **EKS+vLLM applicable** | Yes, for P5/P4d instances |

### 1.5 EC2 Auto Scaling Warm Pools

| Aspect | Detail |
|--------|--------|
| **What** | Pre-initialized EC2 instances in Stopped state, ready to join ASG in ~30 seconds |
| **Time savings** | Launch pre-initialized instances in ~30s vs. 60-120s from scratch |
| **Maturity** | Production-ready. Supported with EKS self-managed node groups |
| **EKS+vLLM applicable** | Yes, but requires careful bootstrapping — instances must skip kubelet registration while entering warm pool, then run bootstrap on scale-out |

**Caveat:** If used with EKS managed node groups, warm pool instances may prematurely register with the cluster and receive workloads before they're ready.

### 1.6 EBS Snapshot-Based Provisioning + Fast Snapshot Restore (FSR)

| Aspect | Detail |
|--------|--------|
| **What** | Pre-populate EBS volumes with model weights, snapshot them, launch nodes from snapshots. FSR pre-warms blocks to eliminate first-read penalty |
| **Time savings** | Eliminates model download during startup for 50+ GB models. FSR adds ~7s savings on first access |
| **Maturity** | Production-ready |
| **EKS+vLLM applicable** | Yes |
| **Cost** | $0.05/GB-month for snapshots + $0.75/hour/AZ for FSR |

**Finding from Phase 4 experiments:** FSR only saves ~7s (not 90s as hoped). The real bottleneck is containerd layer unpack (110s) + readiness probe (120s), not data download speed.

---

## Stage 2: Container Image Pull & Unpack

### 2.1 SOCI (Seekable OCI)

| Aspect | Detail |
|--------|--------|
| **What** | AWS-developed lazy loading that creates separate seekable index (zTOC) artifacts in the registry without modifying the original image |
| **Time savings** | 60% reduction in pull time for 10 GB images; 4x faster pulls reported in production |
| **Maturity** | Production-ready on EKS. Native containerd integration, auto-detects SOCI-indexed images in ECR |
| **EKS+vLLM applicable** | Yes |

**Key advantage over alternatives:** No image conversion required. Image signatures remain valid. Minimum 10 MB layer size threshold for lazy loading.

**Caveat:** Lazy loading redistributes download time — total bytes transferred remains the same. Startup is faster but random I/O during runtime may increase. For GPU inference images where the bulk is PyTorch/CUDA libraries loaded at startup, SOCI benefit may be limited since those libraries are needed immediately.

### 2.2 Nydus (Dragonfly subproject)

| Aspect | Detail |
|--------|--------|
| **What** | Specialized RAFS filesystem with chunk-level deduplication and on-demand fetching |
| **Time savings** | Substantial reduction in launch times when combined with Dragonfly P2P |
| **Maturity** | CNCF project, production-ready at Alibaba scale |
| **EKS+vLLM applicable** | Yes, but requires image conversion and containerd snapshotter plugin setup |

Uses a custom image format (RAFS) optimized for random access. More invasive than SOCI (requires image conversion) but more sophisticated deduplication.

### 2.3 eStargz (Stargz Snapshotter)

| Aspect | Detail |
|--------|--------|
| **What** | Backward-compatible extension of OCI tar.gz format with individually compressed files for random access |
| **Time savings** | Faster initial startup but slower application startup than SOCI (e.g., 25s vs 5s for Airflow) |
| **Maturity** | CNCF containerd subproject |
| **EKS+vLLM applicable** | Possible but SOCI is preferred on AWS |

### 2.4 Dragonfly (P2P Image Distribution)

| Aspect | Detail |
|--------|--------|
| **What** | CNCF Graduated project — P2P file distribution system where every downloading node becomes a seed for peers |
| **Time savings** | Reduced origin traffic from 26 TB to ~130 GB across 200-node cluster. Pull times from minutes to seconds |
| **Maturity** | Production-ready. CNCF Graduated (March 2026). Billions of daily requests at Alibaba |
| **EKS+vLLM applicable** | Yes, installable via Helm |
| **Notable users** | Alibaba, Ant Group |

**Key feature for GPU images:** The seed peer does not need to finish downloading the entire model before sharing — pieces are shared as soon as they're downloaded. Also supports native Hugging Face and ModelScope protocols for AI model distribution.

### 2.5 Spegel (P2P Image Sharing)

| Aspect | Detail |
|--------|--------|
| **What** | Stateless, cluster-local OCI registry mirror. Each node acts as a registry using containerd's existing cache |
| **Time savings** | Eliminates external registry dependency for images already cached on any cluster node |
| **Maturity** | Production-ready. Used by CoreWeave (March 2026). Simpler than Dragonfly |
| **EKS+vLLM applicable** | Yes, runs as DaemonSet |

**How it works:** Deploys as DaemonSet, reads from containerd's content store, serves via OCI registry HTTP interface, uses Kademlia DHT for content discovery. Zero persistent storage needed.

**Best for:** Clusters where the same GPU inference image is pulled to many nodes — after the first pull from the external registry, all subsequent nodes pull from peers.

### 2.6 containerd Tuning

| Aspect | Detail |
|--------|--------|
| **What** | Increase parallel download/unpack settings in containerd config |
| **Time savings** | 60% improvement in image download time (60s to 24s on fresh node) |
| **Maturity** | Production-ready |
| **EKS+vLLM applicable** | Yes |

Specific tuning:
- `max_concurrent_downloads_per_image`: 5 → 10
- `max_concurrent_unpacks_per_image`: 3 → 10
- `concurrent_download_chunk_size`: 8 MB → 16 MB

### 2.7 Image Splitting (Separate Model from Runtime)

| Aspect | Detail |
|--------|--------|
| **What** | Keep the inference runtime image small (1-2 GB) and load model weights separately via init container, CSI driver, or volume mount |
| **Time savings** | Drastically reduces image pull time; runtime images cache efficiently across nodes |
| **Maturity** | Production best practice |
| **EKS+vLLM applicable** | Yes, recommended by AWS EKS best practices |

---

## Stage 3: Model Loading

### 3.1 NVIDIA Run:ai Model Streamer

| Aspect | Detail |
|--------|--------|
| **What** | Open-source C++ backend that concurrently reads model tensors from storage and streams them directly into GPU memory via multiple threads |
| **Time savings** | **4.88s** from S3 (concurrency 32) vs. 47s with HF Safetensors loader (9.6x faster). Total vLLM readiness: **23.18s** from S3 vs. 65s with alternatives |
| **Maturity** | Production-ready. Integrated into vLLM via `--load-format runai_streamer` |
| **EKS+vLLM applicable** | Yes, directly supported |

**Performance by storage backend (Llama 3 8B, 15 GB):**
| Storage | Model Streamer | HF Safetensors | CoreWeave Tensorizer |
|---------|---------------|-----------------|---------------------|
| S3 (conc 32) | 4.88s | N/A | 37.36s |
| IO2 SSD (conc 8) | 7.53s | 47s | 10.36s |
| GP3 SSD (conc 16) | 14.34s | 47.99s | 16.11s |

### 3.2 fastsafetensors

| Aspect | Detail |
|--------|--------|
| **What** | Python library that loads safetensors directly to GPU memory, optionally using GPUDirect Storage (GDS) to bypass CPU/DRAM entirely via NVMe DMA |
| **Time savings** | 4.8x to 7.5x faster for models from 7B to 176B parameters |
| **Maturity** | Integrated into vLLM via `--load-format fastsafetensors` |
| **EKS+vLLM applicable** | Yes. GDS mode requires specific NVMe hardware configuration |

Key innovation: Offloads tensor sharding (for tensor parallelism) to the GPU instead of doing it on the CPU first. Sends full tensors to GPU, slices there.

### 3.3 Mountpoint for Amazon S3 CSI Driver v2

| Aspect | Detail |
|--------|--------|
| **What** | Mounts S3 buckets as POSIX-like filesystem for Kubernetes pods. v2 (2025) adds shared caching across pods on the same node |
| **Time savings** | Eliminates explicit download step; streams on read. Shared cache means second pod on same node gets instant access |
| **Maturity** | Production-ready (GA) |
| **EKS+vLLM applicable** | Yes, recommended by AWS for model weight loading |

**Pro tip:** Use S3 VPC Gateway endpoints (free) to avoid NAT Gateway charges ($0.045/GB). Combine with `hf_transfer` for Rust-based parallel downloads reaching 1 GB/s.

### 3.4 Node-Local NVMe Caching

| Aspect | Detail |
|--------|--------|
| **What** | Use instance-store NVMe SSDs (free, included in instance price) for model weight caching |
| **Time savings** | Sub-millisecond latency, multi-GB/s throughput. p4d.24xlarge: 8x 1TB NVMe at ~65 GB/s aggregate vs. EBS gp3 max 1 GB/s |
| **Maturity** | Production-ready |
| **EKS+vLLM applicable** | Yes, especially with Spot instances (70% cost reduction) |

Requires node-aware scheduling and cache invalidation logic. First pod downloads from S3; subsequent pods read from local NVMe cache.

### 3.5 ServerlessLLM (sllm-store)

| Aspect | Detail |
|--------|--------|
| **What** | Multi-tier checkpoint loading system (GPU → DRAM → SSD) with O_DIRECT I/O and optimized loading format |
| **Time savings** | 6-8x faster startup vs. existing methods. Sequential chunk-based reading maximizes PCIe bandwidth |
| **Maturity** | Research/early-production (OSDI 2024 paper). Open source |
| **EKS+vLLM applicable** | Requires integration work; not a drop-in vLLM extension |
| **Notable** | Published at USENIX OSDI 2024 |

### 3.6 Quantization (Reduce Transfer Volume)

| Aspect | Detail |
|--------|--------|
| **What** | Reduce model precision from FP16 to INT8/INT4/FP8, cutting weight size by 2-4x |
| **Time savings** | 7B model: 14 GB (FP16) → 3.5 GB (INT4) — proportional reduction in load time |
| **Maturity** | Production-ready. vLLM supports AWQ, GPTQ, FP8, INT8, etc. |
| **EKS+vLLM applicable** | Yes |

Dual benefit: faster loading AND lower VRAM requirements, enabling use of cheaper GPU instances.

### 3.7 FSx for Lustre

| Aspect | Detail |
|--------|--------|
| **What** | High-throughput parallel filesystem with S3-backed data repository |
| **Time savings** | Multi-GB/s parallel reads, shared across pods and nodes |
| **Maturity** | Production-ready |
| **EKS+vLLM applicable** | Yes |
| **Cost** | ~$0.14/GB-month minimum for SSD storage |

Best for: Multiple GPU pods across AZs needing simultaneous access to same model weights.

---

## Stage 4: Runtime Initialization

### 4.1 vLLM Sleep Mode

| Aspect | Detail |
|--------|--------|
| **What** | Keeps vLLM process alive with CUDA context, graphs, and JIT caches preserved. Offloads weights to CPU (L1) or discards them (L2) |
| **Time savings** | L1: 0.26-0.90s wake-up (58-203x faster than cold start). L2: 0.85-2.58s wake-up (23-44x faster) |
| **Maturity** | Production-ready in vLLM. Requires `VLLM_SERVER_DEV_MODE=1` |
| **EKS+vLLM applicable** | Yes |

**Level comparison:**
| | L1 Sleep | L2 Sleep |
|--|---------|---------|
| Weights | Offloaded to CPU RAM | Discarded entirely |
| CPU RAM needed | 10-100+ GB per model | MBs per model |
| Wake speed | ~0.3-0.9s | ~0.9-2.6s |
| Best for | Frequent switching, ample RAM | Limited RAM, cost optimization |

**Critical detail:** Sleep mode provides 61-88% faster FIRST INFERENCE (not just wake-up) because CUDA graphs and JIT kernels are preserved. A cold-started model has 5-7x slower first inference even after weights are loaded.

### 4.2 CUDA Checkpoint/Restore (CRIUgpu)

| Aspect | Detail |
|--------|--------|
| **What** | Transparent GPU container checkpointing — freezes entire GPU state (memory, contexts, streams) to disk and restores it |
| **Time savings** | Modal demonstrated 10x cold start improvement (20s → 2s for Parakeet, 45s → 5s for vLLM). |
| **Maturity** | Alpha/experimental. Requires CUDA driver 570/575+. CRIUgpu merged into upstream CRIU 4.0+ |
| **EKS+vLLM applicable** | Not yet — requires driver support on EKS AMIs and Kubernetes checkpoint API (beta in K8s 1.30) |
| **Notable users** | Modal (production), research prototypes |

**How it works:**
1. `cuCheckpointProcessLock()` — prevent new CUDA operations
2. `cuCheckpointProcessCheckpoint()` — copy GPU memory/objects to host
3. Create unified CPU+GPU snapshot
4. On restore: reverse the process

**vLLM RFC (issue #34303):** Active proposal to integrate CUDA checkpoint/restore for near-zero cold starts in multi-model serving scenarios.

### 4.3 Compilation Cache (torch.compile)

| Aspect | Detail |
|--------|--------|
| **What** | Cache FX graphs and Triton kernels from torch.compile to persistent storage. Subsequent cold starts skip recompilation |
| **Time savings** | Compilation: 42s → 13s (with cache). Graph capture: 54s → 7s. Total: 72% reduction when combined with model caching |
| **Maturity** | Production-ready in vLLM V1 |
| **EKS+vLLM applicable** | Yes. Mount `/root/.cache` to EFS/PVC for cross-pod sharing |

### 4.4 CUDA Graph Size Limiting

| Aspect | Detail |
|--------|--------|
| **What** | Instead of capturing graphs for all batch sizes, specify only the sizes your workload uses |
| **Time savings** | Graph capture: 54s → 7s (87% reduction) |
| **Maturity** | Production-ready |
| **EKS+vLLM applicable** | Yes. `--cuda-graph-sizes 1,2,4,8,16,24,32,64` |

### 4.5 CUDA Context Optimization via cgroups

| Aspect | Detail |
|--------|--------|
| **What** | Use cgroups to isolate visible GPUs instead of `CUDA_VISIBLE_DEVICES`. Eliminates initialization of unwanted GPU contexts |
| **Time savings** | Faster init on multi-GPU nodes where only one GPU is needed per container |
| **Maturity** | Production-ready (NVIDIA blog documented approach) |
| **EKS+vLLM applicable** | Yes, via Kubernetes device plugin resource limits |

### 4.6 Target Architecture Pinning

| Aspect | Detail |
|--------|--------|
| **What** | Set `TORCH_CUDA_ARCH_LIST` to match your exact GPU (e.g., '8.9' for L40S) to skip compilation for other architectures |
| **Time savings** | Reduces JIT compilation scope |
| **Maturity** | Production-ready |
| **EKS+vLLM applicable** | Yes |

---

## Stage 5: Orchestration & Scaling Intelligence

### 5.1 Warm Pod Pools (minReplicas > 0)

| Aspect | Detail |
|--------|--------|
| **What** | Maintain minimum running inference pods to eliminate all cold start stages |
| **Time savings** | Eliminates 3-8 minutes of cold start entirely |
| **Maturity** | Production standard |
| **EKS+vLLM applicable** | Yes |
| **Cost** | $2-32/hour continuous GPU cost per warm replica |

### 5.2 Predictive Autoscaling (Kedify / Prophet)

| Aspect | Detail |
|--------|--------|
| **What** | Use time-series forecasting (Prophet, LSTM) to predict demand and pre-scale before traffic arrives |
| **Time savings** | Eliminates reactive scaling delay. Smooths cold start impact by pre-provisioning |
| **Maturity** | Production-ready (Kedify). Research/experimental (FlashServe) |
| **EKS+vLLM applicable** | Yes, via KEDA + Kedify predictive scaler |

Kedify integrates with KEDA, continuously collects metrics, trains Prophet models, and scales with a configurable forecast horizon (e.g., 30 minutes ahead).

### 5.3 NVIDIA Dynamo (Disaggregated Serving + SLO-Based Autoscaling)

| Aspect | Detail |
|--------|--------|
| **What** | Separates prefill and decode into independently scalable GPU pools. SLO-aware planner autoscales each pool targeting TTFT and ITL thresholds |
| **Time savings** | Reduces perceived cold start by routing new requests to existing decode workers while prefill workers scale. Up to 4x performance improvement |
| **Maturity** | v1.0 released (production-ready). Open source |
| **EKS+vLLM applicable** | Yes, blueprint available on AWS AI-on-EKS |
| **Notable** | Announced at GTC 2025, Dynamo 0.4 added SLO-based autoscaling with Kubernetes integration |

**Key capabilities:**
- KV-aware routing: Routes requests based on KV cache overlap to avoid redundant prefill
- KV block manager: Offloads KV cache across GPU, CPU, SSD, and remote storage
- Disaggregated P/D: Prefill (compute-bound) and decode (memory-bound) scale independently

### 5.4 llm-d (Kubernetes-Native Inference Framework)

| Aspect | Detail |
|--------|--------|
| **What** | Kubernetes-native distributed LLM inference with Gateway API Inference Extension, KV cache indexer, and intelligent scheduler |
| **Time savings** | Reduces perceived latency via cache-aware routing. Prefix cache hits eliminate redundant prefill |
| **Maturity** | v0.5 (active development). Backed by Red Hat |
| **EKS+vLLM applicable** | Yes, built on top of vLLM |

**Key features:**
- KV cache indexer maintains global near-real-time view of cache block locality
- Scheduler scores pods by KV cache overlap, load, P/D awareness
- v0.5 adds scale-to-zero autoscaling, hierarchical KV offloading, cache-aware LoRA routing

### 5.5 GKE Inference Gateway / Gateway API Inference Extension

| Aspect | Detail |
|--------|--------|
| **What** | Prefix cache-aware load balancer that routes requests to pods with matching KV cache, plus load-aware routing |
| **Time savings** | Up to 96% TTFT improvement at peak throughput for prefix-heavy workloads. Doubled prefix cache hit rate (35% → 70%) for Vertex AI |
| **Maturity** | GA on GKE. Gateway API Inference Extension is a Kubernetes SIG project |
| **EKS+vLLM applicable** | The Gateway API Inference Extension is cloud-agnostic. GKE-specific features don't apply to EKS, but the extension will |
| **Notable users** | Google Vertex AI (35% faster responses, 2x better tail latency) |

### 5.6 HydraServe (Parallelized Cold Start)

| Aspect | Detail |
|--------|--------|
| **What** | Distributes model across multiple GPU workers during cold start via pipeline parallelism. Each worker loads only a portion, aggregating network bandwidth |
| **Time savings** | 1.7-4.7x cold start reduction. 2.6x average TTFT reduction |
| **Maturity** | Research (NSDI 2026 paper). Open source |
| **EKS+vLLM applicable** | Requires integration work |

**Three-level approach:**
1. **Cluster level:** Network-contention-aware worker placement
2. **Worker level:** Overlaps container creation, library loading, and model fetching
3. **Inference level:** Pipeline consolidation — serves requests with partial model while loading remainder

### 5.7 Knative Serverless Scaling

| Aspect | Detail |
|--------|--------|
| **What** | Request-based autoscaling with scale-to-zero. Configurable pod retention period after last request |
| **Time savings** | None (adds cold start). But pod retention annotation keeps pods warm for configurable duration |
| **Maturity** | Production-ready, but not ideal for GPU inference cold starts |
| **EKS+vLLM applicable** | Yes, but standard Deployment is recommended for generative inference |

### 5.8 Priority-Based Preemption

| Aspect | Detail |
|--------|--------|
| **What** | Use Kubernetes PriorityClasses to ensure high-priority inference pods preempt low-priority batch workloads |
| **Time savings** | Eliminates node provisioning wait by reclaiming existing capacity |
| **Maturity** | Production-ready |
| **EKS+vLLM applicable** | Yes |

### 5.9 Gang Scheduling (Kueue / NVIDIA KAI Scheduler)

| Aspect | Detail |
|--------|--------|
| **What** | Ensures all pods for a multi-GPU inference deployment are schedulable before any are started. Prevents partial allocation waste |
| **Time savings** | Prevents wasted GPU-hours from partial deployments that can't serve traffic |
| **Maturity** | Kueue: GA. KAI Scheduler: Open source (Apache 2.0, 2025) |
| **EKS+vLLM applicable** | Yes |

---

## Cross-Cutting / Emerging Techniques

### Tangram (GPU Memory Reuse)

| Aspect | Detail |
|--------|--------|
| **What** | Unified GPU memory pool with tensor-level parameter sharing across models. GPU-affinity-aware scheduler maximizes reuse |
| **Time savings** | Load latency 23-56% of ServerlessLLM when model pool exceeds 200 GB |
| **Maturity** | Research (December 2025 paper) |
| **Stage** | Model Loading + Orchestration |

### FlashServe (Tiered Memory + Predictive Scaling)

| Aspect | Detail |
|--------|--------|
| **What** | Combines tiered memory snapshotting (pre-stages checkpoints in host DRAM, high-speed DMA via PCIe to GPU), predictive autoscaling (Prophet-LSTM), and adapter multiplexing |
| **Maturity** | Research (December 2025 preprint) |
| **Stage** | Model Loading + Orchestration |

### Progressive Weight Loading (Experimental)

| Aspect | Detail |
|--------|--------|
| **What** | Serve requests with a lightweight model variant (quantized/truncated) while streaming full-precision weights in the background |
| **Time savings** | Reduces time-to-first-token for large LLM deployments |
| **Maturity** | Experimental. No native support in production serving frameworks as of early 2026 |
| **Stage** | Model Loading + Runtime Init |

### LADF (Local Accelerated Data Fetching)

| Aspect | Detail |
|--------|--------|
| **What** | Pre-caches remote storage contents in local GPFS before workload starts. Integrates with Kueue admission checks |
| **Time savings** | Up to 40% reduction in data loading latency (ICML 2024 benchmark) |
| **Maturity** | Emerging (Kubernetes v1.36 target, April 2026) |
| **Stage** | Model Loading |

### DaemonSet Image Pre-Pull

| Aspect | Detail |
|--------|--------|
| **What** | Deploy a DaemonSet that pulls GPU inference images to every node proactively |
| **Time savings** | Eliminates image pull time on nodes that already have the image cached |
| **Maturity** | Production standard |
| **Stage** | Image Pull |
| **EKS+vLLM applicable** | Yes |

---

## Recommended Stack for EKS + Karpenter + vLLM

Based on this survey, here is the recommended optimization stack ordered by impact and ease of implementation:

### Quick Wins (days to implement)
1. **CUDA graph size limiting** — `--cuda-graph-sizes 1,2,4,8,16,24,32,64` (saves ~47s)
2. **Compilation cache on EFS** — mount `/root/.cache` as shared PVC (saves ~30s)
3. **Target arch pinning** — `TORCH_CUDA_ARCH_LIST=8.9` (saves variable)
4. **Model caching on local NVMe** — first pod downloads, subsequent pods read cache (saves 60-180s)
5. **containerd tuning** — increase parallel download/unpack (saves ~36s on fresh nodes)
6. **Image splitting** — separate runtime image from model weights (reduces image size by 10-100+ GB)

### Medium Effort (weeks to implement)
7. **Run:ai Model Streamer** — `--load-format runai_streamer` from S3 (23s total readiness vs 65s)
8. **SOCI lazy loading** — index existing images in ECR (60% pull time reduction)
9. **vLLM Sleep Mode** — for multi-model or scale-to-zero scenarios (sub-second wake)
10. **Predictive autoscaling** — KEDA + Kedify Prophet scaler (pre-provisions ahead of demand)
11. **Spegel P2P** — DaemonSet for peer-to-peer image sharing (eliminates registry dependency)
12. **Warm pools** — EC2 ASG warm pools for pre-initialized GPU instances (~30s launch)

### Strategic Investment (months to implement)
13. **NVIDIA Dynamo** — disaggregated prefill/decode with SLO-based autoscaling
14. **llm-d** — Kubernetes-native inference with KV cache-aware routing
15. **Gateway API Inference Extension** — intelligent request routing
16. **CUDA checkpoint/restore** — when driver support matures on EKS (near-zero cold starts)
17. **Dragonfly + Nydus** — for large multi-cluster deployments

---

## Key Timing Reference

**Before optimization** (typical vLLM cold start): **~294 seconds (4.9 minutes)**

| Component | Before | After (all optimizations) |
|-----------|--------|--------------------------|
| Model download | 61s | 0s (cached) |
| Weight loading | 33s | 5s (Run:ai Streamer) |
| Compilation | 42s | 13s (cached) |
| Graph capture | 54s | 7s (size limiting) |
| Engine init | 94s | 34s (V1 engine) |
| **Total** | **294s** | **~59s** |

**With Sleep Mode L1:** **0.3-0.9 seconds** (existing process, wake from sleep)

**With CUDA checkpoint/restore:** **2-5 seconds** (full restore from snapshot)

---

## Sources

### Node Provisioning
- [Reducing GPU Cold Start Times in Kubernetes - ScaleOps](https://scaleops.com/blog/reducing-gpu-cold-start-times-in-kubernetes-patterns-and-solutions/)
- [EKS AI/ML Compute Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html)
- [Karpenter Best Practices - EKS](https://docs.aws.amazon.com/eks/latest/best-practices/karpenter.html)
- [Ultimate Guide to GPU Scaling With Karpenter](https://cloudnativenow.com/contributed-content/the-ultimate-guide-to-gpu-scaling-with-karpenter/)
- [AWS EC2 Auto Scaling Warm Pools](https://aws.amazon.com/blogs/compute/scaling-your-applications-faster-with-ec2-auto-scaling-warm-pools/)
- [EKS Node Group with Warm Pool (GitHub)](https://github.com/aws-samples/eks-node-group-with-warm-pool)
- [Bottlerocket + NVIDIA on EKS](https://developer.nvidia.com/blog/deploy-ai-workloads-at-scale-with-bottlerocket-and-nvidia-powered-amazon-ec2-instances/)
- [EC2 Capacity Reservations](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-reservations.html)
- [EKS Auto Mode with ODCRs](https://docs.aws.amazon.com/eks/latest/userguide/auto-odcr.html)

### Container Image Pull/Unpack
- [Lazy Loading Performance for GPU Inference - Tensorfuse](https://tensorfuse-docs.mintlify.dev/docs/blogs/lazy_loading_performance_degradation)
- [Docker Lazy Loading at Grab (eStargz/Nydus/SOCI comparison)](https://engineering.grab.com/docker-lazy-loading)
- [Loading Model Weights for GPU Inference on EKS - Gary Stafford](https://garystafford.medium.com/loading-multi-gigabyte-model-weights-for-gpu-inference-on-amazon-eks-8efa93631bba)
- [Dragonfly P2P AI Model Downloads](https://d7y.io/blog/2026/03/11/p2p-accelerated-ai-model-downloads-native-hugging-face-and-modelscope-protocols-in-dragonfly/)
- [CNCF Dragonfly Graduation](https://www.infoq.com/news/2026/03/cncf-dragonfly-graduation/)
- [Spegel Architecture](https://spegel.dev/docs/architecture/)
- [Spegel on CoreWeave CKS](https://docs.coreweave.com/changelog/release-notes/cks-spegel-mirror)
- [Reduce Container Startup Time with Bottlerocket Data Volume](https://aws.amazon.com/blogs/containers/reduce-container-startup-time-on-amazon-eks-with-bottlerocket-data-volume/)

### Model Loading
- [NVIDIA Run:ai Model Streamer (NVIDIA Blog)](https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer/)
- [Run:ai Model Streamer on Google Cloud](https://cloud.google.com/blog/products/containers-kubernetes/nvidia-runai-model-streamer-supports-cloud-storage)
- [vLLM Run:ai Streamer Docs](https://docs.vllm.ai/en/stable/models/extensions/runai_model_streamer/)
- [fastsafetensors Paper](https://arxiv.org/html/2505.23072v1)
- [vLLM fastsafetensors Docs](https://docs.vllm.ai/en/stable/models/extensions/fastsafetensor/)
- [Mountpoint S3 CSI Driver (GitHub)](https://github.com/awslabs/mountpoint-s3-csi-driver)
- [ServerlessLLM (OSDI 2024)](https://www.usenix.org/system/files/osdi24-fu.pdf)
- [ServerlessLLM (GitHub)](https://github.com/ServerlessLLM/ServerlessLLM)

### Runtime Initialization
- [vLLM Sleep Mode Blog](https://vllm.ai/blog/sleep-mode)
- [vLLM Sleep Mode Docs](https://docs.vllm.ai/en/latest/features/sleep_mode/)
- [GPU Memory Snapshots - Modal](https://modal.com/blog/gpu-mem-snapshots)
- [CRIUgpu: GPU Container Checkpoint/Restore](https://www.devzero.io/blog/gpu-container-checkpoint-restore)
- [vLLM CUDA Checkpoint/Restore RFC (#34303)](https://github.com/vllm-project/vllm/issues/34303)
- [Reducing GPU Cold Start with vLLM - Tensorfuse](https://tensorfuse.io/docs/blogs/reducing_gpu_cold_start)
- [torch.compile Integration - vLLM Blog](https://blog.vllm.ai/2025/08/20/torch-compile.html)
- [vLLM CUDA Graphs Design](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
- [CUDA Initialization via cgroups (NVIDIA Blog)](https://developer.nvidia.com/blog/improving-cuda-initialization-times-using-cgroups-in-certain-scenarios/)

### Orchestration & Scaling
- [NVIDIA Dynamo Announcement](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [Dynamo 0.4 SLO-Based Autoscaling](https://developer.nvidia.com/blog/dynamo-0-4-delivers-4x-faster-performance-slo-based-autoscaling-and-real-time-observability/)
- [NVIDIA Dynamo on EKS Blueprint](https://awslabs.github.io/ai-on-eks/docs/blueprints/inference/GPUs/nvidia-dynamo)
- [llm-d (GitHub)](https://github.com/llm-d/llm-d)
- [llm-d KV Cache-Aware Routing (Red Hat)](https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference)
- [GKE Inference Gateway](https://cloud.google.com/blog/products/containers-kubernetes/how-gke-inference-gateway-improved-latency-for-vertex-ai)
- [Gateway API Inference Extension (Kubernetes Blog)](https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/)
- [HydraServe (NSDI 2026)](https://arxiv.org/html/2502.15524v2)
- [Kedify Predictive Autoscaling](https://kedify.io/resources/blog/predictive-autoscaling/)
- [GKE Autoscaling Best Practices for LLM Inference](https://docs.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling)
- [Taming the Chaos: Coordinated Autoscaling for Disaggregated LLM Inference](https://arxiv.org/html/2508.19559v1)

### Cross-Cutting
- [Tangram: GPU Memory Reuse for Serverless LLM Loading](https://arxiv.org/abs/2512.01357)
- [Addressing AI Container Cold Start with Kubernetes 2026](https://dasroot.net/posts/2026/02/addressing-ai-container-cold-start-kubernetes-2026/)
- [NVIDIA KAI Scheduler](https://www.cio.com/article/4152554/how-kubernetes-is-finally-solving-the-gpu-utilization-crisis-to-save-your-ai-budget.html)
- [Kubernetes GPU Optimization - Collabnix](https://collabnix.com/kubernetes-and-gpu-the-complete-guide-to-ai-ml-acceleration-in-2025/)
- [vLLM Optimization Docs](https://docs.vllm.ai/en/stable/configuration/optimization/)
