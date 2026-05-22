# Phase 4: Autoscaling & Cold Start Optimization — Learnings

## The Cold Start Problem

GPU inference cold start is the time from "new replica needed" to "serving traffic." In our stack (EKS + Karpenter + vLLM), it's a 5-stage waterfall:

```
Stage 1: Node Provisioning    ─── 90s ──→  GPU instance exists, kubelet registered
Stage 2: Container Image Pull ── 110s ──→  3.8GB vLLM image decompressed to overlayfs
Stage 3: Model Loading         ─── 48s ──→  Weights transferred from disk to GPU VRAM
Stage 4: Runtime Initialization ── 60s ──→  CUDA context + JIT kernels + CUDA graphs
Stage 5: Readiness Detection    ── 120s ──→  Probes pass, pod marked Ready
                                   ─────
                                   ~428s total (7+ minutes)
```

**AWS analogy:** This is the GPU equivalent of ASG instance launch time. But where an EC2 instance typically boots in 30-60s, a GPU inference pod takes 7 minutes because of the model loading and CUDA compilation stages that have no equivalent in traditional compute.

### Why each stage takes so long

**Stage 1 — Node Provisioning (~90s)**
Karpenter calls the EC2 Fleet API, an instance boots, GPU drivers initialize, kubelet registers with the cluster, and the NVIDIA device plugin exposes the GPU. Spot instances add 10-20s for capacity search across pools.
- *AWS analogy:* ASG LaunchTime — the time from API call to InService.

**Stage 2 — Image Pull/Unpack (~110s)**
The vLLM image is 3.8GB compressed. containerd must download AND decompress all layers into overlayfs snapshots. On gp3 EBS (125 MB/s throughput), the decompression — not the download — is the bottleneck at ~110s.
- *Key finding:* Pre-baking the AMI with the image cached in containerd only saved ~7s. containerd still decompresses layers from the local cache. The bottleneck is I/O, not network.

**Stage 3 — Model Loading (~48s)**
vLLM's default safetensors loader reads weight files sequentially from disk, deserializes tensors in Python, and copies them to GPU VRAM one at a time.
- *AWS analogy:* Like downloading application data from S3 on instance boot instead of baking it into the AMI.

**Stage 4 — Runtime Initialization (~60s)**
First CUDA call allocates the GPU context (~5s). PyTorch JIT-compiles fused attention, GEMM, and layernorm kernels for the specific GPU architecture (~20-30s). If CUDA graphs are enabled, vLLM captures execution graphs for each batch size (~54s for all sizes, ~7s if limited to only used sizes).
- *AWS analogy:* Like Lambda cold start — the first invocation compiles and loads the function, subsequent invocations reuse the warm environment.

**Stage 5 — Readiness Detection (~120s, pre-v3)**
Fixed `initialDelaySeconds: 120` meant kubelet didn't even START checking `/health` for 2 minutes. Even if vLLM was ready in 40s, the pod sat idle for 80s.
- *AWS analogy:* Like setting an ELB HealthCheckGracePeriod of 120s when the app boots in 40s.

---

## Optimizations Implemented (Phase 4v3)

### Optimization 1: Startup Probe (Stage 5 — saves ~115s)

**Problem:** `initialDelaySeconds: 120` is a blind timer. It doesn't know when vLLM is actually ready.

**Solution:** K8s **startup probes** run instead of liveness/readiness until they succeed. We poll `/health` every 5s with 60 attempts (5 min max). The moment vLLM is ready, the startup probe passes and readiness/liveness probes activate.

**Added a second layer:** file-based readiness probe on the queue-worker container (`test -f /tmp/worker-ready`). The worker creates this file after model warmup completes. Pod is NOT ready until both vLLM is healthy AND CUDA kernels are compiled.

```yaml
# vLLM container — detect model readiness instantly
startupProbe:
  httpGet: {path: /health, port: 8000}
  periodSeconds: 5
  failureThreshold: 60   # 5 min max

# Queue-worker container — detect warmup completion
readinessProbe:
  exec:
    command: ["test", "-f", "/tmp/worker-ready"]
  periodSeconds: 5
```

**AWS analogy:**
- Startup probe = EC2 instance status check (immediate, succeed on first pass)
- Worker readiness file = ELB health check on the application endpoint
- Old initialDelaySeconds = fixed InService warmup timer regardless of actual health

### Optimization 2: SOCI Lazy Image Loading (Stage 2 — saves ~95s)

**Problem:** containerd decompresses ALL 3.8GB of vLLM layers before starting the container. Most files (test utilities, unused Python packages) are never accessed.

**Solution:** SOCI (Seekable OCI) creates a secondary index artifact in ECR that maps every file in every layer to its byte offset in the compressed blob. The SOCI containerd snapshotter creates a FUSE mount instead of decompressing — files are fetched via HTTP range requests on demand.

```
Without SOCI:  [download 3.8GB] → [decompress to overlayfs: 110s] → [start container]
With SOCI:     [mount FUSE skeleton: ~15s] → [start container] → [fetch files on access]
```

**Implementation:** Three new components:
1. `setup-soci.sh` — pushes vLLM image to ECR + creates SOCI index
2. `gpu-node-soci.pkr.hcl` — AMI with SOCI snapshotter daemon + containerd proxy config
3. `gpu-nodepool.yaml` — dual amiSelectorTerms (SOCI first, non-SOCI fallback)

**AWS analogy:** SOCI is like S3 Select vs downloading the whole object. Instead of decompressing a 3.8GB tarball to read a few Python files, it range-GETs specific byte ranges.

**Caveat:** SOCI redistributes the I/O — total bytes transferred is the same. For GPU inference images where PyTorch/CUDA libraries are needed immediately at startup, the benefit may be less than for web application images. Background prefetching mitigates this.

### Optimization 3: Run:ai Model Streamer (Stage 3 — saves ~34s)

**Problem:** vLLM's default safetensors loader reads weight files sequentially in Python. Single-threaded, no direct GPU mapping.

**Solution:** `--load-format runai_streamer` — NVIDIA's C++ backend that uses concurrent threads to stream tensors directly from storage to GPU VRAM.

| Storage | Default Loader | Run:ai Streamer | Speedup |
|---------|---------------|-----------------|---------|
| gp3 SSD | 48s | 14s (16 threads) | 3.3x |
| S3 | N/A | 5s (32 threads) | N/A |

**Implementation:** One flag in the vLLM args + `RUNAI_STREAMER_CONCURRENCY=16` env var.

**AWS analogy:** Like S3 Transfer Acceleration or multi-part upload — same data, more parallel pipes.

### Optimization 4: Compilation Cache (Stage 4 — saves ~30s on 2nd+ starts)

**Problem:** CUDA kernels and Triton graphs are compiled on every cold start. PyTorch JIT compilation takes 20-30s, CUDA graph capture takes 7-54s.

**Solution:** Mount a `hostPath` volume at `/root/.cache/vllm` so compiled artifacts persist on the node across pod restarts. Second cold start on the same node skips compilation entirely.

Also removed `--enforce-eager` — with the compilation cache, the one-time cost of CUDA graph capture (7s with size limiting) is worth the ~30% runtime inference speedup.

**AWS analogy:** Lambda SnapStart — pre-initialized execution environment cached across invocations. First invoke compiles, subsequent invokes reuse.

### Optimization 5: containerd Parallel Tuning (Stage 2 — saves ~36s)

**Problem:** containerd defaults to 3 concurrent layer unpacks. The vLLM image has ~30 layers, so decompression is serialized.

**Solution:** Set `max_concurrent_downloads = 10` and `max_concurrent_uploaded_layers = 10` in the containerd config (via Packer AMI bake). Benchmarked at 60% improvement (60s → 24s on fresh nodes).

**Implementation:** Added to both Packer templates (SOCI and non-SOCI).

**AWS analogy:** Like increasing ASG DesiredCapacity for a parallel batch job — more workers finish faster.

### Optimization 6: vLLM Sleep Mode (Wake latency — 295s → 0.5s)

**Problem:** After traffic stops, either: (a) keep the pod running and waste GPU, or (b) terminate and face 295s cold start when traffic returns.

**Solution:** vLLM Sleep Mode keeps the process alive but offloads weights from GPU VRAM to CPU RAM. CUDA graphs, JIT kernels, and the CUDA context are preserved. Wake-up copies weights back to GPU in 0.3-0.9s.

**Two levels:**
| | L1 Sleep | L2 Sleep |
|--|---------|---------|
| Weights | CPU RAM (fast wake) | Discarded (slow wake) |
| Wake time | 0.3-0.9s | 0.9-2.6s |
| CPU RAM needed | ~4GB per model | MBs |
| Best for | Frequent wake, ample RAM | Many sleeping models |

**Implementation:** The queue-worker has an idle monitor: after 300s of no messages, it calls `POST /sleep` on vLLM. When a new message arrives, it calls `POST /wake_up` before processing.

**AWS analogy:** EC2 Hibernate — same instance, same warm caches, just paused. Key difference: on dedicated EC2 instances, sleep mode is a LATENCY optimization (you still pay for the instance). Cost savings only apply on platforms that bill per GPU-second or for multi-model GPU packing.

**Billing reality on EC2:**
```
vLLM active:   g4dn.xlarge = $0.16/hr
vLLM sleeping: g4dn.xlarge = $0.16/hr  ← SAME (billing is per-instance)
```
Sleep mode saves money when packing multiple models on one GPU (3 instances → 1) or on serverless GPU platforms that bill per GPU-second.

### Optimization 7: vLLM Version Upgrade (v0.4.1 → v0.19.0)

All optimizations above require vLLM v0.7.0+. Upgraded from v0.4.1 (April 2024) to v0.19.0 (April 2026).

**Key changes in v0.19.0:**
- V1 engine (default since v0.8.0) — isolates scheduler and EngineCore into separate processes
- torch.compile integration — better kernel optimization
- Run:ai Model Streamer support (v0.6.4+)
- Sleep Mode (v0.7.0+)
- CUDA graph size limiting via compilation config (v0.7.0+)

**Compatibility verified:** AWQ quantization, Qwen2.5-7B-Instruct-AWQ, T4 GPU (CC 7.5) all supported.

---

## Cold Start Timeline: Before vs After

| Stage | Phase 4v2 (before) | Phase 4v3 (after) | Technique |
|-------|-------------------|-------------------|-----------|
| Node provision | 90s | 90s | (no change — Karpenter already optimal) |
| Image unpack | 110s | ~15s (SOCI) or ~44s (containerd tuning) | SOCI lazy loading + parallel unpack |
| Model loading | 48s | ~14s | Run:ai Model Streamer (16 threads) |
| Runtime init | 60s | 7s (first) / 0s (cached) | Compilation cache + CUDA graph sizing |
| Readiness | 120s | ~5s | Startup probe + worker readiness file |
| **Total cold start** | **~428s** | **~131s** (SOCI) or **~160s** (no SOCI) | |
| **Wake from sleep** | N/A | **~0.5s** | vLLM Sleep Mode L1 |

---

## Techniques Surveyed But Not Implemented

### High-impact future options

**CUDA Checkpoint/Restore (CRIUgpu)** — Snapshot entire GPU state (VRAM, contexts, streams) to disk, restore on any node in 2-5s. Skips stages 3+4 on true cold starts. vLLM RFC #34303 active. Requires CUDA driver 570+ (not yet on EKS AMIs).

**NVIDIA Dynamo** — Disaggregated prefill/decode serving. Prefill (compute-bound) and decode (memory-bound) scale independently. New requests hit existing decode workers while prefill scales. AWS EKS blueprint available.

**Gateway API Inference Extension** — KV cache-aware routing at the Kubernetes level. Routes requests to pods with matching prefix cache. GKE saw 96% TTFT improvement. K8s SIG project (cloud-agnostic).

**Spegel P2P** — DaemonSet that turns every node's containerd cache into a peer registry. After 1 node pulls the 8GB vLLM image, all others pull from peers at NVMe speed. Zero storage, simpler than Dragonfly.

**Predictive autoscaling (Kedify)** — Prophet time-series forecasting integrated with KEDA. Pre-scales 30 minutes ahead of predicted demand, eliminating reactive scaling delay entirely.

See `stage3_cold_start_survey.md` for the complete survey of 30+ techniques across all 5 stages.

---

## Scaling Policy Results (Phase 4v2)

Four KEDA scaling policies tested with Pareto workloads (80/15/4/1 distribution):

**Winner: Composite Aggressive (Policy D, KV>0.65)** — the only policy that detected GPU memory pressure from expensive requests at low queue depths.

| Policy | KV Stress Success | Burst TTFT | How It Scales |
|--------|------------------|-----------|---------------|
| A: Queue-only (5) | 17% (5/30) | 24-27s | Queue depth only |
| B: Queue-eager (3) | 17% (5/30) | 24-27s | Queue depth (lower threshold) |
| C: Composite (KV>0.80) | 63% (19/30) | 0.49s | Queue + KV cache > 80% |
| D: Composite (KV>0.65) | 100% (23/23) | 0.50s | Queue + KV cache > 65% |

**Key insight:** Queue-only policies are blind to GPU memory pressure. A few XL requests at low queue depth consume all KV cache without triggering scale-up. Composite policies watch KV cache utilization via Prometheus and scale preemptively.

**AWS analogy:** Queue-only = ASG scaling on SQS ApproximateNumberOfMessagesVisible. Composite = scaling on BOTH SQS depth AND target group average response time. The multi-signal approach catches problems that a single metric misses.

### Queue Duration vs Queue Depth (Next Experiment)

Research (backpressure_research.md, Part 2.3) identifies a potentially better signal than either queue depth or KV cache: **queue duration** — the age of the oldest message. This is the LLM inference equivalent of SQS `ApproximateAgeOfOldestMessage`.

**Why depth lies with variable-cost requests:**
- Scenario A: 50 short requests in queue (200ms each) → depth=50 (looks scary), oldest=10s (users fine)
- Scenario B: 3 XL requests in queue (30s each) → depth=3 (looks fine), oldest=90s (users furious)
- Depth says A is 16x worse than B. Duration says B is 9x worse. Duration is right.

**SLO derivation:** If SLO is P95 latency < 10s, reserve 30% headroom (3s), scale at 7s queue duration. Cleaner than guessing "what queue depth corresponds to my SLO" — which depends on the unpredictable request mix.

**Implementation:** RabbitMQ exposes `rabbitmq_queue_head_message_timestamp` via Prometheus. KEDA Prometheus trigger query: `time() - rabbitmq_queue_head_message_timestamp{queue="inference_queue"}`. Policy E ScaledObject created but not yet tested. Hypothesis: should trigger scale-up during KV stress (where depth=3 but duration=30s+), achieving composite-like benefits with a simpler single-metric signal.

**Status:** Plan written, v2-stack frozen deployment created for apples-to-apples comparison. Pending execution.

---

## Architecture Decisions

### Why hostPath for caches (not EFS/EBS PVC)

Model cache and compilation cache both use `hostPath` volumes, not EFS or PersistentVolumeClaims:
- **hostPath pro:** zero provisioning overhead, survives pod restarts on same node
- **hostPath con:** lost on node replacement (Spot interruption, consolidation)
- **EFS pro:** survives node replacement, shared across nodes
- **EFS con:** higher latency (~2-5ms vs <1ms for local disk), $0.30/GB/month, EFS provisioning adds to cold start

For a learning lab with Spot instances, hostPath is the right tradeoff. The compilation cache is small (~100MB) and regenerates in 7-30s. The model cache has S3 fallback. Neither justifies EFS overhead.

### Why `--enforce-eager` was removed

Phase 4v2 used `--enforce-eager` to disable CUDA graphs, avoiding 54s of graph capture on first request. Phase 4v3 removed it because:
1. Compilation cache makes capture cost one-time per node (~7s with size limiting)
2. CUDA graphs provide ~30% faster inference at runtime
3. The warmup request absorbs any remaining first-request latency

Net: 7s one-time cost (amortized to 0s on subsequent starts) for 30% faster inference.

### Why sleep mode is a latency optimization on EC2

On dedicated EC2 instances, you pay per-instance-hour regardless of GPU utilization. vLLM sleeping doesn't reduce the bill. The value is:
1. **Fast wake:** 0.5s vs 295s when traffic resumes after idle
2. **Multi-model packing:** 3 models on 1 GPU (only active model loaded, others sleeping)
3. **Serverless platforms:** GPU-second billing makes sleep a true cost optimizer

For our single-model learning lab, sleep mode eliminates the "should I keep a warm pod?" question: yes, keep it, but let it sleep. Wake is instant when needed.

---

## Admission Control: Don't Double-Gate

**Finding: Static admission (no checks) beats threshold and per-request strategies for single-model serving.**

vLLM's continuous batching + PagedAttention already handles request queuing internally. Adding application-level admission control on top creates double-gating: the worker rejects requests that vLLM could have handled fine. With threshold strategy, sustained success rate was 74% (69 NACKs). With static, it was 100% (0 NACKs).

**Why:** The threshold strategy polls vLLM `/metrics` and rejects if KV cache > 80%. But vLLM's internal scheduler handles this gracefully — it queues requests, applies PagedAttention to manage memory, and preempts if needed. The external gate rejected requests that vLLM's scheduler would have served fine.

**AWS analogy:** Like putting a rate limiter in front of an ALB that already has connection draining and surge queue. The extra gate hurts more than it helps.

**When admission control DOES help:** Multi-model deployments where different models compete for GPU memory, or when request costs vary by 100x and you want to prioritize cheap requests.

---

## Graceful Drain: Track Active Tasks

**Phase 3 bug:** `asyncio.create_task(process_job(...))` was fire-and-forget. On SIGTERM, in-flight tasks were abandoned — clients saw broken SSE streams mid-response.

**Phase 4 fix:**
1. Track active tasks in `active_tasks: set[asyncio.Task]` with `add_done_callback` for auto-cleanup
2. On SIGTERM: stop consuming new messages, then `asyncio.wait(active_tasks, timeout=DRAIN_TIMEOUT)`
3. DRAIN_TIMEOUT (170s) must be < terminationGracePeriodSeconds (180s) — 10s buffer for connection cleanup
4. vLLM sidecar gets `preStop: exec: ["sleep", "15"]` — keeps vLLM alive while the worker drains

**Result:** 100% request completion on pod kill (tested 10/10).

**AWS analogy:** ALB connection draining. SIGTERM = deregister from target group. DRAIN_TIMEOUT = deregistration delay.

---

## Karpenter Disruption Management

**Consolidation timing is critical.** Karpenter's `consolidateAfter` must exceed KEDA's `cooldownPeriod`:
- If consolidateAfter (60s in Phase 1) < cooldownPeriod (300s), Karpenter kills the GPU node while KEDA still considers scaling down. The pod is evicted, KEDA creates a replacement, Karpenter provisions a new node — 5-minute thrashing loop.
- **Fix:** consolidateAfter: 300s (matches KEDA cooldown). Sequence: load stops → KEDA cooldown (300s) → pod terminated → Karpenter consolidation (300s) → node terminated.

**Disruption budgets prevent cascading eviction.** `budgets: [{nodes: "1"}]` ensures Karpenter only disrupts 1 GPU node at a time. Without this, a cluster-wide consolidation could kill all GPU workers simultaneously.

**Cooldown must be ≥ 2x scale-up latency.** Phase 2's cooldown was 60s. KEDA would scale a new worker, the queue drained in ~30s, and 60s later KEDA killed the worker that was still downloading model weights. Setting cooldown to 300s prevents this.

---

## EKS Operational Gotchas

**aws-auth ConfigMap role mismatch silently blocks node registration.** When the managed node group's IAM role doesn't match the `rolearn` in aws-auth, nodes launch as EC2 instances but kubelet can't authenticate. No errors in the EKS console — instances show "healthy" but never appear as K8s nodes. Diagnosed by comparing `aws eks describe-nodegroup --query nodegroup.nodeRole` against `kubectl get configmap aws-auth -n kube-system`.

**EKS security group rules can break node→control plane communication.** The cluster SG must allow inbound from the node SG. If this rule is missing, nodes can reach the internet but not the API server. Symptoms identical to the aws-auth issue.

**Prometheus retentionSize format: `5GB` not `5Gi`.** Kubernetes uses binary suffixes (Gi) but Prometheus uses SI suffixes with a mandatory B suffix (GB). `5Gi` fails CRD validation silently.

**Grafana sidecar only watches its own namespace by default.** Dashboard ConfigMaps must be in the same namespace as Grafana (`monitoring`), or set `searchNamespace: ALL` in Helm values.

**DCGM Exporter needs `/dev/nvidia*` device mounts.** Without mounting `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, DCGM starts but reports 0% for all GPU metrics.

**ConfigMap envFrom vs explicit env vars:** `envFrom: configMapRef` injects all ConfigMap keys as env vars, but explicit `env` entries take precedence. When Phase 3's `kubectl set env` left behind an explicit `ADMISSION_STRATEGY=per_request`, the ConfigMap value was silently ignored. Always clean up leftover explicit env vars.

---

## Experiment Design Learnings

**Pareto workloads reveal pathologies that uniform workloads hide.** Cycling short/medium/long equally gives every request the same weight. Real traffic follows power-law distributions: 80% cheap, 15% moderate, 4% expensive, 1% enormous. The 1% XL requests create disproportionate KV cache pressure that only GPU-aware scaling policies detect.

**Graduated load reveals transition behavior.** A flat sustained rate either overwhelms or underwhelms — you miss the exact moment each policy decides to scale. Graduated phases (0.8 → 1.5 → 2.5 → 3.0 req/s) observe the system at each threshold.

**SLA targets give an objective function.** Without P99 complete time targets per prompt type, you can't say which policy is "best." SLA compliance percentage is the metric infrastructure leadership cares about.

**Cost-per-token connects scaling to economics.** $0.24/1M output tokens is the headline number. More aggressive scaling = higher GPU-hours but better SLA compliance — the tradeoff is quantifiable.

**The 5x throughput gap was a misleading comparison.** Stage 2 local vLLM at 60 concurrent: 1131 tok/s. Phase 4 EKS pipeline at 1.5 req/s: 232 tok/s. Diagnostic A/B test (same 60 concurrent on EKS): direct vLLM = 390 tok/s, pipeline = 244 tok/s. Real overhead is **1.6x**, from queue serialization + Redis pub/sub per token.

---

## Kubernetes Patterns for Inference

**PodDisruptionBudget with `minAvailable: 1`** (not `maxUnavailable: 1`). With 1 worker, `maxUnavailable: 1` allows disrupting it (1 - 1 = 0). `minAvailable: 1` is an absolute floor.

**preStop hook on sidecar containers:** K8s sends SIGTERM to all containers simultaneously. The queue-worker needs vLLM alive on localhost to drain in-flight streaming responses. `preStop: ["sleep", "15"]` on vLLM delays its shutdown so it outlives the worker's drain phase.

**Port-forward reliability:** kubectl port-forward drops connections during 60+ minute test runs. Always verify before each test phase. Use `127.0.0.1` explicitly — some kubectl versions only bind IPv6.

**Python stdout buffering in background commands:** `python3 script.py 2>&1` in background buffers stdout. Use `python3 -u script.py` for real-time output.

---

## Cold Start Benchmark Design

**Per-stage instrumentation requires multiple data sources.** A single `time.time()` delta gives total cold start but not WHERE the time went. Breaking into 5 stages requires:
- K8s events (`Scheduled`, `Pulled`) for node provision and image pull boundaries
- Worker log markers (`WARMUP: vLLM is ready`, `WARMUP: Complete`, `READY: Created`) for model loading and runtime init
- Pod condition `Ready=True lastTransitionTime` for readiness detection
- Each source has ~1-5s precision — acceptable for benchmarks measured in minutes.

**Don't hardcode baselines — measure them live.** The original plan was to hardcode Phase 2 baseline numbers (300s) and only measure v3 live. This is wrong because the vLLM version upgrade (v0.4.1 → v0.19.0) changed the baseline — the 9.5GB image takes 4.5 min to pull vs 1.8 min for 3.8GB. The true baseline for v0.19.0 is ~558s, not 300s. The benchmark must deploy 8 separate configurations, each adding one optimization cumulatively, all measured live on the same cluster. No derived or estimated numbers.

**Scenario ordering creates dependency chains.** Config 1 (naked baseline) through 7 (+streamer) each do a full cold start: delete all GPU nodes, scale to 1, measure. Config 8 (cache hit) keeps the same node from config 7 and only deletes the pod. Config 9 (wake from sleep) keeps the pod from config 8 and waits for idle sleep.

**vLLM version upgrades can REGRESS cold start.** v0.19.0 is 9.5GB compressed vs v0.4.1's 3.8GB. First image pull on a fresh node: 4.5 min vs 1.8 min. This makes SOCI lazy loading even more critical — the bigger the image, the bigger the savings from lazy loading. Always check image size when upgrading inference engine versions.

**Run:ai Model Streamer is NOT in the stock vLLM image.** `--load-format runai_streamer` requires `pip install runai-model-streamer runai-model-streamer-s3`. The stock `vllm/vllm-openai:v0.19.0` image doesn't include it. Need a custom Dockerfile that extends the base image with the pip install.

**Production uses minReplicas: 3 for GPU workloads.** Research (backpressure_research.md Part 2.3) recommends never dropping below 3 replicas because GPU pods take 5 min to start. This means the warm-node cold start (config 8: cache hit) is the scenario operators actually experience, not the full cold start (config 1). The full cold start matters for initial deployment and disaster recovery, not steady-state scaling.

---

## Operational Discipline

**Always verify `kubectl config current-context` before modifying a cluster.** Phase 4v3 deployment was accidentally applied to the Phase 3 cluster (`inference-phase3`), corrupting 4 of 6 configs in a running compound experiment. Root cause: kubeconfig was pointing to `inference-phase3` and wasn't checked. The `deploy.sh` script now has a pre-flight context check that refuses to run against the wrong cluster.

**Phase 4 needs its own Terraform stack.** Phase 1 (`inference-lab`, VPC 10.0.0.0/16) and Phase 3 (`inference-phase3`, VPC 10.1.0.0/16) are completely separate clusters with separate Terraform state. Phase 4 must create `inference-phase4` (VPC 10.4.0.0/16) — never deploy Phase 4 manifests to another phase's cluster. S3 bucket and ECR repos are shared (referenced via `data` sources, not created).

**`kubectl set env` creates stale overrides that survive `kubectl apply`.** Phase 3 experiments used `kubectl set env` to set `ADMISSION_STRATEGY=reactive`. These explicit env vars override ConfigMap `envFrom` values silently. `kubectl apply` of a new Deployment spec does NOT remove them. Fix: `kubectl delete deployment` then `kubectl apply` for a clean slate.

**`snap-PLACEHOLDER` in Karpenter NodePool causes instant EC2 launch failure.** Never leave placeholder values in Karpenter EBS block device mappings — they pass validation at `kubectl apply` but fail at EC2 Fleet API call time. Comment out the entire block device mapping if the snapshot hasn't been created.

**Run the benchmark:** `python3 phase4/tests/cold_start_benchmark.py --host http://localhost:8080`
