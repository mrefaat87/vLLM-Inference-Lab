# Phase 4.5 — NVIDIA Dynamo Research Spike

**Status:** Pre-spike research (1-2 day time-box)
**Goal:** Hands-on Dynamo experience to speak credibly in an Anthropic Cloud Inference EM interview. NOT a platform rebuild.
**Author:** Mohamed
**Date:** 2026-05-01

---

## 1. What Dynamo Is and Why It Exists

NVIDIA Dynamo (announced at GTC 2025, currently v1.0.x as of April 2026) is an **open-source, datacenter-scale orchestration layer for LLM inference**. It does not replace inference engines like vLLM, TensorRT-LLM, or SGLang — it coordinates fleets of them into a single multi-node serving system. Think of it as the control plane that sits above engines and below the API gateway: it handles request admission, KV-cache-aware routing, prefill/decode disaggregation, and SLA-driven autoscaling. Built in Rust (hot path) with Python (extensibility).

### The problem it solves vs. running vLLM alone

A single vLLM instance gives you continuous batching and PagedAttention, but it owns its KV cache as an island. If you run N vLLM replicas behind a round-robin LB:

- The same prompt prefix may be re-prefilled on every replica that sees it (no cross-replica cache sharing).
- Prefill and decode contend for the same GPU — bursty long prompts starve decode tokens (head-of-line blocking on TBT).
- Autoscaling is reactive on coarse metrics (queue depth, GPU util) — by the time you scale, the SLA is already broken.

Dynamo addresses each of these:
- **KV-aware Smart Router** picks the worker with the best cache overlap (avoids redundant prefill).
- **Disaggregated serving** splits prefill GPUs from decode GPUs, with **NIXL** moving KV blocks GPU-to-GPU at wire speed.
- **SLA Planner** is predictive (ARIMA/Prophet) and aware of P/D imbalance — it scales prefill and decode pools independently.

### AWS Auto Scaling analogy

| Dynamo concept | AWS Auto Scaling analogy |
|---|---|
| Smart Router (KV-aware) | ALB with sticky sessions / consistent-hash routing to warm capacity |
| Prefill workers vs decode workers | Two ASGs with different instance types — compute-optimized for prefill, memory-optimized for decode |
| NIXL KV transfer | Cross-AZ replication over a dedicated network fabric (think VPC peering with placement groups) |
| Planner | Predictive Scaling on CloudWatch + custom metrics, NOT reactive Target Tracking |
| KVBM (KV Block Manager) | Tiered storage: GPU HBM = hot, CPU = warm, NVMe/S3 = cold (like S3 Intelligent-Tiering) |
| etcd + NATS | Service Discovery + EventBridge for the inference control plane |
| DynamoGraphDeployment CRD | Like CDK / CFN stack, but inference-topology-aware |

---

## 2. Architecture

### Component diagram (logical)

```
                   ┌─────────────────────────────┐
   client ───────► │ Frontend (OpenAI API)       │
                   │ /v1/chat/completions        │
                   └──────────────┬──────────────┘
                                  │
                   ┌──────────────▼──────────────┐
                   │ Smart Router                │  ◄── etcd (worker discovery)
                   │ - Radix tree of KV blocks   │  ◄── NATS (KV events)
                   │ - overlap_score + load_cost │
                   └──┬───────────────────┬──────┘
                      │ best worker       │
            ┌─────────▼──────┐   ┌────────▼────────┐
            │ Prefill Worker │   │ Decode Worker   │
            │ (vLLM/TRT/SGL) │──►│ (vLLM/TRT/SGL)  │
            │ KV cache pool  │   │ KV cache pool   │
            └────────┬───────┘   └────────┬────────┘
                     │                    │
                     └─── NIXL ──────────►┘
                          (KV transfer:
                           RDMA / NVLink /
                           UCX / NVMe-oF)
                                  │
                   ┌──────────────▼──────────────┐
                   │ KVBM (multi-tier KV cache)  │
                   │ HBM ─► CPU ─► NVMe ─► S3    │
                   └─────────────────────────────┘

                   ┌─────────────────────────────┐
                   │ Planner (SLA-driven)        │  ◄── /metrics (TTFT, ITL, ISL, OSL)
                   │ ARIMA/Prophet forecasts     │  ──► scale prefill/decode replicas
                   └─────────────────────────────┘
```

### 2.1 Frontend (OpenAI-compatible API)

- Exposes `/v1/chat/completions`, `/v1/completions`, embeddings, etc.
- Reports per-request metrics on `/metrics`: ISL (input seq len), OSL (output seq len), TTFT, ITL — which the Planner consumes.
- This is the only piece clients see. Drop-in replacement for OpenAI SDK.

### 2.2 Smart Router (KV-aware routing)

The core insight: every LLM worker has a **KV cache** — a list of token-block hashes corresponding to prompts it has already prefilled. The router maintains a **Radix tree** mapping prefix-block-hashes → set of workers that hold them. On each incoming request:

1. Tokenize the prompt, hash into KV blocks (typically 16 tokens per block).
2. For each candidate worker, compute:
   - `overlap_score` = number of cached blocks that match this request's prefix
   - `prefill_cost` = blocks that would need to be newly computed
   - `decode_cost` = function of active in-flight requests on that worker
3. Pick the worker minimizing `prefill_cost + α·decode_cost - β·overlap_score`.

Knob: `--router-kv-overlap-score-weight` — higher = better TTFT (favor cache hits), lower = better load balance.

**Signal sources:**
- **KV events over NATS**: workers publish `block_added`/`block_evicted` events; router updates its radix tree.
- **Approximate mode** (`--no-router-kv-events`): router predicts cache state with TTL expiration — lower overhead, risk of stale assumptions.
- **In-flight request count**: tracked router-side from request issue/complete callbacks.

**Relation to consistent hashing:** It is NOT consistent hashing. Consistent hashing pins a key to a worker. Dynamo dynamically reassigns based on real-time cache state — the same prompt can route to different workers across calls if cache contents shift. Closer in spirit to **least-loaded-with-affinity-bonus**.

**Standalone mode:** Router can run without the frontend (e.g., to route into a prefill pool only) for multi-tier architectures.

### 2.3 Disaggregated Serving (Prefill / Decode split)

LLM inference has two phases with opposite resource profiles:
- **Prefill**: compute-bound, parallelizes well (one big matmul over the prompt). Wants high FLOPs, can use lower TP.
- **Decode**: memory-bandwidth-bound, serialized (one token at a time). Wants high HBM bandwidth, benefits from higher TP and continuous batching.

Co-locating them on the same GPU means:
- A long prefill blocks decode tokens → TBT spikes (head-of-line blocking).
- You overprovision GPUs to handle the worst-case mix.

Disaggregation: dedicate one pool of GPUs to prefill, another to decode. Request flow:
1. Frontend → Router → **Prefill worker** runs the prompt, produces KV cache.
2. **NIXL** transfers KV blocks GPU-to-GPU to a chosen decode worker (non-blocking; prefill GPU continues serving other requests).
3. Decode worker streams tokens back through frontend.

Dynamo claims **30x throughput gains** on DeepSeek-R1 with disaggregation on Blackwell. Real-world chat workloads see **3-5x cost-per-token reduction** because decode can run on cheaper GPUs.

### 2.4 NIXL (NVIDIA Inference Transfer Library)

A high-throughput, low-latency, asynchronous point-to-point library for moving KV tensors. Built on **UCX** under the hood, with multiple backend transports:

| Transport | Latency | Use case |
|---|---|---|
| RDMA / InfiniBand NDR (400 Gbps) | <1ms | Production multi-node H100/B200 |
| RoCE via UCX | 1-3ms | Datacenter Ethernet with RDMA |
| **TCP fallback via UCX** | 5-20ms | Dev/test, **AWS EFA/non-RDMA** |
| NVMe-oF | 2-10ms | KV tiering and eviction (KVBM cold storage) |
| S3-compatible | 50-500ms | Cross-cluster KV reuse, archiving |
| NVLink (intra-node) | sub-µs | Same-host GPU-to-GPU |

**How it avoids GPU↔CPU↔GPU bouncing:** RDMA and NVLink kernel-bypass paths let one GPU's HBM be DMA'd directly into another GPU's HBM without traversing host memory. For TCP fallback, there IS a CPU bounce — performance suffers.

**AWS gotcha:** EC2 GPU instances use **EFA (Elastic Fabric Adapter)**, not InfiniBand. EFA has libfabric/UCX support but is NOT plain RDMA-Verbs. NIXL on AWS will likely fall back to UCX-over-EFA or TCP — performance won't match a bare-metal H100 cluster with InfiniBand. This is something the spike must measure empirically.

### 2.5 Planner (SLA-driven autoscaler)

This is what differentiates Dynamo from KEDA + composite KV triggers.

**What it does:**
- Reads frontend `/metrics`: requests, ISL, OSL, TTFT, ITL.
- **Forecasts future load** using one of: ARIMA, Prophet, or constant predictor.
- **Scales prefill and decode pools independently** based on which SLO is at risk:
  - TTFT breach → add prefill workers
  - ITL (inter-token latency) breach → add decode workers
- Uses **performance interpolation tables** built from a **pre-deployment profiling** step (you run AIConfigurator/AIPerf once; it sweeps batch sizes and seq lengths and learns the throughput surface).

**Knobs:**
- `adjustment_interval`: time between scaling decisions (must be > scale-up settling time, otherwise overlapping ops).
- TTFT/ITL targets.
- Predictor algorithm.

**vs KEDA composite KV trigger (Phase 4):** KEDA is reactive — when KV utilization crosses a threshold, scale. Planner is **predictive** (ARIMA forecasts the next interval) AND **inference-aware** (separates P/D scaling, knows that adding a prefill worker won't help an ITL breach). Limitation: SLA Planner currently **only supports disaggregated setups**; load-based planner is the fallback for co-located.

### 2.6 Workers (vLLM, TRT-LLM, SGLang)

Workers are inference engines wrapped in a Dynamo runtime that:
- Registers itself with etcd.
- Publishes KV block events to NATS.
- Implements the prefill-handoff protocol over NIXL.
- Reports per-request metrics.

All three engines support disaggregated, KV-aware routing, and SLA Planner as of v1.0. KVBM is GA on TRT-LLM and vLLM, in-progress on SGLang.

### 2.7 KVBM (KV Block Manager)

Multi-tier KV cache, conceptually like Linux page cache + swap:
- **HBM** (hot): active blocks
- **CPU RAM** (warm): recently evicted, fast re-promotion
- **NVMe** (cool): per-node SSD
- **S3 / Azure Blob** (cold): cross-cluster reuse

Lets you serve much longer effective contexts than HBM alone by paging KV blocks. Eviction policy balances re-prefill cost vs storage latency.

### 2.8 etcd + NATS (control plane + event bus)

- **etcd**: service discovery — workers register; router and planner watch for membership changes. Same role as in vanilla K8s control plane.
- **NATS**: pub/sub messaging — KV events (`block_added`, `block_evicted`), request routing decisions, planner control signals. Lower-overhead than gRPC streaming for fan-out events.

These are standard CNCF components, not bespoke. A minimal Dynamo deploy needs both.

---

## 3. Comparison to Mohamed's Current Stack (Phase 3 / Phase 4)

| Capability | Phase 3/4 today | Dynamo equivalent | Verdict |
|---|---|---|---|
| Routing | RabbitMQ pull-based (workers consume from queue) | Smart Router push-based + KV-aware | **Replaced.** RabbitMQ goes away; KV awareness is a strict superset of pull's load-balancing properties for prefix-heavy workloads. |
| Autoscaling | KEDA composite KV trigger (KV util + queue depth) | SLA Planner (predictive, P/D-aware) | **Replaced** for disaggregated mode. Could keep KEDA as a backup safety scaler. |
| Disaggregation | None (vLLM does P+D on same GPU) | Built-in via NIXL | **New capability.** This is the headline reason to even look at Dynamo. |
| Cold start | FSR + SOCI + streamer (Phase 4.1 work) | ModelExpress (NIXL-based weight streaming, claims 7x) | **Complementary.** Worth comparing head-to-head with Phase 4.1 numbers (212-242s on streamer/D/E config). |
| Admission control | RabbitMQ adaptive strategy (drop on vLLM error, requeue=False) | None built in (Phase 7 territory) | **Phase 3 stays relevant.** Dynamo doesn't do SLO-aware admission like Mooncake/QLM. |
| Observability | Prometheus + Grafana + DCGM | Dynamo exports `/metrics` in Prom format; same Grafana works | **Complementary.** |
| K8s orchestration | Plain Deployments + KEDA ScaledObject | Dynamo Operator + DynamoGraphDeployment CRD + Grove | **Replaced** if going all-in. |

**What stays from Phase 3/4:**
- Karpenter for GPU node provisioning.
- DCGM + Prometheus + Grafana stack.
- The whole admission-control / failed-request-retry research from Phase 3 — Dynamo doesn't address this.
- AMI / FSR / cold-start optimizations from Phase 4.1 (worth A/B-ing against ModelExpress).

**What gets replaced:**
- RabbitMQ + worker pull-loop.
- KEDA ScaledObjects (or kept as belt-and-suspenders).
- Custom routing logic.

---

## 4. Deployment Options on Kubernetes

### 4.1 Install paths

1. **Container** (fastest): Prebuilt runtime images per backend:
   - `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.x`
   - `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.0.x`
   - `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.x`
   - These are large (likely 15-25 GB; not officially documented — verify in spike).
2. **PyPI**: `uv pip install --prerelease=allow "ai-dynamo[vllm]"` for local dev.
3. **Helm chart + Operator** (production): installs the Operator, API Store, NATS, PostgreSQL, MinIO via a single command from the `ai-dynamo/dynamo` repo's `deploy/cloud/helm` path.
4. **AWS Labs blueprint**: `awslabs/ai-on-eks` repo has a Dynamo blueprint that provisions EKS + Karpenter + EFS/FSx + the operator in 15-30 minutes via `./install.sh`. **This is the path of least resistance for the spike.**

### 4.2 CRDs

- **DynamoGraphDeployment** (DGD): the primary CRD. Specifies model, backend, replicas (prefill/decode separately), SLA targets, GPU type. The operator translates this into Deployments, Services, etc.
- **DynamoGraphDeploymentRequest** (DGDR): higher-level "I want model X with SLA Y" — operator + AIConfigurator auto-profile and emit an optimized DGD.
- **Grove API**: topology-aware scheduling for rack-scale (NVL72-style) deployments. Probably not relevant on g5/g6.

### 4.3 GPU node requirements

- Officially supported on AWS: **P6, P5, P4d, P4de, G5, G6, G6e**.
- For Phase 4.5 spike: **g6.xlarge or g6.12xlarge** is the cheapest path (L4 GPUs). **g5.xlarge** also works.
- Disaggregation needs **at least 2 GPUs** to be meaningful (one prefill, one decode). Smallest sensible spike: 2x g6.xlarge or 1x g6.12xlarge (4 GPUs) so prefill and decode can be on different physical GPUs.
- For NIXL to be fast: same-node = NVLink (great), cross-node on AWS = EFA + UCX (decent) or TCP (slow). **g6.xlarge is single-GPU per host**, so cross-pod KV transfer would go over EFA/TCP.

### 4.4 Dependencies

Required on the cluster:
- etcd (via Helm)
- NATS (via Helm)
- PostgreSQL + MinIO (for API Store / artifacts)
- NVIDIA GPU Operator (drivers, device plugin, DCGM) — already on Phase 4 cluster
- Optionally: EFA device plugin for AWS, EFS CSI for shared model storage

### 4.5 Image size / pull-time gotcha

Phase 3 saw 9.5 GB images cause cold-start pain. Dynamo runtime images are likely **larger** (vLLM + Dynamo Rust runtime + NIXL + UCX + CUDA). Expect 15-25 GB. **Use ECR pull-through cache** (already set up in Phase 4) or pre-bake into AMI.

---

## 5. Interview-Relevant Talking Points

Concrete things to be ready to discuss:

1. **Why disaggregation matters: the prefill/decode resource mismatch.** Prefill is FLOPs-bound, decode is bandwidth-bound. Co-located = head-of-line blocking on TBT when a long prompt arrives. Disaggregated = SLAs hold independently. Cost win: decode runs on cheaper GPUs.

2. **When KV-aware routing helps vs. hurts.**
   - Helps: long shared system prompts (RAG, agentic workflows), multi-turn chat with cache reuse, prefix caching workloads. 2x TTFT improvement is realistic.
   - Hurts: random short prompts (no cache to share — overhead of routing logic for nothing), high-churn workloads where cache state changes faster than the router learns.
   - Failure mode: greedy local optimization can hot-spot a worker that has the best cache, starving others.

3. **NIXL avoids GPU-CPU-GPU copies via RDMA/NVLink kernel-bypass paths.** On AWS, no InfiniBand → falls back to UCX-over-EFA or TCP, which IS a CPU bounce. This is the AWS-specific tradeoff vs DGX Cloud.

4. **Planner vs reactive HPA/KEDA.** KEDA scales when a metric crosses a threshold — by which time SLA may already be broken. Planner forecasts next interval (ARIMA/Prophet) AND knows TTFT breach → scale prefill, ITL breach → scale decode. Requires offline profiling step, which is the cost.

5. **Failure modes to discuss:**
   - Smart Router stale state: if a worker crashes silently, router's load count is wrong until heartbeat.
   - NIXL transfer failures during disagg: a request can be stuck after prefill if decode worker dies before KV arrives. Dynamo 1.0 added canary health checks + request migration.
   - Hot-spotting on KV-aware routing: one worker holds the popular prompt, gets all traffic, starves on decode. Mitigation: `--router-kv-overlap-score-weight` knob.
   - Head-of-line blocking still possible *within* a prefill worker (one giant 100k-token prompt). Disagg helps the decode side, not the prefill side itself — that's still a chunked-prefill problem.

6. **Why Anthropic might care.** Anthropic runs long-context Claude with huge KV caches and high prefix sharing (system prompts, tool defs, doc-grounded chat). KV-aware routing + disaggregation is exactly the workload shape this targets. But Anthropic has its own internal stack — Dynamo knowledge maps the *concepts* to whatever they call them.

7. **What I'd skip in production.** KVBM with S3 backing is interesting but cross-region S3 latency (50-500ms) makes it useful only for cold archive, not hot tiers — unlikely to beat just re-prefilling for most workloads.

8. **How this composes with my Phase 3 admission control work.** Dynamo doesn't do SLO-aware admission (Mooncake-style early rejection) or starvation prevention (LtR + aging). Those remain my contribution at the gateway tier — Dynamo runs *behind* that gateway.

9. **Operational maturity caveat.** Dynamo is ~14 months old as of GTC 2025. v1.0 just shipped. Production deployments outside of NVIDIA reference designs are still rare. An EM hire would advocate "evaluate, don't bet the farm yet" — exactly the Tier-0 instinct.

10. **The "router as a database" framing.** The Smart Router's radix tree is essentially a distributed cache index. It has all the same problems: consistency under churn (rebalancing), hotness (some keys get all traffic), staleness (events lag). This is the AWS systems-design lens — and where the interesting tradeoffs live.

---

## 6. Hands-On Spike Scope (1-2 days)

**Smallest thing that exercises the interesting parts:** disaggregated vLLM with KV-aware routing, on the existing Phase 4 EKS cluster. NOT a multi-node InfiniBand setup — that's not what AWS is, and it's not the day job.

### Recommended setup

- **Cluster:** Existing `inference-phase4` EKS (re-provision; current state is torn down per `project_cluster_state.md`).
- **Nodes:** 2x g6.12xlarge (4x L4 GPUs each, same-node NVLink possible) **OR** 4x g6.xlarge (cheaper, but cross-pod KV goes over network — actually a more interesting test of the AWS gotcha).
- **Install path:** `awslabs/ai-on-eks` Dynamo blueprint. Don't hand-roll the Helm chart on day 1.
- **Model:** Qwen2.5-7B (already used in Phase 3/4, known baselines). Consider Llama-3.1-8B for a second data point.

### Three concrete experiments

**Experiment A: Co-located baseline (no disagg, no KV router)**
- Deploy DynamoGraphDeployment with 2 vLLM workers, round-robin routing.
- Run the Phase 3 graduated load ramp.
- Capture: TTFT (p50/p99), TBT, throughput tokens/sec, GPU util.
- Purpose: establish that Dynamo-without-its-features matches naked vLLM.

**Experiment B: KV-aware routing (still co-located)**
- Same topology, enable Smart Router with KV events.
- **Workload:** prompts with shared 2KB system prompt prefix (simulate RAG / agentic).
- Capture: cache hit rate (Dynamo `/metrics`), TTFT delta, p99 TTFT.
- **Hypothesis:** 30-50% TTFT reduction on shared-prefix workload, neutral on random prompts.

**Experiment C: Disaggregated (prefill/decode split)**
- 2 prefill workers + 2 decode workers, NIXL transport (UCX/TCP since no IB on AWS).
- Same workload mix as B, plus a "long prompt + short output" stress test (8k input, 100 output) to expose head-of-line blocking that disagg fixes.
- Capture: TTFT, ITL, NIXL transfer time per request, decode GPU util vs prefill GPU util.
- **Hypothesis:** ITL stable under bursty long-prompt arrivals; co-located baseline (A) shows ITL spikes.

### Metrics dashboard

Reuse Phase 2 Grafana. Add panels for:
- Dynamo router cache hit rate (`dynamo_router_kv_overlap_blocks_total`)
- Per-pool replica counts (prefill vs decode)
- NIXL transfer latency histogram
- TTFT p50/p99 split by router decision (cache hit vs miss)

### Gotchas to watch for

1. **Image pull time:** runtime images are big. Pre-pull or use ECR pull-through. Phase 4 lessons apply.
2. **NIXL on AWS:** no InfiniBand. Will use UCX-over-EFA or TCP. Expect 5-20ms KV transfer overhead — this changes the disagg math. Document it; don't try to "fix" it in the spike.
3. **Karpenter GPU nodes:** Phase 4 already provisions g6 in <60s. Make sure nodepools tag the GPU type Dynamo expects.
4. **etcd + NATS:** small but stateful. Use the Helm chart defaults, don't try to share with anything else.
5. **Pre-deployment profiling for SLA Planner:** required step. Plan to run AIPerf for ~30 min before turning on the planner. **Or skip Planner entirely in this spike** — it's not needed for the core "does KV routing + disagg work?" question.
6. **NCCL between pods:** for multi-GPU TP within a pod = fine (intra-host NCCL). For TP across pods on different nodes = NCCL over EFA, needs the EFA device plugin and `hostNetwork: true` or proper IPC. Avoid cross-pod TP in the spike.
7. **Storage for model weights:** EFS works but slow on first load. Pre-stage onto FSx or use Phase 4.1 streamer setup if compatible.
8. **DGD vs DGDR:** start with hand-written DGD. Auto-generated DGDR is a bonus if time permits.

### Out of scope for the spike

- KVBM tiered storage (S3 backend) — too much setup for the value.
- Multi-node tensor parallelism — irrelevant on g6.
- TRT-LLM or SGLang backends — stick to vLLM (consistency with Phase 2-4).
- Production hardening (mTLS, RBAC, network policy) — that's Phase 7.
- Replacing KEDA — keep it disabled but installed; don't tear out existing infra.

---

## 7. Open Questions / Risks

Things research couldn't fully answer; the spike will have to figure out empirically:

1. **What's the actual NIXL transfer time over EFA vs TCP on AWS?** Docs assume InfiniBand. AWS blog post is silent on this. **Probably the single biggest unknown.** If NIXL-over-TCP is 50ms+, disagg's economics on AWS collapse for short-prompt workloads.

2. **Runtime image size?** Not officially documented. Could be 15-25 GB. Affects cold-start meaningfully.

3. **How invasive is DGD config?** Sample DGDs aren't in the docs I could fetch (some doc URLs 404). Need to read the operator source / examples directory.

4. **SLA Planner profiling time and disk footprint?** AIPerf sweep duration, where profile artifacts live, whether they survive worker upgrades — all unclear.

5. **Does Smart Router fail open?** If etcd is unreachable, does routing degrade to round-robin or stop entirely? Tier-0 question.

6. **Can the Dynamo Operator coexist with KEDA on the same workloads?** Or do they fight over scaling? (Likely fight; need to disable KEDA on Dynamo-managed Deployments.)

7. **vLLM version coupling.** Dynamo 1.0.2 ships its own vLLM runtime. Can we pin to a specific vLLM version we benchmarked in Phase 2? Unclear from docs.

8. **Cost.** g6.12xlarge spot is ~$1.60/hr. 4 of them for a 1-2 day spike = ~$130-260. Reasonable but not free.

9. **Planner vs KEDA in the same cluster.** If we keep both for safety, who wins? Need to test or set Planner as the only owner of Dynamo-namespaced replicas.

10. **What metrics does Smart Router emit to Prometheus by default?** Need exact metric names to wire into Grafana.

---

## 8. Sources

- [GitHub — ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)
- [NVIDIA Blog — Introducing NVIDIA Dynamo (GTC 2025 launch)](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA Blog — How Dynamo 1.0 Powers Multi-Node Inference at Production Scale](https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/)
- [NVIDIA Blog — Dynamo 0.4 (4x perf, SLO autoscaling, observability)](https://developer.nvidia.com/blog/dynamo-0-4-delivers-4x-faster-performance-slo-based-autoscaling-and-real-time-observability/)
- [NVIDIA Dynamo product page](https://www.nvidia.com/en-us/ai/dynamo/)
- [NVIDIA Dynamo Developer page](https://developer.nvidia.com/dynamo)
- [Dynamo Docs — KV Cache Aware Routing](https://docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing)
- [Dynamo Docs — Router Component](https://docs.nvidia.com/dynamo/components/router)
- [Dynamo Docs — Disaggregated Serving](https://docs.nvidia.com/dynamo/v-0-7-1/design-docs/disaggregated-serving)
- [Dynamo Docs — KV Cache Transfer (TRT-LLM)](https://docs.nvidia.com/dynamo/latest/backends/trtllm/kv-cache-transfer.html)
- [Dynamo Docs — SLA Planner](https://docs.nvidia.com/dynamo/latest/architecture/sla_planner.html)
- [Dynamo Docs — Autoscaling on Kubernetes](https://docs.nvidia.com/dynamo/dev/kubernetes/autoscaling.html)
- [AWS Blog — Accelerate generative AI inference with NVIDIA Dynamo and Amazon EKS](https://aws.amazon.com/blogs/machine-learning/accelerate-generative-ai-inference-with-nvidia-dynamo-and-amazon-eks/)
- [AKS Blog — Scaling multi-node LLM inference with Dynamo on AKS](https://blog.aks.azure.com/2025/10/24/dynamo-on-aks)
- [AKS Blog — Dynamo on AKS Part 2](https://blog.aks.azure.com/2026/01/22/dynamo-on-aks-part-2)
- [Spheron Blog — NVIDIA NIXL and Disaggregated Inference](https://www.spheron.network/blog/nvidia-nixl-disaggregated-inference-guide/)
- [WEKA Blog — WEKA Accelerates AI Inference with Dynamo and NIXL](https://www.weka.io/blog/ai-ml/weka-accelerates-ai-inference-with-dynamo-and-nvidia-nixl/)
- [VAST Data Blog — Accelerating Inference](https://www.vastdata.com/blog/accelerating-inference)
- [vLLM Docs — NixlConnector Usage](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/)
