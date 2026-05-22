# LLM Inference Infrastructure Learning Lab — Comprehensive Project Summary

A hands-on, multi-stage build of a production-style LLM inference platform on AWS, designed to develop deep operational understanding of model serving infrastructure. Built by a Senior EM from AWS Auto Scaling preparing for an Engineering Manager, Cloud Inference role at Anthropic. Every stage built to *understand*, not to ship — every artifact explainable in an interview.

**Cumulative work:** 4 stages, 4 isolated EKS clusters, 8 vLLM tests on M4, 29 Stage-2 vLLM experiments, 4 backpressure rounds, 4 KEDA scaling policies, 7-stage cold-start optimization, ~$30–50 GPU spend across ~6 weeks of calendar time.

---

## 1. What We Built

### Stage 1 — Local baseline (Ollama on Apple M4)
- **Hardware:** Apple M4, 10 cores (4P + 6E), 16 GB unified memory
- **Engine:** Ollama v0.18.0 on `localhost:11434`
- **Model:** Qwen2.5:7B (Q4 GGUF, 4.7 GB on disk)
- **Tooling:** `llmfit` to validate fit (50% memory, "Perfect"), custom Python streaming load tester capturing TTFT, TBT, and per-token timing
- **Goal:** establish bottom-line "what does serving feel like with no batching, no scheduling, single-threaded llama.cpp"

### Stage 2 — vLLM on AWS GPU (g4dn.xlarge Spot, NVIDIA T4 16 GB)
- **Hardware:** g4dn.xlarge Spot (~$0.16–0.22/hr), us-east-1f
- **Engine:** vLLM v0.17.1 in Docker, OpenAI-compatible API on `:8000`
- **Models tried:** Qwen2.5-7B-Instruct (FP16 OOM'd), Qwen2.5-7B-Instruct-AWQ (4-bit, ~4 GB), Qwen2.5-7B-Instruct-GPTQ-Int8, Qwen2.5-7B FP8 / FP8-E4M3 / FP8-E5M2 KV cache variants
- **Persistence:** `~/.cache/huggingface` mounted as Docker volume so model survives container restarts
- **Test harness:** custom Python with `aiohttp`, streaming SSE parsing, structured timing, 60-question quality benchmark across 7 categories (Math, Factual Recall, Instruction Following, Code, MMLU-style, GSM8K-style, HumanEval-style), saved as JSON per test
- **30+ experiment scripts**: `stage2_test1` through `stage2_test30c` — each one a self-contained Python file with results stored alongside

### Stage 3 — Production Inference Platform on EKS

**Phase 1 ✅ — EKS + Karpenter + queue architecture** (`phase1/`)
- **Cluster:** `inference-lab` (later updated to K8s 1.30), VPC `10.0.0.0/16`, 2× t3.medium control plane nodes
- **Karpenter v0.34.0** provisioning g4dn.xlarge Spot for inference, NVIDIA device plugin exposing `nvidia.com/gpu: 1`
- **Async architecture:**
  ```
  Client → FastAPI gateway → RabbitMQ queue → vLLM worker (sidecar pod) → Redis pub/sub → SSE stream back
  ```
- **Sidecar pattern:** vLLM container + Python `worker.py` in same pod — worker hits vLLM at `localhost:8000`, atomic scaling, simpler graceful drain, no cluster networking overhead
- **`prefetch_count=5`** on the worker → flow-control mechanism (broker pushes via `basic_consume`, holds queue depth as scaling signal)
- **Terraform stack:** 35 files, 7 milestone gates with explicit acceptance criteria
- **IAM:** 10 managed policies + EKS inline policy on `vLLM-spot-lab` user; IRSA for Karpenter, EBS, ECR
- **Storage:** EBS PV for HuggingFace model cache (S3 Mountpoint dropped because it doesn't support file locking → switched to `emptyDir`)

**Phase 2 ✅ — Observability + KEDA autoscaling** (`phase2/`)
- **Prometheus + Grafana:** kube-prometheus-stack via Helm, `retentionSize: 5GB` (NB: Prometheus uses SI suffix `GB`, not K8s binary `Gi`)
- **DCGM Exporter:** `/dev/nvidia*` device mounts required (without them, DCGM starts but reports 0% for all GPU metrics)
- **vLLM `/metrics`** scraped natively: queue depth, TTFT, TBT, KV cache %, running requests, prefix cache hit rate, num_preempted
- **Custom Grafana dashboard** with 4 rows: GPU Hardware, vLLM Internals, Request Flow, Autoscaling State (later expanded to 5 rows / 18 panels in Phase 4)
- **KEDA** installed for queue-depth-based pod autoscaling → `ScaledObject` per worker deployment
- **Karpenter NodePool** with disruption budget `nodes: 1` and `consolidateAfter: 300s` to match KEDA cooldown

**Phase 3 ✅ — Admission control & backpressure** (`phase3/`, isolated cluster `inference-phase3`, VPC `10.1.0.0/16`)
- **Pluggable admission strategies** in `worker.py` via `ADMISSION_STRATEGY` env var: `static`, `threshold`, `per_request`, `predictive`, `reactive`
- **`AdmissionTracker` class:** local atomic state replacing the original (broken) remote `/metrics` polling architecture — tracks `running` jobs dict by `job_id` with `estimated_kv`, `remaining_tokens`, `try_admit()`, `update_progress()`, `release()`
- **Test runner** (`backpressure_comparison.py`): per-workload rates, Pareto random distributions (seeded for reproducibility), burst patterns, SLA tracking, cost-per-token tracking, incremental JSON saves, Grafana screenshot capture
- **HTML dashboards:** `phase3_results_dashboard.html`, `phase3_results_v3_dashboard.html`, `phase3_complete_results.html` — narrative-driven explanation of each round
- **Models tried:** 7B AWQ (everywhere), 14B AWQ for Round D KV-stress (the only setup where predictive admission wins)
- **Rounds executed:** v1 (9 runs uniform), v2 (15 runs varied rates), v3 (12 runs Pareto), Round A/B/C/D second-pass with cleaner admission tracker

**Phase 4 ✅ — Scaling policy comparison + cold start optimization** (`phase4/`, isolated cluster `inference-phase4`, VPC `10.4.0.0/16`)
- **Karpenter v0.37.0** (upgraded from v0.34.0 for K8s 1.30 compat — v0.34.0 marks EC2NodeClass for deletion every reconcile loop on 1.30)
- **EKS access entries** (v20 module) replacing the old aws-auth ConfigMap — Karpenter custom node role must be explicitly added or kubelet fails to authenticate silently
- **4 KEDA scaling policies:** Policy A (queue-only depth=5), Policy B (queue-eager depth=3), Policy C (composite KV>0.80), Policy D (composite aggressive KV>0.65)
- **Pareto load generator** (80% short / 15% medium / 4% long / 1% XL) replacing uniform workloads
- **7-phase graduated load ramps:** 0.8 → 1.5 → 2.5 → 3.0 req/s with SLA + cost-per-token tracking per phase
- **Cold-start benchmark harness:** 8 cumulative configs, per-stage timing via K8s events (`Scheduled`, `Pulled`) + worker log markers (`WARMUP: vLLM is ready`, `WARMUP: Complete`, `READY: Created`) + pod conditions (`Ready=True lastTransitionTime`)
- **Custom AMIs via Packer:** SOCI snapshotter AMI (`ami-094977f5e7ac485f8`) + non-SOCI fallback (`ami-08d6b00352f604240`)
- **Custom vLLM image:** extends `vllm/vllm-openai:v0.19.0` with `pip install runai-model-streamer runai-model-streamer-s3`
- **vLLM Sleep Mode L1** wired into worker idle monitor (300s idle → `POST /sleep`; new message → `POST /wake_up`)
- **Graceful drain:** tracked `active_tasks: set[asyncio.Task]` + `add_done_callback` for auto-cleanup, `asyncio.wait` on SIGTERM, `DRAIN_TIMEOUT=170s < terminationGracePeriodSeconds=180s`, `preStop: ["sleep", "15"]` on vLLM sidecar
- **Compilation cache:** `hostPath` mount at `/root/.cache/vllm` survives pod restarts on the same node (regenerates in 7–30 s if lost; not worth EFS overhead at $0.30/GB/month + 2–5 ms latency penalty)

**Stage 4 (planned)** — Inference Playground: interactive platform comparing optimization techniques, engines (vLLM, TensorRT-LLM, SGLang), and scaling strategies

---

## 2. What We Measured

### Stage 1 — Ollama on M4 (3 concurrent, sequential)
- **Cold start (model load):** 52.5 s (4.7 GB → Metal GPU memory)
- **Per-request:** TTFT 14.6 s / 99.7 s / 177.5 s — proves single-threaded queuing
- **Throughput:** 0.3–0.4 tok/s vs theoretical max ~21 tok/s (M4 base 100 GB/s memory bandwidth ÷ 4.7 GB model). The 70× gap is llama.cpp's bolted-on Metal backend, not the hardware.
- **Chat template overhead:** 6-word prompt expands to 35 tokens after Ollama wraps it in Qwen's `<|im_start|>` template

### Stage 2 — vLLM on T4, apples-to-apples vs Ollama on T4
| | Ollama on T4 (3 concurrent) | vLLM on T4 (5 concurrent) |
|---|---|---|
| TTFT spread | 1.32 s | **0.176 s** |
| Gap between consecutive first-tokens | 0.66 s | **0.009 s** |
| Per-request tok/s | ~41 | ~44 |
| Gap/Total ratio | ~0.72 (queuing) | **0.01** (batching) |
- **Tok/s nearly identical** — hardware dictates decode speed. Difference is entirely scheduling.
- **Prefix cache hit rate:** 32.7% when 5 requests share a prompt
- **Theoretical max concurrency at 2048 tokens:** **38.5×** (KV budget = 10.5 GB / ~802 MB per request at full context)

### Stage 2 — Quantization comparison (5 concurrent, T4)
| Metric | AWQ INT4 | AWQ_Marlin INT4 | GPTQ INT8 |
|---|---|---|---|
| Tok/s per request | 43.8–44.7 | 45.2–45.4 | 26.6–27.6 |
| TTFT (min) | 0.547 s | **0.291 s** | 0.831 s |
| Avg TBT | 22.5 ms | 22.1 ms | 37.5 ms |
| Model VRAM | ~4 GB | ~4 GB | 8.3 GB |
| KV headroom | ~10 GB | ~9.3 GB | ~4.2 GB |
- Marlin: ~3% decode improvement, **47% faster prefill** — same data on disk, different GPU kernel reading INT4 in a bandwidth-optimized layout

### Stage 2 — Quality benchmark (60 questions, 7 categories, temperature=0)
| Category | Q | AWQ INT4 | GPTQ INT8 | INT4 latency | INT8 latency |
|---|---|---|---|---|---|
| Math Reasoning | 10 | 10/10 | 10/10 | 0.46 s | 0.73 s |
| Factual Recall | 10 | 10/10 | 10/10 | 0.43 s | 0.68 s |
| Instruction Following | 10 | 10/10 | 10/10 | 0.36 s | 0.62 s |
| Code Generation | 5 | 5/5 | 5/5 | 2.84 s | 4.14 s |
| MMLU-Style | 10 | 10/10 | 10/10 | 0.23 s | 0.39 s |
| GSM8K-Style | 10 | 10/10 | 10/10 | 4.22 s | 7.08 s |
| HumanEval-Style | 5 | 5/5 | 5/5 | 5.39 s | 7.77 s |
| **OVERALL** | 60 | **60/60** | **60/60** | **0.77 s** | **2.58 s** |
- INT4 = INT8 quality at 7B; INT4 ~60% lower avg latency

### Stage 2 Test 9 — Prefix caching A/B
- Cold TTFT: 0.335 s, warm TTFT: 0.192 s → **1.75× speedup** on repeated prompt prefix (3 trials × 10 requests serial)
- Concurrent 10× same prefix: TTFT p50 0.277 s, p99 0.293 s, **0 failures over 30 requests**

### Stage 2 Test 12 — Concurrency cliff (max_model_len=512)
- vLLM reported max concurrency: **249**
- Concurrent 10: 26.3 tok/s system; concurrent 20: 423.7 tok/s; concurrent 60: 765 tok/s; concurrent 80: 769 tok/s (saturation point)
- TTFT p50 grew gracefully: 10.6 s → 0.37 s → 0.57 s (the 10-concurrent number is anomalous — first-batch cold path)

### Stage 2 Test 16 — Chunked prefill A/B (300-token decode "bomb" injected mid-decode)
- Without chunked prefill: TBT spike ratio 1.67× during prefill bomb (decode latency more than doubled)
- With chunked prefill: TBT spike ratio 1.02× — **zero perceptible degradation**, exactly Sarathi-Serve's claim

### Stage 2 Test 21 — Sampling parameters at fixed 200-token output
- Greedy / temp 0.3 / temp 0.7 / temp 1.0 / top_k=50: TTFT 0.19–0.21 s, tok/s 46.1–46.7, TBT 21.5–21.8 ms
- **Sampling has near-zero perf cost** — the bottleneck is decode kernel, not sampler logic

### Stage 2 Test 25 — Latency curve at sustained RPS
| RPS | Total req | TTFT p50 | TTFT p99 | total p50 | total p99 | system tok/s |
|---|---|---|---|---|---|---|
| 1 | 60 | 0.21 s | 0.25 s | 1.98 s | 2.45 s | 79.4 |
| 2 | 120 | 0.22 s | 0.24 s | 2.04 s | 2.54 s | 156.0 |
| 4 | 240 | 0.22 s | 0.24 s | 2.23 s | 2.86 s | 313.2 |
| 8 | 480 | 0.24 s | 0.27 s | (...) | (...) | 600+ |

### Stage 2 Test 26 — 15-min soak at 4 RPS
- 3,600 requests, **0 failures**
- Steady state from minute 2 onward: TTFT p99 ~0.26 s, total p99 ~3.6 s, ~430 tok/s sustained
- VRAM steady at 14,527 MB — no leak over 15 min

### Stage 2 Test 29C — FP8 KV cache (E5M2) on T4 — quality collapse
| Category | Pass | Accuracy | Notes |
|---|---|---|---|
| MMLU-Style | 10/10 | 100% | Short answers fine |
| Math Reasoning | 5/10 | 50% | "17 × 23 = 391" → "177 + 22 = 199" |
| Factual Recall | 6/10 | 60% | Truncates / hallucinates |
| Instruction Following | 4/10 | 40% | Fails complex instructions |
| GSM8K-Style | 1/10 | 10% | Multi-step reasoning broken |
| HumanEval-Style | 0/5 | 0% | Code completely garbled |
| **OVERALL** | **29/60** | **48.3%** | (vs 100% baseline) |
- Throughput extends to N=128 concurrent but accuracy unusable — **never use FP8 E5M2 KV on T4 for production**
- E4M3 variant (Test 30c) was tested and behaved better but still problematic — recommendation: use FP16 KV + prefix caching instead

### Phase 2 — Grafana under sustained N=20 load
- **GPU compute: 95%**, **Memory bandwidth: 94%** (T4's 320 GB/s near-saturated → confirms decode is BW-bound)
- **VRAM: ~14 GB / 16 GB** (4 GB model + ~10 GB KV)
- **Throughput: 118 tok/s** peak batched (vs 33 single-request)
- **TTFT p50 / p99:** 150 ms / 1 s; **TBT p50: 47 ms**
- **GPU temp: 46°C** (well below T4's 83°C thermal throttle)
- **KV cache: 1.66%** at peak — for 7B AWQ on T4, KV cache is *not* the bottleneck (this discovery later invalidated Phase 3 v1/v2 strategy designs)

### Phase 2 — KEDA autoscaling validation (N=15 burst)
- 1 worker (prefetch=5): max queue wait **16,798 ms**
- 2 workers: max queue wait **1,722 ms** = **90% reduction**
- Throughput per worker: ~33 tok/s (linear); total: ~66 tok/s = 2× scale-out

### Phase 2 — GPU scale-up latency breakdown
| Phase | Duration | Cumulative |
|---|---|---|
| KEDA detects queue_depth > 5 | 0–15 s | 15 s |
| Karpenter selects cheapest Spot | ~15 s | 30 s |
| EC2 Spot launch + boot | ~60 s | 90 s |
| Node joins cluster (kubelet + CNI) | ~10 s | 100 s |
| vLLM image pull (8 GB) | ~90 s | 190 s |
| Model download from HF (4 GB) | ~60 s | 250 s |
| Model load into VRAM | ~50 s | 300 s |
| **Total Pending → Ready** | | **~300 s (5 min)** |
- **70% of latency is model-related** (image + download + GPU load); infra is only 30%

### Phase 3 Round D — 14B AWQ KV-stress (kv_stress_14b at 0.15 req/s, the only round that differentiates strategies)
| Strategy | Tok/s | TPOT | Tracker | NACKs | SLA (124.6 ms TPOT) |
|---|---|---|---|---|---|
| Static pf=2 | 7.6 | 64.8 ms | 20.1% | 0 | ✅ |
| Static pf=3 | 14.9 | 67.9 ms | 30.1% | 0 | ✅ |
| Static pf=5 | 22.8 | 118.5 ms | 50.2% | 0 | ✅ |
| Static pf=8 | 23.2 | 130.1 ms | 80.4% | 0 | ❌ |
| Static pf=10 | 22.8 | 170.6 ms | 100.5% (over-saturated) | 0 | ❌ |
| Reactive 0.80 | 30.3 | 157 ms | 80.4% | **121K** | ❌ |
| **Predictive 0.50** | 22.7 | **94.8 ms** | 50.2% | 66K | ✅ |
| Predictive 0.80 | 30.3 | 149 ms | 80.4% | 120K | ❌ |
- **Only predictive @ threshold 0.50 keeps TPOT under SLA**; reactive has highest throughput but violates SLA with massive NACKs

### Phase 3 Rounds A/B/C — 7B AWQ on T4 (Static wins)
| Round | Metric | Winner | Tok/s | SLA |
|---|---|---|---|---|
| A | decode_throughput | Static pf=3 | 88 | 100% |
| B | gpu_compute | Static ≈ Reactive | 86 | 100% |
| C | compound | Static pf=3 | 88 | 100% |
- Reactive has NACK churn; predictive too conservative on 7B
- **Lesson: vLLM's internal scheduler is sufficient for 7B AWQ on T4 — external admission adds overhead, not value**

### Phase 4v2 — KV-stress phase across 4 KEDA policies
| Policy | Reqs Served | Tok/s | Burst TTFT |
|---|---|---|---|
| Queue-only A (depth=5) | 5/30 (17%) | 228 | 24–27 s |
| Queue-eager B (depth=3) | 5/30 (17%) | 228 | 24–27 s |
| Composite C (KV>0.80) | 19/30 (63%) | 825 | 0.49 s |
| **Composite D (KV>0.65)** | **23/23 (100%)** | **968** (4.3×) | **0.50 s** |
- Composite D + 7-phase graduated load: **$0.24 per 1M output tokens** end-to-end pipeline cost
- Scale-up latency across policies: 825–1,028 s

### Phase 4 — Throughput diagnostic A/B
- Direct vLLM on EKS (60 concurrent): 389.8 tok/s
- Full pipeline through queue + Redis pub/sub per token: 244.4 tok/s
- **Real overhead: 1.6×** (not the 5× initially feared from comparing apples-to-oranges with Stage 2's 1131 tok/s at 60 concurrent)

### Phase 4v3 — Cold-start optimization journey
| Stage | Phase 4v2 baseline | Phase 4v3 target | Technique |
|---|---|---|---|
| Node provision | 90 s | 90 s | (Karpenter already optimal) |
| Image unpack | 110 s | ~15 s (SOCI) / 44 s (containerd tuning) | Lazy load + parallel unpack |
| Model loading | 48 s | **14 s** (gp3) / **5 s** (S3) | Run:ai Model Streamer, 16/32 threads |
| Runtime init | 60 s | 7 s first / 0 s cached | Compilation cache + CUDA graph sizing |
| Readiness | 120 s | ~5 s | Startup probe + worker readiness file |
| **Total** | **~428 s** | **~131 s** (SOCI) / **~160 s** (no SOCI) | |
| Wake from sleep | N/A | **~0.5 s** (L1) / **~1.5 s** (L2) | vLLM Sleep Mode |

### Phase 4v3 — Actual measurements (sometimes worse than planned)
- **Warm node, all caches hit:** Pod scheduled → 2/2 Ready in **72 s** (init 5 s + model load 40 s + startup probe 5 s + warmup 20 s + readiness 2 s)
- **Warm node, S3 model download:** ~180 s
- **Full cold start on fresh node (vLLM v0.19.0 9.5 GB image):** **~510 s** — *worse* than v2 because the version upgrade ballooned the image
- **First image pull on fresh node:** 4.5 min for v0.19.0 (9.5 GB) vs 1.8 min for v0.4.1 (3.8 GB) — 2.5× regression from version bump alone

### Phase 4 cold-start benchmark — invalidated
- 8 cumulative configs ran but raw totals were non-monotonic: naked=743.9 s, +s3-init=625.8 s, +prebaked-ami=644.3 s, +warmup=661.1 s, +startup-probe=637.4 s, +soci=555.5 s, +streamer=574.6 s, +cache-hit=519.0 s
- **Three flaws identified:**
  1. Per-config fresh Spot instance → ±50–100 s infra variance > 7–34 s optimization deltas
  2. `parse_k8s_timestamp()` NoneType bug killed per-stage data
  3. Original assumption that `initialDelaySeconds:120` was additive was wrong — it overlaps with model loading; replacing with startup probe saves ~5 s, not ~115 s
- Redesign plan filed (`sprightly-shimmying-pumpkin`): build slim vLLM from source for `sm_75` only (~4 GB vs 9.5 GB) + reuse nodes within same-AMI groups (3 provisions vs 8)

---

## 3. What Surprised / Taught Something Non-Obvious

### Findings that flipped my mental model
1. **Ollama and vLLM tok/s on the same T4 are nearly identical (~41 vs ~44).** Hardware dictates decode speed. The 100× difference in user-perceived latency is *entirely* scheduling. This was the moment "continuous batching = a city bus that picks up at every stop" clicked.
2. **Decode is memory-bandwidth bound, not compute bound.** Confirmed in Phase 2 Grafana: GPU compute 95% AND memory bandwidth 94% simultaneously — the BW is the wall, compute is just along for the ride. This is why H100 HBM3 = 3.35 TB/s matters more than its FLOPS for inference.
3. **Pre-baking AMIs only saved ~7 s of cold start, not ~90 s.** The bottleneck wasn't network download — it was containerd decompressing layers to overlayfs on gp3 EBS at 125 MB/s. AWS-side mental model said "cache the artifact upstream"; reality said "the unpack is the wall." `containerd journal` confirmed NO `PullImage` call — image found locally; the 110 s was pure I/O.
4. **Adding admission control on top of vLLM made things worse, not better.** Static (no app-level gate) hit 100% success / 0 NACKs. Threshold (gate at KV>80%) hit 74% success / 69 NACKs. Double-gating: vLLM's PagedAttention + continuous batching already handle queueing. Counter to my "more guardrails = safer" instinct from large-fleet operations.
5. **Queue depth lies with variable-cost requests.** 50 short requests look scary (depth=50, oldest=10 s — users fine). 3 XL requests look fine (depth=3, oldest=90 s — users furious). LLM-inference equivalent of choosing SQS `ApproximateAgeOfOldestMessage` over `ApproximateNumberOfMessagesVisible`. Same lesson, new domain. Will switch to queue-duration via `rabbitmq_queue_head_message_timestamp` in Phase 5.
6. **The `MAX_BATCH_SIZE` check via remote `/metrics` polling was logically self-defeating.** If the gate enforces the limit, `running_requests` can never reach the limit — so the gate only fires through race conditions between async coroutines reading the same stale snapshot. Discovered through discussion, not Grafana.
7. **vLLM Sleep Mode is a latency optimization on EC2, NOT a cost optimizer.** You still pay $0.16/hr whether vLLM is active or sleeping. It's only a cost lever on per-GPU-second platforms (Modal, RunPod) or when packing multiple models on one GPU. Easy to mis-pitch in a "save 80% on GPU costs!" Slack post.
8. **`kubectl set env` overrides survive `kubectl apply`.** Phase 3 set `ADMISSION_STRATEGY=reactive` via `kubectl set env`; Phase 4's ConfigMap `envFrom` was silently ignored because explicit env entries take precedence. K8s strategic merge patches *add* to the env list, never replace. Fix: `kubectl delete deployment` then `kubectl apply` for a clean slate.
9. **Pareto workloads exposed pathologies that uniform load hid completely.** Phase 3 v1 (uniform short/medium/long at 2 req/s × 9 runs) showed zero strategy differentiation. Same code, same cluster, same runs — but with 80/15/4/1 Pareto + a 14B AWQ model in v3, predictive admission was the only config that kept TPOT under SLA. The 1% XL requests create disproportionate KV pressure that's invisible in uniform tests.
10. **vLLM version upgrade can REGRESS cold start.** v0.19.0 image is 9.5 GB compressed vs v0.4.1's 3.8 GB. First image pull on a fresh node: 4.5 min vs 1.8 min. The version bump alone added ~2.7 min to cold start before any optimizations could recover it. Always check image size before upgrading inference engine versions.
11. **The "5× pipeline overhead" was actually 1.6× — apples-to-oranges comparison fooled me.** Stage 2 local vLLM at 60 concurrent: 1,131 tok/s. Phase 4 EKS pipeline at 1.5 req/s: 232 tok/s. Looks like 5× overhead. Ran direct vLLM on EKS at 60 concurrent (390 tok/s) vs pipeline at same load (244 tok/s). Real overhead = 1.6×. Distinguished load shape from architecture cost.
12. **GPU utilization is the WRONG autoscaling signal.** Decode is BW-bound → SM utilization reads 20–40% even when saturated. Industry research (cited by Red Hat / Kedify): an autoscaler using SM utilization once scaled DOWN during decode-heavy load, spiking p99 from 200 ms to 1,200 ms.
13. **vLLM's waiting queue is unbounded.** No max queue length. No request timeout. Under sustained overload, queue grows until CPU OOM. RFC #18826 proposes `--max-waiting-queue-length` → HTTP 503, but PR #27064 still open. SGLang already has `--max-queued-requests` and `SGLANG_REQ_WAITING_TIMEOUT` — vLLM is behind here.
14. **Production GPU scaling defaults are wildly different from CPU.** GPU cooldown should be ~600 s (10 min, not 60 s). `minReplicas: 3` (not 1) because GPU pods take 5 min to start. Over-provisioning is the cost of meeting SLOs — the math doesn't work the other way.
15. **Static admission beats sophisticated admission for 7B on T4.** This was the most counterintuitive Phase 3 result. The "smart" approach (threshold/per-request gates polling vLLM metrics) lost on every 7B round. The "dumb" approach (just set prefetch=3 and let vLLM handle the rest) won every time. Lesson: trust the engine's scheduler unless you have evidence it's failing.
16. **Apple Silicon's M4 unified memory makes Ollama's slowness *more* surprising, not less.** Unified memory means no CPU↔GPU copies. MLX-native runtimes hit 15–20 tok/s on the same hardware where Ollama gets 0.3 tok/s. The bottleneck isn't physics — it's that llama.cpp was designed for CUDA's split memory model and Metal is a bolt-on.
17. **Karpenter consolidation can fight KEDA cooldown.** If `consolidateAfter` (60 s) < KEDA `cooldownPeriod` (300 s), Karpenter kills GPU nodes while KEDA still considers scaling — 5-min thrash loop. Same class of bug as ASG cooldown < scaling policy interval.
18. **PodDisruptionBudget `minAvailable: 1` deadlocks single-replica GPU rollouts.** Old pod can't be evicted (PDB blocks), new pod can't start (no second GPU). Rolling update hangs forever. Fix: either accept maxSurge=1 (need transient 2nd GPU node) or delete PDB during rollout.
19. **EKS module v20 silently breaks Karpenter node registration.** Replaced aws-auth ConfigMap with EKS access entries API. Managed node groups get auto-entries; Karpenter custom role does NOT. Without explicit `access_entries = { karpenter_node = ... }`, GPU nodes launch but kubelet can't authenticate — NodeClaims show "Node not registered" forever. Three-hour debug session learning this.
20. **`asyncio.create_task(...)` without tracking is fire-and-forget at SIGTERM.** Phase 3 abandoned in-flight SSE streams mid-token on pod kill — clients saw broken responses. Trivial bug, real-world impact. Fix: track tasks in a set, await with timeout on shutdown.

### AWS mental models that mapped cleanly
- KV cache = warm instance pool
- PagedAttention = OS virtual memory paging
- Continuous batching = bus with stops vs charter
- TTFT = time-to-first-byte
- Tensor parallelism = sharding across multi-AZ ASG
- Cold-start waterfall = ASG launch + bootstrap + service init
- Admission control = surge queue + connection draining
- Karpenter consolidation thrash = ASG cooldown vs scaling policy interval mismatch
- KEDA queue-depth scaling = SQS `ApproximateNumberOfMessagesVisible` target tracking
- KV-cache-aware routing = session-affine ALB (sticky sessions on prefix hash)
- Disaggregated P/D = separate ASGs for web (compute-bound) vs worker (I/O-bound)
- vLLM Sleep Mode = EC2 Hibernate (state preserved, instance still billed)
- Spot capacity exhaustion = the same problem Karpenter's instance diversification solves

### AWS mental models that broke or needed adjustment
- **"More replicas = better availability"** is muted by 5-min GPU cold start — research recommends `minReplicas: 3` so the cache-hit path (warm node, new pod) is the steady-state experience, not the full cold start. Cold start is for disaster recovery, not steady-state scaling.
- **"Stateless workers behind a load balancer"** doesn't fit when prefix caching means routing to the *right* worker matters. Pure round-robin destroys cache hit rate. Phase 5 territory: cache-aware routing.
- **"Reactive scaling on queue depth"** is the canonical pattern but fails with variable-cost requests. Need composite signals (queue + KV%) or queue-duration.
- **"Hard rate limit at the door"** (TGI's model) loses to predictive reservation (SGLang's `new_token_ratio`) under variable workloads — predictive smooths degradation; hard limits create cliffs.
- **"Continuous batching = always better"** isn't quite true — batching helps when GPU has memory headroom; under KV pressure, the same batching causes preemption cascades that hurt P99 more than serial would.
- **"Just provision more capacity to absorb bursts"** doesn't work for GPUs because Spot pools for V100 are *empty* (legacy hardware being decommissioned). G-family is the only viable Spot option for inference today.

---

## 4. Trade-Offs & Decisions

### Hardware / model choices
- **AWQ INT4 over GPTQ INT8 over FP16.** FP16 7B = 14.25 GB on a 14.56 GB usable T4 → 110 MB headroom = OOM. AWQ kept quality identical on the 60-question benchmark (60/60 vs 60/60) while leaving 10 GB for KV cache and giving ~60% lower latency. INT8 leaves only 4.2 GB for KV → ~2× fewer concurrent requests.
- **AWQ_Marlin over plain AWQ** when the kernel is available (vLLM logs literally say "Use quantization=awq_marlin for faster inference") — same data on disk, 47% faster prefill from a memory-access-pattern-optimized GPU kernel.
- **g4dn.xlarge T4 over A10G/L4** for Spot availability. P-family (V100) Spot pools are *empty* across all regions — legacy hardware decommissioning. G-family has abundant capacity. T4 is the cheapest GPU AWS still sells reliably on Spot.
- **Spot with on-demand fallback** in Karpenter capacity-type. Spot exhaustion in us-east-1a/1b during Phase 4 testing forced fallback config: `capacity-type: ["spot", "on-demand"]` instead of Spot-only.

### Architecture choices
- **Sidecar pattern (vLLM + Python worker in one pod) over separate Deployments.** Worker reaches vLLM on `localhost:8000` — no cluster networking, atomic scaling, simpler graceful drain. Trade-off: can't independently scale prefill vs decode (deferred to Phase 8 disaggregated serving).
- **RabbitMQ over Redis Streams** for queue semantics (acks, prefetch flow control, DLQ in future). Redis used for pub/sub token streaming back to client (low latency, no persistence needed).
- **Broker pushes via `basic_consume` with `prefetch_count`** (not pulling). This is the right model for keeping GPU 100% utilized — but limits cache reuse because broker round-robins to consumers without knowing which has cached prefixes. Resolved in Phase 5 with per-replica local queues behind a smart router.
- **`hostPath` for caches over EFS / PVC.** Model + compilation cache survive pod restarts on the same node, lost on node replacement. EFS adds 2–5 ms latency, $0.30/GB/month, and provisioning time. For a Spot-heavy lab, hostPath is the right call. Compilation cache (~100 MB) regenerates in 7–30 s anyway.
- **Removed `--enforce-eager` in Phase 4v3.** Phase 4v2 disabled CUDA graphs to skip 54 s of capture. With compilation cache + `--cuda-graph-sizes 1,2,4,8,16,24,32,64`, capture is one-time per node (~7 s) for 30% faster runtime inference. Net positive.
- **Separate Terraform stacks per phase.** Phase 1 (`inference-lab`, 10.0.0.0/16), Phase 3 (`inference-phase3`, 10.1.0.0/16), Phase 4 (`inference-phase4`, 10.4.0.0/16). After a Phase 4 deploy accidentally clobbered 4 of 6 configs in a running Phase 3 experiment, `deploy.sh` now refuses to run against the wrong context (pre-flight `kubectl cluster-info` check).

### Scaling decisions
- **Karpenter `consolidateAfter: 300 s` to match KEDA `cooldownPeriod: 300 s`.** Initial 60 s consolidation killed nodes while KEDA still considered scaling — 5-min thrash loop.
- **Composite KV trigger over queue-only.** Adds Prometheus dependency to autoscaling path, but queue-only is blind to GPU memory pressure when a few XL requests dominate. Worth the operational complexity for variable-cost workloads. Queue-only is fine if all your requests are uniform.
- **Disruption budget `nodes: 1`** so Karpenter only disrupts one GPU node at a time. Without this, cluster-wide consolidation could kill all GPU workers simultaneously.
- **`expireAfter: 6h` on GPU NodePool** to force Spot interruption simulation and prevent silently long-lived nodes from masking issues.
- **Dual `amiSelectorTerms`** (SOCI AMI tagged first, non-SOCI as fallback) so Karpenter prefers the optimized AMI but doesn't fail if it's missing.

### Things tried that didn't work
- **P-family Spot (V100):** zero capacity across all regions and instance types (p2, p3). Legacy hardware. Forced move to G-family.
- **SOCI on small images:** overhead > savings. Useful for 9.5 GB vLLM image, not for slim runtime images.
- **FP8 KV cache E5M2 on T4:** quality collapse to 48% (29/60). E4M3 better but still problematic. Recommendation: FP16 KV + prefix caching.
- **vLLM v0.19.0 vs v0.4.1:** image grew 3.8 GB → 9.5 GB, baseline cold start regressed before optimizations recovered it.
- **Pre-baked AMI alone:** saved 7 s, not the expected 90 s. Bottleneck is unpack, not download.
- **Remote `/metrics` polling in admission hot path** (Phase 3 v1/v2): added 2–5 ms per request, returned stale state, created positive feedback loop under load. Replaced with local `AdmissionTracker`.
- **`MAX_BATCH_SIZE` check via vLLM metrics:** logically self-defeating (gate enforcing the limit prevents the metric from reaching the limit). Discovered through discussion, removed from design.
- **S3 Mountpoint for HF cache:** doesn't support file locking → switched to `emptyDir`.
- **Karpenter v0.34 on K8s 1.30:** EC2NodeClass continuously marked for deletion, "karpenter version is not compatible with K8s version 1.30" error. Forced upgrade to v0.37.0.
- **Packer template with both `instance_type` + `spot_instance_types`:** plugin rejects with mutual exclusivity error. Use `spot_instance_types` only with `spot_price = "auto"`.
- **`snap-PLACEHOLDER` left in NodePool EBS config:** passes `kubectl apply` validation, fails at EC2 Fleet API call time with instant launch failure.
- **`maxSurge=0` + `minAvailable=1` PDB + 1 GPU available:** rolling update deadlock forever.
- **Karpenter v0.37.0 with Terraform-installed CRDs:** missing `status.conditions` field, "object is awaiting reconciliation" forever. Must apply CRDs manually from matching GitHub release tag.

---

## 5. Current State

### What's working today
- **Stage 1 ✅** complete and documented (`stage1_learnings.md`)
- **Stage 2 ✅** all 29 experiments complete (Tests 1–28 done 2026-03-19, Test 29 FP8 KV done 2026-04-05). JSON results + comprehensive `stage2_learnings.md`. Quantization comparison + quality benchmark results.
- **Phase 1 ✅** EKS + Karpenter + queue architecture validated on 3 load-test rounds with 0 failures. Queue wait < 5 ms at N=5, 12 s max at N=15 (proves need for autoscaling).
- **Phase 2 ✅** Prometheus + Grafana + DCGM + KEDA. 90% queue-wait reduction proven on autoscale (16,798 ms → 1,722 ms). Custom 4-row dashboard.
- **Phase 3 ✅** all 4 backpressure rounds complete. Key result: predictive admission only differentiates on 14B + KV stress; static wins on 7B. Local `AdmissionTracker` replaces flawed `/metrics` polling.
- **Phase 4 ✅** scaling policy comparison done (composite KV>0.65 wins by 4.3× under stress); cold-start optimization stack reduces 428 s → 131 s on paper; Sleep Mode wake = 0.5 s. v0.19.0 with Run:ai Streamer + SOCI infrastructure built.

### What's incomplete / open
- **Phase 4 cold-start benchmark results invalidated** — infra variance dominated 7–34 s optimization deltas. Redesign plan exists (`sprightly-shimmying-pumpkin.md`): build slim vLLM from source for `sm_75` only (~4 GB vs 9.5 GB), reuse nodes within AMI groups (3 provisions vs 8). Pending execution.
- **SOCI lazy loading not validated** — code complete, AMI built (`ami-094977f5e7ac485f8`), but never measured against non-SOCI baseline cleanly. SOCI AMI has duplicate `[proxy_plugins.soci]` in containerd config that needs dedup on first boot.
- **Sleep mode disabled during debugging** — needs verification in clean run.
- **Compilation cache cross-pod sharing** — works for same-node restart, not validated for cross-node EFS share.
- **Run:ai Model Streamer S3 path** — gp3 measurement done (14 s); S3 measurement (5 s expected) not yet captured.
- **Phase 4 prerequisites built but FSR ($1.50/hr) needs to be disabled** when not actively benchmarking.

### What's planned (Phase 5–8, research-driven)
- **Phase 5 — Smart routing & inference optimization:** push-based routing with per-replica local queues behind a smart router; cache-aware routing (hash system prompt → warm replica) inspired by SGLang's RadixAttention; speculative decoding (0.5B draft + 7B target); SGLang comparison (predictive reservation vs vLLM admit-everything); deficit-based fairness (D2LPM); evaluate Gateway API Inference Extension (K8s SIG)
- **Phase 6 — Multi-model serving & graceful degradation:** GPU bin-packing (1.5B classification + 7B generation co-resident, both fit on T4); independent KEDA scalers per model queue; tiered model fallback (7B saturated → route to 1.5B); CUDA Checkpoint/Restore for fast model swaps when driver support matures on EKS AMIs
- **Phase 7 — Production hardening:** API gateway with token bucket rate limiting (RPM + input/output TPM separately, like Anthropic's three-tier system); priority queues with starvation prevention (aging/boosting, ProServe-style adaptive urgency, NOT strict priority); prediction-based early rejection (Mooncake pattern: estimate `queue_wait + prefill + decode`, reject with 503 if > SLO at admission); token-aware rate limiting (cache reads 0.1×, cache writes 1.25× / 2.0× — Anthropic's weighted accounting); response length limiting under load; queue duration as scaling signal (RabbitMQ `rabbitmq_queue_head_message_timestamp`); KV cache eviction to CPU (QLM pattern); failed request retry strategies (requeue with retry counter, RabbitMQ DLX TTL backoff, priority boost on retry)
- **Phase 8 — Disaggregated inference:** prefill pool (compute-optimized) + decode pool (memory-optimized) via Ray Serve or NVIDIA Dynamo; KV cache transfer via NIXL/RDMA/NVLink; Llumnix live migration as lighter alternative (12× P99 improvement claimed); compare monolithic vs live-migration vs disaggregated

### Cost discipline
- **All clusters scaled to 0** between sessions ($0.10/hr control plane, ~$0.19/hr with NAT Gateway + S3)
- **Resume scripts** in each phase's `scripts/` directory (`setup-cluster.sh`, `teardown-cluster.sh`, `port-forward.sh`)
- **Cumulative spend across all phases: ~$30–50** of GPU time, mostly Spot at $0.16–0.22/hr
- **Three idle clusters today** (inference-lab, inference-phase3, inference-phase4) at ~$0.55/hr combined when GPUs run, ~$0.30/hr with GPUs at 0

---

## 6. Technical Depth Moments (genuine hands-on)

### Calculated max concurrency from first principles, then verified empirically
```
KV per token = 2 (K+V) × 28 layers × 128 head_dim × 28 num_kv_heads × 2 bytes = ~401 KB
At 2048 tokens: ~802 MB per request
KV budget: 16 GB - 4 GB (AWQ weights) - 1.5 GB (overhead) = 10.5 GB
Theoretical max: 10.5 GB / 802 MB ≈ 13 concurrent at full context
```
Then proved with PagedAttention pages-on-demand that 200-token requests use ~80 MB → **38× actual concurrency**. Theory + empirical match.

### Diagnosed the "5× throughput gap" as actually 1.6× via controlled A/B
- Stage 2 local vLLM at 60 concurrent: 1,131 tok/s (uniform short prompts, hot system)
- Phase 4 EKS pipeline at 1.5 req/s: 232 tok/s (Pareto, with cold paths)
- Looks like 5× overhead. Wrong inference.
- Direct vLLM on EKS at 60 concurrent (same load): 390 tok/s
- Pipeline at same load: 244 tok/s → real overhead = **1.6×** (RabbitMQ serialization + Redis per-token pub/sub)
- Distinguished load-shape effect from architecture cost. Without this A/B, would have wrongly attributed 3× of overhead to the wrong layer.

### AWQ vs AWQ_Marlin distinction
- AWQ = INT4 weight format on disk (with scaling factors per channel, calibrated via activation profiling)
- AWQ_Marlin = the GPU compute kernel that multiplies INT4 weights with FP16 activations at runtime
- Same data, different execution path. Marlin restructures memory access patterns (mixed-precision tensor cores, optimal LDS layout) to maximize bandwidth utilization
- vLLM logs literally said "Use quantization=awq_marlin for faster inference" — and the 47% prefill speedup matched the prediction

### Per-stage cold-start instrumentation
Built timing harness pulling from three data sources because no single source had the resolution:
- **K8s events** (`Scheduled` / `Pulled`) for node provision + image pull boundaries
- **Worker log markers** (`WARMUP: vLLM is ready`, `WARMUP: Complete`, `READY: Created`) for model load + runtime init
- **Pod condition** `Ready=True lastTransitionTime` for readiness detection

Each source has 1–5 s precision — fine for benchmarks measured in minutes.

### Graceful drain via tracked asyncio tasks
```python
active_tasks: set[asyncio.Task] = set()
task = asyncio.create_task(process_job(msg))
active_tasks.add(task)
task.add_done_callback(active_tasks.discard)

# On SIGTERM:
await rabbitmq_consumer.cancel()  # stop consuming new
await asyncio.wait(active_tasks, timeout=DRAIN_TIMEOUT)  # let in-flight finish
```
Plus `preStop: ["sleep", "15"]` on the vLLM sidecar so it outlives the worker drain (K8s sends SIGTERM to all containers simultaneously; worker needs vLLM alive on localhost to finish streaming).
- `DRAIN_TIMEOUT=170s < terminationGracePeriodSeconds=180s` — 10 s buffer for connection cleanup
- Result: **10/10 requests completed on pod kill**

### KEDA composite trigger config (not just queue depth)
```yaml
triggers:
  - type: rabbitmq
    metadata:
      queueName: inference
      value: "5"
      mode: QueueLength
  - type: prometheus
    metadata:
      query: vllm:gpu_cache_usage_perc
      threshold: "0.65"
```
Two triggers, OR semantics — scales on whichever fires first. The KV cache trigger catches the 1% XL Pareto requests that look fine on queue depth but consume all GPU memory.

### `AdmissionTracker` local atomic state (Phase 3 v3 redesign)
```python
class AdmissionTracker:
    def __init__(self, kv_budget_bytes, max_concurrent):
        self.running = {}  # job_id → {estimated_kv, remaining_tokens, ...}
        self.kv_budget_bytes = kv_budget_bytes
        self.max_concurrent = max_concurrent

    def try_admit(self, job_id, estimated_kv, max_tokens):
        # Instant, atomic, no HTTP call to vLLM
        ...

    def update_progress(self, job_id, tokens_generated):
        # Called as tokens stream — remaining budget shrinks in real-time
        ...

    def release(self, job_id):
        # Called on ACK — frees budget
        ...
```
Replaces `GET http://vllm:8000/metrics` polling that added 2–5 ms per request, returned stale state, and created a positive feedback loop under load. Strategies become:
- **Static:** prefetch_count=N, no tracker (unchanged)
- **Threshold:** `tracker.running_count < N` (accurate local count instead of RabbitMQ's prefetch)
- **Per-request:** `tracker.kv_remaining > this_request_cost` (cost-aware gate)
- **Predictive:** estimate per-request KV cost from input length + max_tokens, gate proactively
- **Reactive:** allow over-admission, let vLLM preempt, NACK on 5xx

The key insight: the only real difference between threshold and per-request is whether the gate accounts for request size — same local state, same zero overhead.

### vLLM Sleep Mode integration
Worker idle monitor: after 300 s of no messages, calls `POST /sleep` on vLLM sidecar with `level=1`. New message → `POST /wake_up` before processing.
- **L1:** weights → CPU RAM (~4 GB), preserves CUDA context + JIT kernels + CUDA graphs. Wake = 0.3–0.9 s.
- **L2:** weights discarded, only context preserved. Wake = 0.9–2.6 s.
- Chose L1 for single-model lab (RAM is cheap, latency is the value).
- Critical detail: Sleep mode provides 61–88% **faster first inference** (not just wake) because CUDA graphs and JIT kernels survive — a cold-started model has 5–7× slower first inference even after weights load.
- Requires `VLLM_SERVER_DEV_MODE=1`.

### Run:ai Model Streamer wiring
Stock `vllm/vllm-openai:v0.19.0` doesn't include it. Custom Dockerfile:
```dockerfile
FROM vllm/vllm-openai:v0.19.0
RUN pip install runai-model-streamer runai-model-streamer-s3
```
Then `--load-format runai_streamer` + `RUNAI_STREAMER_CONCURRENCY=16` env var.
- **Result:** 48 s → 14 s on gp3 (3.3×); ~5 s from S3 with 32 threads (9.6×)
- For Llama 3 8B (15 GB) per benchmarks: total vLLM readiness 23.18 s from S3 vs 65 s with HF safetensors

### SOCI (Seekable OCI) lazy image loading stack
Three components:
1. **`setup-soci.sh`** — pushes vLLM image to ECR + creates SOCI index artifact (`zTOC`)
2. **`gpu-node-soci.pkr.hcl`** — Packer AMI with SOCI snapshotter daemon + containerd proxy config
3. **`gpu-nodepool.yaml`** — dual `amiSelectorTerms` (SOCI tagged first, non-SOCI fallback)

How it works: containerd configures SOCI as a proxy snapshotter; on `PullImage`, SOCI creates a FUSE mount pointing at ECR with the zTOC; container starts immediately; files fetched via HTTP range requests on demand.

Insight: SOCI redistributes I/O, doesn't reduce total bytes. For PyTorch/CUDA images where everything is needed at startup, the lazy benefit is muted vs web app images. Still: cut image-unpack stage from 110 s to ~15 s.

### containerd parallel tuning
```toml
# /etc/containerd/config.toml
max_concurrent_downloads = 10
max_concurrent_unpacks_per_image = 10
concurrent_download_chunk_size = 16777216  # 16 MB, up from 8 MB
```
Defaults are 3 concurrent unpacks. The vLLM image has ~30 layers, so default config serializes most of them. Benchmarked at 60% improvement (60 s → 24 s on fresh nodes). Baked into both Packer templates.

### CUDA graph size limiting
```
--cuda-graph-sizes 1,2,4,8,16,24,32,64
```
vLLM defaults to capturing graphs for *every* batch size from 1 to `max_num_seqs`. For our workload, 8 sizes covers >95% of actual batches. Capture time: 54 s → 7 s (87% reduction).

### Diagnostic verification for "image cached but still slow"
- Confirmed via `journalctl -u containerd | grep -i pullimage` that NO `PullImage` call was made for vLLM on the FSR-restored node — image found locally
- Yet K8s reported "Successfully pulled in 1m49s"
- Conclusion: that 1m49s is *pure unpack time*, not download. The pull time field in K8s events is misleading.

### Composite Karpenter NodePool config (production-grade)
```yaml
disruption:
  consolidationPolicy: WhenUnderutilized
  consolidateAfter: 300s        # Match KEDA cooldown
  expireAfter: 6h               # Force Spot interruption simulation
  budgets:
    - nodes: "1"                # Disrupt only 1 GPU node at a time
limits:
  cpu: 32
  memory: 128Gi
requirements:
  - key: node.kubernetes.io/instance-type
    operator: In
    values: ["g4dn.xlarge", "g5.xlarge"]
  - key: karpenter.sh/capacity-type
    operator: In
    values: ["spot", "on-demand"]   # Fallback essential after Spot exhaustion
```

### Quality benchmark suite design (60 questions, 7 categories, temperature=0)
Custom benchmark not just to measure quantization quality, but to make quality testable:
- Temperature 0 for reproducibility
- 7 categories isolate failure modes (math reasoning fails differently than code generation)
- Substring scoring with manual review for false positives (Q54 "train meeting time" flagged)
- Same harness used to compare AWQ INT4 / GPTQ INT8 / FP8 KV E5M2 / FP8 KV E4M3
- Result quality matrix above shows quant tradeoffs with rigor (60/60 vs 60/60 vs 29/60)

### The 5 strategies of admission control as a design space
After Phase 3, articulated the design space cleanly:
1. **Static:** fixed prefetch, no gate — trust the engine
2. **Threshold:** local running_count < N — accurate concurrency limit
3. **Per-request:** kv_remaining > this_request_cost — cost-aware admission
4. **Predictive:** estimate per-request KV cost from input + max_tokens — proactive gate
5. **Reactive:** over-admit, let engine preempt, NACK on 5xx — pessimistic

Industry research (Mooncake, SGLang, QLM) maps onto this taxonomy: SGLang is "predictive with adaptive `new_token_ratio`"; vLLM V1 default is "reactive with recompute preemption"; TGI is "static with hard semaphore."

---

## Cross-Cutting Skills Demonstrated

- **Build-to-understand discipline**: every stage produces a working, testable artifact + a written `learnings.md` that maps inference concepts to AWS Auto Scaling primitives.
- **Research-before-build**: each phase preceded by an industry-research markdown (`backpressure_research.md`, `stage3_cold_start_survey.md`, `aws_gpu_instance_comparison.md`) covering academic papers, engine source, production blog posts. This is how we discovered push-vs-pull routing, Mooncake early rejection, ProServe starvation prevention, the SGLang `new_token_ratio` pattern — all before writing the wrong code.
- **Operational discipline**: scale-to-zero between sessions, pre-flight context checks in deploy scripts, separate Terraform stacks per phase to prevent cross-contamination, IAM least-privilege (Karpenter node role is ECR ReadOnly; ECR push gets a temp inline policy that's deleted after).
- **Experiment design rigor**: graduated load ramps over flat sustained rates, Pareto distributions over uniform, SLA targets defined upfront, cost-per-token tracked, JSON results + HTML dashboards for narrative explanation, A/B isolation when diagnosing perf gaps, separate result files per sweep run (after losing 6 data points to overwrite).
- **Diagnosis from first principles**: when results don't match prediction, instrument deeper. The "AMI doesn't help" finding came from `journalctl` showing no PullImage call. The "5× overhead" debunk came from the controlled direct-vs-pipeline A/B. The "predictive admission self-defeating" finding came from reading the code and noticing the metric used to gate the limit could never reach the limit.
- **Cost-conscious infrastructure**: $30–50 total spend over 6 weeks, three isolated EKS clusters, full observability stack, 4 cold-start optimizations, 50+ JSON result files. All Spot, all scale-to-zero between sessions.
