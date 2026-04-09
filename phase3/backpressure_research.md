# Backpressure & Admission Control: Industry Research

Research compiled from academic papers, production blog posts, engine source code, and framework documentation. April 2026.

## TL;DR — What the Industry Actually Does

Production LLM serving uses **multi-tier admission control** — not one gate, but a cascade of gates at different layers. No single mechanism handles overload alone. Here's the stack, from outermost to innermost:

```
┌─────────────────────────────────────────────────────────┐
│  TIER 1: API Gateway                                     │
│  Token bucket rate limiting (RPM + input TPM + output    │
│  TPM, independently). HTTP 429. Circuit breakers.        │
│  Fallback chains to smaller models.                      │
│  WHO: Anthropic, OpenAI, Azure, Portkey, Envoy AI GW     │
├─────────────────────────────────────────────────────────┤
│  TIER 2: Inference Router / Load Balancer                │
│  KV-cache-aware routing (hash prefix → replica).         │
│  Deficit-based fairness (break affinity when overloaded). │
│  Queue duration as scaling signal (not queue depth).      │
│  WHO: vLLM Router, SGLang sgl-router, llm-d, Dynamo     │
├─────────────────────────────────────────────────────────┤
│  TIER 3: Engine Scheduler                                │
│  Token budget per forward pass. KV block allocation      │
│  check before admission. Preemption (recompute/swap)     │
│  when memory pressure. Priority scheduling.              │
│  WHO: vLLM scheduler, SGLang scheduler, TGI router       │
├─────────────────────────────────────────────────────────┤
│  TIER 4: Request-Level Controls                          │
│  Per-request timeout. Client disconnect cancellation.    │
│  Starvation prevention (priority boost after N seconds). │
│  Prediction-based early rejection (Mooncake pattern).    │
│  WHO: custom middleware, emerging in engines              │
└─────────────────────────────────────────────────────────┘
```

**Key insight**: Our Phase 3 experiments were operating at Tier 3 only (engine-level admission). The industry puts the most weight on Tiers 1 and 2, because by the time the engine is overloaded it's already too late.

---

## Part 1: Engine-Level Mechanisms (What vLLM/SGLang Actually Do)

### vLLM V1: "Admit everything, preempt when necessary"

**Admission**: Each scheduling step has a token budget (`max_num_batched_tokens`, default ~2048). Scheduler services running requests first (1 token each for decode), then admits waiting requests with remaining budget. For each candidate, calls `allocate_slots()` on the KV block pool. If blocks available → admitted. If not → skip (request stays in unbounded waiting deque).

**Preemption**: When running requests can't get blocks, lowest-priority running request is evicted:
- **Recompute** (V1 default): KV blocks freed, request goes back to waiting queue, must re-prefill entirely (prefix caching reuses shared prefix blocks, so overhead is ~20% of full recompute)
- **Swap** (V0 only, removed from V1): KV blocks moved to CPU memory

**Critical gap**: vLLM's waiting queue is an **unbounded deque in CPU memory**. No max queue length. No request timeout. Under sustained overload, queue grows until CPU OOM. RFC #18826 proposes `--max-waiting-queue-length` → HTTP 503, but PR #27064 is still open.

**Config knobs that matter**:
| Flag | Default | What it controls |
|------|---------|------------------|
| `--max-num-seqs` | 128 | Max concurrent sequences per iteration |
| `--max-num-batched-tokens` | ~2048 | Token budget per forward pass |
| `--gpu-memory-utilization` | 0.9 | Fraction of GPU memory for KV cache |
| `--scheduling-policy` | "fcfs" | "fcfs" or "priority" |
| `--scheduler-reserve-full-isl` | True | Anti-thrashing: only admit if full sequence fits |
| `--enable-chunked-prefill` | True (V1) | Split long prompts across steps |

**Priority scheduling** (merged): `priority` field on sequences, strict priority with FCFS tiebreaker. Lower-priority running requests preempted for higher-priority waiting ones. <4% overhead.

**SLA-tiered scheduling** (RFC, not merged): Three tiers: `interactive > batch > background`. Preemption order: background first. Includes `max_interactive_batch_tokens` to prevent monopolization.

### SGLang: "Predict and reserve"

**Core innovation — `new_token_ratio`**: Before admitting new prefills, SGLang estimates how many tokens running requests will still generate and *reserves memory* for them. This is proactive — like predictive auto-scaling that reserves capacity headroom.

```
available_for_new_requests = total_kv_memory - currently_used - reserved_for_future_decode
```

The ratio starts conservatively high (~0.7), decays as the system learns actual decode patterns, and resets upward when retraction (preemption) occurs. `--schedule-conservativeness` (default 1.0) scales this.

**Retraction**: When decode memory check fails, SGLang evicts running requests sorted by `(output_tokens_generated, -input_tokens)` — most-progress-least-input first. If even the last request can't fit, it's aborted with HTTP 500.

**Queue management (ahead of vLLM)**:
- `--max-queued-requests`: Hard limit, rejects with HTTP 503 when full
- `SGLANG_REQ_WAITING_TIMEOUT`: Abort waiting requests after N seconds
- Priority scheduling with preemption threshold

**Known failure mode**: Under extreme pressure, SGLang can enter "fake dead" state — health checks return 200 but GPU utilization drops to 0%, no requests processed. Requires manual restart.

### TGI: "Hard gate at the door"

Simplest model. Rust router holds a semaphore of `max-concurrent-requests` (default 128). Request arrives → `try_acquire()` → if semaphore exhausted → HTTP 429 immediately. No queuing, no preemption complexity.

**Batching knobs**:
- `--waiting-served-ratio` (1.2): min ratio of waiting-to-served before interrupting decode for new prefill
- `--max-waiting-tokens` (20): max decode tokens before forcing new request admission (starvation prevention)

**TGI entered maintenance mode Dec 2025.** HuggingFace now recommends vLLM/SGLang.

### Engine Comparison Under Overload

| Aspect | vLLM | SGLang | TGI |
|--------|------|--------|-----|
| Queue limit | Unbounded (no limit) | Configurable (`--max-queued-requests`) | Semaphore (128) |
| Rejection | None (PR pending) | HTTP 503 when full/timeout | HTTP 429 immediately |
| Preemption | Recompute (V1) | Retract-decode + priority | None |
| KV exhaustion | Preempt lowest-priority running | Retract running to waiting | Block allocation fail |
| Admission philosophy | Admit all, preempt later | Predict + reserve, retract if wrong | Hard cap at door |

**AWS analogy**: vLLM is an SQS queue with no DLQ and infinite retention — keeps accepting until the system falls over. SGLang is an ASG with predictive scaling that reserves capacity. TGI is an ALB with a max-connections limit that returns 503 immediately.

---

## Part 2: Infrastructure-Level Patterns (Above the Engine)

### 2.1 KV-Cache-Aware Load Balancing

**Why traditional LB fails for LLM inference**:
- CPU utilization is meaningless (inference is GPU-bound)
- GPU utilization is misleading (95% with 1 request = room for dozens more)
- VRAM appears constant (pre-allocated by vLLM)
- Request cost varies 1000x (100 tokens vs 128K tokens)

**The right signal: KV cache utilization.** Engines expose `{capacity_tokens, used_tokens, utilization}`. Route to the replica with most available token capacity.

**Prefix-cache-aware routing**: Hash the system prompt → route to the replica that has it cached. Avoids redundant prefill computation. Results:
- llm-d: 87.4% cache hit rate, 88% faster TTFT (340ms warm vs 2,850ms cold)
- SGLang sgl-router: 1.9x throughput, 3.8x cache hit rate (20% → 75%)
- vLLM Router: consistent hashing with circuit breaker fallback

**Deficit-based fairness (D2LPM)**: When to break cache affinity? When a tenant is monopolizing a GPU with cached prefixes. Deficit counters per-client per-worker, inspired by deficit round-robin. Occasionally prioritizes under-served clients over those with longer prefix matches.

**AWS analogy**: Session-affine ALB where the session key is the prompt prefix hash. Break stickiness when the target becomes unhealthy (high queue depth).

### 2.2 Gateway-Level Rate Limiting

**Anthropic's production approach** (3 tiers):
- **Priority**: committed capacity (contracts), minimizes "server overloaded" errors
- **Standard**: best-effort, default
- **Batch**: async, processed outside normal capacity

Critical detail: **token accounting is weighted**:
- Cache reads = 0.1x per token (incentivizes reuse)
- Cache writes (5min TTL) = 1.25x
- Cache writes (1hr TTL) = 2.0x

Rate limiting uses token bucket (continuous replenishment, not fixed-window). Limits enforced on **three axes simultaneously**: RPM, input TPM, output TPM. Separate input vs output reflects different resource profiles (prefill vs decode).

**OpenAI**: RPM + TPM + RPD + TPD. Token bucket with 60s and 24h windows. Limits can be quantized (RPM 600 enforced as 10 req/s — short bursts trigger 429).

**Envoy AI Gateway** (CNCF): Token-aware rate limiting via CEL expressions: `input_tokens * 0.5 + output_tokens * 1.5`. Can zero out request count cost so only token usage draws down quota.

### 2.3 Autoscaling Signals

**Why GPU utilization is the WRONG signal**:
- GPU at 100% can mean 1 request OR 50 — can't distinguish "fine" from "catastrophically overloaded"
- Decode is memory-bandwidth-bound → SM utilization reads 20-40% even when saturated
- An autoscaler using SM utilization once scaled DOWN during decode-heavy load, spiking p99 from 200ms to 1200ms

**What works**:

| Signal | Why | Threshold derivation |
|--------|-----|---------------------|
| **Queue duration** (primary) | Directly measures user pain, unlike depth which is misleading with variable request lengths | SLO (10s p95) - headroom (3s) = 7s threshold |
| **TTFT** (secondary) | Users perceive <2s as "working", >5s as "broken" | Alert >5s, scale >2s |
| **KV cache utilization** (tertiary) | Approaching capacity cliff | >80% = scale up |
| Throughput floor | Prevents false positives from intentional batching | Below baseline = scale |

**Production HPA config for GPU workloads**:
```yaml
scaleUp:
  stabilizationWindowSeconds: 30      # React fast
  policies:
    - type: Percent, value: 50        # +50% capacity per step
scaleDown:
  stabilizationWindowSeconds: 600     # 10 MINUTES, not 60s
  policies:
    - type: Pods, value: 1, periodSeconds: 120  # -1 pod every 2 min

minReplicas: 3   # NOT 1. GPU pods take 5 minutes to start.
```

| Aspect | CPU workloads | GPU workloads |
|--------|---------------|---------------|
| Scale-up time | 10 seconds | 5 minutes |
| Scale-down window | 60 seconds | 600 seconds |
| Concurrency per replica | 100s of requests | 1-10 requests |
| Request duration | 10-100ms | 1-30 seconds |
| Over-provisioning | Minimize | Accept it (SLO > cost) |

### 2.4 Graceful Degradation Patterns

When demand exceeds capacity and scaling can't keep up:

1. **Tiered model fallback**: Primary (large) → secondary (small) → cached responses. Anthropic's Priority Tier falls to Standard when committed capacity exhausted.

2. **Request classification + differential treatment**: Short prompts (<1K) → fast path. Long prompts (>32K) → quality path with longer SLO. QLM uses three tiers: Interactive (20s), Batch-1 (1min), Batch-2 (1hr).

3. **Prediction-based early rejection** (Mooncake, 100B+ tokens/day): When system predicts it can't meet SLO for a request, reject immediately. Better than accepting and timing out — saves GPU cycles, gives client instant feedback to retry.

4. **Response length limiting**: Under load, reduce `max_tokens` for lower-priority requests. 4K-token response costs 4x the decode time of 1K.

5. **KV cache eviction to CPU** (QLM): Pause batch request, move its KV to CPU memory (not discard). Resume later without recomputation. The inference equivalent of swap space.

---

## Part 3: Academic Systems Worth Simulating

### Sarathi-Serve (OSDI 2024): Chunked Prefill

**Problem**: Long prefills stall ongoing decodes, causing latency spikes.
**Solution**: Split prefill into equal chunks across iterations. Decode tokens piggyback on idle GPU cores during chunk processing.
**Result**: 2.6-5.6x higher serving capacity. Predictable decode latency.
**Why it matters for us**: Eliminates the primary cause of latency spikes that trigger backpressure. vLLM V1 already has `--enable-chunked-prefill=True` — this is Sarathi's idea adopted.

### DistServe (OSDI 2024): Prefill/Decode Disaggregation

**Problem**: Prefill (compute-bound, 90-95% GPU util) and decode (memory-bandwidth-bound, 20-40% GPU util) interfere on shared GPUs. Decode latency inflates 2-30x during large prefills.
**Solution**: Separate GPU pools. Independent scaling.
**Result**: 7.4x more requests, 12.6x tighter SLO at >90% attainment.
**AWS analogy**: Separate ASGs for web tier (compute-bound) vs worker tier (I/O-bound).

### Llumnix (OSDI 2024, Alibaba): Live Request Migration

**Problem**: Load imbalance across replicas. One hot, one cold.
**Solution**: Migrate running requests between instances by transferring KV cache state. Like OS context switching across CPU cores.
**Use cases**: Load balancing, memory defragmentation, priority differentiation.
**Result**: Tail latency improved by an order of magnitude. P99 TTFT up to 12.1x better vs round-robin.

### Learning-to-Rank Scheduling (NeurIPS 2024)

**Problem**: FCFS is suboptimal because short requests get stuck behind long ones.
**Solution**: Lightweight ML model predicts relative output length ranking. SJF-like scheduling without exact predictions. Priority boost after wait threshold (starvation prevention).
**Result**: 6.9x lower mean latency at 64 req/s vs FCFS. <2% overhead.

### ProServe: Multi-Priority Without Starvation

**Problem**: Strict priority scheduling starves low-priority requests and reduces total system gain.
**Solution**: Adaptive urgency partitioning — deadline-first under low load, gain-maximization under high load.
**Result**: 35% higher system gain, 52% higher SLO attainment vs strict priority.
**Key lesson**: Every production system implements starvation prevention. Pure priority is dangerous.

### QLM (IBM, ACM SoCC 2024): Queue Management for Multi-SLO

**Problem**: How to schedule across models and SLO tiers.
**Solution**: Request Waiting Time estimator (CLT-based, R^2=0.99) + linear programming scheduler minimizing total SLO violation.
**Result**: 40-90% higher SLO attainment, 2-5x GPU reduction.
**Innovation**: KV cache migration to CPU when evicting batch requests (preserve work, not discard).

---

## Part 4: Patterns Mapped to AWS Auto Scaling

| LLM Inference Pattern | AWS Auto Scaling Analogy |
|---|---|
| KV cache block pool | Warm instance pool / capacity reservation |
| `max_num_seqs` / `max_num_batched_tokens` | DesiredCapacity / MaxSize |
| Preemption (recompute) | Spot interruption + relaunch from scratch |
| Preemption (swap to CPU) | Hibernate instance to EBS |
| Unbounded waiting queue | SQS queue with no DLQ and no max-receive-count |
| Priority scheduling | Priority-based scaling policies |
| KV-cache-aware routing | Capacity-optimized-prioritized placement |
| Prefix-cache affinity | Session-affine ALB sticky sessions |
| Token budget per step | SQS batch window |
| SLA-tiered scheduling | Tier-0 / Tier-1 / Tier-2 service levels |
| Circuit breaker at gateway | Route 53 health checks + failover |
| Disaggregated P/D | Separate ASGs for web vs worker tier |
| `new_token_ratio` (SGLang) | Predictive scaling policy that reserves capacity headroom |
| `scheduler_reserve_full_isl` | Never admit a job to an ASG if the instance can't handle it fully |
| Queue duration autoscaling | Target tracking on `ApproximateAgeOfOldestMessage` |
| Token bucket rate limiting | API Gateway usage plans with per-key throttling |

---

## Part 5: What We Can Simulate on g4dn.xlarge

Given our constraints (single T4 16GB, 7B AWQ model, KV cache peaks at ~11%), here are patterns ranked by simulability and learning value:

### High Value, Easy to Simulate

**1. Multi-tier gateway admission control**
Add a FastAPI gateway layer that enforces token bucket rate limits (RPM + TPM) per client. Classify requests into tiers (interactive/batch) with separate quotas. Return HTTP 429 when quota exhausted. This is the highest-leverage pattern and requires zero GPU changes.

**2. Queue duration as scaling signal (vs queue depth)**
Our KEDA config uses queue depth. Switch to queue duration. Inject variable-length requests (100-token vs 4K-token prompts). Show that queue depth gives false positives (many fast requests look scary) and false negatives (few slow requests look fine but users are waiting).

**3. Prediction-based early rejection**
Estimate request cost at the gateway: `estimated_seconds = input_tokens / prefill_rate + estimated_output_tokens / decode_rate`. If estimated completion time + current queue wait exceeds SLO → reject immediately with 503. Compare against blind admission. This is the Mooncake pattern.

**4. Priority scheduling with starvation prevention**
Use vLLM's `--scheduling-policy=priority` with a gateway that assigns priority. Then add starvation prevention: if a request waits >N seconds, boost its priority. Show that without boosting, low-priority requests starve. With boosting, everyone gets served. Compare against ProServe's adaptive urgency.

### Medium Value, Moderate Setup

**5. SGLang's `new_token_ratio` vs vLLM's admit-everything**
Deploy SGLang on the same hardware, same model. Inject the same overload workload. Compare degradation curves. SGLang should show smoother degradation (predictive reservation) vs vLLM's cliff (admit until preemption cascade).

**6. Chunked prefill interference**
vLLM V1 has `--enable-chunked-prefill`. Run mixed workload (long prefills + ongoing decodes) with chunked prefill on vs off. Measure decode latency variance. Should show Sarathi's insight: chunked prefill eliminates decode stalls.

**7. Request classification + differential max_tokens**
Under load, limit `max_tokens` for batch-tier requests while keeping interactive-tier unlimited. Show that reducing output length is a direct knob on per-request GPU cost. Measure throughput gain from output length limiting.

### High Value, Needs Multi-Replica (Phase 4+)

**8. Cache-aware routing vs round-robin**
Two vLLM replicas. Shared system prompt workload. Compare: (a) round-robin routing, (b) hash(system_prompt) → replica affinity. Measure TTFT difference — llm-d saw 88% improvement.

**9. Deficit-based fairness**
Cache-aware routing but with multiple "tenants" sending different system prompts. One tenant sends 10x more traffic. Show that pure cache affinity starves the low-volume tenant. Add deficit counters to rebalance.

**10. Disaggregated prefill/decode simulation**
Two replicas: one configured as prefill-optimized (high `max_num_batched_tokens`, low `max_num_seqs`), one as decode-optimized (low `max_num_batched_tokens`, high `max_num_seqs`). Route new requests to prefill replica, transfer to decode replica after prefill. Compare latency variance vs co-located.

---

## Part 6: What Our Experiments Got Right and Wrong

### What we got right
- **Queue-based architecture** — the entire industry does this. Async decoupling is table stakes.
- **Testing multiple gate signals** (KV cache, decode throughput, gpu_compute) — we proved that KV cache isn't always the right signal. The industry confirms: the signal must match the bottleneck (KV for large-model-small-GPU, bandwidth for 7B-AWQ-on-T4).
- **Priority queues from day one** — every production system implements tiered scheduling.

### What the research reveals we missed
1. **No gateway-level admission** — we went straight to engine-level gates. Production puts the heaviest gate at the API layer (token buckets, per-client quotas). By the time the engine is stressed, it's too late.
2. **Queue depth not queue duration** — our KEDA signal is queue depth. Variable-length requests make depth misleading. Duration directly measures user pain.
3. **No starvation prevention** — our priority experiments didn't test what happens when low-priority requests never get served. Every production system ages/boosts starved requests.
4. **Static threshold tuning** — we hand-tuned admission thresholds. SGLang's `new_token_ratio` auto-adapts. QLM uses CLT-based wait time estimation. The industry trend is toward adaptive, not static, admission.
5. **Broker-push model limits cache reuse** — our architecture has RabbitMQ pushing to workers via `basic_consume` with `prefetch_count` for flow control (not polling/pulling). But the broker distributes round-robin across consumers — it doesn't know which replica has a cached prefix. Cache-aware routing requires a *smart router* that pushes to a specific replica based on prefix hash. Production resolves this with per-replica local queues behind a cache-aware router.

---

## Sources

### Engine Source / Docs
- [vLLM V1 Scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- [vLLM Priority Scheduling RFC #6077](https://github.com/vllm-project/vllm/issues/6077)
- [vLLM Max Queue RFC #18826](https://github.com/vllm-project/vllm/issues/18826)
- [vLLM SLA-Tiered RFC #30256](https://github.com/vllm-project/vllm/issues/30256)
- [SGLang Scheduler Source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)
- [SGLang Hyperparameter Tuning](https://docs.sglang.io/advanced_features/hyperparameter_tuning.html)
- [TGI Architecture](https://huggingface.co/docs/text-generation-inference/architecture)

### Production Systems
- [Anthropic Service Tiers](https://platform.claude.com/docs/en/api/service-tiers)
- [Anthropic Rate Limits](https://platform.claude.com/docs/en/api/rate-limits)
- [OpenAI Rate Limits](https://developers.openai.com/api/docs/guides/rate-limits)
- [vLLM Router Release](https://vllm.ai/blog/vllm-router-release)
- [llm-d Architecture](https://llm-d.ai/docs/architecture)
- [llm-d KV Cache Routing](https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference)
- [Portkey AI Gateway](https://github.com/Portkey-AI/gateway)
- [Envoy AI Gateway](https://aigateway.envoyproxy.io/docs/0.1/capabilities/usage-based-ratelimiting/)

### Academic Papers
- [Sarathi-Serve (OSDI 2024)](https://www.usenix.org/conference/osdi24/presentation/agrawal) — chunked prefill
- [DistServe (OSDI 2024)](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) — P/D disaggregation
- [Llumnix (OSDI 2024)](https://www.usenix.org/conference/osdi24/presentation/sun-biao) — live request migration
- [Learning-to-Rank (NeurIPS 2024)](https://haoailab.com/blogs/vllm-ltr/) — output length prediction for SJF
- [ProServe](https://arxiv.org/html/2512.12928v1) — multi-priority without starvation
- [QLM (ACM SoCC 2024)](https://arxiv.org/html/2407.00047v2) — queue management for multi-SLO
- [D2LPM](https://arxiv.org/html/2501.14312v1) — deficit-based fair scheduling with cache locality
- [Preble (ICLR 2025)](https://arxiv.org/abs/2407.00023) — distributed prefix-aware scheduling
- [Mooncake](https://arxiv.org/abs/2407.00079) — prediction-based early rejection

### Infrastructure
- [K8s Gateway API Inference Extension](https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/)
- [GPU Autoscaling for CPU Engineers](https://medium.com/@kinjaldand/gpu-autoscaling-for-cpu-engineers-why-everything-you-know-is-wrong-db95ce92624c)
- [KServe + KEDA Autoscaling](https://developers.redhat.com/articles/2025/09/23/how-set-kserve-autoscaling-vllm-keda)
- [Ray Serve LLM Architecture](https://docs.ray.io/en/latest/serve/llm/architecture/overview.html)
- [Disaggregated Inference 18 Months Later](https://haoailab.com/blogs/distserve-retro/)
