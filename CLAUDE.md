# Project: LLM Inference Learning Lab

## Owner
Mohamed — Senior Engineering Manager, AWS Auto Scaling. Infrastructure leader learning LLM inference infrastructure. Target role: Engineering Manager, Cloud Inference at Anthropic.

## Project Goal
Build a hands-on inference serving system that goes from zero to observable, auto-scaling model serving. Every stage should deepen understanding of how inference infrastructure works — not just make code run.

## Stages
1. ✅ Ollama: local model serving + manual concurrent load (Apple M4, 16GB)
2. ✅ vLLM on AWS: g4dn.xlarge spot (~$0.16/hr), continuous batching, FP8/AWQ quantization benchmarks
3. Production Inference Platform on EKS (8 phases):
   - Phase 1 ✅ EKS + Karpenter foundation
   - Phase 2 ✅ Observability (Prometheus + Grafana + DCGM)
   - Phase 3 ✅ Pod autoscaling (KEDA) + admission control experiments
   - Phase 4 ✅ Scaling policy comparison (composite KV triggers) + cold start optimization
   - Phase 5: Smart routing & inference optimization (cache-aware routing, prefix caching, speculative decoding)
   - Phase 6: Multi-model serving & graceful degradation (bin-packing, tiered fallback, CUDA checkpoint/restore)
   - Phase 7: Production hardening (gateway admission, priority queues, failure injection). Key strategies from research:
     - **Wait time prediction (QLM):** predict queue wait via CLT-based output length distributions, LP-optimize scheduling across SLO tiers
     - **SLO feasibility / early rejection (Mooncake):** estimate total time (queue_wait + prefill + decode) at admission, reject with 503 if > SLO
     - **Request cost ordering (Learning-to-Rank):** lightweight ML model predicts relative output lengths for SJF-like scheduling + starvation prevention via aging
     - **Failed request retry strategies:** Phase 3 adaptive strategy drops failed requests (requeue=False on vLLM errors). Explore: (A) requeue=True with retry counter + max retries to prevent loops, (B) RabbitMQ dead-letter TTL for delayed redelivery (e.g., 5s backoff) so system stabilizes before retry, (C) priority boost on retry so failed requests don't starve behind new arrivals
   - Phase 8: Disaggregated inference (prefill/decode separation via Ray Serve or Dynamo)
4. Inference Playground: interactive platform to compare optimization techniques, engines (vLLM, TensorRT-LLM, SGLang), and scaling strategies

## Explanation Style
Always map inference concepts to AWS Auto Scaling / distributed systems analogies before explaining them in isolation. Mohamed knows: capacity pools, queue depth, scaling policies, latency SLAs, Spot vs On-Demand, session affinity, consistent hashing, Tier-0 availability patterns.

## Code Style
- Inline comments on every non-obvious line (explain WHY, not what)
- Small focused files, one concept per file where possible
- After each working stage: ask Mohamed to explain what's happening before you explain it

## Key Concepts Mapped (reference these when they come up)
- KV cache = warm instance pool
- PagedAttention = OS virtual memory paging
- Continuous batching = city bus with stops (not charter waiting to fill)
- TTFT = time to first byte; TBT = streaming throughput
- Tensor parallelism = sharding load across nodes (like multi-AZ Auto Scaling)

## Metrics That Matter
TTFT, TBT, P99 latency, throughput (tokens/sec), GPU utilization, queue depth

## Learning Principles
- Build to understand, not to ship
- Every artifact should be explainable in an Anthropic engineering interview
- Prefer working simple systems over clever complex ones
- **Research before building:** Before starting each phase, do online research (blogs, papers, engine docs, production post-mortems) to understand current best practices and avoid reinventing solved problems. Document findings in a research markdown file per phase. This is how we discovered the push-vs-pull routing tension, the Mooncake early rejection pattern, and the ProServe starvation prevention insight — all before writing code.
