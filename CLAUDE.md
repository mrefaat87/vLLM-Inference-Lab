# Project: LLM Inference Learning Lab

## Owner
Mohamed — Senior Engineering Manager, AWS Auto Scaling. Infrastructure leader learning LLM inference infrastructure. Target role: Engineering Manager, Cloud Inference at Anthropic.

## Project Goal
Build a hands-on inference serving system that goes from zero to observable, auto-scaling model serving. Every stage should deepen understanding of how inference infrastructure works — not just make code run.

## Stages
1. ✅ Ollama: local model serving + manual concurrent load (Apple M4, 16GB)
2. vLLM on AWS g4dn.xlarge spot (~$0.16/hr): compare behavior vs Ollama, understand continuous batching
3. Observability: Prometheus metrics (queue depth, TTFT, TBT, P99), Grafana dashboard (same instance, docker-compose)
4. Auto-scaling: EKS + Karpenter, GPU node provisioning, scaling based on queue depth / P99 thresholds

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
