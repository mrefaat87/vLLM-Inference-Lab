# Stage 2: vLLM — Continuous Batching on GPU

## Setup
- **Instance:** AWS g4dn.xlarge spot (~$0.22/hr)
- **GPU:** NVIDIA T4, 16GB VRAM, 320 GB/s memory bandwidth
- **Model:** Qwen2.5-7B-Instruct-AWQ (4-bit quantized, ~4GB)
- **Server:** vLLM v0.17.1 (Docker), OpenAI-compatible API on port 8000
- **Region:** us-east-1f

## Why AWQ Quantization Was Required
FP16 Qwen2.5-7B weights = 14.25GB. The T4 has 14.56GB usable VRAM. After loading weights, only 110MB remained — not enough for KV cache, CUDA graph compilation, or any actual serving. vLLM OOM'd repeatedly.

Solution: AWQ 4-bit quantized model (~4GB weights), leaving ~10GB for KV cache. This enabled **38.5x max concurrency** at 2048 token context window.

**Auto Scaling analogy:** Like right-sizing instances. A c5.large that uses 95% memory at idle can't handle any traffic spikes. You either need a bigger instance (more VRAM) or a more efficient payload (quantization).

## Apples-to-Apples Comparison: Same Hardware, Different Engine

Ran both Ollama and vLLM on the same g4dn.xlarge T4 GPU to isolate the batching effect.

### Ollama on T4 (3 concurrent requests)
```
Req    Start  1st Token      End     TTFT   Avg TBT    Total   Tok/s  Tokens
  1    0.001      0.316    0.923    0.315    0.0241    0.922    41.2      25
  2    0.001      0.950    1.608    0.949    0.0242    1.607    41.0      27
  0    0.000      1.636    2.391    1.636    0.0242    2.391    41.1      31
```

### vLLM on T4 (5 concurrent requests)
```
Req    Start  1st Token      End     TTFT   Avg TBT    Total   Tok/s  Tokens
  0    0.001      0.707    1.423    0.706    0.0224    1.422    44.7      32
  1    0.001      0.724    1.422    0.723    0.0225    1.421    44.4      31
  2    0.183      0.741    1.289    0.557    0.0228    1.106    43.8      24
  3    0.190      0.741    1.424    0.551    0.0226    1.234    44.0      30
  4    0.194      0.741    1.422    0.547    0.0227    1.228    44.1      30
```

### Key Comparison

| Metric | Ollama (T4) | vLLM (T4) | Insight |
|--------|-------------|-----------|---------|
| TTFT spread | 1.321s | 0.176s | vLLM serves all requests together |
| Gap between first-tokens | 0.660s | 0.009s | Sequential vs parallel |
| Tokens/sec | ~41 | ~44 | Same hardware = same decode speed |
| Gap/Total ratio | ~0.72 | 0.01 | 1.0=queuing, 0.0=batching |

**Critical insight:** Tok/s is nearly identical (~41 vs ~44). The hardware dictates decode speed. The difference is entirely in **scheduling** — how the server manages concurrent requests.

## vLLM Features Observed in Logs

### Prefix Caching (enabled by default)
- All 5 requests used the same prompt
- `Prefix cache hit rate: 32.7%` during the test
- Requests 2-4 had lower TTFT (~0.55s vs ~0.71s) — they reused the KV cache computed by request 0 instead of recomputing prefill from scratch
- This is like **AMI caching** — don't rebuild what's already been built

### Chunked Prefill (enabled by default)
- `Chunked prefill is enabled with max_num_batched_tokens=2048`
- Allows prefill computation to be broken into chunks and interleaved with decode steps
- Prevents long-prompt prefills from blocking decode for already-running requests

### KV Cache Capacity
- `Available KV cache memory: 4.21 GiB`
- `Maximum concurrency for 2,048 tokens per request: 38.50x`
- With 5 requests, memory was never a constraint (GPU KV cache usage: 0.1-0.5%)

## Lessons Learned

### Spot Capacity Reality
- P-family (V100) spot: **zero capacity** across all regions and instance types (p2, p3). Legacy hardware being decommissioned.
- G-family (T4, A10G): abundant spot capacity
- This is exactly the problem Karpenter solves in Stage 4 — diversifying across instance types and AZs

### Model Sizing for VRAM
- Rule of thumb: FP16 model size in GB ≈ 2 × parameters in billions (7B → ~14GB)
- vLLM needs headroom beyond weights: KV cache, CUDA graphs, activation memory
- On a 16GB GPU, a 7B FP16 model leaves no room. Quantization (AWQ/GPTQ) is essential.
- AWQ 4-bit: ~4GB weights → ~10GB for KV cache → 38x concurrency

### Prefill vs Decode Bottlenecks
- **Prefill** (TTFT): compute-bound. Matrix-matrix multiply across all input tokens. Benefits from GPU parallelism.
- **Decode** (TBT): memory-bandwidth-bound. Matrix-vector multiply for one token, bottlenecked by reading model weights from VRAM each step.
- TTFT and TBT measure fundamentally different bottlenecks.

**Auto Scaling analogy:** TTFT = instance launch time (CPU-bound initialization). TBT = per-request latency (I/O-bound, limited by reading state).

## What's Next — Stage 3
- Add Prometheus metrics: queue depth, TTFT, TBT, P99 latency, throughput, GPU utilization
- Grafana dashboard on the same instance (docker-compose)
- vLLM exposes a `/metrics` endpoint with Prometheus-format metrics out of the box
