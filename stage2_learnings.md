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

## Quantization Deep Dive

### What Is `--dtype half`?
`dtype` controls the data type for model weights. `half` = FP16 (16-bit floating point, 2 bytes per parameter).

| dtype | Bytes/param | 7B Model Size | Precision |
|-------|------------|---------------|-----------|
| float32 | 4 | ~28 GB | Highest |
| half / float16 | 2 | ~14 GB | Standard for inference |
| INT8 | 1 | ~7 GB | Slight quality loss |
| INT4 | 0.5 | ~3.5 GB | More quality loss |

FP16 is the production default — virtually identical quality to FP32 at half the memory.

### AWQ (Activation-Aware Weight Quantization)
AWQ is a **quantization algorithm** that compresses FP16 weights to INT4 (4-bit). It's smarter than naive rounding:

1. **Profile** — run calibration data through the model, identify which weight channels produce the largest activations (the important ones)
2. **Scale and quantize** — scale up important channels (preserving precision when rounded to INT4), aggressively quantize the rest

**Auto Scaling analogy:** Like right-sizing instances in a fleet. Profile which services are latency-sensitive (important weights), give them more resources (higher precision), downsize everything else.

Other quantization methods:
- **GPTQ** — similar 4-bit, uses layer-wise optimization. Older, widely supported.
- **GGUF/Q4_K_M** — what Ollama uses (llama.cpp). Mixed quantization per layer.
- **FP8** — 8-bit float, H100 supports natively in hardware. Less compression but zero quality loss.

### AWQ vs AWQ_Marlin
- **AWQ** = the data format on disk (INT4 weights with scaling factors)
- **AWQ_Marlin** = the GPU compute kernel that multiplies those INT4 weights with activations at runtime

Same weights, different execution speed. Marlin restructures how INT4 data is read from GPU memory to maximize bandwidth utilization. vLLM logs even told us: "Use quantization=awq_marlin for faster inference."

**Auto Scaling analogy:** Same data in S3, but one path uses a generic instance and the other uses a Graviton-optimized binary. Same input, same output, different throughput.

### HuggingFace Cache = Model Storage
When vLLM downloads a model from HuggingFace, it stores files in `~/.cache/huggingface/`. There is no separate "install" vs "cache" — the cache IS where the model lives on disk. Clearing it means re-downloading on next launch.

We mounted it as a Docker volume (`-v /home/ubuntu/hf-cache:/root/.cache/huggingface`) to persist across container restarts.

## Quantization Performance Comparison

### Load Test Results: AWQ 4-bit vs AWQ_Marlin 4-bit vs GPTQ Int8

All tests: 5 concurrent requests, same prompt, same hardware (T4 16GB).

| Metric | AWQ 4-bit | AWQ_Marlin 4-bit | GPTQ Int8 |
|--------|-----------|-----------------|-----------|
| Model VRAM | ~4 GB | ~4 GB | 8.3 GB |
| KV cache room | ~10 GB | ~9.3 GB | ~4.2 GB |
| Tok/s (per request) | 43.8-44.7 | 45.2-45.4 | 26.6-27.6 |
| TTFT (min) | 0.547s | 0.291s | 0.831s |
| TTFT spread | 0.176s | 0.015s | 0.017s |
| Avg TBT | 0.0225s | 0.0221s | 0.0375s |
| Gap/Total ratio | 0.01 | 0.01 | 0.00 |

**Key insights:**
- Marlin kernel gives ~3% decode improvement but **47% faster prefill** (TTFT: 0.29s vs 0.55s)
- INT8 is ~40% slower on decode because model is 2x larger in memory — more data to read per token (memory-bandwidth bound)
- INT8 uses 2x VRAM for weights, leaving half the room for KV cache = fewer concurrent requests
- All three batch perfectly (gap/total ≈ 0)

### Quality Benchmark Results (60 questions)

Custom benchmark suite covering 7 categories. Temperature=0 for reproducibility.

| Category | Questions | AWQ INT4 | GPTQ INT8 | INT4 Avg Latency | INT8 Avg Latency |
|----------|-----------|----------|-----------|-----------------|-----------------|
| Math Reasoning | 10 | 10/10 | 10/10 | 0.46s | 0.73s |
| Factual Recall | 10 | 10/10 | 10/10 | 0.43s | 0.68s |
| Instruction Following | 10 | 10/10 | 10/10 | 0.36s | 0.62s |
| Code Generation | 5 | 5/5 | 5/5 | 2.84s | 4.14s |
| MMLU-Style | 10 | 10/10 | 10/10 | 0.23s | 0.39s |
| GSM8K-Style | 10 | 10/10 | 10/10 | 4.22s | 7.08s |
| HumanEval-Style | 5 | 5/5 | 5/5 | 5.39s | 7.77s |
| **OVERALL** | **60** | **60/60 (100%)** | **60/60 (100%)** | **0.77s** | **2.58s** |

**Note:** Q54 (train meeting time) flagged as a likely false positive in both runs — substring scorer matched despite potentially wrong answer. Real score may be 59/60 for both.

**Conclusion:** At 7B scale with modern quantization (AWQ/GPTQ), INT4 and INT8 are quality-equivalent on practical tasks. INT4 is ~60% faster with 2x more KV cache headroom. Quality degradation from 4-bit shows up on smaller models (1-3B), extreme quantization (2-bit), or adversarial evaluations.

### How Teams Measure Model Quality
- **Benchmarks:** MMLU (general knowledge), HumanEval (code), GSM8K (math), HellaSwag (common sense), TruthfulQA (factual accuracy)
- **Perplexity:** statistical measure of how "surprised" the model is by expected text. <1% increase = generally considered lossless
- **Human eval:** run on domain-specific tasks, have humans judge quality. Benchmarks don't always capture domain-specific regressions

## vLLM Features: Default vs Opt-In

### Enabled by Default (server-side, not per-request)
| Feature | What It Does |
|---------|-------------|
| PagedAttention | Virtual memory paging for KV cache — core architecture |
| Continuous batching | Add/remove requests from batch at every decode step |
| Chunked prefill | Break long prefills into chunks, interleave with decode |
| Prefix caching | Reuse KV cache when requests share prompt prefixes |
| Dynamic memory allocation | KV pages allocated on demand, freed on completion |

These are **engine-level features**, not per-request options. The client doesn't ask for batching — same way an HTTP client doesn't ask an ALB to load-balance.

### Opt-In Features (require flags)
| Feature | Flag | When to Use |
|---------|------|-------------|
| Tensor parallelism | `--tensor-parallel-size N` | Model too large for one GPU |
| Pipeline parallelism | `--pipeline-parallel-size N` | Very large models across GPUs |
| Speculative decoding | `--speculative-model <small>` | Latency-sensitive single-request |
| Quantization | `--quantization awq/gptq/fp8` | Memory-constrained GPUs |
| LoRA serving | `--enable-lora` | Multi-tenant fine-tuned adapters |
| CUDA graphs | Default on, `--enforce-eager` disables | Disable for debugging/OOM |
| Guided decoding | Per-request `response_format` | Force JSON schema/regex output |

**Auto Scaling analogy:** PagedAttention and continuous batching = kernel scheduler and virtual memory (always on). Tensor parallelism = multi-AZ deployment (configured based on capacity needs).

## Calculating Max Concurrent Requests

### Step 1: VRAM budget for KV cache
```
Total VRAM - Model Weights - vLLM Overhead = KV Cache Budget
16GB       - 4GB (AWQ)    - ~1.5GB         = ~10.5GB
```

### Step 2: KV cache per request (Qwen2.5-7B, FP16 KV)
```
KV per token = 2 (K+V) × 28 layers × 128 head_dim × 28 num_kv_heads × 2 bytes
             = ~401 KB per token

At max context (2048 tokens):
             = 401KB × 2048 = ~802 MB per request
```

### Step 3: Theoretical max
```
Max concurrent = 10.5GB / 802MB ≈ 13 requests at full context
```

PagedAttention improves this in practice — pages allocated on demand, so a 200-token request uses ~80MB, not 802MB.

### Other limiting factors
| Factor | How It Limits |
|--------|--------------|
| KV cache memory | Hard ceiling — OOM if exceeded |
| Memory bandwidth | Soft ceiling — more batched requests = more KV reads per step = higher latency per token |
| Compute (prefill) | Burst of new requests hitting prefill can bottleneck GPU FLOPS |
| Scheduler overhead | CPU overhead per request per step, matters at hundreds of concurrent |

**Production answer:** Don't calculate theoretically — load test and measure where latency becomes unacceptable. Same as determining ASG max-size.

## The Inference Tradeoff Diamond

The traditional "golden triangle" (cost / latency / quality) is incomplete. Concurrency is a first-class dimension.

```
        Quality
          ▲
         / \
        /   \
   Cost ◄─────► Latency
        \   /
         \ /
          ▼
      Concurrency
```

Why concurrency doesn't fold into the other three:
1. **Hard limits that don't trade off smoothly** — KV cache creates a cliff at N concurrent requests. N is fine, N+1 causes preemption/rejection.
2. **Independent scaling signal** — in Stage 4, we'll scale on queue depth (pending requests exceeding capacity), not just latency or cost.
3. **A system can be within cost and latency SLA but unable to handle a traffic spike** — that's a concurrency problem, not a cost or latency problem.

Every knob affects all four:
- More quantization → lower quality, lower cost, lower latency, **higher concurrency**
- Bigger GPU → higher cost, lower latency, same quality, higher concurrency
- More replicas → higher cost, same per-request latency, same quality, higher concurrency
- Longer context → same quality, higher latency, same cost, **lower concurrency**

## What's Next — Stage 3
- Add Prometheus metrics: queue depth, TTFT, TBT, P99 latency, throughput, GPU utilization
- Grafana dashboard on the same instance (docker-compose)
- vLLM exposes a `/metrics` endpoint with Prometheus-format metrics out of the box
