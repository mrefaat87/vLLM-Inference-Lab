I now have a thorough understanding of the existing codebase patterns. Let me provide the complete plan.

---

# vLLM Inference Lab: 18 New Experiments Plan

## Part 1: Complete Reorganized Test Numbering (1-28)

### Group A: Fundamentals -- Batching & Scheduling (Tests 1-5)

| New # | Old # | Test Name | Status |
|-------|-------|-----------|--------|
| 1 | old-1 | Ollama Sequential Queuing (Stage 1 baseline, local M4) | DONE |
| 2 | old-2 | vLLM Continuous Batching (5 concurrent, AWQ 4-bit) | DONE |
| 3 | old-5 | Max Concurrency Scaling (1-80 concurrent, short prompt) | DONE |
| 4 | old-6 | Variable Prompt Scheduling (S/M/L mixed, scheduling order) | DONE |
| 5 | new-30 | **Batch Size vs Throughput Curve** | NEW |

### Group B: Quantization & Memory Efficiency (Tests 6-8)

| New # | Old # | Test Name | Status |
|-------|-------|-----------|--------|
| 6 | old-3 | Quantization Speed Comparison (AWQ vs Marlin vs GPTQ Int8) | DONE |
| 7 | old-4 | Quantization Quality Benchmark (60 questions) | DONE |
| 8 | new-15 | **KV Cache Quantization (FP16 vs FP8)** | NEW |

### Group C: Caching & Memory Management (Tests 9-13)

| New # | Old # | Test Name | Status |
|-------|-------|-----------|--------|
| 9 | new-13 | **Prefix Caching -- Explicit On vs Off** | NEW |
| 10 | old-7 | KV Cache Cliff with Prefix Caching ON | DONE |
| 11 | old-8 | Preemption & Queuing (Int8 + no prefix cache + unique prompts) | DONE |
| 12 | new-16 | **Context Window Scaling (max_model_len effect)** | NEW |
| 13 | new-18 | **Preemption Policy -- Recompute vs Swap** | NEW |

### Group D: Prefill & Decode Mechanics (Tests 14-17)

| New # | Old # | Test Name | Status |
|-------|-------|-----------|--------|
| 14 | old-9 | Prefill vs Decode Bottlenecks (vary input vs output) | DONE |
| 15 | old-10 | Input vs Output Token Cost Asymmetry | DONE |
| 16 | new-19 | **Chunked Prefill -- Direct Interleaving Measurement** | NEW |
| 17 | new-36 | **max_num_batched_tokens -- Prefill Budget Control** | NEW |

### Group E: Decode Acceleration (Tests 18-19)

| New # | Old # | Test Name | Status |
|-------|-------|-----------|--------|
| 18 | new-11 | **Speculative Decoding -- Draft Model Speedup** | NEW |
| 19 | new-12 | **Speculative Decoding -- Varying num_speculative_tokens** | NEW |

### Group F: Output Control & Sampling (Tests 20-21)

| New # | Old # | Test Name | Status |
|-------|-------|-----------|--------|
| 20 | new-22 | **Guided/Structured Output (JSON Schema)** | NEW |
| 21 | new-23 | **Temperature/Sampling Effects on Speed** | NEW |

### Group G: Model Management & Operations (Tests 22-24)

| New # | Old # | Test Name | Status |
|-------|-------|-----------|--------|
| 22 | new-24 | **LoRA Adapter Serving -- Multi-Tenant** | NEW |
| 23 | new-26 | **Cold Start and Model Loading Time (incl. S3 comparison)** | NEW |
| 24 | new-35 | **Multi-Turn Conversation Simulation** | NEW |

### Group H: Load Testing & Production Patterns (Tests 25-28)

| New # | Old # | Test Name | Status |
|-------|-------|-----------|--------|
| 25 | new-20 | **Tail Latency Under Sustained Load (P50 vs P99)** | NEW |
| 26 | new-31 | **Soak Test (30+ min sustained)** | NEW |
| 27 | new-32 | **Graceful Degradation Under Overload** | NEW |
| 28 | new-37 | **Full Production Simulation** | NEW |

---

## Part 2: Detailed Plans for Each of the 18 New Experiments

---

### Test 5 (old-30): Batch Size vs Throughput Curve

**Validated Scenario**

This test measures how system throughput (total tokens/sec across all requests) changes as the number of concurrent requests (effective batch size) scales up. Unlike Test 3 (max concurrency) which used short prompts and measured latency breakdown, this test uses medium-length prompts (200 input, 200 output) to make the throughput curve more realistic. The goal is to find the "throughput knee" -- the concurrency level where adding more requests stops improving system throughput.

No special vLLM flags needed. Uses the standard AWQ config.

**T4 Limitations**: None specific. Standard operation.

**vLLM Docker Run Command**

```bash
docker run -d \
    --name vllm-server \
    --restart unless-stopped \
    --gpus all \
    -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin \
    --dtype half \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048
```

This is the same server config as Tests 1-4 (Group A). No restart needed if running sequentially within the group.

**Test Script Approach**

- Concurrency levels: [1, 2, 4, 8, 16, 24, 32, 48, 64]
- Each level sends N concurrent requests, all with ~200 input tokens and max_tokens=200
- Measure wall-clock time for all N to complete, compute system_tok_s = total_tokens_generated / wall_time
- 3 trials per concurrency level, report median
- Use the same filler-text prompt generation pattern from `stage2_exp2_prefill_vs_decode.py`

**Result JSON Schema**

```json
[
  {
    "concurrency": 1,
    "wall_start": "2026-03-20 10:00:00",
    "wall_end": "2026-03-20 10:00:05",
    "wall_duration_sec": 5.0,
    "successes": 1,
    "failures": 0,
    "total_tokens_generated": 200,
    "system_tok_s": 40.0,
    "per_request_tok_s_avg": 40.0,
    "per_request_tok_s_min": 40.0,
    "ttft_p50": 0.25,
    "ttft_p99": 0.25,
    "ttft_max": 0.25,
    "total_p50": 5.0,
    "total_p99": 5.0,
    "total_max": 5.0,
    "avg_tokens_generated": 200.0,
    "trial": 1
  }
]
```

**Notebook Cell Plan**

- **Chart type**: Dual-axis line chart. X = concurrency. Left Y = system throughput (tok/s). Right Y = per-request throughput (tok/s).
- **Insight**: System throughput rises linearly then plateaus. Per-request throughput declines throughout. The crossover region is the "throughput knee" -- optimal operating point for max system efficiency.
- **Second chart**: Bar chart showing throughput efficiency = system_tok_s / (concurrency * single_request_tok_s) to show how GPU utilization efficiency changes.

**Dependencies**: Same server config as Tests 1-4. No restart needed.

---

### Test 8 (old-15): KV Cache Quantization (FP16 vs FP8)

**Validated Scenario**

vLLM supports `--kv-cache-dtype fp8` to store KV cache values in FP8 format (E4M3 or E5M2), reducing per-token KV cache memory by 50%. This should roughly double the number of concurrent requests possible at the same VRAM budget.

**T4 Limitation**: The T4 has compute capability 7.5 and does NOT have native FP8 compute hardware (that requires sm_89+ / Ada Lovelace). However, vLLM can still store KV cache in FP8 and cast to FP16 for attention computation. This has been supported since vLLM v0.4.x. The `--kv-cache-dtype fp8` flag is confirmed in vLLM v0.17.1. The cast overhead may reduce the memory savings benefit slightly on T4 vs native FP8 GPUs.

**Important caveat**: There is a risk this flag is silently ignored or errors on T4. The test script should verify KV cache capacity actually changed by comparing the reported "Available KV cache memory" and "Maximum concurrency" in server logs between the two runs.

**vLLM Docker Run Commands**

Run A (baseline FP16 KV -- same as standard config):
```bash
docker run -d \
    --name vllm-fp16kv \
    --gpus all \
    -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin \
    --dtype half \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048
```

Run B (FP8 KV cache):
```bash
docker run -d \
    --name vllm-fp8kv \
    --gpus all \
    -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin \
    --dtype half \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048 \
    --kv-cache-dtype fp8
```

**Test Script Approach**

1. Start FP16 KV server, capture startup logs to extract "Available KV cache memory" and "Maximum concurrency"
2. Run the same cliff test from Test 10 (concurrency levels [20, 30, 40, 50, 60, 70, 80, 90, 100]) with long prompts + max_tokens=1500
3. Restart with FP8 KV, repeat
4. Also run a quality sanity check: 20 questions at temperature=0 to verify FP8 KV cache does not degrade output quality
5. 1 trial per concurrency level per config (cliff tests are deterministic enough)

**Result JSON Schema**

```json
{
  "fp16_kv": {
    "kv_cache_memory_gib": 10.0,
    "max_concurrency_theoretical": 38.5,
    "cliff_results": [
      {
        "concurrency": 20,
        "wall_start": "...",
        "successes": 20,
        "failures": 0,
        "ttft_p50": 0.5,
        "ttft_p99": 0.6,
        "ttft_max": 0.6,
        "total_p50": 56.0,
        "total_p99": 57.0,
        "total_max": 57.0,
        "per_request_tok_s_avg": 26.8,
        "system_tok_s": 536.0,
        "avg_tokens_generated": 1500,
        "kv_usage_pct_peak": 15.0
      }
    ]
  },
  "fp8_kv": {
    "kv_cache_memory_gib": 10.0,
    "max_concurrency_theoretical": 77.0,
    "cliff_results": [ "..." ],
    "quality_check": {
      "questions": 20,
      "pass_fp16": 20,
      "pass_fp8": 20,
      "mismatches": 0
    }
  }
}
```

**Notebook Cell Plan**

- **Chart type**: Overlaid line charts. X = concurrency. Y = system throughput (tok/s). Two lines: FP16 KV vs FP8 KV.
- **Insight**: FP8 KV should sustain throughput to ~2x higher concurrency before hitting the cliff. The cliff itself should be just as sharp.
- **Second chart**: Stacked bar comparing KV capacity (tokens) for FP16 vs FP8.

**Dependencies**: Requires 2 server restarts (FP16 then FP8). Cannot share config with other groups.

---

### Test 9 (old-13): Prefix Caching -- Explicit On vs Off

**Validated Scenario**

Prefix caching is ON by default in vLLM v0.17.1. It can be disabled with `--no-prefix-caching` (or `--enable-prefix-caching false` in older versions; confirm the exact flag). This test directly measures the TTFT benefit when multiple requests share the same prompt prefix.

The test sends bursts of requests that share an identical system prompt prefix, then measures TTFT for the first request (cold -- prefix must be computed) vs subsequent requests (warm -- prefix KV can be reused).

**T4 Limitations**: None. Prefix caching works on all GPUs.

**vLLM Docker Run Commands**

Run A (prefix caching ON -- default):
```bash
docker run -d \
    --name vllm-prefix-on \
    --gpus all \
    -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin \
    --dtype half \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048
```

Run B (prefix caching OFF):
```bash
docker run -d \
    --name vllm-prefix-off \
    --gpus all \
    -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin \
    --dtype half \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048 \
    --no-prefix-caching
```

**Test Script Approach**

1. Create a long system prompt (~500 tokens) shared across all requests
2. Create 10 different user questions (different suffixes)
3. Send requests serially first (request 1 is cold, requests 2-10 should benefit from cached prefix)
4. Then send 10 concurrently (all share prefix, but only first to arrive computes it)
5. Repeat for prefix caching OFF -- all requests must compute the full prefix
6. 3 trials each
7. Measure: TTFT per request, track which request was "cold" vs "warm"

**Result JSON Schema**

```json
{
  "prefix_on": {
    "serial": {
      "cold_ttft_avg": 0.45,
      "warm_ttft_avg": 0.22,
      "speedup_ratio": 2.05,
      "trials": 3
    },
    "concurrent_10": {
      "wall_start": "...",
      "successes": 10,
      "failures": 0,
      "ttft_p50": 0.25,
      "ttft_p99": 0.50,
      "ttft_max": 0.50,
      "total_p50": 1.2,
      "total_p99": 1.5,
      "total_max": 1.5,
      "prefix_cache_hit_rate": 82.0,
      "trials": 3
    }
  },
  "prefix_off": {
    "serial": {
      "cold_ttft_avg": 0.45,
      "warm_ttft_avg": 0.44,
      "speedup_ratio": 1.02,
      "trials": 3
    },
    "concurrent_10": {
      "wall_start": "...",
      "successes": 10,
      "failures": 0,
      "ttft_p50": 0.48,
      "ttft_p99": 0.55,
      "ttft_max": 0.55,
      "total_p50": 1.5,
      "total_p99": 1.8,
      "total_max": 1.8,
      "prefix_cache_hit_rate": 0.0,
      "trials": 3
    }
  }
}
```

**Notebook Cell Plan**

- **Chart type**: Grouped bar chart. X = request number (1-10, serial). Y = TTFT. Two groups: prefix ON vs OFF. Request 1 should be similar; requests 2-10 should diverge significantly.
- **Insight**: Prefix caching is the inference analog of CDN caching -- identical prefix = identical "static content" that doesn't need recomputation. The warm/cold TTFT ratio quantifies the ROI.

**Dependencies**: Requires 2 server restarts (ON then OFF).

---

### Test 12 (old-16): Context Window Scaling (max_model_len effect)

**Validated Scenario**

`--max-model-len` controls the maximum sequence length vLLM will allocate KV cache for. Smaller values mean less KV cache per request, allowing more concurrent requests. This test measures the tradeoff: shorter max context = higher concurrency capacity but limits what the model can handle.

**T4 Limitations**: None specific. The model (Qwen2.5-7B) supports up to 32K context natively, but T4 VRAM constrains practical limits.

**vLLM Docker Run Commands**

Four configurations (4 server restarts):

```bash
# max_model_len = 512
docker run -d --name vllm-ctx512 --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 512

# max_model_len = 1024
docker run -d --name vllm-ctx1024 --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 1024

# max_model_len = 2048 (current default)
# (same as standard config)

# max_model_len = 4096
docker run -d --name vllm-ctx4096 --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 4096
```

**Test Script Approach**

For each config:
1. Capture startup logs for "Maximum concurrency for X tokens per request"
2. Send 10 concurrent requests with short prompts (50 tokens) + max_tokens=100 to get baseline throughput
3. Send a scaling test: ramp concurrency [10, 20, 40, 60, 80, 100, 120] and find where failures begin
4. 1 trial per level (cliff behavior is deterministic)

**Result JSON Schema**

```json
[
  {
    "max_model_len": 512,
    "max_concurrency_reported": 154,
    "kv_cache_memory_gib": 10.0,
    "baseline_10_concurrent": {
      "wall_start": "...",
      "successes": 10,
      "failures": 0,
      "ttft_p50": 0.20,
      "ttft_p99": 0.25,
      "total_p50": 2.0,
      "total_p99": 2.5,
      "system_tok_s": 450.0
    },
    "cliff_level": 120,
    "first_failure_level": 140
  }
]
```

**Notebook Cell Plan**

- **Chart type**: Bar chart. X = max_model_len [512, 1024, 2048, 4096]. Y = max supported concurrency before cliff.
- **Second chart**: Line chart showing system throughput at 10 concurrent for each config (should be similar -- max_model_len doesn't affect throughput below the capacity limit).
- **Insight**: max_model_len is like setting the max connection pool size per instance. Lower values = more total connections but each is lighter weight.

**Dependencies**: 4 server restarts needed. Can partially share with Test 10 (2048 config).

---

### Test 13 (old-18): Preemption Policy -- Recompute vs Swap

**Validated Scenario**

vLLM has a `--preemption-mode` flag with two options:
- `recompute`: When a request is preempted (KV cache evicted), its KV cache is discarded and must be recomputed from the prompt when resumed.
- `swap`: Preempted KV cache is swapped to CPU RAM, then swapped back when the request can resume.

**CRITICAL T4/V1 ENGINE CONCERN**: In vLLM v0.17.1, the V1 engine (default) may not support `--preemption-mode` at all. The V0 engine supported it. The test script must first check if the flag is accepted. If V1 does not support it, we would need to add `--enforce-eager` or use `VLLM_USE_V1=0` environment variable to fall back to V0 engine. The test should document this discovery.

To trigger preemption, we need to exceed KV cache capacity, which we already know how to do from Test 11 (old-8). Use the GPTQ Int8 model with its smaller KV cache (4.21 GiB = ~39 max concurrent) and send 45+ concurrent long-context requests.

**vLLM Docker Run Commands**

```bash
# Recompute mode (V0 engine fallback if needed)
docker run -d --name vllm-preempt-recompute --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    -e VLLM_USE_V1=0 \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8 \
    --quantization gptq --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 2048 \
    --no-prefix-caching \
    --preemption-mode recompute

# Swap mode
docker run -d --name vllm-preempt-swap --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    -e VLLM_USE_V1=0 \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8 \
    --quantization gptq --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 2048 \
    --no-prefix-caching \
    --preemption-mode swap
```

**Test Script Approach**

1. For each mode, send N=42 concurrent requests (slightly above the ~39 limit) with unique long prompts + max_tokens=1500
2. Capture server logs for preemption events, waiting queue depth, KV cache usage
3. Measure: total wall time, per-request latency, any failures
4. Compare: recompute should have lower VRAM-to-CPU bandwidth usage but higher GPU compute usage; swap should use more CPU RAM but avoid recomputation
5. 1 trial per mode (long-running tests)

**Result JSON Schema**

```json
{
  "recompute": {
    "concurrency": 42,
    "wall_start": "...",
    "wall_duration_sec": 350.0,
    "successes": 42,
    "failures": 0,
    "ttft_p50": 3.0,
    "ttft_p99": 5.0,
    "ttft_max": 6.0,
    "total_p50": 340.0,
    "total_p99": 350.0,
    "total_max": 350.0,
    "avg_tok_per_sec": 4.2,
    "preemption_count": 15,
    "kv_peak_pct": 100,
    "waiting_max": 3
  },
  "swap": {
    "concurrency": 42,
    "wall_start": "...",
    "wall_duration_sec": 320.0,
    "successes": 42,
    "failures": 0,
    "ttft_p50": 3.0,
    "ttft_p99": 5.0,
    "ttft_max": 6.0,
    "total_p50": 310.0,
    "total_p99": 320.0,
    "total_max": 320.0,
    "avg_tok_per_sec": 4.6,
    "preemption_count": 15,
    "kv_peak_pct": 100,
    "waiting_max": 3
  }
}
```

**Notebook Cell Plan**

- **Chart type**: Side-by-side bar chart comparing recompute vs swap on: wall duration, avg tok/s, preemption count, tail latency (p99).
- **Insight**: Swap trades CPU RAM for GPU compute. On g4dn.xlarge (16GB CPU RAM), swap space is limited. Recompute avoids CPU-GPU transfer but wastes GPU cycles redoing prefill. Auto Scaling analogy: swap = session draining to standby instances; recompute = stateless restart.

**Dependencies**: 2 server restarts. Uses GPTQ Int8 model (same as Test 11). Requires V0 engine likely.

---

### Test 16 (old-19): Chunked Prefill -- Direct Interleaving Measurement

**Validated Scenario**

Chunked prefill is enabled by default in vLLM v0.17.1. It can be disabled with `--enable-chunked-prefill false` (or `--no-chunked-prefill`). When enabled, a long prefill is broken into chunks so that decode steps for already-running requests can interleave between chunks. The hypothesis: with chunked prefill enabled, ongoing decode requests should see stable TBT even when a new long-prefill request arrives; without it, decode TBT should spike during prefill.

**T4 Limitations**: None. Chunked prefill works on all GPUs.

**vLLM Docker Run Commands**

```bash
# Chunked prefill ON (default)
docker run -d --name vllm-chunked-on --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 2048

# Chunked prefill OFF
docker run -d --name vllm-chunked-off --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 2048 \
    --enable-chunked-prefill false
```

**Note on V1 engine**: In vLLM V1, chunked prefill may always be on and the flag may be ignored. If so, fall back to V0 with `VLLM_USE_V1=0`. The test script should detect this.

**Test Script Approach**

1. Start a "background" decode request: send a request with a short prompt (~20 tokens) and max_tokens=500. This request will be actively decoding tokens.
2. While the background request is streaming, measure its inter-token times (TBT per token).
3. After ~1 second (background request has started generating), inject a "prefill bomb": send a new request with a ~1500-token prompt and max_tokens=10.
4. Continue measuring the background request's TBT. With chunked prefill, TBT should remain stable. Without it, TBT should spike dramatically when the prefill bomb arrives.
5. 5 trials per configuration.

**Result JSON Schema**

```json
{
  "chunked_on": {
    "background_tbt_before_bomb_ms": [22, 22, 23, 22, 22],
    "background_tbt_during_bomb_ms": [24, 25, 23, 24, 25],
    "background_tbt_after_bomb_ms": [22, 22, 22, 23, 22],
    "tbt_spike_ratio": 1.09,
    "bomb_ttft_sec": 0.8,
    "trials": 5
  },
  "chunked_off": {
    "background_tbt_before_bomb_ms": [22, 22, 23, 22, 22],
    "background_tbt_during_bomb_ms": [150, 180, 160, 170, 155],
    "background_tbt_after_bomb_ms": [22, 22, 22, 23, 22],
    "tbt_spike_ratio": 7.3,
    "bomb_ttft_sec": 0.4,
    "trials": 5
  }
}
```

**Notebook Cell Plan**

- **Chart type**: Time-series line chart. X = token index from the background request. Y = inter-token time (ms). Vertical dashed line at "bomb injected" point. Two lines: chunked ON vs OFF.
- **Insight**: Chunked prefill is like the difference between a non-preemptive and preemptive task scheduler. Without chunking, a long prefill monopolizes the GPU -- just like a long-running Lambda freezing the event loop.

**Dependencies**: 2 server restarts (ON vs OFF). If V1 always has chunked on, only one meaningful run possible.

---

### Test 17 (old-36): max_num_batched_tokens -- Prefill Budget Control

**Validated Scenario**

`--max-num-batched-tokens` controls the maximum number of tokens that can be processed in a single forward pass (prefill batch budget). Higher values allow more tokens to be prefilled at once (faster TTFT for large prompts) but can cause TBT stalls for decode requests. Lower values protect decode latency but slow down prefill.

The flag is confirmed in vLLM. Default value is 2048 in the existing setup (visible in stage2_learnings.md: "Chunked prefill is enabled with max_num_batched_tokens=2048").

**T4 Limitations**: Very high values (>8192) may OOM on T4 due to activation memory during prefill.

**vLLM Docker Run Commands**

```bash
# max_num_batched_tokens = 512 (aggressive decode protection)
docker run -d --name vllm-mbt512 --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 2048 \
    --max-num-batched-tokens 512

# max_num_batched_tokens = 2048 (default)
# (standard config)

# max_num_batched_tokens = 4096 (aggressive prefill)
docker run -d --name vllm-mbt4096 --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 4096 \
    --max-num-batched-tokens 4096
```

**Test Script Approach**

1. For each config, use the same "prefill bomb + background decode" pattern from Test 16
2. Additionally, measure TTFT for a large prompt (~1500 tokens) in isolation -- lower max_num_batched_tokens should increase TTFT because the prefill must be chunked into more pieces
3. Then measure under load: 20 concurrent requests with mixed prompt sizes and track TBT stability
4. 3 trials per config

**Result JSON Schema**

```json
[
  {
    "max_num_batched_tokens": 512,
    "isolated_large_prompt_ttft_sec": 1.2,
    "background_tbt_spike_ratio": 1.05,
    "concurrent_20_ttft_p50": 1.5,
    "concurrent_20_ttft_p99": 2.0,
    "concurrent_20_total_p50": 12.0,
    "concurrent_20_total_p99": 15.0,
    "concurrent_20_system_tok_s": 400.0,
    "wall_start": "...",
    "successes": 20,
    "failures": 0,
    "trials": 3
  }
]
```

**Notebook Cell Plan**

- **Chart type**: Grouped bar chart. X = max_num_batched_tokens [512, 2048, 4096]. Three bars per group: isolated TTFT, TBT spike ratio, system throughput at 20 concurrent.
- **Insight**: This is the prefill-decode scheduling knob. Production systems tune this based on SLA: if P99 TBT matters (streaming chat), keep it low; if TTFT matters (batch processing), keep it high.

**Dependencies**: 3 server restarts. The 2048 config is shared with standard.

---

### Test 18 (old-11): Speculative Decoding -- Draft Model Speedup

**Validated Scenario**

Speculative decoding uses a small "draft" model to generate candidate tokens quickly, then the main model verifies them in a single forward pass. If the draft model's predictions are accepted, you get multiple tokens per main-model forward pass, reducing latency.

vLLM v0.17.1 supports speculative decoding via `--speculative-model`. For Qwen2.5 family:
- Target model: `Qwen/Qwen2.5-7B-Instruct-AWQ` (~4GB)
- Draft model: `Qwen/Qwen2.5-0.5B-Instruct` (~1GB FP16) or `Qwen/Qwen2.5-1.5B-Instruct` (~3GB FP16)
- Combined: 4 + 1 = ~5GB (fits in 16GB with room for KV cache) or 4 + 3 = ~7GB

The draft and target model must share the same tokenizer (vocabulary). The Qwen2.5 family shares the same tokenizer.

**T4 Limitation**: Speculative decoding adds overhead per step. On T4's limited bandwidth, the verification step for rejected tokens can negate gains. Benefit is most likely with simple/predictable text (high acceptance rate). The 0.5B draft model is strongly recommended to minimize VRAM usage.

**vLLM Docker Run Commands**

```bash
# Baseline (no speculative decoding)
docker run -d --name vllm-nospec --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 2048

# With speculative decoding (0.5B draft)
docker run -d --name vllm-spec --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 2048 \
    --speculative-model Qwen/Qwen2.5-0.5B-Instruct \
    --num-speculative-tokens 5
```

**Test Script Approach**

1. Single request latency comparison: send 10 identical requests one at a time, measure TTFT and total latency for ~200 output tokens
2. Test with 3 prompt types: (a) simple factual question (high acceptance expected), (b) creative writing (lower acceptance), (c) code generation (medium acceptance)
3. Compare per-request tok/s between baseline and speculative
4. Also test at concurrency 5 and 10 to see if speculative decoding degrades under batch load (it typically does -- speculative decoding is best for single-request latency)
5. 3 trials per configuration per prompt type

**Result JSON Schema**

```json
{
  "baseline": {
    "single_request": {
      "factual": {"ttft_avg": 0.25, "total_avg": 4.5, "tok_s_avg": 45.0, "trials": 3},
      "creative": {"ttft_avg": 0.26, "total_avg": 4.6, "tok_s_avg": 44.0, "trials": 3},
      "code": {"ttft_avg": 0.28, "total_avg": 4.8, "tok_s_avg": 42.0, "trials": 3}
    },
    "concurrent_5": {
      "wall_start": "...",
      "successes": 5,
      "failures": 0,
      "ttft_p50": 0.30,
      "ttft_p99": 0.35,
      "total_p50": 5.0,
      "total_p99": 5.5,
      "per_request_tok_s_avg": 40.0,
      "system_tok_s": 200.0
    }
  },
  "speculative": {
    "draft_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "num_speculative_tokens": 5,
    "single_request": {
      "factual": {"ttft_avg": 0.30, "total_avg": 3.2, "tok_s_avg": 63.0, "acceptance_rate": 0.78, "trials": 3},
      "creative": {"ttft_avg": 0.32, "total_avg": 4.0, "tok_s_avg": 50.0, "acceptance_rate": 0.55, "trials": 3},
      "code": {"ttft_avg": 0.31, "total_avg": 3.6, "tok_s_avg": 56.0, "acceptance_rate": 0.65, "trials": 3}
    },
    "concurrent_5": {
      "wall_start": "...",
      "successes": 5,
      "failures": 0,
      "ttft_p50": 0.40,
      "ttft_p99": 0.50,
      "total_p50": 6.0,
      "total_p99": 7.0,
      "per_request_tok_s_avg": 35.0,
      "system_tok_s": 175.0
    }
  }
}
```

**Notebook Cell Plan**

- **Chart type**: Grouped bar chart. X = prompt type [factual, creative, code]. Y = per-request tok/s. Two bars: baseline vs speculative.
- **Second chart**: Bar chart showing acceptance rate by prompt type.
- **Third chart**: Line chart comparing single-request latency vs concurrent latency, showing speculative decoding's "sweet spot" is low concurrency.
- **Insight**: Speculative decoding is a single-request latency optimization. Under batch load, the draft model competes for GPU resources. Analogy: speculative execution in CPUs -- great for single-thread but does not help multi-core throughput.

**Dependencies**: 2 server restarts (baseline + speculative). Baseline config is same as standard.

---

### Test 19 (old-12): Speculative Decoding -- Varying num_speculative_tokens

**Validated Scenario**

`--num-speculative-tokens` controls how many tokens the draft model proposes before the target model verifies. Higher values = more tokens per verification pass (more speedup if accepted) but higher rejection risk (wasted compute if wrong). This test sweeps the parameter to find the optimal value.

**T4 Limitations**: Higher values increase memory usage per step. On T4, values above 7-8 may cause OOM or significant slowdown.

**vLLM Docker Run Commands**

```bash
# num_speculative_tokens = 1
docker run -d --name vllm-spec1 --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 2048 \
    --speculative-model Qwen/Qwen2.5-0.5B-Instruct \
    --num-speculative-tokens 1

# Repeat for num_speculative_tokens = 2, 3, 5, 7
# (5 server restarts total)
```

**Test Script Approach**

1. For each value of num_speculative_tokens [1, 2, 3, 5, 7]:
   - Send 10 single requests (factual prompt, max_tokens=200)
   - Measure TTFT, total time, tok/s
2. 3 trials per value
3. Compare against the baseline (no speculative decoding) from Test 18

**Result JSON Schema**

```json
[
  {
    "num_speculative_tokens": 1,
    "ttft_avg": 0.28,
    "total_avg": 4.2,
    "tok_s_avg": 47.6,
    "wall_start": "...",
    "successes": 10,
    "failures": 0,
    "trials": 3
  },
  {
    "num_speculative_tokens": 2,
    "ttft_avg": 0.29,
    "total_avg": 3.8,
    "tok_s_avg": 52.6,
    "wall_start": "...",
    "successes": 10,
    "failures": 0,
    "trials": 3
  }
]
```

**Notebook Cell Plan**

- **Chart type**: Line chart. X = num_speculative_tokens [0 (baseline), 1, 2, 3, 5, 7]. Y = per-request tok/s.
- **Insight**: There should be a clear optimum. Too few = overhead without benefit. Too many = high rejection rate. The optimal value depends on draft model quality and text predictability.

**Dependencies**: 5 server restarts (one per value). Must run after Test 18 (uses same draft model).

---

### Test 20 (old-22): Guided/Structured Output (JSON Schema)

**Validated Scenario**

vLLM v0.17.1 supports guided decoding through the OpenAI-compatible API's `response_format` parameter. When `response_format={"type": "json_schema", "json_schema": {...}}` is provided, vLLM constrains the model's token sampling to only produce valid JSON matching the schema. This is powered by the `outlines` or `lm-format-enforcer` backend.

**T4 Limitations**: None specific. Guided decoding adds CPU overhead for token masking but no additional GPU requirements.

**vLLM Docker Run Command**

```bash
docker run -d --name vllm-guided --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 2048 \
    --guided-decoding-backend outlines
```

Note: `--guided-decoding-backend` defaults to `outlines` in v0.17.1. Can also try `lm-format-enforcer` or `xgrammar`.

**Test Script Approach**

1. Define a JSON schema:
   ```json
   {
     "type": "object",
     "properties": {
       "name": {"type": "string"},
       "age": {"type": "integer"},
       "skills": {"type": "array", "items": {"type": "string"}},
       "summary": {"type": "string"}
     },
     "required": ["name", "age", "skills", "summary"]
   }
   ```
2. Send 20 requests asking "Generate a profile for a software engineer" with `response_format` set to the JSON schema
3. Send 20 identical requests without guided decoding, with the prompt modified to say "respond in JSON format"
4. Compare: (a) parsing success rate (is the output valid JSON matching the schema?), (b) latency difference, (c) token count difference
5. Test at concurrency 1 (serial) and concurrency 10
6. Use `/v1/chat/completions` endpoint (guided decoding works via chat API)

**Result JSON Schema**

```json
{
  "guided": {
    "serial": {
      "valid_json_count": 20,
      "schema_match_count": 20,
      "ttft_avg": 0.35,
      "total_avg": 3.0,
      "tok_s_avg": 40.0,
      "avg_tokens_generated": 120,
      "wall_start": "...",
      "successes": 20,
      "failures": 0,
      "trials": 1
    },
    "concurrent_10": {
      "valid_json_count": 10,
      "schema_match_count": 10,
      "ttft_p50": 0.40,
      "ttft_p99": 0.55,
      "total_p50": 4.0,
      "total_p99": 5.0,
      "system_tok_s": 350.0,
      "wall_start": "...",
      "successes": 10,
      "failures": 0
    }
  },
  "unguided": {
    "serial": {
      "valid_json_count": 15,
      "schema_match_count": 12,
      "ttft_avg": 0.25,
      "total_avg": 3.5,
      "tok_s_avg": 44.0,
      "avg_tokens_generated": 150,
      "wall_start": "...",
      "successes": 20,
      "failures": 0,
      "trials": 1
    },
    "concurrent_10": {
      "valid_json_count": 7,
      "schema_match_count": 5,
      "ttft_p50": 0.30,
      "ttft_p99": 0.40,
      "total_p50": 4.5,
      "total_p99": 5.5,
      "system_tok_s": 380.0,
      "wall_start": "...",
      "successes": 10,
      "failures": 0
    }
  }
}
```

**Notebook Cell Plan**

- **Chart type**: Stacked bar chart. X = [guided, unguided]. Y = count. Stacks: valid JSON, valid schema, invalid.
- **Second chart**: Side-by-side latency comparison (guided vs unguided tok/s).
- **Insight**: Guided decoding guarantees schema compliance at a small latency cost (~5-15%). For production APIs that consume JSON, this eliminates retry logic and parsing errors. Analogy: schema validation at the API gateway vs hoping clients send valid payloads.

**Dependencies**: Same standard server config. No restart needed if `--guided-decoding-backend` is not specified (outlines is default).

---

### Test 21 (old-23): Temperature/Sampling Effects on Speed

**Validated Scenario**

Different sampling parameters (temperature, top_p, top_k) should theoretically not affect decode speed since the GPU computation is the same -- sampling happens on CPU after logits are computed. However, temperature=0 (greedy) may enable CUDA graph optimizations that skip sampling entirely. This test validates whether sampling parameters have measurable latency impact.

**T4 Limitations**: None.

**vLLM Docker Run Command**

Same standard config (no server restart needed):
```bash
docker run -d --name vllm-server --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 2048
```

**Test Script Approach**

1. Send requests with identical prompts (~100 tokens input, max_tokens=200) varying only sampling parameters:
   - temperature=0 (greedy)
   - temperature=0.3, top_p=0.9
   - temperature=0.7, top_p=0.9
   - temperature=1.0, top_p=1.0
   - temperature=1.0, top_p=0.9, top_k=50
   - temperature=1.5, top_p=0.95
2. 10 requests per config, serial (single request at a time to isolate sampling overhead)
3. Measure TTFT, total time, tok/s
4. Note: different temperatures will produce different length outputs, so also track actual tokens generated and normalize to tok/s

**Result JSON Schema**

```json
[
  {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "ttft_avg": 0.24,
    "total_avg": 4.2,
    "tok_s_avg": 47.5,
    "avg_tokens_generated": 200,
    "wall_start": "...",
    "successes": 10,
    "failures": 0,
    "trials": 1
  }
]
```

**Notebook Cell Plan**

- **Chart type**: Bar chart. X = sampling config label. Y = tok/s.
- **Insight**: If all bars are roughly equal (~1-2% variance), it proves sampling is not the bottleneck -- GPU computation and memory bandwidth dominate. If greedy is significantly faster, CUDA graphs may be optimizing the greedy path.

**Dependencies**: Same standard server config. No restart.

---

### Test 22 (old-24): LoRA Adapter Serving -- Multi-Tenant

**Validated Scenario**

vLLM supports serving multiple LoRA adapters from a single base model with `--enable-lora`. Different requests can specify different LoRA adapters via the `model` field in the API request. This enables multi-tenant serving where each tenant has a fine-tuned adapter.

For LoRA adapters on Qwen2.5-7B, searching HuggingFace shows several options. Alternatively, we can use a different model family with well-known LoRA adapters. One practical approach: use a model like `meta-llama/Llama-3.1-8B-Instruct` with publicly available LoRA adapters, but that model may not fit on T4 without quantization. A safer approach: use the existing Qwen2.5-7B-Instruct-AWQ base and load small dummy LoRA adapters, or find actual Qwen2.5-7B LoRA adapters on HuggingFace.

**Practical recommendation**: Search HuggingFace for "qwen2.5-7b lora" adapters. Several exist for chat, coding, math tasks. If none are suitable, we can train a trivial LoRA adapter using the `peft` library (a 5-minute job on any GPU).

**T4 Limitations**: LoRA adapters are small (typically 10-50MB each) and stored in GPU memory alongside the base model. With AWQ base at ~4GB, we have ample room for dozens of adapters. `--max-loras` controls how many can be loaded simultaneously.

**vLLM Docker Run Command**

```bash
docker run -d --name vllm-lora --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 2048 \
    --enable-lora \
    --max-loras 4 \
    --lora-modules lora-math=/path/to/math-lora lora-code=/path/to/code-lora
```

**Note**: The `--lora-modules` flag syntax is `name=path name2=path2`. Paths can be HuggingFace model IDs (auto-downloaded) or local paths.

**IMPORTANT COMPATIBILITY NOTE**: vLLM LoRA serving with quantized base models (AWQ/GPTQ) has historically had issues. Verify at test time whether `--enable-lora` works with `--quantization awq_marlin`. If not, fall back to `--quantization awq` or use the FP16 base model `Qwen/Qwen2.5-3B-Instruct` (which fits in 16GB as FP16).

**Test Script Approach**

1. Verify LoRA loading by calling `/v1/models` endpoint -- should list base model + LoRA adapters
2. Send 20 requests: 10 to base model, 5 to lora-math, 5 to lora-code
3. Measure: first request to each adapter (cold LoRA load time), subsequent requests (warm)
4. Then send mixed concurrent requests (10 concurrent: mix of base and LoRA)
5. Compare latency between base and LoRA-adapted requests
6. 3 trials

**Result JSON Schema**

```json
{
  "adapters_loaded": ["base", "lora-math", "lora-code"],
  "lora_load_time_sec": {"lora-math": 0.5, "lora-code": 0.4},
  "serial_results": {
    "base": {"ttft_avg": 0.25, "total_avg": 4.5, "tok_s_avg": 44.0, "trials": 3},
    "lora-math": {"ttft_avg": 0.27, "total_avg": 4.6, "tok_s_avg": 43.5, "trials": 3},
    "lora-code": {"ttft_avg": 0.26, "total_avg": 4.5, "tok_s_avg": 44.0, "trials": 3}
  },
  "concurrent_10_mixed": {
    "wall_start": "...",
    "successes": 10,
    "failures": 0,
    "ttft_p50": 0.35,
    "ttft_p99": 0.50,
    "total_p50": 5.0,
    "total_p99": 6.0,
    "system_tok_s": 380.0,
    "per_adapter_tok_s": {"base": 42.0, "lora-math": 40.0, "lora-code": 41.0}
  }
}
```

**Notebook Cell Plan**

- **Chart type**: Grouped bar chart. X = adapter. Y = tok/s. Groups: serial vs concurrent.
- **Insight**: LoRA adapters should have <5% latency overhead vs base model. This enables multi-tenant serving from a single GPU -- analogous to running multiple Lambda functions on the same execution environment with different handler code.

**Dependencies**: 1 server restart (needs --enable-lora). Requires LoRA adapters to be identified/created beforehand.

---

### Test 23 (old-26): Cold Start and Model Loading Time

**Validated Scenario**

Measures the time from `docker run` to the first successful API response. This is the "cold start" latency for model serving. Additionally, compares loading from local EBS storage vs an S3-mounted filesystem using `mountpoint-s3`.

**T4 Limitations**: None specific. g4dn.xlarge has a 125GB NVMe SSD for EBS.

**vLLM Docker Run Commands**

```bash
# Test A: Load from local EBS (model already cached)
time docker run --rm --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 2048

# Test B: Load from S3 (model stored in S3, mounted via mountpoint-s3)
# Prerequisites:
#   apt-get install -y mountpoint-s3
#   mkdir -p /mnt/s3-models
#   mount-s3 your-bucket-name /mnt/s3-models --region us-east-1
time docker run --rm --gpus all -p 8000:8000 \
    -v /mnt/s3-models/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 2048

# Test C: Cold start with no cache (download from HuggingFace)
time docker run --rm --gpus all -p 8000:8000 \
    -v /tmp/empty-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 2048
```

**Test Script Approach**

1. For each source (local EBS, S3 mount, fresh download):
   - Record timestamp when `docker run` is executed
   - Poll `http://localhost:8000/health` every 0.5 seconds until it returns 200
   - Send one test request and record TTFT
   - Record total cold start time = docker run to first successful response
   - Stop the container, clear any warm state
2. 3 trials per source
3. Also measure just the docker image startup overhead (run with a non-existent model to see how fast the Python/vLLM initialization is before model loading)

**Result JSON Schema**

```json
{
  "local_ebs": {
    "docker_to_healthy_sec": [25.0, 24.5, 25.2],
    "docker_to_first_response_sec": [26.0, 25.5, 26.2],
    "first_request_ttft_sec": [0.3, 0.28, 0.32],
    "avg_cold_start_sec": 25.9,
    "trials": 3
  },
  "s3_mountpoint": {
    "docker_to_healthy_sec": [45.0, 48.0, 44.0],
    "docker_to_first_response_sec": [46.0, 49.0, 45.0],
    "first_request_ttft_sec": [0.3, 0.29, 0.31],
    "avg_cold_start_sec": 46.7,
    "trials": 3
  },
  "fresh_download": {
    "docker_to_healthy_sec": [120.0, 115.0, 125.0],
    "docker_to_first_response_sec": [121.0, 116.0, 126.0],
    "first_request_ttft_sec": [0.3, 0.28, 0.30],
    "avg_cold_start_sec": 121.0,
    "trials": 3
  }
}
```

**Notebook Cell Plan**

- **Chart type**: Horizontal bar chart. Y = storage source. X = cold start time (seconds). Error bars from 3 trials.
- **Insight**: Cold start is the #1 enemy of auto-scaling. EBS local is fastest (pure disk read), S3 adds network latency, fresh download adds download time. For Karpenter (Stage 4), pre-caching models on EBS snapshots or using S3 Express One Zone will be critical.

**Dependencies**: Requires multiple full server lifecycle cycles. Must be run independently.

---

### Test 24 (old-35): Multi-Turn Conversation Simulation

**Validated Scenario**

Real chat applications involve multi-turn conversations where each turn includes all previous messages. This means the input grows with each turn. Combined with prefix caching, subsequent turns should see fast TTFT because the prefix (all previous turns) is cached.

Uses the `/v1/chat/completions` endpoint with a growing messages array.

**T4 Limitations**: At 2048 max_model_len, conversations are limited to ~2048 total tokens. With ~500 tokens per turn (input + output), that is roughly 4 turns before hitting the context limit.

**vLLM Docker Run Command**

Same standard config. No restart needed.

```bash
docker run -d --name vllm-server --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 2048
```

**Test Script Approach**

1. Simulate 5 conversations, each with 4-6 turns
2. Each turn: append the model's response to messages array, then add a new user message
3. Measure TTFT per turn (should decrease with prefix caching as more prefix is cached)
4. Measure total latency per turn (should increase as context grows)
5. Track actual token counts per turn
6. Also simulate 5 concurrent conversations (different users, different topics)
7. 3 trials

**Result JSON Schema**

```json
{
  "single_conversation": [
    {
      "turn": 1,
      "input_tokens_approx": 50,
      "output_tokens": 120,
      "ttft_sec": 0.25,
      "total_sec": 2.8,
      "tok_s": 43.0,
      "cumulative_context_tokens": 170,
      "prefix_cache_hit_pct": 0
    },
    {
      "turn": 2,
      "input_tokens_approx": 200,
      "output_tokens": 100,
      "ttft_sec": 0.15,
      "total_sec": 2.5,
      "tok_s": 40.0,
      "cumulative_context_tokens": 470,
      "prefix_cache_hit_pct": 65
    }
  ],
  "concurrent_5_conversations": {
    "wall_start": "...",
    "wall_duration_sec": 45.0,
    "total_turns_completed": 20,
    "successes": 20,
    "failures": 0,
    "ttft_p50_turn1": 0.30,
    "ttft_p50_turn4": 0.20,
    "total_p50_turn1": 3.0,
    "total_p50_turn4": 2.5
  }
}
```

**Notebook Cell Plan**

- **Chart type**: Line chart. X = turn number. Y1 (left axis) = TTFT. Y2 (right axis) = cumulative context tokens.
- **Insight**: TTFT should decrease with each turn (prefix caching benefit grows), while total latency per turn stays roughly stable (output length dominates). This validates prefix caching's value for chat workloads.

**Dependencies**: Same standard server config. No restart.

---

### Test 25 (old-20): Tail Latency Under Sustained Load (P50 vs P99)

**Validated Scenario**

Measures how P50 vs P99 latency diverge under sustained load. In production, P99 is what determines SLA compliance. This test sends a constant stream of requests at various request-per-second rates and measures the latency distribution over time.

**T4 Limitations**: None.

**vLLM Docker Run Command**

Same standard config. No restart needed.

**Test Script Approach**

1. Open-loop load generation: send requests at fixed rates (not waiting for responses before sending next)
2. Request rates: [1, 2, 4, 8, 12, 16, 20] requests per second
3. Each rate sustained for 60 seconds
4. Requests: same prompt (~100 input tokens), max_tokens=100
5. Measure per-request TTFT and total latency, compute P50, P90, P95, P99, P99.9
6. 1 trial per rate (60 seconds provides enough samples)

**Result JSON Schema**

```json
[
  {
    "rps": 1,
    "duration_sec": 60,
    "total_requests": 60,
    "wall_start": "...",
    "successes": 60,
    "failures": 0,
    "ttft_p50": 0.22,
    "ttft_p90": 0.25,
    "ttft_p95": 0.28,
    "ttft_p99": 0.32,
    "ttft_p999": 0.40,
    "ttft_max": 0.45,
    "total_p50": 2.3,
    "total_p90": 2.5,
    "total_p95": 2.7,
    "total_p99": 3.0,
    "total_p999": 3.5,
    "total_max": 4.0,
    "system_tok_s": 44.0,
    "queue_depth_avg": 0.5,
    "queue_depth_max": 2
  }
]
```

**Notebook Cell Plan**

- **Chart type**: Multi-line chart. X = RPS. Multiple lines: P50, P90, P95, P99, P99.9 for TTFT. Shows how the percentile fan spreads under increasing load.
- **Second chart**: Same for total request latency.
- **Insight**: At low RPS, all percentiles cluster together. As RPS approaches GPU capacity, P99 diverges dramatically from P50. The divergence point is the practical capacity limit for SLA-bound workloads.

**Dependencies**: Same standard server config. No restart.

---

### Test 26 (old-31): Soak Test (30+ min sustained)

**Validated Scenario**

Runs sustained load for 30+ minutes to detect memory leaks, performance degradation over time, CUDA OOM events, or other stability issues that don't appear in short tests.

**T4 Limitations**: None.

**vLLM Docker Run Command**

Same standard config. No restart needed.

**Test Script Approach**

1. Send requests at a constant rate of 4 RPS (below saturation) for 30 minutes
2. Every 60 seconds, record a "checkpoint" with: P50/P99 latency for that window, GPU memory usage (via `nvidia-smi`), success/failure count, system throughput
3. Also periodically query the `/metrics` endpoint for vLLM Prometheus metrics
4. At the end, compare first 5 minutes vs last 5 minutes to detect degradation
5. 1 trial (30+ minutes is long enough)

**Result JSON Schema**

```json
{
  "duration_minutes": 30,
  "total_requests": 7200,
  "wall_start": "...",
  "wall_end": "...",
  "successes": 7200,
  "failures": 0,
  "checkpoints": [
    {
      "minute": 1,
      "requests_this_window": 240,
      "ttft_p50": 0.22,
      "ttft_p99": 0.30,
      "total_p50": 2.3,
      "total_p99": 3.0,
      "system_tok_s": 440.0,
      "gpu_memory_used_mb": 14200,
      "gpu_utilization_pct": 85
    }
  ],
  "degradation_check": {
    "first_5min_ttft_p99": 0.30,
    "last_5min_ttft_p99": 0.31,
    "drift_pct": 3.3,
    "memory_leak_detected": false
  }
}
```

**Notebook Cell Plan**

- **Chart type**: Time-series line chart. X = minute. Y = P99 latency. Second line: GPU memory usage. Horizontal dashed line at the first-5-minute P99 baseline.
- **Insight**: If the lines are flat, vLLM's memory management (PagedAttention) is working correctly with no leaks. Any upward drift indicates a problem. Production systems should soak-test before deployment.

**Dependencies**: Same standard server config. No restart. Long-running test (30 min).

---

### Test 27 (old-32): Graceful Degradation Under Overload

**Validated Scenario**

Tests what happens when request rate exceeds server capacity. Does vLLM queue gracefully? Return 429s? OOM? This test ramps load from comfortable to extreme and monitors failure modes.

**T4 Limitations**: None.

**vLLM Docker Run Command**

Same standard config. No restart needed.

**Test Script Approach**

1. Ramp load in stages:
   - Stage 1 (0-60s): 4 RPS (comfortable)
   - Stage 2 (60-120s): 10 RPS (moderate)
   - Stage 3 (120-180s): 20 RPS (heavy)
   - Stage 4 (180-240s): 40 RPS (overload)
   - Stage 5 (240-300s): 4 RPS (recovery)
2. Track per-stage: success rate, error types (timeout, 429, 500, connection refused), latency percentiles, queue depth
3. Key question: does the system recover in Stage 5 or is it "stuck" after overload?
4. 1 trial

**Result JSON Schema**

```json
{
  "stages": [
    {
      "stage": 1,
      "rps": 4,
      "duration_sec": 60,
      "wall_start": "...",
      "total_requests": 240,
      "successes": 240,
      "failures": 0,
      "error_types": {},
      "ttft_p50": 0.22,
      "ttft_p99": 0.30,
      "total_p50": 2.3,
      "total_p99": 3.0,
      "system_tok_s": 440.0
    },
    {
      "stage": 4,
      "rps": 40,
      "duration_sec": 60,
      "wall_start": "...",
      "total_requests": 2400,
      "successes": 2100,
      "failures": 300,
      "error_types": {"timeout": 280, "429": 20},
      "ttft_p50": 5.0,
      "ttft_p99": 30.0,
      "total_p50": 15.0,
      "total_p99": 55.0,
      "system_tok_s": 500.0
    },
    {
      "stage": 5,
      "rps": 4,
      "duration_sec": 60,
      "wall_start": "...",
      "total_requests": 240,
      "successes": 240,
      "failures": 0,
      "error_types": {},
      "ttft_p50": 0.23,
      "ttft_p99": 0.32,
      "total_p50": 2.4,
      "total_p99": 3.1,
      "system_tok_s": 435.0,
      "recovery_note": "Latency returned to baseline within 15 seconds"
    }
  ]
}
```

**Notebook Cell Plan**

- **Chart type**: Multi-panel time-series. Panel 1: RPS over time (step function). Panel 2: P50 and P99 latency. Panel 3: Success rate (%). Panel 4: Queue depth.
- **Insight**: A well-designed system degrades gracefully (increased latency, queuing) rather than catastrophically (OOM, crash). The recovery phase is equally important -- does the system return to normal when load drops? Analogy: ASG target tracking under traffic spikes.

**Dependencies**: Same standard server config. No restart. ~5 min test.

---

### Test 28 (old-37): Full Production Simulation

**Validated Scenario**

Combines all learnings into a realistic production traffic simulation: mixed prompt lengths, multi-turn conversations, varying request rates (with burstiness), and structured output requests. This is the "final exam" for the T4 inference setup.

**T4 Limitations**: None.

**vLLM Docker Run Command**

Standard config with guided decoding support:

```bash
docker run -d --name vllm-prod --gpus all -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin --dtype half \
    --gpu-memory-utilization 0.9 --max-model-len 2048 \
    --guided-decoding-backend outlines
```

**Test Script Approach**

1. Traffic mix (realistic distribution):
   - 40% short queries (20-50 input, 50-100 output tokens)
   - 30% medium queries (100-200 input, 100-300 output tokens)
   - 15% long queries (300-500 input, 200-500 output tokens)
   - 10% JSON output requests (guided decoding)
   - 5% multi-turn (3-4 turns per conversation)
2. Request rate: Poisson process with mean 6 RPS, with 2 burst periods (15 RPS for 30 seconds)
3. Duration: 10 minutes
4. Track all metrics: per-request type breakdown, overall throughput, tail latencies, failure rate
5. 1 trial

**Result JSON Schema**

```json
{
  "duration_minutes": 10,
  "wall_start": "...",
  "wall_end": "...",
  "total_requests": 3600,
  "successes": 3580,
  "failures": 20,
  "overall": {
    "ttft_p50": 0.30,
    "ttft_p99": 1.20,
    "ttft_max": 3.50,
    "total_p50": 3.0,
    "total_p99": 12.0,
    "total_max": 25.0,
    "system_tok_s_avg": 520.0,
    "failure_rate_pct": 0.56
  },
  "by_type": {
    "short": {"count": 1440, "ttft_p50": 0.22, "ttft_p99": 0.40, "total_p50": 2.0, "total_p99": 3.5},
    "medium": {"count": 1080, "ttft_p50": 0.28, "ttft_p99": 0.60, "total_p50": 4.0, "total_p99": 8.0},
    "long": {"count": 540, "ttft_p50": 0.40, "ttft_p99": 1.50, "total_p50": 8.0, "total_p99": 18.0},
    "json": {"count": 360, "ttft_p50": 0.35, "ttft_p99": 0.80, "total_p50": 5.0, "total_p99": 10.0, "schema_valid_pct": 100.0},
    "multi_turn": {"count": 180, "ttft_p50": 0.20, "ttft_p99": 0.50, "total_p50": 3.5, "total_p99": 7.0}
  },
  "burst_periods": [
    {"start_sec": 180, "end_sec": 210, "rps": 15, "failures": 5, "ttft_p99": 2.5},
    {"start_sec": 420, "end_sec": 450, "rps": 15, "failures": 10, "ttft_p99": 3.5}
  ]
}
```

**Notebook Cell Plan**

- **Chart type**: Dashboard with 4 panels:
  1. Time-series: throughput (tok/s) and RPS over time, with burst periods highlighted
  2. CDF plot: latency distribution by request type (5 overlapping CDF curves)
  3. Heatmap: latency by time bucket (1-min windows) and request type
  4. Summary table: SLA compliance (% of requests under 1s TTFT, under 5s total)
- **Insight**: This is the production readiness assessment. Can we define SLAs (e.g., P99 TTFT < 1s, P99 total < 10s) and meet them? What percentage of requests violate the SLA during bursts?

**Dependencies**: Same standard config. No restart.

---

## Part 3: Dependency Graph -- Server Configurations

### Configuration Groups (tests that can share a server instance)

**Config A: Standard AWQ Marlin (default)** -- No restart between these tests:
- Tests 1-5 (Group A: Fundamentals)
- Tests 6-7 (Group B: Quantization -- AWQ/Marlin runs)
- Test 9a (Prefix caching ON -- default)
- Test 10 (KV cache cliff, prefix ON)
- Test 12c (Context window 2048 -- default)
- Tests 14-15 (Prefill vs decode)
- Test 16a (Chunked prefill ON -- default)
- Test 17b (max_num_batched_tokens=2048 -- default)
- Test 18a (Speculative baseline -- no spec)
- Test 20 (Guided output)
- Test 21 (Temperature/sampling)
- Test 24 (Multi-turn conversation)
- Tests 25-28 (Load testing & production)

**Config B: Standard AWQ Marlin, prefix caching OFF** -- 1 restart:
- Test 9b (Prefix caching OFF)

**Config C: FP8 KV cache** -- 1 restart:
- Test 8b (FP8 KV cache)

**Config D: Context window variants** -- 3 restarts:
- Test 12a (max_model_len=512)
- Test 12b (max_model_len=1024)
- Test 12d (max_model_len=4096)

**Config E: GPTQ Int8, no prefix cache, V0 engine** -- 2 restarts:
- Test 11 (Preemption with GPTQ Int8 -- already done)
- Test 13a (Preemption mode: recompute)
- Test 13b (Preemption mode: swap)

**Config F: Chunked prefill OFF** -- 1 restart (possibly V0 engine):
- Test 16b (Chunked prefill OFF)

**Config G: max_num_batched_tokens variants** -- 2 restarts:
- Test 17a (max_num_batched_tokens=512)
- Test 17c (max_num_batched_tokens=4096)

**Config H: Speculative decoding** -- 6 restarts total:
- Test 18b (Speculative, num_spec=5)
- Test 19a-e (Speculative, num_spec=1,2,3,5,7)

**Config I: LoRA serving** -- 1 restart:
- Test 22 (LoRA multi-tenant)

**Config J: Cold start** -- Multiple container lifecycle cycles:
- Test 23 (Cold start timing -- uses --rm, not -d)

---

## Part 4: Optimal Execution Order (Minimizing Server Restarts)

The following order minimizes server restarts. Total restarts: approximately 18-20 (some are unavoidable due to mutually exclusive configurations).

**Phase 1: Standard Config A** (0 restarts -- longest run, most tests)
```
START server: Config A (standard AWQ Marlin, default settings)
  Run Test 2 (vLLM batching -- already done, but verify)
  Run Test 3 (Max concurrency -- already done)
  Run Test 4 (Variable prompt -- already done)
  Run Test 5 (Batch size vs throughput curve) -- NEW
  Run Test 6 (Quantization speed -- AWQ/Marlin portion -- already done)
  Run Test 10 (KV cache cliff, prefix ON -- already done)
  Run Test 14 (Prefill vs decode -- already done)
  Run Test 15 (Input vs output cost -- already done)
  Run Test 21 (Temperature/sampling effects) -- NEW
  Run Test 9a (Prefix caching ON measurement) -- NEW
  Run Test 20 (Guided/structured output) -- NEW
  Run Test 24 (Multi-turn conversation) -- NEW
  Run Test 25 (Tail latency under sustained load) -- NEW
  Run Test 26 (Soak test, 30 min) -- NEW
  Run Test 27 (Graceful degradation) -- NEW
  Run Test 28 (Full production simulation) -- NEW
```

**Phase 2: Prefix caching OFF** (1 restart)
```
RESTART server: Config B (--no-prefix-caching)
  Run Test 9b (Prefix caching OFF measurement) -- NEW
```

**Phase 3: FP8 KV cache** (1 restart)
```
RESTART server: Config C (--kv-cache-dtype fp8)
  Run Test 8 (FP8 KV cache vs FP16) -- NEW
```

**Phase 4: Context window scaling** (3 restarts)
```
RESTART: Config D-512 (--max-model-len 512)
  Run Test 12a

RESTART: Config D-1024 (--max-model-len 1024)
  Run Test 12b

RESTART: Config D-4096 (--max-model-len 4096)
  Run Test 12d
```

**Phase 5: Chunked prefill + batched tokens tuning** (3 restarts)
```
RESTART: Config F (chunked prefill OFF, possibly V0 engine)
  Run Test 16b (Chunked prefill OFF)

RESTART: Config G-512 (--max-num-batched-tokens 512)
  Run Test 17a

RESTART: Config G-4096 (--max-num-batched-tokens 4096)
  Run Test 17c
```

**Phase 6: Return to standard for chunked-on and batched-tokens-default** (1 restart)
```
RESTART: Config A (standard -- for Test 16a and 17b)
  Run Test 16a (Chunked prefill ON background TBT measurement)
  Run Test 17b (max_num_batched_tokens=2048 baseline)
```

**Phase 7: Preemption policies** (2 restarts)
```
RESTART: Config E-recompute (GPTQ Int8, V0 engine, --preemption-mode recompute)
  Run Test 13a

RESTART: Config E-swap (GPTQ Int8, V0 engine, --preemption-mode swap)
  Run Test 13b
```

**Phase 8: Speculative decoding** (6 restarts)
```
RESTART: Config H-spec5 (--speculative-model Qwen2.5-0.5B, --num-speculative-tokens 5)
  Run Test 18b (Speculative decoding speedup)

RESTART: Config H-spec1 (--num-speculative-tokens 1)
  Run Test 19a

RESTART: Config H-spec2 (--num-speculative-tokens 2)
  Run Test 19b

RESTART: Config H-spec3 (--num-speculative-tokens 3)
  Run Test 19c

RESTART: Config H-spec7 (--num-speculative-tokens 7)
  Run Test 19d
```

**Phase 9: LoRA serving** (1 restart)
```
RESTART: Config I (--enable-lora --max-loras 4)
  Run Test 22 (LoRA multi-tenant)
```

**Phase 10: Cold start benchmarking** (3 full container lifecycles)
```
STOP all containers
  Run Test 23a (Cold start from local EBS) -- 3 trials
  Run Test 23b (Cold start from S3 mount) -- 3 trials
  Run Test 23c (Cold start fresh download) -- 3 trials
```

**Total estimated restarts: ~22**
**Total estimated execution time: ~3-4 hours** (dominated by soak test at 30 min, cold start tests at ~15 min total, and speculative decoding sweep at ~20 min total)

---

### Critical Files for Implementation

- `/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_cliff_test.py` - Pattern for concurrent load testing with server log capture and SSH integration. Best template for Tests 5, 8, 12, 25-27.
- `/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_exp2_prefill_vs_decode.py` - Pattern for single-request precision measurement with filler prompt generation. Best template for Tests 16, 17, 18, 19, 21.
- `/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_exp1_preemption.py` - Pattern for unique prompts, preemption detection, V0 engine testing. Best template for Tests 13, 26.
- `/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_results.json` - Canonical JSON schema pattern with metadata, per-request fields (wall_start, successes, failures, p50/p99/max). All 18 new result files should follow this structure.
- `/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/CLAUDE.md` - Project context, explanation style guidelines, Auto Scaling analogies to maintain throughout all new experiment documentation.