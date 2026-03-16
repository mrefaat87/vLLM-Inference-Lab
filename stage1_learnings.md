# Stage 1: Ollama — Local Model Serving & Concurrent Load

## Setup
- **Machine:** Apple M4, 10 cores (4P + 6E), 16GB unified memory
- **Model:** Qwen2.5:7B (4.7GB on disk, Q4 quantized)
- **Server:** Ollama v0.18.0, serving on localhost:11434
- **Tool used:** [llmfit](https://github.com/AlexsJones/llmfit) to validate model fits hardware (50% memory, "Perfect" fit)

## Key Concepts Learned

### Inference Request Lifecycle
Every inference request goes through two phases:
1. **Prefill** — process the entire input prompt, build the KV cache. This is compute-bound (matrix multiplications across all input tokens in parallel).
2. **Decode** — generate output tokens one at a time, each depending on the previous. This is memory-bandwidth-bound (reading KV cache + model weights for each token).

**Auto Scaling analogy:** Prefill is like instance launch (heavy upfront work). Decode is like steady-state request processing (ongoing, sequential).

### Cold Start vs Warm Requests
First request to a model incurs a **load_duration** — time to load model weights from disk into GPU memory. Subsequent requests skip this.

- Cold start observed: **52.5 seconds** (loading 4.7GB into Metal GPU memory)
- This disappears on subsequent requests while the model stays loaded

**Auto Scaling analogy:** Identical to EC2 cold start. First request to a new instance pays the AMI boot + app startup cost. Warm pool eliminates this.

### Ollama API Response Metadata
Ollama returns timing breakdown automatically with every response (values in nanoseconds):
- `load_duration` — model weight loading time (cold start penalty)
- `prompt_eval_duration` — prefill phase time
- `prompt_eval_count` — number of input tokens processed
- `eval_duration` — decode phase time
- `eval_count` — number of output tokens generated
- `total_duration` — end-to-end

### Chat Templates
The model has no built-in system prompt. Ollama wraps your prompt in a **chat template** — formatting tokens the model was trained to expect (e.g., `<|im_start|>`, role labels). A 6-word prompt becomes 35 tokens after template wrapping. This is an Ollama-layer behavior, not a model-layer one. You can override or remove the default template.

### TTFT and TBT Measurement
- **TTFT (Time to First Token)** — requires `stream: true`. Time from request sent to first token received. Captures the full prefill phase.
- **TBT (Time Between Tokens)** — interval between consecutive streamed tokens. Measures decode throughput.
- With `stream: false`, you only get aggregate timing — can't distinguish user-perceived wait from generation speed.

**Auto Scaling analogy:** TTFT = time-to-first-byte on HTTP response. TBT = streaming transfer rate after headers.

### Ollama Does Not Batch
Ollama processes requests **one at a time, sequentially**. Concurrent requests queue behind the active request.

- No parallel prefill across requests
- No continuous batching
- Single-threaded request processing model

**Auto Scaling analogy:** Single-worker Flask app behind an ALB. The ALB accepts all connections, but the backend processes them one by one.

### KV Cache and Context Window
- Qwen2.5:7B supports up to **128K context** tokens
- Ollama defaults to **2048 tokens** context window
- Each request builds its own KV cache (pre-computed attention state for all previous tokens)
- Ollama sidesteps memory management complexity by only serving one request at a time

**Auto Scaling analogy:** KV cache = warm instance pool. Pre-computed state you don't want to recompute on every request.

### Decode Is Memory-Bandwidth Bound
To generate each output token, the model must:
1. Read the **entire model weights** (~4.7GB) from memory
2. Read the **entire KV cache** (grows with each token generated)
3. Do a relatively small matrix-vector multiply (trivial compute)
4. Write the new KV entry back to memory

The GPU cores sit idle waiting for data to arrive. Throughput is governed by `model_size / memory_bandwidth`, not FLOPS.

- M4 base memory bandwidth: ~100 GB/s
- Theoretical max: ~21 tok/s
- Observed with Ollama (llama.cpp): **0.3-0.4 tok/s** — massive gap due to runtime inefficiency

**Auto Scaling analogy:** Like an application where every request reads a 5GB config from EBS before doing 1ms of CPU work. It's I/O-bound, not CPU-bound. Faster CPUs won't help; faster storage will.

**Key interview insight:** Inference hardware is specced by memory bandwidth (H100 HBM3 = 3.35 TB/s), not just FLOPS. This is why. And it's why batching helps — read the weights once, apply to N requests' tokens simultaneously, amortizing the memory read cost.

### Ollama vs MLX vs vLLM — Different Layers of the Stack
These three are **not alternatives** — they operate at different layers:

| Layer | Analogy | MLX | Ollama | vLLM |
|-------|---------|-----|--------|------|
| **Compute framework** | Nitro hypervisor | ✅ | ❌ (uses llama.cpp) | ❌ (uses CUDA/PyTorch) |
| **Inference engine** | Application runtime | Can be | llama.cpp | ✅ (batching, PagedAttention) |
| **Serving layer** | Web server + API | ❌ | ✅ | ✅ |

- **MLX** — Apple's native ML framework for Apple Silicon. Like AWS Nitro: purpose-built for the hardware, eliminates overhead. Uses unified memory natively (no CPU↔GPU copies). Would give ~15-20 tok/s on M4 vs Ollama's 0.3 tok/s.
- **Ollama** — Convenience wrapper (model downloads, chat templates, API). Like Elastic Beanstalk: easy to deploy, but you don't control the runtime. Uses llama.cpp under the hood, which was designed for CUDA and bolted on Metal support later.
- **vLLM** — Production inference engine. Solves the problems Stage 1 exposed: sequential processing → continuous batching, memory fragmentation → PagedAttention, no scheduling → preemptive request scheduling. Built for NVIDIA GPUs (CUDA). Like custom fleet management with EC2 Auto Scaling + hand-tuned ALB.

### Why Ollama Is Slow on Apple Silicon
Ollama uses **llama.cpp**, which was designed for CUDA's split CPU/GPU memory model. Its Metal backend is a bolt-on, not native. This architectural mismatch means it doesn't saturate the M4's memory bandwidth — explaining the 0.3 tok/s vs the ~20 tok/s an MLX-native runtime achieves on the same hardware.

## Load Test Results

### Test: 3 Concurrent Requests
```
Req    Start  1st Token      End     TTFT   Avg TBT    Total   Tok/s  Tokens
  0    0.000     14.568   97.947   14.568    3.2162   97.947     0.3      26
  1    0.004     99.711  176.355   99.707    2.7607  176.351     0.4      27
  2    0.005    177.547  277.795  177.542    3.6158  277.790     0.3      28
```

**Key observations:**
- All 3 requests sent at t=0 (within 5ms)
- Gap between consecutive first-tokens: **85s, 78s** — matches single request duration
- This proves **sequential queuing**: each request waits for the previous to fully complete
- Request 2's TTFT of 177s is pure queuing delay, not model slowness
- Tokens/sec consistent across requests (~0.3-0.4) — no degradation, just waiting

**Auto Scaling analogy:** Single-instance target group with no ASG. The "latency" problem isn't processing speed — it's queue depth. Exactly the problem horizontal scaling (or batching) solves.

## What's Next — Stage 2
- Move to vLLM on a cloud GPU (NVIDIA) to observe continuous batching behavior
- Same model (Qwen2.5:7B), same load test, different engine
- Expect to see: parallel request processing, dramatically lower TTFT spread, higher throughput
