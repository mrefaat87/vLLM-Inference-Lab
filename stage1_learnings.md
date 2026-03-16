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

## What's Next
- Load test: send concurrent requests, observe queuing behavior
- Measure TTFT and TBT under load
- Compare sequential vs concurrent latency to prove Ollama's single-request model
