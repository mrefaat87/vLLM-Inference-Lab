# worker.py — Queue-based inference worker with pluggable admission control
#              and graceful drain on shutdown.
#
# PHASE 4 CHANGES (from Phase 3):
#   1. Graceful drain: track active asyncio.Tasks, await them on SIGTERM
#   2. DRAIN_TIMEOUT env var: configurable drain window (must be < terminationGracePeriodSeconds)
#   3. Drain stats logging: DRAIN_STATS line for test harness verification
#
# The admission strategies (static, threshold, per_request) are UNCHANGED from
# Phase 3. The only difference is shutdown behavior.
#
# SHUTDOWN SEQUENCE (Phase 4):
#   1. SIGTERM received (K8s sends this on pod eviction or scale-down)
#   2. shutdown_event is set — consume loop stops pulling NEW messages
#   3. Current message is NACK'd + requeued (another worker will pick it up)
#   4. Wait for all in-flight asyncio.Tasks to complete (up to DRAIN_TIMEOUT)
#   5. If timeout: cancel remaining tasks (pod is about to be force-killed anyway)
#   6. Close connections (RabbitMQ, Redis, HTTP client)
#   7. Log DRAIN_STATS for test verification
#
# PHASE 3 BUG FIXED: asyncio.create_task() was fire-and-forget. On SIGTERM,
# the loop broke but in-flight tasks were abandoned — clients saw broken SSE
# streams mid-response. Now tasks are tracked and awaited.
#
# ANALOGY: ALB connection draining. SIGTERM = deregister from target group.
# DRAIN_TIMEOUT = deregistration delay. In-flight tasks = active connections.

import os
import json
import time
import signal
import asyncio
import logging
import re
from typing import Optional

import aio_pika
import redis.asyncio as aioredis
import httpx

# ---------------------------------------------------------------------------
# Configuration from env vars — K8s ConfigMap injects these via envFrom
# ---------------------------------------------------------------------------
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")

# WHY prefetch_count matters differently per strategy:
#   Strategy 1 (static): THIS IS the concurrency limiter. N=10 means at most 10 in flight.
#   Strategy 2+3: prefetch_count is a soft upper bound; admission control is the real gate.
#   We still set it to prevent unbounded message buffering in the consumer.
PREFETCH_COUNT = int(os.getenv("PREFETCH_COUNT", "10"))

# Admission control thresholds (used by strategies 2 and 3)
GPU_CACHE_THRESHOLD = float(os.getenv("GPU_CACHE_THRESHOLD", "0.80"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))

# WHY "threshold" as default: preserves backward compatibility with Phase 1/2 worker
ADMISSION_STRATEGY = os.getenv("ADMISSION_STRATEGY", "threshold")

# Per-request admission control constants (Strategy 3)
# WHY 401KB per token: each KV cache entry stores key+value vectors for all
# attention heads across all layers. For Qwen2.5-7B-AWQ:
#   layers=28, heads=28, head_dim=128, kv_dtype=float16
#   per_token = 28 x 28 x 128 x 2bytes x 2(k+v) = 401,408 bytes
KV_BYTES_PER_TOKEN = int(os.getenv("KV_BYTES_PER_TOKEN", "401408"))
# WHY 4 chars per token: rough tokenizer approximation for English text.
CHARS_PER_TOKEN = int(os.getenv("CHARS_PER_TOKEN", "4"))
# Total KV cache capacity in bytes — T4 16GB x 0.85 util - 4GB model = 9.6GB
TOTAL_KV_CACHE_BYTES = int(os.getenv("TOTAL_KV_CACHE_BYTES", str(9_600_000_000)))

# Phase 4: Graceful drain timeout in seconds. Must be < terminationGracePeriodSeconds
# (180s) to leave buffer for connection cleanup before Kubernetes force-kills the pod.
DRAIN_TIMEOUT = int(os.getenv("DRAIN_TIMEOUT", "170"))

# Phase 4v3: Readiness file path — the queue-worker's readiness probe checks for this
# file. Created after warmup completes, deleted on shutdown. Lives in the container's
# ephemeral filesystem so it's automatically cleaned up on container restart.
# WHY file-based (not HTTP): the worker is a queue consumer, not a web server.
# A file probe is zero-overhead — no socket, no port, no dependencies.
READINESS_FILE = "/tmp/worker-ready"

# Phase 4v2: Model warmup — send a synthetic request before consuming real traffic.
# WHY: first inference pays ~20-30s extra TTFT for CUDA kernel compilation. By warming
# up BEFORE consuming from the queue, real user requests never see this penalty.
# Analogy: ALB slow start mode — new instances ramp up before receiving full traffic.
WARMUP_ENABLED = os.getenv("WARMUP_ENABLED", "true").lower() == "true"

# Phase 4v3: vLLM Sleep Mode — free GPU memory during idle periods, wake on demand.
# WHY: after SLEEP_IDLE_TIMEOUT seconds of no traffic, the worker calls vLLM's /sleep
# endpoint. vLLM offloads model weights from GPU VRAM to CPU RAM (L1) or discards them
# (L2), freeing the GPU. When a new message arrives, the worker calls /wake_up before
# processing. Wake-up takes 0.3-0.9s (L1) vs ~295s full cold start.
# On dedicated EC2 instances this is a LATENCY optimization (billing is per-instance),
# but it eliminates cold start pain when traffic resumes after idle gaps.
# Analogy: EC2 hibernate — same instance, same warm caches, just paused.
SLEEP_ENABLED = os.getenv("SLEEP_ENABLED", "true").lower() == "true"
SLEEP_IDLE_TIMEOUT = int(os.getenv("SLEEP_IDLE_TIMEOUT", "300"))  # 5 min default
SLEEP_LEVEL = int(os.getenv("SLEEP_LEVEL", "1"))  # L1=CPU offload, L2=discard

logger = logging.getLogger("worker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# Global shutdown flag — set by signal handlers for graceful termination
shutdown_event = asyncio.Event()

# NACK counter — tracks how many times this worker rejected+requeued a message.
nack_count = 0
admit_count = 0

# Phase 4: Track active tasks for graceful drain.
# Using a set because tasks are added/removed frequently and we need O(1) add/discard.
active_tasks: set[asyncio.Task] = set()


# ---------------------------------------------------------------------------
# vLLM metrics parsing
# ---------------------------------------------------------------------------
async def get_vllm_metrics(http_client: httpx.AsyncClient) -> dict:
    """Fetch and parse Prometheus metrics from vLLM's /metrics endpoint.

    Returns:
        Dict with 'gpu_cache_usage' (0.0-1.0) and 'running_requests' (int).
        Returns safe defaults on error so we don't block work on metrics failure.
    """
    try:
        response = await http_client.get(
            f"http://{VLLM_HOST}:{VLLM_PORT}/metrics",
            timeout=2.0,
        )
        text = response.text

        cache_match = re.search(r"vllm:gpu_cache_usage_perc\s+([\d.]+)", text)
        gpu_cache_usage = float(cache_match.group(1)) if cache_match else 0.0

        running_match = re.search(r"vllm:num_requests_running\s+([\d.]+)", text)
        running_requests = int(float(running_match.group(1))) if running_match else 0

        return {
            "gpu_cache_usage": gpu_cache_usage,
            "running_requests": running_requests,
        }

    except Exception as e:
        logger.warning("Failed to fetch vLLM metrics: %s — using safe defaults", e)
        return {
            "gpu_cache_usage": 0.0,
            "running_requests": 0,
        }


# ---------------------------------------------------------------------------
# Admission strategies — each returns (admitted: bool, reason: str)
# ---------------------------------------------------------------------------

async def admit_static(
    http_client: httpx.AsyncClient, prompt: str, max_tokens: int
) -> tuple[bool, str]:
    """Strategy 1: Static Max Concurrent — always admit.

    The concurrency limit is enforced by RabbitMQ's prefetch_count alone.
    If we got the message, we have a slot. No runtime checks needed.

    WHY this works: prefetch_count=N means RabbitMQ delivers at most N unacked
    messages. The worker processes them concurrently via asyncio.create_task().
    When one finishes (ACK), RabbitMQ delivers the next. Natural flow control
    without any /metrics polling overhead.

    Limitation: all requests are treated equal regardless of size. A 20-token
    request and a 2048-token request both consume one slot.
    """
    return True, ""


async def admit_threshold(
    http_client: httpx.AsyncClient, prompt: str, max_tokens: int
) -> tuple[bool, str]:
    """Strategy 2: Utilization Thresholds — check GPU state before admitting.

    Polls vLLM /metrics and admits if:
      - gpu_cache_usage < GPU_CACHE_THRESHOLD (default 80%)
      - running_requests < MAX_BATCH_SIZE (default 8)

    WHY this is better than static: adapts to runtime conditions. If a previous
    large request consumed lots of KV cache, this strategy sees it and backs off.

    Limitation: point-in-time check. A request admitted at 65% KV usage could
    push actual usage to 90% if it has a long prompt.
    """
    metrics = await get_vllm_metrics(http_client)

    if metrics["gpu_cache_usage"] > GPU_CACHE_THRESHOLD:
        return False, (
            f"GPU cache usage {metrics['gpu_cache_usage']:.1%} exceeds "
            f"threshold {GPU_CACHE_THRESHOLD:.1%}"
        )

    if metrics["running_requests"] >= MAX_BATCH_SIZE:
        return False, (
            f"Running requests {metrics['running_requests']} >= "
            f"max batch size {MAX_BATCH_SIZE}"
        )

    return True, ""


async def admit_per_request(
    http_client: httpx.AsyncClient, prompt: str, max_tokens: int
) -> tuple[bool, str]:
    """Strategy 3: Per-Request Admission Control — estimate cost, project state.

    Steps:
      1. Estimate input tokens from prompt character length
      2. Calculate KV cache cost: total_tokens x KV_BYTES_PER_TOKEN
      3. Fetch current GPU state from /metrics
      4. Project: would admitting this request exceed the KV threshold?
      5. Admit only if projected KV < threshold AND running < max_batch

    WHY this is better than threshold: a 20-token request at 70% KV usage is
    safe to admit, but a 2048-token request at 70% is not. This strategy knows
    the difference.
    """
    metrics = await get_vllm_metrics(http_client)

    if metrics["running_requests"] >= MAX_BATCH_SIZE:
        return False, (
            f"Running requests {metrics['running_requests']} >= "
            f"max batch size {MAX_BATCH_SIZE}"
        )

    estimated_input_tokens = len(prompt) / CHARS_PER_TOKEN
    estimated_total_tokens = estimated_input_tokens + max_tokens
    estimated_kv_bytes = estimated_total_tokens * KV_BYTES_PER_TOKEN

    current_kv_bytes = metrics["gpu_cache_usage"] * TOTAL_KV_CACHE_BYTES
    projected_kv_bytes = current_kv_bytes + estimated_kv_bytes
    projected_kv_fraction = projected_kv_bytes / TOTAL_KV_CACHE_BYTES

    if projected_kv_fraction > GPU_CACHE_THRESHOLD:
        return False, (
            f"Projected KV usage {projected_kv_fraction:.1%} would exceed "
            f"threshold {GPU_CACHE_THRESHOLD:.1%} "
            f"(current={metrics['gpu_cache_usage']:.1%}, "
            f"request_cost={estimated_kv_bytes / 1_000_000:.1f}MB, "
            f"est_tokens={estimated_total_tokens:.0f})"
        )

    return True, ""


# Strategy dispatch — maps env var value to admission function
STRATEGIES = {
    "static": admit_static,
    "threshold": admit_threshold,
    "per_request": admit_per_request,
}


# ---------------------------------------------------------------------------
# Job processing — the core inference loop
# ---------------------------------------------------------------------------
async def process_job(
    message: aio_pika.abc.AbstractIncomingMessage,
    redis_client: aioredis.Redis,
    http_client: httpx.AsyncClient,
    admit_fn,
) -> None:
    """Process a single inference job from the queue.

    Steps:
    1. Parse and validate the job JSON
    2. Run the pluggable admission check
    3. Stream completions from vLLM
    4. Publish each token to Redis pub/sub
    5. ACK or NACK the RabbitMQ message
    """
    global nack_count, admit_count
    dequeue_time = time.time()

    try:
        job = json.loads(message.body.decode())
        job_id = job["job_id"]
        prompt = job["prompt"]
        max_tokens = job.get("max_tokens", 256)
        temperature = job.get("temperature", 0.7)
        enqueue_time = job.get("enqueue_time", dequeue_time)

        queue_wait_ms = (dequeue_time - enqueue_time) * 1000
        logger.info(
            "Processing job %s (queue_wait: %.0fms, prompt_len: %d, max_tokens: %d, strategy: %s)",
            job_id, queue_wait_ms, len(prompt), max_tokens, ADMISSION_STRATEGY,
        )

        channel = f"job:{job_id}"

    except (json.JSONDecodeError, KeyError) as e:
        logger.error("Invalid job message: %s — dropping", e)
        await message.nack(requeue=False)
        return

    # --- Admission control (strategy-dependent) ---
    admitted, reason = await admit_fn(http_client, prompt, max_tokens)
    if not admitted:
        nack_count += 1
        logger.info(
            "Job %s NACK'd [%s] (#%d): %s — requeuing",
            job_id, ADMISSION_STRATEGY, nack_count, reason,
        )
        await message.nack(requeue=True)
        # WHY sleep 100ms: prevents tight requeue loops. Gives GPU time to free cache.
        await asyncio.sleep(0.1)
        return

    admit_count += 1
    logger.info(
        "Job %s ADMITTED [%s] (admit #%d, nack #%d)",
        job_id, ADMISSION_STRATEGY, admit_count, nack_count,
    )

    # --- Stream inference from vLLM ---
    tokens_generated = 0
    ttft_ms: Optional[float] = None
    inference_start = time.time()

    try:
        async with http_client.stream(
            "POST",
            f"http://{VLLM_HOST}:{VLLM_PORT}/v1/completions",
            json={
                "model": VLLM_MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
            timeout=300.0,
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                raise RuntimeError(
                    f"vLLM returned {response.status_code}: {error_body.decode()}"
                )

            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)

                    for line in event.strip().split("\n"):
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]

                        if data_str.strip() == "[DONE]":
                            continue

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.debug("Skipping non-JSON SSE data: %s", data_str[:100])
                            continue

                        choices = data.get("choices", [])
                        if not choices:
                            continue

                        token_text = choices[0].get("text", "")
                        if not token_text:
                            continue

                        if ttft_ms is None:
                            ttft_ms = (time.time() - inference_start) * 1000
                            logger.info("Job %s TTFT: %.0fms", job_id, ttft_ms)

                        tokens_generated += 1

                        await redis_client.publish(
                            channel,
                            json.dumps({"token": token_text}),
                        )

        # --- Completion ---
        total_ms = (time.time() - inference_start) * 1000
        completion_msg = {
            "status": "complete",
            "metrics": {
                "queue_wait_ms": round(queue_wait_ms, 1),
                "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
                "total_ms": round(total_ms, 1),
                "tokens_generated": tokens_generated,
                "tokens_per_sec": round(
                    tokens_generated / (total_ms / 1000), 1
                ) if total_ms > 0 else 0,
                "strategy": ADMISSION_STRATEGY,
                "worker_nack_count": nack_count,
                "worker_admit_count": admit_count,
            },
        }
        await redis_client.publish(channel, json.dumps(completion_msg))
        await message.ack()

        logger.info(
            "Job %s complete [%s]: %d tokens in %.0fms (%.1f tok/s, TTFT: %.0fms, wait: %.0fms)",
            job_id, ADMISSION_STRATEGY, tokens_generated, total_ms,
            completion_msg["metrics"]["tokens_per_sec"],
            ttft_ms or 0, queue_wait_ms,
        )

    except Exception as e:
        logger.error("Job %s failed: %s", job_id, e, exc_info=True)

        error_msg = {"status": "error", "error": str(e)}
        try:
            await redis_client.publish(channel, json.dumps(error_msg))
        except Exception as pub_err:
            logger.error("Failed to publish error for job %s: %s", job_id, pub_err)

        await message.nack(requeue=False)


# ---------------------------------------------------------------------------
# vLLM Sleep Mode — free GPU during idle, wake on demand
# ---------------------------------------------------------------------------

# Track whether vLLM is currently sleeping so we know when to wake it
_vllm_sleeping = False
# Track last activity time for idle detection
_last_activity_time = time.time()


async def sleep_vllm(http_client: httpx.AsyncClient) -> None:
    """Put vLLM to sleep — offload weights from GPU VRAM to CPU RAM.

    WHY: frees GPU memory during idle periods. On dedicated EC2 instances this
    doesn't save money (billing is per-instance), but it means wake-up is 0.3-0.9s
    instead of ~295s if we had terminated and cold-started.

    Analogy: EC2 Stop (hibernate) — the instance is paused, EBS volume retains state,
    start resumes from where you left off. vLLM sleep keeps CUDA graphs and JIT cache
    in CPU RAM, so wake doesn't need to recompile anything.
    """
    global _vllm_sleeping
    if _vllm_sleeping:
        return

    try:
        response = await http_client.post(
            f"http://{VLLM_HOST}:{VLLM_PORT}/sleep",
            json={"level": SLEEP_LEVEL},
            timeout=30.0,
        )
        if response.status_code == 200:
            _vllm_sleeping = True
            logger.info(
                "SLEEP: vLLM entered L%d sleep — GPU VRAM freed, CUDA context preserved",
                SLEEP_LEVEL,
            )
        else:
            logger.warning("SLEEP: vLLM returned %d — may not support sleep mode", response.status_code)
    except Exception as e:
        logger.warning("SLEEP: Failed to put vLLM to sleep: %s", e)


async def wake_vllm(http_client: httpx.AsyncClient) -> None:
    """Wake vLLM from sleep — reload weights to GPU VRAM.

    WHY: must be called before processing any inference request. The wake call
    copies model weights from CPU RAM back to GPU (L1: ~0.3-0.9s) or reloads
    from disk (L2: ~0.9-2.6s). CUDA graphs and JIT kernels are preserved in
    both levels — no recompilation needed.
    """
    global _vllm_sleeping
    if not _vllm_sleeping:
        return

    wake_start = time.time()
    try:
        response = await http_client.post(
            f"http://{VLLM_HOST}:{VLLM_PORT}/wake_up",
            timeout=30.0,
        )
        wake_ms = (time.time() - wake_start) * 1000
        if response.status_code == 200:
            _vllm_sleeping = False
            logger.info("WAKE: vLLM woke from L%d sleep in %.0fms", SLEEP_LEVEL, wake_ms)
        else:
            logger.warning("WAKE: vLLM returned %d (%.0fms)", response.status_code, wake_ms)
    except Exception as e:
        logger.warning("WAKE: Failed to wake vLLM: %s", e)
        # If wake fails, try to process anyway — vLLM might not be sleeping
        _vllm_sleeping = False


async def idle_sleep_monitor(http_client: httpx.AsyncClient) -> None:
    """Background task that puts vLLM to sleep after SLEEP_IDLE_TIMEOUT of inactivity.

    WHY: instead of requiring an external controller to manage sleep/wake, the worker
    itself tracks idle time. After N seconds without processing a message, it calls
    /sleep on vLLM. When a message arrives, process_job calls wake_vllm() first.

    Analogy: ASG scale-in cooldown — after N seconds of low utilization, scale down.
    Except here we don't terminate the instance, we hibernate it.
    """
    global _last_activity_time
    logger.info(
        "SLEEP_MONITOR: Started — will sleep vLLM after %ds idle (level=%d)",
        SLEEP_IDLE_TIMEOUT, SLEEP_LEVEL,
    )

    while not shutdown_event.is_set():
        await asyncio.sleep(10)  # check every 10s
        idle_seconds = time.time() - _last_activity_time
        if idle_seconds >= SLEEP_IDLE_TIMEOUT and not _vllm_sleeping:
            logger.info(
                "SLEEP_MONITOR: No activity for %ds — putting vLLM to sleep",
                int(idle_seconds),
            )
            await sleep_vllm(http_client)

    logger.info("SLEEP_MONITOR: Stopped (shutdown requested)")


# ---------------------------------------------------------------------------
# Model warmup — absorb CUDA compilation before real traffic
# ---------------------------------------------------------------------------
async def warmup_model(http_client: httpx.AsyncClient) -> None:
    """Send a synthetic request to trigger CUDA kernel compilation.

    WHY: The first inference request after vLLM starts pays a one-time CUDA
    compilation cost (~20-30s extra TTFT) because --enforce-eager disables
    CUDA graphs and PyTorch compiles kernels on first use. By sending a warmup
    request BEFORE consuming from the queue, real user requests never hit this.

    Analogy: ALB slow start mode — gradually increase traffic to new instances
    so they can warm up JIT caches before receiving full load.
    """
    logger.info("WARMUP: Waiting for vLLM to be ready before sending warmup request...")
    warmup_start = time.time()

    # Wait for vLLM to be ready (it starts in the same pod but takes 2-3 min)
    for attempt in range(60):  # up to 5 min (60 × 5s)
        try:
            probe = await http_client.get(
                f"http://{VLLM_HOST}:{VLLM_PORT}/health", timeout=3.0
            )
            if probe.status_code == 200:
                logger.info("WARMUP: vLLM is ready (attempt %d)", attempt + 1)
                break
        except Exception:
            pass
        await asyncio.sleep(5)
    else:
        logger.warning("WARMUP: vLLM not ready after 5 min — skipping warmup")
        return

    logger.info("WARMUP: Sending synthetic inference request to trigger CUDA compilation...")

    try:
        async with http_client.stream(
            "POST",
            f"http://{VLLM_HOST}:{VLLM_PORT}/v1/completions",
            json={
                "model": VLLM_MODEL,
                "prompt": "Explain what a warmup request is in one sentence.",
                "max_tokens": 50,
                "temperature": 0.1,
            },
            timeout=120.0,  # generous timeout for first-request compilation
        ) as response:
            token_count = 0
            async for chunk in response.aiter_text():
                if '"text"' in chunk:
                    token_count += 1

            warmup_ms = (time.time() - warmup_start) * 1000
            logger.info(
                "WARMUP: Complete — %d tokens in %.0fms (%.1f tok/s). CUDA kernels compiled.",
                token_count, warmup_ms,
                token_count / (warmup_ms / 1000) if warmup_ms > 0 else 0,
            )
    except Exception as e:
        warmup_ms = (time.time() - warmup_start) * 1000
        logger.warning(
            "WARMUP: Failed after %.0fms: %s — proceeding anyway (first real request will be slow)",
            warmup_ms, e,
        )


# ---------------------------------------------------------------------------
# Main loop — with Phase 4 graceful drain + warmup
# ---------------------------------------------------------------------------
async def main() -> None:
    """Worker entry point — connect to services and consume from the queue."""
    # Validate strategy before connecting to anything
    if ADMISSION_STRATEGY not in STRATEGIES:
        logger.error(
            "Invalid ADMISSION_STRATEGY '%s'. Must be one of: %s",
            ADMISSION_STRATEGY, list(STRATEGIES.keys()),
        )
        return

    admit_fn = STRATEGIES[ADMISSION_STRATEGY]

    logger.info("Worker starting (Phase 4 — graceful drain enabled)...")
    logger.info(
        "Config: strategy=%s VLLM=%s:%s model=%s prefetch=%d cache_threshold=%.0f%% max_batch=%d drain_timeout=%ds",
        ADMISSION_STRATEGY, VLLM_HOST, VLLM_PORT, VLLM_MODEL,
        PREFETCH_COUNT, GPU_CACHE_THRESHOLD * 100, MAX_BATCH_SIZE, DRAIN_TIMEOUT,
    )
    if ADMISSION_STRATEGY == "per_request":
        logger.info(
            "Per-request config: kv_bytes_per_token=%d chars_per_token=%d total_kv_cache=%dMB",
            KV_BYTES_PER_TOKEN, CHARS_PER_TOKEN, TOTAL_KV_CACHE_BYTES // 1_000_000,
        )

    # --- Connect to RabbitMQ ---
    logger.info("Connecting to RabbitMQ at %s:%s ...", RABBITMQ_HOST, RABBITMQ_PORT)
    rabbitmq_connection = await aio_pika.connect_robust(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        login=RABBITMQ_USER,
        password=RABBITMQ_PASS,
        heartbeat=30,
    )
    channel = await rabbitmq_connection.channel()
    await channel.set_qos(prefetch_count=PREFETCH_COUNT)
    logger.info("RabbitMQ connected, prefetch_count=%d", PREFETCH_COUNT)

    # --- Connect to Redis ---
    logger.info("Connecting to Redis at %s:%s ...", REDIS_HOST, REDIS_PORT)
    redis_client = aioredis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_timeout=5.0,
        retry_on_timeout=True,
    )
    await redis_client.ping()
    logger.info("Redis connected")

    # --- HTTP client for vLLM ---
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
    )

    # --- Phase 4v2: Model warmup ---
    if WARMUP_ENABLED:
        await warmup_model(http_client)
    else:
        logger.info("WARMUP: Disabled (WARMUP_ENABLED=false)")

    # --- Phase 4v3: Signal readiness via file ---
    # WHY after warmup: the K8s readiness probe on this container checks for this file.
    # The pod isn't marked "ready" until ALL containers pass readiness. By creating
    # this file only after warmup, we guarantee:
    #   1. vLLM model is loaded (startup probe already passed on vLLM container)
    #   2. CUDA kernels are compiled (warmup sent a synthetic request)
    #   3. The worker is about to start consuming from the queue
    # Without this, the old initialDelaySeconds: 120 was a blind guess that often
    # overshot (wasted 80s) or undershot (pod marked ready before warmup finished).
    # Analogy: ALB target group — instance is InService only after health check passes
    # on the actual application endpoint, not after a fixed warm-up timer.
    try:
        with open(READINESS_FILE, "w") as f:
            f.write(str(time.time()))
        logger.info("READY: Created %s — pod readiness probe will pass", READINESS_FILE)
    except Exception as e:
        logger.warning("READY: Failed to create %s: %s — pod may not become ready", READINESS_FILE, e)

    # --- Declare queue (idempotent — same params as gateway) ---
    queue = await channel.declare_queue(
        "inference_queue",
        durable=True,
        arguments={"x-max-length": 1000},
    )
    logger.info(
        "Queue 'inference_queue' declared, consuming with strategy '%s'...",
        ADMISSION_STRATEGY,
    )

    # --- Phase 4v3: Start sleep monitor ---
    # WHY background task: the sleep monitor runs alongside the consume loop,
    # checking idle time every 10s. When idle exceeds SLEEP_IDLE_TIMEOUT, it
    # calls /sleep on vLLM. The consume loop calls wake_vllm() when a new
    # message arrives. This is self-contained — no external controller needed.
    sleep_monitor_task = None
    if SLEEP_ENABLED:
        sleep_monitor_task = asyncio.create_task(idle_sleep_monitor(http_client))
    else:
        logger.info("SLEEP: Disabled (SLEEP_ENABLED=false)")

    # --- Consume messages ---
    # Phase 4: track tasks in active_tasks set for drain on shutdown
    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            if shutdown_event.is_set():
                logger.info("Shutdown requested — stopping consumption")
                await message.nack(requeue=True)
                break

            # Phase 4v3: wake vLLM if it's sleeping before processing
            # WHY here (not in process_job): wake once per batch of messages,
            # not once per concurrent task. Multiple messages arriving in quick
            # succession share the same wake call.
            if _vllm_sleeping:
                await wake_vllm(http_client)

            # Update activity timestamp for idle sleep detection
            global _last_activity_time
            _last_activity_time = time.time()

            # Phase 4 FIX: track the task so we can await it during drain.
            # Phase 3 used bare asyncio.create_task() — fire-and-forget.
            task = asyncio.create_task(
                process_job(message, redis_client, http_client, admit_fn)
            )
            active_tasks.add(task)
            # WHY add_done_callback: automatically remove completed tasks from the set.
            # Without this, the set grows unboundedly and drain would wait for already-
            # finished tasks (which would return immediately, but still — cleaner).
            task.add_done_callback(active_tasks.discard)

    # --- Phase 4: Graceful drain — wait for in-flight tasks ---
    drain_start = time.time()
    in_flight = len(active_tasks)

    if in_flight > 0:
        logger.info(
            "DRAIN_START: waiting for %d in-flight tasks (timeout=%ds)",
            in_flight, DRAIN_TIMEOUT,
        )
        # WHY asyncio.wait with timeout: gives in-flight requests time to finish
        # streaming their tokens to Redis. Without this, the process exits and
        # clients see broken SSE streams.
        done, pending = await asyncio.wait(
            active_tasks, timeout=DRAIN_TIMEOUT
        )
        drain_seconds = time.time() - drain_start

        if pending:
            # WHY cancel: terminationGracePeriodSeconds is about to expire.
            # Better to cancel cleanly than have the kernel SIGKILL the process.
            logger.warning(
                "Drain timeout after %.1fs — cancelling %d remaining tasks",
                drain_seconds, len(pending),
            )
            for task in pending:
                task.cancel()
            # Give cancelled tasks a moment to handle CancelledError
            await asyncio.gather(*pending, return_exceptions=True)
        else:
            drain_seconds = time.time() - drain_start
            logger.info("All in-flight tasks drained in %.1fs", drain_seconds)
    else:
        drain_seconds = 0.0
        done = set()
        pending = set()
        logger.info("No in-flight tasks to drain")

    # Phase 4: Log drain stats for test harness verification
    # The test harness greps for "DRAIN_STATS:" to verify graceful shutdown worked.
    logger.info(
        "DRAIN_STATS: in_flight=%d drained=%d cancelled=%d drain_seconds=%.1f",
        in_flight, len(done), len(pending), drain_seconds,
    )

    # --- Phase 4v3: Stop sleep monitor ---
    if sleep_monitor_task and not sleep_monitor_task.done():
        sleep_monitor_task.cancel()
        try:
            await sleep_monitor_task
        except asyncio.CancelledError:
            pass

    # --- Phase 4v3: Remove readiness file on shutdown ---
    # WHY remove: if the container somehow survives shutdown (unlikely but possible
    # with zombie processes), removing the file makes the readiness probe fail,
    # which tells K8s to stop routing new work to this pod.
    try:
        os.remove(READINESS_FILE)
        logger.info("READY: Removed %s — pod readiness will fail if re-probed", READINESS_FILE)
    except FileNotFoundError:
        pass

    # --- Cleanup ---
    logger.info("Cleaning up connections...")
    logger.info(
        "Final stats [%s]: admits=%d nacks=%d",
        ADMISSION_STRATEGY, admit_count, nack_count,
    )
    await http_client.aclose()
    await redis_client.close()
    await rabbitmq_connection.close()
    logger.info("Worker shut down cleanly")


def handle_shutdown(sig: signal.Signals) -> None:
    """Signal handler for graceful shutdown."""
    logger.info("Received %s — initiating graceful shutdown", sig.name)
    shutdown_event.set()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_shutdown, sig)

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — exiting")
    finally:
        loop.close()
