# worker.py — Queue-based inference worker with admission control.
#
# ARCHITECTURE ANALOGY (AWS → inference platform):
#   This worker is like an EC2 instance in a target group that pulls work from SQS
#   instead of receiving routed requests from an ALB. The key difference: it controls
#   its own admission. Before processing a job, it checks GPU memory pressure and
#   decides whether to accept or requeue — like an instance reporting "unhealthy"
#   to the ALB so traffic shifts elsewhere.
#
# WHY pull-based (not push):
#   Push model (ALB → pod): request arrives whether the GPU has capacity or not.
#   Pull model (pod ← queue): worker only takes work when it can handle it.
#   This prevents GPU OOM and KV cache thrashing — the #1 cause of inference
#   latency spikes in production systems.
#
# FLOW:
#   RabbitMQ "inference_queue" → worker.process_job() → vLLM /v1/completions
#                                                          ↓ (streaming)
#                                                     Redis PUBLISH tokens
#                                                          ↓
#                                                     API Gateway → SSE → Client

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
# Configuration from env vars — K8s manifests inject these
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

# WHY prefetch_count: controls how many unacked messages this worker holds.
# Too low (1) = underutilized GPU between message fetches.
# Too high (50) = worker grabs work it can't process, starving other workers.
# 5 is a sweet spot — enough to keep the GPU pipeline full without hoarding.
# Same concept as SQS ReceiveMessageWaitTimeSeconds + MaxNumberOfMessages.
PREFETCH_COUNT = int(os.getenv("PREFETCH_COUNT", "5"))

# Admission control thresholds
# WHY 80%: vLLM's PagedAttention allocator starts evicting KV cache blocks above
# ~85%. At 80% we stop admitting new requests to avoid eviction storms.
# Analogous to an ASG scaling policy: "if CPU > 80%, scale out."
GPU_CACHE_THRESHOLD = float(os.getenv("GPU_CACHE_THRESHOLD", "0.80"))
# WHY max_batch 8: a T4 with 16GB VRAM running a 7B-AWQ model (~4GB) has ~12GB
# for KV cache. Each concurrent request uses ~200MB at 2048 context length.
# 8 * 200MB = 1.6GB, leaving headroom for variable-length requests.
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))

logger = logging.getLogger("worker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# Global shutdown flag — set by signal handlers for graceful termination
shutdown_event = asyncio.Event()


# ---------------------------------------------------------------------------
# vLLM metrics parsing
# ---------------------------------------------------------------------------
async def get_vllm_metrics(http_client: httpx.AsyncClient) -> dict:
    """Fetch and parse Prometheus metrics from vLLM's /metrics endpoint.

    WHY parse Prometheus text format directly: vLLM exposes metrics in
    Prometheus exposition format. We only need 2 gauges, so a regex is
    simpler than pulling in prometheus_client as a dependency.

    Returns:
        Dict with 'gpu_cache_usage' (0.0-1.0) and 'running_requests' (int).
        Returns safe defaults on error (so we don't block work on metrics failure).
    """
    try:
        response = await http_client.get(
            f"http://{VLLM_HOST}:{VLLM_PORT}/metrics",
            timeout=2.0,  # WHY short timeout: metrics fetch shouldn't block job processing
        )
        text = response.text

        # Parse gpu_cache_usage_perc — fraction of GPU KV cache blocks in use
        # WHY this metric: directly measures memory pressure on the GPU.
        # When this approaches 1.0, vLLM starts evicting cached prefixes and
        # preempting running requests — catastrophic for latency.
        cache_match = re.search(r"vllm:gpu_cache_usage_perc\s+([\d.]+)", text)
        gpu_cache_usage = float(cache_match.group(1)) if cache_match else 0.0

        # Parse num_requests_running — current batch size on the GPU
        running_match = re.search(r"vllm:num_requests_running\s+([\d.]+)", text)
        running_requests = int(float(running_match.group(1))) if running_match else 0

        return {
            "gpu_cache_usage": gpu_cache_usage,
            "running_requests": running_requests,
        }

    except Exception as e:
        # WHY return safe defaults instead of raising: if vLLM metrics endpoint
        # is temporarily unavailable (e.g., during model loading), we don't want
        # to NACK every message. Allow work to proceed with conservative assumptions.
        logger.warning("Failed to fetch vLLM metrics: %s — using safe defaults", e)
        return {
            "gpu_cache_usage": 0.0,
            "running_requests": 0,
        }


# ---------------------------------------------------------------------------
# Admission control
# ---------------------------------------------------------------------------
async def should_admit(http_client: httpx.AsyncClient) -> tuple[bool, str]:
    """Decide whether to accept a new inference request based on GPU state.

    ANALOGY: This is the worker's equivalent of an ALB health check combined
    with a scaling policy. Instead of waiting for an external system to detect
    overload, the worker self-regulates. Like a circuit breaker in a microservice.

    Returns:
        (True, "") if request should be processed.
        (False, reason) if request should be requeued.
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


# ---------------------------------------------------------------------------
# Job processing — the core inference loop
# ---------------------------------------------------------------------------
async def process_job(
    message: aio_pika.abc.AbstractIncomingMessage,
    redis: aioredis.Redis,
    http_client: httpx.AsyncClient,
) -> None:
    """Process a single inference job from the queue.

    Steps:
    1. Parse and validate the job JSON
    2. Check admission control (GPU memory + batch size)
    3. Stream completions from vLLM
    4. Publish each token to Redis pub/sub
    5. ACK or NACK the RabbitMQ message

    WHY this is an async function (not sync): every I/O operation (Redis publish,
    vLLM HTTP call, RabbitMQ ack) yields to the event loop, allowing the worker
    to process multiple jobs concurrently up to PREFETCH_COUNT.
    """
    dequeue_time = time.time()

    try:
        # --- Parse job payload ---
        job = json.loads(message.body.decode())
        job_id = job["job_id"]
        prompt = job["prompt"]
        max_tokens = job.get("max_tokens", 256)
        temperature = job.get("temperature", 0.7)
        enqueue_time = job.get("enqueue_time", dequeue_time)

        # WHY queue_wait_ms: this is the #1 signal for capacity planning.
        # If queue wait is consistently > 1s, you need more GPU workers.
        # Same metric as SQS ApproximateAgeOfOldestMessage.
        queue_wait_ms = (dequeue_time - enqueue_time) * 1000
        logger.info(
            "Processing job %s (queue_wait: %.0fms, prompt_len: %d)",
            job_id, queue_wait_ms, len(prompt),
        )

        channel = f"job:{job_id}"

    except (json.JSONDecodeError, KeyError) as e:
        # Malformed message — NACK without requeue (don't poison the queue)
        logger.error("Invalid job message: %s — dropping", e)
        await message.nack(requeue=False)
        return

    # --- Admission control ---
    admitted, reason = await should_admit(http_client)
    if not admitted:
        # WHY requeue (not drop): the GPU is temporarily overloaded, not broken.
        # Requeuing lets another worker pick it up, or this worker can retry
        # after current requests finish. Like returning a 503 with Retry-After.
        logger.info("Job %s rejected (admission control): %s — requeuing", job_id, reason)
        await message.nack(requeue=True)
        # WHY sleep 100ms: prevents tight requeue loops that burn CPU and flood
        # RabbitMQ with NACK/redeliver cycles. Gives the GPU time to free cache.
        await asyncio.sleep(0.1)
        return

    # --- Stream inference from vLLM ---
    tokens_generated = 0
    ttft_ms: Optional[float] = None  # Time to first token
    inference_start = time.time()

    try:
        # WHY streaming: non-streaming would block until all tokens are generated,
        # giving the client a long wait before seeing any output. Streaming lets
        # the user see tokens arrive in real-time (TTFT ~50ms vs ~5s for 256 tokens).
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
            # WHY long timeout: a 4096-token generation at ~30 tokens/sec takes ~136s.
            # 300s gives plenty of headroom for slow batches under load.
            timeout=300.0,
        ) as response:
            if response.status_code != 200:
                # vLLM returned an error — read the body for details
                error_body = await response.aread()
                raise RuntimeError(
                    f"vLLM returned {response.status_code}: {error_body.decode()}"
                )

            # Parse SSE stream from vLLM's OpenAI-compatible API
            # Format: "data: {json}\n\n" or "data: [DONE]\n\n"
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                # WHY manual SSE parsing: httpx doesn't have built-in SSE support.
                # We split on double newlines (SSE event boundary) and parse each.
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)

                    for line in event.strip().split("\n"):
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]  # Strip "data: " prefix

                        # [DONE] signals end of stream from vLLM
                        if data_str.strip() == "[DONE]":
                            continue

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.debug("Skipping non-JSON SSE data: %s", data_str[:100])
                            continue

                        # Extract the generated token text from OpenAI-format response
                        choices = data.get("choices", [])
                        if not choices:
                            continue

                        token_text = choices[0].get("text", "")
                        if not token_text:
                            continue

                        # Record time to first token — the most user-visible latency metric.
                        # WHY TTFT matters: users perceive <200ms TTFT as "instant".
                        # Above 1s, they think the system is broken. It's like TTFB in web perf.
                        if ttft_ms is None:
                            ttft_ms = (time.time() - inference_start) * 1000
                            logger.info("Job %s TTFT: %.0fms", job_id, ttft_ms)

                        tokens_generated += 1

                        # Publish token to Redis so the API gateway can stream it to the client
                        await redis.publish(
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
                # WHY tokens_per_sec: the throughput metric. Helps you see if the
                # GPU is running at expected speed or if batch contention is slowing it.
                "tokens_per_sec": round(
                    tokens_generated / (total_ms / 1000), 1
                ) if total_ms > 0 else 0,
            },
        }
        await redis.publish(channel, json.dumps(completion_msg))

        # WHY ACK after publishing completion: ensures the client receives the
        # completion signal before we tell RabbitMQ to remove the message.
        # If we crash between publish and ACK, the message gets redelivered —
        # the client might see duplicate tokens, but that's better than lost jobs.
        await message.ack()

        logger.info(
            "Job %s complete: %d tokens in %.0fms (%.1f tok/s, TTFT: %.0fms, wait: %.0fms)",
            job_id, tokens_generated, total_ms,
            completion_msg["metrics"]["tokens_per_sec"],
            ttft_ms or 0, queue_wait_ms,
        )

    except Exception as e:
        logger.error("Job %s failed: %s", job_id, e, exc_info=True)

        # Notify the client that the job failed — don't leave them hanging
        error_msg = {"status": "error", "error": str(e)}
        try:
            await redis.publish(channel, json.dumps(error_msg))
        except Exception as pub_err:
            logger.error("Failed to publish error for job %s: %s", job_id, pub_err)

        # WHY NACK without requeue: the job failed, likely due to a bad prompt or
        # vLLM internal error. Retrying would probably fail again, wasting GPU time.
        # For transient errors (network blip), the client can resubmit.
        # This is like sending a message to an SQS dead-letter queue.
        await message.nack(requeue=False)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def main() -> None:
    """Worker entry point — connect to services and consume from the queue."""
    logger.info("Worker starting...")
    logger.info(
        "Config: VLLM=%s:%s model=%s prefetch=%d cache_threshold=%.0f%% max_batch=%d",
        VLLM_HOST, VLLM_PORT, VLLM_MODEL, PREFETCH_COUNT,
        GPU_CACHE_THRESHOLD * 100, MAX_BATCH_SIZE,
    )

    # --- Connect to RabbitMQ ---
    logger.info("Connecting to RabbitMQ at %s:%s ...", RABBITMQ_HOST, RABBITMQ_PORT)
    # WHY robust connection: auto-reconnects if RabbitMQ restarts during rolling update.
    # Without this, a RabbitMQ pod restart would kill all workers permanently.
    rabbitmq_connection = await aio_pika.connect_robust(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        login=RABBITMQ_USER,
        password=RABBITMQ_PASS,
        heartbeat=30,
    )
    channel = await rabbitmq_connection.channel()

    # WHY set QoS (prefetch): controls concurrency at the consumer level.
    # Without this, RabbitMQ pushes ALL pending messages to the worker, which
    # would overwhelm the GPU. prefetch_count=5 means "send me at most 5 unacked
    # messages." Same concept as SQS MaxNumberOfMessages per ReceiveMessage call.
    await channel.set_qos(prefetch_count=PREFETCH_COUNT)
    logger.info("RabbitMQ connected, prefetch_count=%d", PREFETCH_COUNT)

    # --- Connect to Redis ---
    logger.info("Connecting to Redis at %s:%s ...", REDIS_HOST, REDIS_PORT)
    redis = aioredis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_timeout=5.0,
        retry_on_timeout=True,
    )
    await redis.ping()
    logger.info("Redis connected")

    # --- HTTP client for vLLM ---
    # WHY a shared client: connection pooling — reuses TCP connections to vLLM
    # across requests. Creating a new connection per request adds ~1ms overhead
    # and can exhaust ephemeral ports under load.
    http_client = httpx.AsyncClient(
        # WHY http2=False: vLLM's OpenAI-compatible server doesn't support HTTP/2.
        # Trying h2 would cause connection errors.
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
    )

    # --- Declare queue (idempotent — same params as gateway) ---
    queue = await channel.declare_queue(
        "inference_queue",
        durable=True,
        arguments={"x-max-length": 1000},
    )
    logger.info("Queue 'inference_queue' declared, starting consumption...")

    # --- Consume messages ---
    # WHY async iteration over queue: aio-pika delivers messages as they arrive,
    # up to prefetch_count. Each message triggers process_job as a coroutine,
    # allowing concurrent processing within the event loop.
    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            if shutdown_event.is_set():
                # WHY check shutdown before processing: graceful shutdown means
                # finishing in-flight work but not starting new work. Like
                # connection draining in an ALB target group.
                logger.info("Shutdown requested — stopping consumption")
                # NACK with requeue so another worker can pick it up
                await message.nack(requeue=True)
                break

            # Process the job — errors are handled inside process_job,
            # so the consumer loop continues regardless
            asyncio.create_task(process_job(message, redis, http_client))

    # --- Cleanup ---
    logger.info("Cleaning up connections...")
    await http_client.aclose()
    await redis.close()
    await rabbitmq_connection.close()
    logger.info("Worker shut down cleanly")


def handle_shutdown(sig: signal.Signals) -> None:
    """Signal handler for graceful shutdown.

    WHY handle SIGTERM: K8s sends SIGTERM during pod eviction (Spot reclaim,
    rolling update, scale-down). We need to finish in-flight inference before
    exiting, or the user gets an incomplete response.
    """
    logger.info("Received %s — initiating graceful shutdown", sig.name)
    shutdown_event.set()


if __name__ == "__main__":
    # Register signal handlers before starting the event loop
    loop = asyncio.new_event_loop()

    # WHY both SIGTERM and SIGINT: SIGTERM comes from K8s, SIGINT from Ctrl+C
    # during local development. Both should trigger graceful shutdown.
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_shutdown, sig)

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — exiting")
    finally:
        loop.close()
