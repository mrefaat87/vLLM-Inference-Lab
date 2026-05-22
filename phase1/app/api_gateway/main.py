# main.py — FastAPI gateway that decouples clients from GPU workers via a queue.
#
# ARCHITECTURE ANALOGY (AWS → inference platform):
#   This gateway plays the role of an Application Load Balancer + SQS producer.
#   Instead of routing requests directly to backends (which would couple throughput
#   to GPU count), it enqueues work into RabbitMQ and returns immediately.
#   Clients get a stream URL to watch their job complete — like polling an
#   SQS-backed Step Functions execution via its ARN.
#
# REQUEST FLOW:
#   Client → POST /generate → RabbitMQ "inference_queue"
#                                    ↓
#                           Worker picks up job
#                                    ↓
#                           Worker → vLLM /v1/completions (streaming)
#                                    ↓
#                           Worker → Redis PUBLISH "job:{id}" (each token)
#                                    ↓
#   Client ← GET /stream/{id} ← SSE ← Redis SUBSCRIBE "job:{id}"

import os
import uuid
import json
import time
import logging
from contextlib import asynccontextmanager

import aio_pika
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from redis_client import RedisSubscriber

# ---------------------------------------------------------------------------
# Configuration from env vars — K8s manifests inject these
# ---------------------------------------------------------------------------
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

logger = logging.getLogger("api_gateway")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# ---------------------------------------------------------------------------
# Shared state — initialized in lifespan, cleaned up on shutdown
# ---------------------------------------------------------------------------
# WHY module-level state: FastAPI's lifespan context manager runs once, and
# we need these connections accessible from all request handlers. Using app.state
# would also work, but module-level is simpler for a single-worker process.
rabbitmq_connection: aio_pika.abc.AbstractRobustConnection | None = None
rabbitmq_channel: aio_pika.abc.AbstractChannel | None = None
redis_client: aioredis.Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage connection lifecycle — connect on startup, close on shutdown.

    WHY lifespan over on_event: FastAPI deprecated @app.on_event in favor of
    lifespan context managers. This pattern ensures cleanup runs even on
    ungraceful shutdowns (SIGTERM from K8s during pod eviction).
    """
    global rabbitmq_connection, rabbitmq_channel, redis_client

    # --- Startup ---
    logger.info("Connecting to RabbitMQ at %s:%s ...", RABBITMQ_HOST, RABBITMQ_PORT)
    # WHY RobustConnection: auto-reconnects on transient failures (e.g., RabbitMQ
    # pod restart during rolling update). Same concept as RDS Proxy connection pooling.
    rabbitmq_connection = await aio_pika.connect_robust(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        login=RABBITMQ_USER,
        password=RABBITMQ_PASS,
        # WHY heartbeat 30s: detect dead connections faster than the default 60s.
        # In K8s, network partitions during node drains are common.
        heartbeat=30,
    )
    rabbitmq_channel = await rabbitmq_connection.channel()

    # WHY declare queue here: ensures the queue exists before workers start consuming.
    # durable=True means the queue survives RabbitMQ restarts (messages in transit
    # are also persisted if published as persistent). Same idea as an SQS FIFO queue
    # with message retention.
    await rabbitmq_channel.declare_queue(
        "inference_queue",
        durable=True,
        arguments={
            # WHY x-max-length: backpressure — if the queue grows beyond 1000 messages,
            # RabbitMQ drops the oldest. Prevents OOM if GPU workers fall behind.
            # Analogous to SQS maximum receives + DLQ, but simpler.
            "x-max-length": 1000,
        },
    )
    logger.info("RabbitMQ connected, queue 'inference_queue' declared")

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

    yield  # --- App runs here ---

    # --- Shutdown ---
    logger.info("Shutting down connections...")
    if redis_client:
        await redis_client.close()
    if rabbitmq_channel:
        await rabbitmq_channel.close()
    if rabbitmq_connection:
        await rabbitmq_connection.close()
    logger.info("All connections closed")


app = FastAPI(
    title="Inference Lab API Gateway",
    description="Queue-based inference serving with SSE streaming",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    """Input for a text generation job."""
    prompt: str = Field(..., min_length=1, description="The input text to complete")
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=4096,
        # WHY cap at 4096: prevents a single request from monopolizing GPU memory.
        # A 7B model with 4096 output tokens uses ~512MB KV cache — about 3% of
        # a T4's 16GB. Keeps the cache pool healthy for concurrent requests.
        description="Maximum tokens to generate (1-4096)",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0=greedy, 2=creative)",
    )


class GenerateResponse(BaseModel):
    """Returned immediately after enqueuing — client uses stream_url to get results."""
    job_id: str
    stream_url: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/generate", response_model=GenerateResponse, status_code=202)
async def generate(request: GenerateRequest):
    """Enqueue an inference job and return a stream URL.

    Returns 202 Accepted — the job is queued, not complete. This decouples
    the client from GPU availability. If all GPUs are busy, the job waits
    in the queue rather than getting a 503.

    WHY 202 (not 200): HTTP semantics — 202 means "request accepted for
    processing" which is exactly what's happening. The client must poll
    the stream_url to get the actual result.
    """
    if not rabbitmq_channel:
        raise HTTPException(status_code=503, detail="RabbitMQ not connected")

    # Generate a unique job ID — used as the Redis pub/sub channel name
    # and returned to the client for correlation
    job_id = str(uuid.uuid4())

    # Build the job payload — everything the worker needs to run inference
    job = {
        "job_id": job_id,
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        # WHY enqueue_time: the worker computes queue_wait_ms from this.
        # Measures how long the job sat in RabbitMQ before a worker picked it up.
        # This is the #1 metric for knowing if you need more GPU capacity.
        "enqueue_time": time.time(),
    }

    # WHY delivery_mode=2 (persistent): survives RabbitMQ restarts. Without it,
    # a RabbitMQ pod restart loses all queued jobs. Trade-off: ~2ms extra latency
    # for the disk fsync. Worth it — losing user requests is worse than 2ms.
    await rabbitmq_channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(job).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            # WHY content_type: helps debugging when inspecting messages in
            # the RabbitMQ management UI
            content_type="application/json",
        ),
        routing_key="inference_queue",
    )

    logger.info("Enqueued job %s (prompt length: %d chars)", job_id, len(request.prompt))

    return GenerateResponse(
        job_id=job_id,
        stream_url=f"/stream/{job_id}",
    )


@app.get("/stream/{job_id}")
async def stream(job_id: str):
    """Stream inference results via Server-Sent Events (SSE).

    The client connects here after receiving the job_id from POST /generate.
    Each event is either a token chunk or a completion signal with metrics.

    WHY SSE (not WebSocket): SSE is simpler — unidirectional, auto-reconnects,
    works through HTTP proxies. We don't need bidirectional communication.
    Same pattern as CloudWatch Logs live tail.
    """

    async def event_generator():
        """Yield SSE events from Redis pub/sub for this job."""
        subscriber = RedisSubscriber()
        try:
            await subscriber.connect()
            await subscriber.subscribe(job_id)

            # WHY async generator: sse-starlette consumes this generator and
            # sends each yielded dict as an SSE "data:" line. When the generator
            # exits, the SSE connection closes.
            async for message in subscriber.listen(timeout=120.0):
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON message on job %s: %s", job_id, message)
                    continue

                # Yield the raw JSON to the SSE stream
                yield {"data": message}

                # WHY check for "complete" or "error": these are terminal events.
                # After this, no more tokens will arrive — close the stream to
                # free the Redis subscription and the client's HTTP connection.
                if data.get("status") in ("complete", "error"):
                    logger.info("Job %s finished with status: %s", job_id, data["status"])
                    break

        except Exception as e:
            logger.error("Stream error for job %s: %s", job_id, e)
            # Send error event so the client knows something went wrong
            yield {"data": json.dumps({"status": "error", "error": str(e)})}

        finally:
            # Always clean up the subscription — prevents Redis from tracking
            # dead subscribers (memory leak on the Redis side)
            await subscriber.cleanup()

    return EventSourceResponse(event_generator())


@app.get("/health")
async def health():
    """Health check endpoint for K8s liveness/readiness probes.

    WHY check both dependencies: a gateway that can't reach RabbitMQ can't
    enqueue jobs, and one that can't reach Redis can't stream results.
    Both must be healthy for the gateway to be useful.

    K8s uses this to decide whether to route traffic to this pod (readiness)
    and whether to restart it (liveness). Same pattern as ELB health checks
    on an EC2 target group.
    """
    status = {"service": "api-gateway"}
    errors = []

    # Check RabbitMQ — can we still publish messages?
    try:
        if rabbitmq_connection and not rabbitmq_connection.is_closed:
            status["rabbitmq"] = "healthy"
        else:
            errors.append("RabbitMQ connection closed")
            status["rabbitmq"] = "unhealthy"
    except Exception as e:
        errors.append(f"RabbitMQ: {e}")
        status["rabbitmq"] = "unhealthy"

    # Check Redis — can we still subscribe to channels?
    try:
        if redis_client:
            # WHY ping with short implicit timeout (socket_timeout=5s from init):
            # prevents the health check from blocking the event loop
            await redis_client.ping()
            status["redis"] = "healthy"
        else:
            errors.append("Redis not connected")
            status["redis"] = "unhealthy"
    except Exception as e:
        errors.append(f"Redis: {e}")
        status["redis"] = "unhealthy"

    if errors:
        status["status"] = "degraded"
        status["errors"] = errors
        # WHY 503 on degraded: tells K8s readiness probe to stop routing traffic.
        # The pod stays running (liveness uses a different threshold) but doesn't
        # receive new requests until dependencies recover.
        raise HTTPException(status_code=503, detail=status)

    status["status"] = "healthy"
    return status
