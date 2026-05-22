# redis_client.py — Async Redis pub/sub helper for streaming tokens back to clients.
#
# ARCHITECTURE ANALOGY (AWS → inference platform):
#   Redis pub/sub here acts like SNS fan-out — the worker publishes each generated
#   token to a per-job channel, and the API gateway subscribes to relay it via SSE.
#   It's a lightweight push model: no polling, no SQS visibility timeouts to manage.
#
# WHY Redis for token streaming (not RabbitMQ):
#   RabbitMQ is optimized for durable, at-least-once delivery with ack/nack.
#   Token streaming is fire-and-forget (if a client disconnects, tokens are dropped).
#   Redis pub/sub has <1ms overhead per message vs ~5ms for AMQP. For 50+ tokens/sec
#   streaming, that difference matters.

import os
import asyncio
import logging
from typing import AsyncGenerator

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Connection params from env vars — K8s manifests inject these via env or ConfigMap
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


class RedisSubscriber:
    """Wraps redis.asyncio pub/sub for per-job channel subscription.

    Each inference job gets its own channel "job:{job_id}". The API gateway
    subscribes to this channel and yields messages as an async generator,
    which feeds directly into an SSE response stream.
    """

    def __init__(self):
        self._redis: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None

    async def connect(self) -> None:
        """Establish connection to Redis. Called once at gateway startup."""
        self._redis = aioredis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            # WHY decode_responses: pub/sub messages arrive as bytes by default.
            # Decoding here means we don't have to .decode() on every message.
            decode_responses=True,
            # WHY socket_timeout: prevents the gateway from hanging forever if
            # Redis becomes unreachable (e.g., pod eviction during Spot reclaim).
            socket_timeout=5.0,
            # WHY retry_on_timeout: transient network blips in K8s are common
            # during pod rescheduling — retry once before raising.
            retry_on_timeout=True,
        )
        # Verify connectivity — fail fast on startup rather than on first request
        await self._redis.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")

    async def subscribe(self, job_id: str) -> None:
        """Subscribe to the channel for a specific job.

        Args:
            job_id: UUID of the inference job. Channel name is "job:{job_id}".
        """
        if not self._redis:
            raise RuntimeError("Redis not connected — call connect() first")

        self._pubsub = self._redis.pubsub()
        channel = f"job:{job_id}"
        await self._pubsub.subscribe(channel)
        logger.debug(f"Subscribed to channel: {channel}")

    async def listen(self, timeout: float = 120.0) -> AsyncGenerator[str, None]:
        """Async generator that yields messages from the subscribed channel.

        Args:
            timeout: Max seconds to wait for any message before giving up.
                     120s is generous — a single completion should never take this long.
                     Acts as a circuit breaker against leaked subscriptions.

        Yields:
            Raw message string (JSON-encoded token or completion event).
        """
        if not self._pubsub:
            raise RuntimeError("Not subscribed — call subscribe() first")

        deadline = asyncio.get_event_loop().time() + timeout

        while True:
            # Check if we've exceeded the timeout deadline
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning("Subscription timed out — no messages within deadline")
                break

            try:
                # WHY get_message with timeout: non-blocking poll that yields control
                # back to the event loop. 1s timeout balances responsiveness (don't
                # block SSE too long) vs CPU usage (don't spin-loop).
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=min(1.0, remaining),
                )

                if message is not None and message["type"] == "message":
                    # Reset deadline on each received message — the timeout is for
                    # inactivity, not total duration. A long generation is fine as
                    # long as tokens keep flowing.
                    deadline = asyncio.get_event_loop().time() + timeout
                    yield message["data"]

            except asyncio.CancelledError:
                # Client disconnected — SSE stream was closed. Clean up gracefully.
                logger.debug("Subscription cancelled — client disconnected")
                break
            except Exception as e:
                logger.error(f"Error reading from pub/sub: {e}")
                break

    async def cleanup(self) -> None:
        """Unsubscribe and close connections. Called when SSE stream ends."""
        try:
            if self._pubsub:
                # WHY unsubscribe before close: prevents the Redis server from
                # tracking a dead subscriber (saves server-side memory).
                await self._pubsub.unsubscribe()
                await self._pubsub.close()
                self._pubsub = None
        except Exception as e:
            # Log but don't raise — cleanup failures shouldn't crash the request
            logger.warning(f"Error during pub/sub cleanup: {e}")

    async def close(self) -> None:
        """Close the underlying Redis connection entirely."""
        await self.cleanup()
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Redis connection closed")

    async def ping(self) -> bool:
        """Health check — returns True if Redis is reachable."""
        try:
            if self._redis:
                await self._redis.ping()
                return True
        except Exception:
            pass
        return False
