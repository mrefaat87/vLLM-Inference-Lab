# worker.py — Queue-based inference worker with pluggable admission control.
#
# ADMISSION STRATEGIES (set via ADMISSION_STRATEGY env var):
#
#   "static"      → Fixed prefetch_count=N, no runtime checks.
#                   RabbitMQ itself is the concurrency limiter.
#                   AWS analogy: fixed-size ASG — no scaling policy at all.
#
#   "reactive"    → Local tracker monitors current system load against threshold.
#                   "Is the bottleneck metric currently above X? NACK."
#                   Does NOT look at the incoming request's characteristics.
#                   AWS analogy: target tracking on CPU utilization.
#
#   "predictive"  → Same local tracker, but estimates what happens if THIS
#                   specific request is admitted. Uses prompt length + max_tokens
#                   to predict impact on the bottleneck metric.
#                   AWS analogy: predictive scaling — forecast load before routing.
#
# BOTTLENECK METRICS (set via BOTTLENECK_METRIC env var):
#   The metric to gate on depends on hardware/model, not the strategy:
#   - "decode_throughput" — T4 + 7B AWQ: memory bandwidth is the bottleneck
#   - "gpu_compute"       — T4 prefill: long prompts saturate GPU SMs
#   - "kv_cache"          — 14B model: KV cache headroom is tight
#   - "compound"          — gates on max(gpu_fraction, decode_fraction)
#
# All three strategies share the same process_job() loop — only the gate differs.
# NACK+requeue is the common rejection mechanism (no request ever lost).
#
# KEY DESIGN DECISION: No remote /metrics polling. All admission state is local
# and atomic via AdmissionTracker. This eliminates the stale-state and feedback-
# loop problems discovered in the first 3 experiment rounds.
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
from enum import Enum
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

# WHY prefetch_count matters differently per strategy:
#   Static: THIS IS the concurrency limiter. N=5 means at most 5 in flight.
#   Reactive/Predictive: prefetch_count is a soft upper bound; the tracker
#   is the real gate. We still set it to prevent unbounded message buffering.
PREFETCH_COUNT = int(os.getenv("PREFETCH_COUNT", "10"))

# Strategy selection — which admission gate to use
ADMISSION_STRATEGY = os.getenv("ADMISSION_STRATEGY", "reactive")

# WHY a single threshold for all strategies: this is the fraction of bottleneck
# capacity at which we reject. 0.80 = reject when 80% of capacity is consumed.
# Industry best practice: 70-80% is the balanced zone. Above 90% tail latencies
# spike exponentially per queueing theory (M/M/1 hockey stick curve).
ADMISSION_THRESHOLD = float(os.getenv("ADMISSION_THRESHOLD", "0.80"))

# Bottleneck metric — what resource to gate on. Hardware-dependent, not strategy-dependent.
BOTTLENECK_METRIC = os.getenv("BOTTLENECK_METRIC", "decode_throughput")

# Capacity values for each bottleneck dimension.
# WHY separate capacities: compound mode needs both GPU and decode budgets.
# Single-metric modes only use the relevant one.
#
# DECODE_CAPACITY: total concurrent decode tokens the GPU can sustain before
# TPOT degrades past SLA. This is NOT tok/s — it's the total "decode work
# budget" across all in-flight requests.
#
# Calibration for T4 + 7B AWQ, pareto_heavy workload:
#   avg max_tokens ~71 (weighted: 80%×20 + 15%×150 + 4%×300 + 1%×2048)
#   At 5 concurrent: 5×71 = 355 tokens in flight → TPOT ≈ 40ms (within SLA)
#   At 10 concurrent: 10×71 = 710 tokens in flight → TPOT ≈ 80ms (over SLA)
#   So capacity ~500 puts the SLA boundary at 7 concurrent requests.
DECODE_CAPACITY = float(os.getenv("DECODE_CAPACITY", "500.0"))
#
# GPU_COMPUTE_CAPACITY: total concurrent prefill tokens the GPU can sustain.
# Prefill is compute-bound and gets released after first token. At any moment,
# only requests currently in prefill contribute. With avg prompt ~32 tokens
# and 1-2 in prefill at a time, typical load is 32-64 / 300 = 10-20%.
GPU_COMPUTE_CAPACITY = float(os.getenv("GPU_COMPUTE_CAPACITY", "300.0"))
#
# KV_CACHE_CAPACITY_BYTES: total KV cache budget in bytes.
# T4 16GB × 0.85 utilization = 13.6GB available. Model ~4GB → ~9.6GB for KV.
KV_CACHE_CAPACITY_BYTES = int(os.getenv("KV_CACHE_CAPACITY_BYTES", str(9_600_000_000)))

# Per-request cost estimation constants
# WHY 401KB per token: KV cache entry = key+value for all heads across all layers.
# Qwen2.5-7B-AWQ: 28 layers × 28 heads × 128 dim × 2bytes × 2(k+v) = 401,408 bytes
KV_BYTES_PER_TOKEN = int(os.getenv("KV_BYTES_PER_TOKEN", "401408"))
# WHY 4 chars per token: rough tokenizer approximation for English.
# Qwen2.5 averages 3.5-4.5 chars/token. 4 is conservative (overestimates tokens).
CHARS_PER_TOKEN = int(os.getenv("CHARS_PER_TOKEN", "4"))

# Adaptive strategy configuration — SGLang-inspired output ratio learning.
# WHY these knobs: the adaptive strategy learns how much of max_tokens requests
# actually use. alpha controls learning speed, initial_ratio sets the starting
# point (1.0 = identical to predictive), pressure_reset is the snap-back target
# when vLLM errors signal overload.
ADAPTIVE_EMA_ALPHA = float(os.getenv("ADAPTIVE_EMA_ALPHA", "0.1"))
ADAPTIVE_INITIAL_RATIO = float(os.getenv("ADAPTIVE_INITIAL_RATIO", "1.0"))
ADAPTIVE_PRESSURE_RESET = float(os.getenv("ADAPTIVE_PRESSURE_RESET", "0.8"))

logger = logging.getLogger("worker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# Global shutdown flag — set by signal handlers for graceful termination
shutdown_event = asyncio.Event()

# Counters — the load test reads these from completion messages
nack_count = 0
admit_count = 0


# ---------------------------------------------------------------------------
# Bottleneck metric enum
# ---------------------------------------------------------------------------
class BottleneckMetric(Enum):
    DECODE_THROUGHPUT = "decode_throughput"
    GPU_COMPUTE = "gpu_compute"
    KV_CACHE = "kv_cache"
    COMPOUND = "compound"


# ---------------------------------------------------------------------------
# AdmissionTracker — local atomic state, zero HTTP overhead
# ---------------------------------------------------------------------------
class AdmissionTracker:
    """Tracks admission state locally in the worker process.

    WHY this replaces /metrics polling: the first 3 experiment rounds discovered
    that HTTP polling adds 2-5ms per request, returns stale state (multiple async
    coroutines read the same snapshot), and creates a positive feedback loop under
    load (polling delays → queue growth → more polling → cascade).

    This tracker is updated atomically via asyncio.Lock. Every admission decision
    reads state that reflects ALL previous admit/release calls — no stale snapshots.

    AWS analogy: this is like each instance tracking its own connection count
    locally rather than polling a central CloudWatch metric with 60s lag.
    """

    def __init__(
        self,
        metric: BottleneckMetric,
        decode_capacity: float,
        gpu_capacity: float,
        kv_capacity: int,
    ):
        self.metric = metric
        # Each running job: {decode_cost, gpu_cost, kv_cost, remaining_tokens}
        self.running: dict[str, dict] = {}
        # Separate budgets for each resource dimension
        self.decode_capacity = decode_capacity
        self.decode_used = 0.0
        self.gpu_capacity = gpu_capacity
        self.gpu_used = 0.0
        self.kv_capacity = float(kv_capacity)
        self.kv_used = 0.0
        self._lock = asyncio.Lock()
        # Adaptive strategy state — EMA-tracked output ratio.
        # WHY here and not in a separate class: the ratio needs the same lock
        # as the budget tracking (update_output_ratio is called at completion,
        # which also releases budget). Sharing the lock prevents races.
        self.output_ratio = ADAPTIVE_INITIAL_RATIO
        self.ema_alpha = ADAPTIVE_EMA_ALPHA
        self.pressure_reset = ADAPTIVE_PRESSURE_RESET
        self._completion_count = 0

    def _decode_fraction(self) -> float:
        return self.decode_used / self.decode_capacity if self.decode_capacity > 0 else 0.0

    def _gpu_fraction(self) -> float:
        return self.gpu_used / self.gpu_capacity if self.gpu_capacity > 0 else 0.0

    def _kv_fraction(self) -> float:
        return self.kv_used / self.kv_capacity if self.kv_capacity > 0 else 0.0

    def current_load(self) -> float:
        """Current utilization fraction. Used by reactive strategy.

        For compound: returns the worst-case (highest) of the two fractions.
        This means we reject if EITHER resource is overloaded.
        """
        if self.metric == BottleneckMetric.COMPOUND:
            return max(self._gpu_fraction(), self._decode_fraction())
        elif self.metric == BottleneckMetric.DECODE_THROUGHPUT:
            return self._decode_fraction()
        elif self.metric == BottleneckMetric.GPU_COMPUTE:
            return self._gpu_fraction()
        elif self.metric == BottleneckMetric.KV_CACHE:
            return self._kv_fraction()
        return 0.0

    def projected_load(self, costs: dict) -> float:
        """Projected utilization if we admit a request with these costs.

        Used by predictive strategy. Same logic as current_load() but adds
        the incoming request's estimated cost before computing the fraction.
        """
        if self.metric == BottleneckMetric.COMPOUND:
            gpu_proj = (self.gpu_used + costs.get("gpu", 0)) / self.gpu_capacity if self.gpu_capacity > 0 else 0.0
            decode_proj = (self.decode_used + costs.get("decode", 0)) / self.decode_capacity if self.decode_capacity > 0 else 0.0
            return max(gpu_proj, decode_proj)
        elif self.metric == BottleneckMetric.DECODE_THROUGHPUT:
            return (self.decode_used + costs.get("decode", 0)) / self.decode_capacity if self.decode_capacity > 0 else 0.0
        elif self.metric == BottleneckMetric.GPU_COMPUTE:
            return (self.gpu_used + costs.get("gpu", 0)) / self.gpu_capacity if self.gpu_capacity > 0 else 0.0
        elif self.metric == BottleneckMetric.KV_CACHE:
            return (self.kv_used + costs.get("kv", 0)) / self.kv_capacity if self.kv_capacity > 0 else 0.0
        return 0.0

    async def admit(self, job_id: str, costs: dict, max_tokens: int) -> None:
        """Register a request as admitted. Updates budget atomically."""
        async with self._lock:
            self.running[job_id] = {
                "decode_cost": costs.get("decode", 0),
                "gpu_cost": costs.get("gpu", 0),
                "kv_cost": costs.get("kv", 0),
                "remaining_tokens": max_tokens,
                "admitted_at": time.time(),
            }
            self.decode_used += costs.get("decode", 0)
            self.gpu_used += costs.get("gpu", 0)
            self.kv_used += costs.get("kv", 0)

    async def update_progress(self, job_id: str, tokens_generated: int) -> None:
        """Update progress as tokens are generated.

        WHY per-metric behavior differs:
        - decode_throughput: remaining tokens decrease → budget frees progressively
          as the request generates output. This is accurate because decode cost
          is proportional to remaining work.
        - gpu_compute: prefill is a one-time cost at the start. Once prefill is
          done (first token generated), the GPU cost drops. We release GPU budget
          after first token.
        - kv_cache: KV stays allocated until the request completes. No progressive
          release — the cache is needed for the full generation.
        """
        async with self._lock:
            if job_id not in self.running:
                return
            entry = self.running[job_id]

            if self.metric in (BottleneckMetric.DECODE_THROUGHPUT, BottleneckMetric.COMPOUND):
                # Each generated token reduces remaining decode work
                old_remaining = entry["remaining_tokens"]
                new_remaining = max(0, old_remaining - 1)
                freed = old_remaining - new_remaining
                entry["remaining_tokens"] = new_remaining
                self.decode_used = max(0.0, self.decode_used - freed)

            if self.metric in (BottleneckMetric.GPU_COMPUTE, BottleneckMetric.COMPOUND):
                # Prefill is done after first token — release GPU budget
                if tokens_generated == 1 and entry["gpu_cost"] > 0:
                    self.gpu_used = max(0.0, self.gpu_used - entry["gpu_cost"])
                    entry["gpu_cost"] = 0

    async def release(self, job_id: str) -> None:
        """Release all budgets for a completed/failed request."""
        async with self._lock:
            if job_id not in self.running:
                return
            entry = self.running.pop(job_id)
            # Release whatever budget is still held
            self.decode_used = max(0.0, self.decode_used - entry.get("decode_cost", 0))
            self.gpu_used = max(0.0, self.gpu_used - entry.get("gpu_cost", 0))
            self.kv_used = max(0.0, self.kv_used - entry.get("kv_cost", 0))

    async def update_output_ratio(self, max_tokens: int, actual_tokens: int) -> None:
        """Update the output ratio EMA after a request completes.

        SGLang's new_token_ratio concept: learn the actual fraction of max_tokens
        that requests use, then use that fraction for future cost estimates.

        EMA formula: ratio = alpha * sample + (1 - alpha) * ratio
        With alpha=0.1, the ratio is a smoothed trailing average that adapts
        to workload changes within ~10-20 requests.
        """
        async with self._lock:
            if max_tokens > 0 and actual_tokens >= 0:
                sample = min(1.0, max(0.0, actual_tokens / max_tokens))
                self.output_ratio = (
                    self.ema_alpha * sample + (1 - self.ema_alpha) * self.output_ratio
                )
                self._completion_count += 1

    async def signal_pressure(self) -> None:
        """Snap output_ratio upward toward 1.0 on overload signal.

        Called when vLLM returns an error (5xx, timeout, preemption).
        Blends current ratio toward pressure_reset (default 0.8).
        This makes the next admission more conservative immediately,
        without waiting for the EMA to catch up.

        AWS analogy: like a circuit breaker that trips on failure — immediately
        reduces traffic to the backend rather than waiting for the metric to
        reflect the problem.
        """
        async with self._lock:
            self.output_ratio = max(self.output_ratio, self.pressure_reset)

    def stats(self) -> dict:
        """Snapshot for logging and completion messages."""
        return {
            "running_count": len(self.running),
            "decode_used": round(self.decode_used, 1),
            "decode_capacity": self.decode_capacity,
            "gpu_used": round(self.gpu_used, 1),
            "gpu_capacity": self.gpu_capacity,
            "kv_used_mb": round(self.kv_used / 1_000_000, 1),
            "kv_capacity_mb": round(self.kv_capacity / 1_000_000, 1),
            "utilization_pct": round(self.current_load() * 100, 1),
            "output_ratio": round(self.output_ratio, 4),
            "completions": self._completion_count,
        }


# ---------------------------------------------------------------------------
# Cost estimation — what does this request cost in bottleneck units?
# ---------------------------------------------------------------------------
def estimate_request_cost(prompt: str, max_tokens: int, metric: BottleneckMetric) -> dict:
    """Estimate the cost of admitting a request, in metric-appropriate units.

    WHY different metrics have different cost dimensions:
    - decode_throughput: cost is the number of output tokens to generate.
      These tokens consume memory bandwidth one at a time during decode.
    - gpu_compute: cost is the number of input (prompt) tokens to prefill.
      Prefill is compute-bound and proportional to input length.
    - kv_cache: cost is the total KV memory for input + output tokens.
      KV cache holds attention state for ALL tokens (input and output).
    - compound: both gpu and decode costs, checked independently.
    """
    prompt_tokens = len(prompt) / CHARS_PER_TOKEN

    if metric == BottleneckMetric.DECODE_THROUGHPUT:
        return {"decode": float(max_tokens)}
    elif metric == BottleneckMetric.GPU_COMPUTE:
        return {"gpu": prompt_tokens}
    elif metric == BottleneckMetric.KV_CACHE:
        total_tokens = prompt_tokens + max_tokens
        return {"kv": total_tokens * KV_BYTES_PER_TOKEN}
    elif metric == BottleneckMetric.COMPOUND:
        return {"gpu": prompt_tokens, "decode": float(max_tokens)}
    return {}


def estimate_request_cost_adaptive(
    prompt: str, max_tokens: int, metric: BottleneckMetric, output_ratio: float
) -> dict:
    """Like estimate_request_cost but scales output tokens by the learned ratio.

    WHY a separate function: predictive uses max_tokens as worst-case output.
    Adaptive uses max_tokens * output_ratio — the EMA-learned fraction of
    max_tokens that requests actually consume. On 7B where requests use ~35%
    of max_tokens, this reduces cost estimates by ~3x, allowing higher throughput.

    Prompt tokens are still fully counted (they're known exactly from the input).
    GPU_COMPUTE cost is unchanged (prefill is independent of output length).
    """
    prompt_tokens = len(prompt) / CHARS_PER_TOKEN
    estimated_output = max_tokens * output_ratio

    if metric == BottleneckMetric.DECODE_THROUGHPUT:
        return {"decode": estimated_output}
    elif metric == BottleneckMetric.GPU_COMPUTE:
        # WHY not scaled: prefill cost depends on input length, not output.
        return {"gpu": prompt_tokens}
    elif metric == BottleneckMetric.KV_CACHE:
        total_tokens = prompt_tokens + estimated_output
        return {"kv": total_tokens * KV_BYTES_PER_TOKEN}
    elif metric == BottleneckMetric.COMPOUND:
        return {"gpu": prompt_tokens, "decode": estimated_output}
    return {}


# ---------------------------------------------------------------------------
# Admission strategies — each returns (admitted: bool, reason: str)
# ---------------------------------------------------------------------------

async def admit_static(
    tracker: AdmissionTracker, prompt: str, max_tokens: int
) -> tuple[bool, str]:
    """Static strategy — always admit, rely on RabbitMQ prefetch_count.

    WHY this works: prefetch_count=N means RabbitMQ delivers at most N unacked
    messages. The worker processes them concurrently via asyncio.create_task().
    When one finishes (ACK), RabbitMQ delivers the next. Natural flow control
    without any admission overhead.

    Limitation: all requests are treated equal. A 20-token request and a
    2048-token request both consume one slot.
    """
    return True, ""


async def admit_reactive(
    tracker: AdmissionTracker, prompt: str, max_tokens: int
) -> tuple[bool, str]:
    """Reactive strategy — check current system load, reject if above threshold.

    Reads tracker.current_load() which returns the bottleneck utilization
    fraction (0.0-1.0). Does NOT examine the incoming request at all.

    AWS analogy: target tracking on CPU — if CPU > 70%, stop routing.
    The policy doesn't know if the next request is a health check or a
    batch import; it just sees "system is hot, back off."

    WHY this is better than static: adapts to runtime conditions. If previous
    large requests consumed most of the budget, this strategy sees it and
    backs off. Static would keep admitting regardless.

    Limitation: reactive, not predictive. A request admitted at 65% load
    could push actual to 95% if it's a 2048-token request. The strategy
    doesn't account for the COST of the incoming request.
    """
    load = tracker.current_load()
    if load >= ADMISSION_THRESHOLD:
        return False, (
            f"Current load {load:.1%} >= threshold {ADMISSION_THRESHOLD:.1%} "
            f"(metric={tracker.metric.value})"
        )
    return True, ""


async def admit_predictive(
    tracker: AdmissionTracker, prompt: str, max_tokens: int
) -> tuple[bool, str]:
    """Predictive strategy — estimate request cost, project post-admission state.

    Steps:
      1. Estimate this request's cost in bottleneck units
      2. Project: if we admit, what would the load become?
      3. Admit only if projected load < threshold

    WHY this is better than reactive: a 20-token request at 70% load is safe
    to admit, but a 2048-token request at 70% is not. Reactive rejects both
    (or admits both) because it only sees 70%. Predictive sees the difference:
    70% + 20tok_cost < threshold, but 70% + 2048tok_cost > threshold.

    AWS analogy: predictive scaling — "if I route this request here, will
    this instance breach its CPU target?" Rather than waiting to see CPU spike,
    the load balancer pre-computes the impact.

    The cost estimate is approximate:
      - Tokenizer approximation (4 chars/token)
      - max_tokens as worst-case output length
    We intentionally overestimate to be conservative — better to reject one
    safe request than to cause a preemption storm in vLLM.
    """
    metric = BottleneckMetric(BOTTLENECK_METRIC)
    costs = estimate_request_cost(prompt, max_tokens, metric)
    projected = tracker.projected_load(costs)

    if projected >= ADMISSION_THRESHOLD:
        current = tracker.current_load()
        return False, (
            f"Projected load {projected:.1%} >= threshold {ADMISSION_THRESHOLD:.1%} "
            f"(current={current:.1%}, request_cost={costs}, metric={metric.value})"
        )
    return True, ""


async def admit_adaptive(
    tracker: AdmissionTracker, prompt: str, max_tokens: int
) -> tuple[bool, str]:
    """Adaptive strategy — predictive with learned output ratio.

    Same projection logic as predictive, but replaces max_tokens with
    max_tokens * output_ratio in the cost estimate. The output_ratio is
    an EMA that learns from completed requests how much of max_tokens
    is actually used.

    Starts at output_ratio=1.0 (identical to predictive), then converges
    to the true ratio within ~10/alpha completions. For alpha=0.1, that's
    ~100 completions (~2 min at 1 req/s).

    On vLLM errors, the ratio snaps upward (more conservative) via
    signal_pressure(), preventing cascading failures.

    AWS analogy: predictive scaling with a feedback loop — the scaling
    policy adjusts its own target based on observed capacity usage,
    rather than using a fixed target that may be wrong.
    """
    metric = BottleneckMetric(BOTTLENECK_METRIC)
    # WHY read without lock: float read is atomic in CPython (GIL).
    # The ratio may change between read and projected_load(), but the
    # difference is at most one EMA step — negligible.
    output_ratio = tracker.output_ratio
    costs = estimate_request_cost_adaptive(prompt, max_tokens, metric, output_ratio)
    projected = tracker.projected_load(costs)

    if projected >= ADMISSION_THRESHOLD:
        current = tracker.current_load()
        return False, (
            f"Projected load {projected:.1%} >= threshold {ADMISSION_THRESHOLD:.1%} "
            f"(current={current:.1%}, request_cost={costs}, metric={metric.value}, "
            f"output_ratio={output_ratio:.3f})"
        )
    return True, ""


# Strategy dispatch — maps env var value to admission function
STRATEGIES = {
    "static": admit_static,
    "reactive": admit_reactive,
    "predictive": admit_predictive,
    "adaptive": admit_adaptive,
}


# ---------------------------------------------------------------------------
# Job processing — the core inference loop
# ---------------------------------------------------------------------------
async def process_job(
    message: aio_pika.abc.AbstractIncomingMessage,
    redis_client: aioredis.Redis,
    http_client: httpx.AsyncClient,
    admit_fn,
    tracker: AdmissionTracker,
) -> None:
    """Process a single inference job from the queue.

    Steps:
    1. Parse and validate the job JSON
    2. Run the pluggable admission check (local state, no HTTP)
    3. Register with tracker (budget allocated)
    4. Stream completions from vLLM, updating tracker per token
    5. Release tracker budget and ACK/NACK the RabbitMQ message
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

    # --- Admission control (strategy-dependent, local state only) ---
    admitted, reason = await admit_fn(tracker, prompt, max_tokens)
    if not admitted:
        nack_count += 1
        logger.info(
            "Job %s NACK'd [%s] (#%d): %s — requeuing",
            job_id, ADMISSION_STRATEGY, nack_count, reason,
        )
        await message.nack(requeue=True)
        # WHY sleep 100ms: prevents tight requeue loops. Gives GPU time to
        # free resources. Static never reaches here (always admits).
        await asyncio.sleep(0.1)
        return

    admit_count += 1

    # Register with tracker — allocates budget for this request.
    # WHY adaptive uses its own cost function: the budget reservation must match
    # the admission decision. If admit_adaptive() estimated cost using output_ratio,
    # but we reserve using max_tokens, the tracker would over-reserve and the ratio
    # learning would be meaningless.
    metric = BottleneckMetric(BOTTLENECK_METRIC)
    if ADMISSION_STRATEGY == "adaptive":
        costs = estimate_request_cost_adaptive(prompt, max_tokens, metric, tracker.output_ratio)
    else:
        costs = estimate_request_cost(prompt, max_tokens, metric)
    await tracker.admit(job_id, costs, max_tokens)

    logger.info(
        "Job %s ADMITTED [%s] (admit #%d, nack #%d, tracker: %s)",
        job_id, ADMISSION_STRATEGY, admit_count, nack_count,
        tracker.stats(),
    )

    # --- Stream inference from vLLM ---
    # try/finally ensures tracker.release() is called even on error,
    # preventing budget leaks that would permanently reduce capacity.
    tokens_generated = 0
    prefill_ms: Optional[float] = None
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

                        if prefill_ms is None:
                            prefill_ms = (time.time() - inference_start) * 1000
                            logger.info("Job %s TTFT: %.0fms", job_id, prefill_ms)

                        tokens_generated += 1

                        # Update tracker — progressive budget release for decode
                        await tracker.update_progress(job_id, tokens_generated)

                        await redis_client.publish(
                            channel,
                            json.dumps({"token": token_text}),
                        )

        # --- Completion ---
        total_ms = (time.time() - inference_start) * 1000
        estimated_input_tokens = len(prompt) / CHARS_PER_TOKEN
        completion_msg = {
            "status": "complete",
            "metrics": {
                "queue_wait_ms": round(queue_wait_ms, 1),
                # WHY both prefill_ms and ttft_ms: prefill_ms is the server-side
                # time to first token (excludes queue wait). The test runner computes
                # end-to-end TTFT (includes queue) from client timestamps separately.
                "prefill_ms": round(prefill_ms, 1) if prefill_ms else None,
                "ttft_ms": round(prefill_ms, 1) if prefill_ms else None,
                "total_ms": round(total_ms, 1),
                "tokens_generated": tokens_generated,
                "tokens_per_sec": round(
                    tokens_generated / (total_ms / 1000), 1
                ) if total_ms > 0 else 0,
                # Input/output token split for cost analysis
                "input_tokens_est": round(estimated_input_tokens),
                "output_tokens": tokens_generated,
                # Strategy and tracker state for correlation
                "strategy": ADMISSION_STRATEGY,
                "bottleneck_metric": BOTTLENECK_METRIC,
                "worker_nack_count": nack_count,
                "worker_admit_count": admit_count,
                "tracker_utilization": round(tracker.current_load(), 3),
                "output_ratio": round(tracker.output_ratio, 4),
            },
        }

        # Update adaptive output ratio from this request's actual output.
        # WHY after building completion_msg: the ratio in the message reflects
        # the state BEFORE this request's learning, which is what the admission
        # decision was based on. The updated ratio will appear in the NEXT request.
        if ADMISSION_STRATEGY == "adaptive":
            await tracker.update_output_ratio(max_tokens, tokens_generated)

        await redis_client.publish(channel, json.dumps(completion_msg))
        await message.ack()

        logger.info(
            "Job %s complete [%s]: %d tokens in %.0fms (%.1f tok/s, prefill: %.0fms, wait: %.0fms, tracker: %.1f%%)",
            job_id, ADMISSION_STRATEGY, tokens_generated, total_ms,
            completion_msg["metrics"]["tokens_per_sec"],
            prefill_ms or 0, queue_wait_ms,
            tracker.current_load() * 100,
        )

    except Exception as e:
        logger.error("Job %s failed: %s", job_id, e, exc_info=True)

        # Signal pressure to adaptive strategy — vLLM error means the system
        # was overloaded. Snap output_ratio upward so next admissions are more
        # conservative. Requests that fail are dropped (requeue=False) — retry
        # semantics are a Phase 7 production hardening concern.
        if ADMISSION_STRATEGY == "adaptive":
            await tracker.signal_pressure()
            logger.info(
                "Job %s: pressure signal sent, output_ratio snapped to %.3f",
                job_id, tracker.output_ratio,
            )

        error_msg = {"status": "error", "error": str(e)}
        try:
            await redis_client.publish(channel, json.dumps(error_msg))
        except Exception as pub_err:
            logger.error("Failed to publish error for job %s: %s", job_id, pub_err)

        await message.nack(requeue=False)

    finally:
        # ALWAYS release tracker budget — prevents budget leak on error.
        # If process_job crashes between admit() and here, the finally block
        # ensures we don't permanently lose capacity.
        await tracker.release(job_id)


# ---------------------------------------------------------------------------
# Main loop
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

    # Validate bottleneck metric
    try:
        metric = BottleneckMetric(BOTTLENECK_METRIC)
    except ValueError:
        logger.error(
            "Invalid BOTTLENECK_METRIC '%s'. Must be one of: %s",
            BOTTLENECK_METRIC, [m.value for m in BottleneckMetric],
        )
        return

    admit_fn = STRATEGIES[ADMISSION_STRATEGY]

    # Create the admission tracker with configured capacities
    tracker = AdmissionTracker(
        metric=metric,
        decode_capacity=DECODE_CAPACITY,
        gpu_capacity=GPU_COMPUTE_CAPACITY,
        kv_capacity=KV_CACHE_CAPACITY_BYTES,
    )

    logger.info("Worker starting...")
    logger.info(
        "Config: strategy=%s metric=%s threshold=%.0f%% prefetch=%d",
        ADMISSION_STRATEGY, BOTTLENECK_METRIC, ADMISSION_THRESHOLD * 100, PREFETCH_COUNT,
    )
    logger.info(
        "Capacities: decode=%.1f tok/s, gpu_compute=%.1f tok/s, kv_cache=%dMB",
        DECODE_CAPACITY, GPU_COMPUTE_CAPACITY, KV_CACHE_CAPACITY_BYTES // 1_000_000,
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

    # --- Declare queue (idempotent — same params as gateway) ---
    queue = await channel.declare_queue(
        "inference_queue",
        durable=True,
        arguments={"x-max-length": 1000},
    )
    logger.info(
        "Queue 'inference_queue' declared, consuming with strategy '%s', metric '%s'...",
        ADMISSION_STRATEGY, BOTTLENECK_METRIC,
    )

    # --- Consume messages ---
    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            if shutdown_event.is_set():
                logger.info("Shutdown requested — stopping consumption")
                await message.nack(requeue=True)
                break

            asyncio.create_task(
                process_job(message, redis_client, http_client, admit_fn, tracker)
            )

    # --- Cleanup ---
    logger.info("Cleaning up connections...")
    logger.info(
        "Final stats [%s]: admits=%d nacks=%d tracker=%s",
        ADMISSION_STRATEGY, admit_count, nack_count, tracker.stats(),
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
