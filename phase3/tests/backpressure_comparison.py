"""
Phase 3 — Admission Control Pareto Experiment

Compares three admission control strategies under sustained load:
  1. Static (prefetch_count=N, no runtime checks — RabbitMQ is the gate)
  2. Reactive (local tracker monitors current load — "is the system hot?")
  3. Predictive (local tracker + request cost estimation — "will admitting
     this request make the system hot?")

MODES:
  Grid mode (legacy): 3 strategies × 4 workloads = 12 runs
  Sweep mode (new): For each strategy, sweep the aggressiveness parameter
    to generate a Pareto frontier (throughput vs latency tradeoff).

BOTTLENECK METRICS (sweep across rounds):
  Round A: decode_throughput — T4 memory bandwidth during decode
  Round B: gpu_compute — T4 GPU SMs during prefill
  Round C: compound — max(gpu, throughput) — realistic both-phase gate
  Round D: kv_cache — 14B model with tight KV headroom

7 METRICS PER DATA POINT:
  1. TTFT P95 (end-to-end, includes queue wait)
  2. Queue time P95 (RabbitMQ wait only)
  3. Prefill time P95 (server-side, no queue)
  4. TPOT P95 (time per output token during decode)
  5. Throughput (tok/s)
  6. $/input token
  7. $/output token

Usage:
    # Start port-forward first
    bash phase3/scripts/port-forward.sh

    # Grid mode (legacy): 3 strategies × 4 workloads
    python3 phase3/tests/backpressure_comparison.py --host http://localhost:8080

    # Sweep mode: Pareto frontier for decode_throughput bottleneck
    python3 phase3/tests/backpressure_comparison.py --host http://localhost:8080 \
        --sweep --bottleneck-metric decode_throughput

    # Sweep with custom parameters
    python3 phase3/tests/backpressure_comparison.py --host http://localhost:8080 \
        --sweep --sweep-static 1,2,3,5,8,10,15,20 \
        --sweep-reactive 0.50,0.60,0.70,0.80,0.90
"""

import json
import time
import threading
import argparse
import subprocess
import statistics
import sys
import os
import random
from typing import Optional
from datetime import datetime, timezone

import requests

HOST = ""
GRAFANA_URL = ""  # Set from --grafana arg

# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------
# WHY these specific prompts: they create measurably different KV cache pressure.
# Short: fast prefill (~3 tokens), tiny KV footprint → tests throughput ceiling
# Medium: moderate prefill (~50 tokens), moderate KV → baseline production workload
# Long: heavy prefill (~300 tokens), large KV footprint → tests admission control quality

PROMPTS = {
    "short": {
        "text": "What is 2+2?",
        "max_tokens": 20,
        "label": "S",
        "description": "~3 input tokens, 20 max output",
    },
    "medium": {
        "text": (
            "Explain the concept of auto scaling in cloud computing. "
            "Cover the key components including scaling policies, "
            "cooldown periods, health checks, and how metrics like "
            "CPU utilization and queue depth drive scaling decisions. "
            "Be concise."
        ),
        "max_tokens": 150,
        "label": "M",
        "description": "~50 input tokens, 150 max output",
    },
    "long": {
        "text": (
            "You are an expert in distributed systems and cloud infrastructure. "
            "I need you to analyze the following scenario and provide recommendations. "
            "We have a web application serving 10,000 requests per second across "
            "3 availability zones. The application consists of a load balancer layer, "
            "an application tier running on auto-scaled EC2 instances, a caching layer "
            "using ElastiCache Redis, and a database tier using RDS PostgreSQL with "
            "read replicas. Recently we've been experiencing intermittent latency spikes "
            "where P99 latency jumps from 50ms to 500ms during peak hours. The CPU "
            "utilization on application instances stays below 40% during these spikes. "
            "Memory utilization is at 60%. The Redis cache hit rate drops from 95% to "
            "80% during spikes. Database connections occasionally hit the max limit. "
            "What are the most likely root causes and what specific changes would you "
            "recommend to fix this? Prioritize your recommendations by impact."
        ),
        "max_tokens": 300,
        "label": "L",
        "description": "~300 input tokens, 300 max output",
    },
    "very_long": {
        "text": (
            "You are a principal engineer at a large cloud provider designing "
            "the next-generation inference serving platform. The platform must "
            "serve multiple foundation models (7B to 70B parameters) across a "
            "heterogeneous GPU fleet (NVIDIA T4, A10G, A100, H100) with strict "
            "latency SLAs. Write a comprehensive technical design document that "
            "covers the following areas in detail:\n\n"
            "1. REQUEST ROUTING AND LOAD BALANCING: Design a multi-tier routing "
            "system that accounts for model placement, GPU memory availability, "
            "KV cache locality, and request priority. Explain how you would handle "
            "prefix caching across replicas, session affinity for multi-turn "
            "conversations, and graceful degradation when capacity is exhausted. "
            "Compare hash-based routing vs least-loaded vs power-of-two-choices "
            "for this workload.\n\n"
            "2. MEMORY MANAGEMENT AND KV CACHE: Design the KV cache management "
            "strategy including PagedAttention block allocation, cross-request "
            "prefix sharing, preemption policies (FIFO vs priority vs cost-based), "
            "and multi-tier caching (GPU VRAM → CPU RAM → NVMe → network). "
            "Calculate the memory budget for a 70B model on 8xH100 with tensor "
            "parallelism, showing how many concurrent requests of various context "
            "lengths can be served simultaneously.\n\n"
            "3. SCHEDULING AND BATCHING: Design the continuous batching scheduler "
            "that handles heterogeneous request sizes (32 to 32768 context tokens). "
            "Explain chunked prefill, how to balance prefill vs decode throughput, "
            "and how to prevent long-context prefills from starving short decode "
            "steps. Include a mathematical model of throughput as a function of "
            "batch size, sequence length, and model size.\n\n"
            "4. AUTOSCALING AND CAPACITY PLANNING: Design the autoscaling system "
            "that handles GPU provisioning latency (5-10 minutes for new nodes). "
            "Cover predictive scaling based on traffic patterns, warm pools of "
            "pre-loaded models, bin-packing multiple small models on large GPUs, "
            "and Spot instance interruption handling. Define the metrics and "
            "thresholds for scaling decisions.\n\n"
            "5. OBSERVABILITY AND DEBUGGING: Design the metrics, logging, and "
            "tracing infrastructure for debugging inference latency issues. "
            "Cover GPU-level metrics (SM utilization, memory bandwidth, thermal), "
            "engine-level metrics (KV cache hit rate, preemption count, batch "
            "efficiency), and request-level metrics (TTFT, TBT, queue wait). "
            "Explain how to correlate a user-reported latency spike to a specific "
            "GPU, request batch, or model loading event.\n\n"
            "For each area, provide concrete numbers, formulas, and architectural "
            "diagrams. Reference real systems (vLLM, TensorRT-LLM, Triton, SGLang) "
            "where applicable. Justify every design choice with quantitative reasoning."
        ),
        "max_tokens": 2048,
        "label": "XL",
        "description": "~1000 input tokens, 2048 max output — KV cache stress test",
        # WHY 1000+2048=3048 tokens: at 401KB per token, one XL request uses 1.22GB
        # of KV cache. With 8 concurrent (MAX_BATCH_SIZE), that's 9.8GB = 102% of
        # the T4's 9.6GB KV budget. This WILL trigger the 80% admission threshold
        # in strategies 2+3. Finally.
    },
}

# SLA targets per prompt type — the objective function for strategy comparison.
# WHY define SLAs: without targets, you can't say which strategy is "best."
# These are realistic targets for an interactive inference API.
SLAS = {
    "short":     {"complete_time_p99_sec": 5},
    "medium":    {"complete_time_p99_sec": 15},
    "long":      {"complete_time_p99_sec": 30},
    "very_long": {"complete_time_p99_sec": 90},
}

# Cost constants for $/1M token tracking
GPU_COST_PER_HOUR = 0.22  # g4dn.xlarge Spot average (us-east-1)

# Workload patterns — designed to stress different bottlenecks.
#
# WHY Pareto distributions: real production traffic follows power-law distributions
# (80% short, 15% medium, 4% long, 1% very_long). Uniform and fixed-pattern
# mixes hide scheduling pathologies that only appear with realistic variance.
#
# WHY kv_stress and burst_outlier: previous experiments showed 0 NACKs because
# KV cache never exceeded 25%. The very_long (XL) prompt uses 3048 tokens =
# 1.22GB KV per request. At 8 concurrent XL, that's 102% KV → triggers admission.
#
# Each workload can define either "pattern" (cyclic) or "distribution" (weighted
# random with seed=42 for reproducibility).
WORKLOADS = {
    "pareto_light": {
        "description": "0.8 req/s Pareto mix — realistic production at ~80% GPU",
        "distribution": {"short": 0.80, "medium": 0.15, "long": 0.04, "very_long": 0.01},
        "rate": 0.8,
    },
    "pareto_heavy": {
        "description": "1.2 req/s Pareto mix — realistic production overloaded",
        "distribution": {"short": 0.80, "medium": 0.15, "long": 0.04, "very_long": 0.01},
        "rate": 1.2,
    },
    "kv_stress": {
        "description": "0.5 req/s all XL — forces KV cache toward 80% threshold",
        # WHY 0.5 req/s: slow rate but huge per-request cost. Each XL request
        # uses 1.22GB KV. At prefetch=10, up to 10 concurrent = 12.2GB > 9.6GB.
        # Even at 5 concurrent = 6.1GB = 64% KV. With output generation building
        # up more KV over time, this should approach the 80% threshold.
        "pattern": ["very_long"],
        "rate": 0.5,
    },
    "kv_stress_14b": {
        "description": "0.15 req/s all XL — KV stress for 14B (128s/request)",
        # WHY 0.15 req/s: 14B model generates at 15.9 tok/s → each XL request
        # takes ~128s. At 0.5 req/s the queue backs up faster than the worker
        # can drain it, causing SSE timeouts. 0.15 req/s means max backlog of
        # ~19 requests per 128s window, keeping SSE wait under timeout.
        # At prefetch=8+, 8 concurrent XL = 4.7GB KV = 89% of 5.3GB budget.
        "pattern": ["very_long"],
        "rate": 0.15,
    },
    "burst_outlier": {
        "description": "1.0 req/s 90% short + periodic XL bursts every 30s",
        # WHY burst pattern: steady stream of cheap shorts (GPU happy), then a
        # sudden burst of 5 XL requests that consume 6GB+ KV at once.
        # Static admits all 5 XL immediately → KV spike → potential preemption.
        # Per-request estimates cost → defers some XL → smoother KV curve.
        # The "burst" key tells run_sustained_load to inject XL bursts.
        "distribution": {"short": 0.90, "medium": 0.05, "long": 0.04, "very_long": 0.01},
        "burst": {"type": "very_long", "count": 5, "interval_sec": 30},
        "rate": 1.0,
    },
}

STRATEGIES = ["static", "reactive", "predictive", "adaptive"]


# ---------------------------------------------------------------------------
# Request sender — fires one request and collects all metrics
# ---------------------------------------------------------------------------
def send_request(
    request_id: int,
    prompt_key: str,
    results: list,
    run_start_time: float,
) -> None:
    """Send a single request through the queue pipeline, collect timing metrics.

    This function runs in its own thread. Results are stored in results[request_id].
    """
    prompt_config = PROMPTS[prompt_key]
    submit_time = time.perf_counter()
    submit_wall = time.time()

    token_times = []
    tokens = []
    error = None
    server_metrics = {}

    try:
        # Step 1: Submit job to queue
        submit_resp = requests.post(
            f"{HOST}/generate",
            json={
                "prompt": prompt_config["text"],
                "max_tokens": prompt_config["max_tokens"],
                "temperature": 0.7,
            },
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        if submit_resp.status_code != 202:
            raise RuntimeError(
                f"Expected 202, got {submit_resp.status_code}: {submit_resp.text}"
            )

        job_data = submit_resp.json()
        stream_url = job_data["stream_url"]
        job_id = job_data.get("job_id", "unknown")

        # Step 2: Stream results via SSE
        stream_resp = requests.get(
            f"{HOST}{stream_url}",
            stream=True,
            headers={"Accept": "text/event-stream"},
            timeout=(10, 600),
        )

        if stream_resp.status_code != 200:
            raise RuntimeError(
                f"Stream returned {stream_resp.status_code}: {stream_resp.text}"
            )

        for line in stream_resp.iter_lines():
            if not line:
                continue
            line_str = line.decode()
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if "token" in data:
                token_times.append(time.perf_counter())
                tokens.append(data["token"])

            if data.get("status") == "complete":
                server_metrics = data.get("metrics", {})
                break

            if data.get("status") == "error":
                error = data.get("error", "unknown server error")
                break

    except Exception as e:
        error = str(e)

    request_end = time.perf_counter()

    # Build result record
    if error or not token_times:
        results[request_id] = {
            "request_id": request_id,
            "prompt_type": prompt_key,
            "label": prompt_config["label"],
            "error": error or "no tokens received",
            "total_sec": round(request_end - submit_time, 3),
            "submit_offset_sec": round(submit_wall - run_start_time, 2),
        }
        return

    client_ttft = token_times[0] - submit_time
    # TBT = inter-token intervals (time between consecutive tokens)
    tbts = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]

    results[request_id] = {
        "request_id": request_id,
        "prompt_type": prompt_key,
        "label": prompt_config["label"],
        "submit_offset_sec": round(submit_wall - run_start_time, 2),
        "client_ttft_sec": round(client_ttft, 4),
        "client_tbts": [round(t, 5) for t in tbts],
        "client_avg_tbt_sec": round(statistics.mean(tbts), 5) if tbts else None,
        "client_total_sec": round(request_end - submit_time, 3),
        "tokens_out": len(tokens),
        "client_tokens_per_sec": round(
            len(tokens) / (request_end - token_times[0]), 1
        ) if len(token_times) > 1 else 0,
        # For cost-per-token tracking: approximate input tokens from char length
        "prompt_char_len": len(prompt_config["text"]),
        "max_tokens_requested": prompt_config["max_tokens"],
        # Server-side metrics from the worker (via Redis pub/sub completion message)
        "server_queue_wait_ms": server_metrics.get("queue_wait_ms"),
        "server_ttft_ms": server_metrics.get("ttft_ms"),
        "server_prefill_ms": server_metrics.get("prefill_ms"),
        "server_total_ms": server_metrics.get("total_ms"),
        "server_tokens_per_sec": server_metrics.get("tokens_per_sec"),
        "server_input_tokens_est": server_metrics.get("input_tokens_est"),
        "server_output_tokens": server_metrics.get("output_tokens"),
        "server_strategy": server_metrics.get("strategy"),
        "server_bottleneck_metric": server_metrics.get("bottleneck_metric"),
        "server_nack_count": server_metrics.get("worker_nack_count"),
        "server_admit_count": server_metrics.get("worker_admit_count"),
        "server_tracker_utilization": server_metrics.get("tracker_utilization"),
    }


# ---------------------------------------------------------------------------
# Open-loop load generator
# ---------------------------------------------------------------------------
def _select_prompt(workload_config: dict, request_index: int, rng: random.Random,
                    elapsed_sec: float) -> str:
    """Select a prompt type based on workload config (cyclic, distribution, or burst).

    Supports three modes:
      - "pattern": cyclic list (e.g., ["short", "medium", "long"])
      - "distribution": weighted random (e.g., {"short": 0.80, "medium": 0.15, ...})
      - "burst": periodic injection of expensive requests on top of distribution
    """
    burst = workload_config.get("burst")
    if burst:
        interval = burst["interval_sec"]
        count = burst["count"]
        # Inject a burst of XL requests at each interval boundary
        # WHY modular arithmetic: at t=30s, 60s, 90s etc., the next `count` requests
        # after the boundary are forced to the burst type. This creates predictable
        # spikes in an otherwise steady-state stream.
        time_in_cycle = elapsed_sec % interval
        if time_in_cycle < (count / 2.0):  # ~first 2.5s of each 30s window
            return burst["type"]

    if "distribution" in workload_config:
        dist = workload_config["distribution"]
        types = list(dist.keys())
        weights = list(dist.values())
        return rng.choices(types, weights=weights, k=1)[0]

    pattern = workload_config["pattern"]
    return pattern[request_index % len(pattern)]


def run_sustained_load(
    workload_key: str,
    rate: float,
    duration_sec: int,
    run_start_time: float,
) -> list:
    """Send requests at a fixed rate for a given duration (open-loop).

    WHY open-loop: closed-loop testing (wait for response before sending next)
    underestimates latency under load. Open-loop sends at a fixed rate regardless
    of response time, creating realistic backpressure. This is the same principle
    as wrk2 vs wrk, or Vegeta vs ab.

    Supports three prompt selection modes:
      - Cyclic pattern (deterministic)
      - Weighted random distribution (Pareto, seeded for reproducibility)
      - Burst injection (periodic XL spikes on top of distribution)

    Returns list of result dicts (one per request, including errors).
    """
    workload_config = WORKLOADS[workload_key]
    interval = 1.0 / rate
    total_requests = int(rate * duration_sec)
    # WHY fixed seed: Pareto workloads use random selection. Seed ensures
    # all 3 strategies see the exact same request sequence for fair comparison.
    rng = random.Random(42)

    results = [None] * total_requests
    threads = []

    # Describe the workload pattern
    if "distribution" in workload_config:
        dist = workload_config["distribution"]
        desc = " ".join(f"{v:.0%} {k}" for k, v in dist.items())
        if workload_config.get("burst"):
            b = workload_config["burst"]
            desc += f" + burst of {b['count']}x {b['type']} every {b['interval_sec']}s"
        print(f"    Sending {total_requests} requests at {rate} req/s for {duration_sec}s...")
        print(f"    Distribution: {desc}")
    else:
        pattern = workload_config["pattern"]
        print(f"    Sending {total_requests} requests at {rate} req/s for {duration_sec}s...")
        print(f"    Pattern: {' -> '.join(pattern)} (cycling)")

    for i in range(total_requests):
        elapsed = time.time() - run_start_time
        prompt_key = _select_prompt(workload_config, i, rng, elapsed)
        t = threading.Thread(
            target=send_request,
            args=(i, prompt_key, results, run_start_time),
        )
        threads.append(t)
        t.start()

        # Print progress every 20 requests
        if (i + 1) % 20 == 0:
            elapsed = time.time() - run_start_time
            print(f"\r    [{elapsed:>5.0f}s] Sent {i + 1}/{total_requests}", end="", flush=True)

        # WHY sleep(interval) between sends: maintains the target rate.
        # Thread start overhead is ~0.1ms, negligible vs 500ms interval at 2 req/s.
        if i < total_requests - 1:
            time.sleep(interval)

    print(f"\n    All {total_requests} requests submitted. Waiting for completions...")

    # Wait for all requests to complete (with timeout).
    # WHY 600s: 14B XL requests take ~128s to generate. At high prefetch
    # with queue backlog, a request can wait 5+ minutes before being picked up.
    for t in threads:
        t.join(timeout=600)

    # Filter out None entries (shouldn't happen, but defensive)
    completed = [r for r in results if r is not None]
    errors = [r for r in completed if "error" in r]
    successes = [r for r in completed if "error" not in r]

    print(f"    Completed: {len(successes)} success, {len(errors)} errors, "
          f"{total_requests - len(completed)} timeouts")

    return completed


# ---------------------------------------------------------------------------
# Steady-state analysis — the core metrics computation
# ---------------------------------------------------------------------------
def analyze_steady_state(
    results: list,
    warmup_sec: float = 30.0,
    drain_sec: float = 15.0,
    duration_sec: float = 120.0,
) -> dict:
    """Analyze metrics from the steady-state window only.

    WHY discard warm-up and drain:
      - First 30s: GPU is ramping up, queue is building, not representative
      - Last 15s: load generator has stopped, queue is draining, not representative
      - Middle 75s: GPU is fully saturated, strategies show their true behavior

    This is the same principle as JMH warm-up iterations in Java benchmarking.
    """
    steady_start = warmup_sec
    steady_end = duration_sec - drain_sec

    # Filter to requests submitted during steady-state
    steady = [
        r for r in results
        if "error" not in r
        and steady_start <= r["submit_offset_sec"] <= steady_end
    ]
    all_errors = [r for r in results if "error" in r]
    steady_errors = [
        r for r in all_errors
        if steady_start <= r.get("submit_offset_sec", 0) <= steady_end
    ]

    if not steady:
        return {
            "steady_state_requests": 0,
            "note": "No successful requests in steady-state window",
        }

    # TTFT — time to first token (client-side, includes queue wait + network)
    ttfts = sorted([r["client_ttft_sec"] for r in steady])

    # Client-side TBT — kept in raw output but NOT used in comparison table
    # (too noisy through port-forward). Flattened from per-request arrays.
    all_tbts = []
    for r in steady:
        all_tbts.extend(r.get("client_tbts", []))
    all_tbts.sort()

    # Server-side prefill times (time to first token on server, excludes queue)
    prefill_times_ms = sorted([
        r["server_prefill_ms"] for r in steady
        if r.get("server_prefill_ms") is not None
    ])

    # TPOT — Time Per Output Token (decode phase only, excludes prefill).
    # WHY not server_total_ms / tokens: that amortizes prefill across all tokens,
    # inflating the per-token time for short outputs. True TPOT isolates decode:
    #   TPOT = (total_ms - prefill_ms) / (output_tokens - 1)
    # The -1 is because the first output token arrives at prefill_ms, so there
    # are (N-1) inter-token intervals for N output tokens.
    tpot_ms_list = sorted([
        (r["server_total_ms"] - r["server_prefill_ms"]) / max(r.get("server_output_tokens", r.get("tokens_out", 0)) - 1, 1)
        for r in steady
        if r.get("server_total_ms") and r.get("server_prefill_ms") is not None
        and r.get("server_output_tokens", r.get("tokens_out", 0)) > 1
    ])

    # Legacy server-estimated TBT (amortized, kept for backward compat)
    server_est_tbts_ms = sorted([
        r["server_total_ms"] / r["tokens_out"]
        for r in steady
        if r.get("server_total_ms") and r.get("tokens_out", 0) > 0
    ])

    # Server-side queue wait times
    queue_waits = sorted([
        r["server_queue_wait_ms"] for r in steady
        if r.get("server_queue_wait_ms") is not None
    ])

    # Throughput: total tokens generated in steady-state / steady-state duration
    total_tokens = sum(r["tokens_out"] for r in steady)
    steady_duration = steady_end - steady_start
    throughput_tok_per_sec = total_tokens / steady_duration if steady_duration > 0 else 0

    # Per-second throughput for variance analysis
    tokens_per_sec_list = [r.get("client_tokens_per_sec", 0) for r in steady]

    # NACK count — use delta (max - min) to get NACKs for THIS run only.
    # WHY delta: the worker's nack_count is a module-level counter that resets on
    # pod restart (strategy switch) but accumulates across the 3 workload runs
    # within a strategy. max(nack_counts) would give cumulative total, inflating
    # workload-2 and workload-3 counts. max - min gives per-run NACKs.
    nack_counts = [r.get("server_nack_count", 0) for r in steady if r.get("server_nack_count") is not None]
    total_nacks = (max(nack_counts) - min(nack_counts)) if nack_counts else 0

    def percentiles(data, pcts=(50, 95, 99)):
        """Compute percentiles from sorted data."""
        if not data:
            return {f"p{p}": None for p in pcts}
        result = {}
        for p in pcts:
            idx = int(len(data) * p / 100)
            idx = min(idx, len(data) - 1)
            result[f"p{p}"] = round(data[idx], 4)
        return result

    # Tracker utilization snapshots
    tracker_utils = sorted([
        r["server_tracker_utilization"] for r in steady
        if r.get("server_tracker_utilization") is not None
    ])

    return {
        "steady_state_requests": len(steady),
        "steady_state_errors": len(steady_errors),
        "total_requests": len(results),
        "total_errors": len(all_errors),
        "completion_rate": round(len(steady) / (len(steady) + len(steady_errors)), 3) if (len(steady) + len(steady_errors)) > 0 else 0,
        "ttft": {
            **percentiles(ttfts),
            "min": round(min(ttfts), 4),
            "max": round(max(ttfts), 4),
        },
        "prefill_ms": {
            **{k: round(v, 1) if v is not None else None
               for k, v in percentiles(prefill_times_ms).items()},
            "min": round(min(prefill_times_ms), 1) if prefill_times_ms else None,
            "max": round(max(prefill_times_ms), 1) if prefill_times_ms else None,
        },
        "tpot_ms": {
            **{k: round(v, 2) if v is not None else None
               for k, v in percentiles(tpot_ms_list).items()},
            "min": round(min(tpot_ms_list), 2) if tpot_ms_list else None,
            "max": round(max(tpot_ms_list), 2) if tpot_ms_list else None,
        },
        "tbt_client": {
            **percentiles(all_tbts),
            "min": round(min(all_tbts), 5) if all_tbts else None,
            "max": round(max(all_tbts), 5) if all_tbts else None,
            "count": len(all_tbts),
            "note": "Client-side through port-forward — high jitter, use tpot_ms instead",
        },
        "server_est_tbt_ms": {
            **{k: round(v, 1) if v is not None else None
               for k, v in percentiles(server_est_tbts_ms).items()},
            "min": round(min(server_est_tbts_ms), 1) if server_est_tbts_ms else None,
            "max": round(max(server_est_tbts_ms), 1) if server_est_tbts_ms else None,
        },
        "queue_wait_ms": {
            **{k: round(v, 1) if v is not None else None
               for k, v in percentiles(queue_waits).items()},
            "min": round(min(queue_waits), 1) if queue_waits else None,
            "max": round(max(queue_waits), 1) if queue_waits else None,
        },
        "throughput": {
            "total_tokens": total_tokens,
            "tok_per_sec_avg": round(throughput_tok_per_sec, 1),
            "per_request_tok_per_sec_avg": round(statistics.mean(tokens_per_sec_list), 1) if tokens_per_sec_list else 0,
        },
        "tracker_utilization": {
            **{k: round(v * 100, 1) if v is not None else None
               for k, v in percentiles(tracker_utils).items()},
        } if tracker_utils else {},
        "nack_count": total_nacks,
        # SLA compliance — measures against defined targets per prompt type
        "sla": _compute_sla(steady, steady_errors),
        # Cost per token — connects strategy choice to infrastructure economics
        "cost": _compute_cost(steady, steady_duration),
    }


def _compute_sla(steady: list, steady_errors: list) -> dict:
    """Compute SLA compliance per prompt type and overall.

    SLA = P99 complete_time < target for each prompt tier.
    Compliance % = fraction of requests that completed within target.
    """
    sla_result = {}
    all_within = 0
    all_total = 0

    for prompt_type, targets in SLAS.items():
        target_sec = targets["complete_time_p99_sec"]
        # Get all requests of this type (successes + errors)
        type_ok = [r for r in steady if r.get("prompt_type") == prompt_type]
        type_err = [r for r in steady_errors if r.get("prompt_type") == prompt_type]

        if not type_ok and not type_err:
            continue

        # Complete time = client_total_sec for successes; errors count as SLA miss
        complete_times = sorted([r["client_total_sec"] for r in type_ok])
        p99_idx = min(int(len(complete_times) * 0.99), len(complete_times) - 1) if complete_times else 0
        p99_complete = complete_times[p99_idx] if complete_times else None

        # Compliance: requests that completed within SLA target
        within_sla = sum(1 for t in complete_times if t <= target_sec)
        total_for_type = len(type_ok) + len(type_err)
        compliance = within_sla / total_for_type if total_for_type > 0 else 0

        sla_result[prompt_type] = {
            "p99_complete_sec": round(p99_complete, 2) if p99_complete else None,
            "target_sec": target_sec,
            "met": p99_complete <= target_sec if p99_complete else False,
            "compliance_pct": round(compliance * 100, 1),
            "count": total_for_type,
        }

        all_within += within_sla
        all_total += total_for_type

    sla_result["overall_compliance_pct"] = round(
        all_within / all_total * 100, 1
    ) if all_total > 0 else 0

    return sla_result


def _compute_cost(steady: list, steady_duration: float) -> dict:
    """Compute cost-per-token metrics.

    WHY this matters: strategy choice affects throughput, which affects how many
    tokens you get per GPU-hour. Lower throughput = higher $/token = more expensive
    infrastructure. This connects admission control to infrastructure economics.
    """
    if not steady or steady_duration <= 0:
        return {}

    gpu_cost = (steady_duration / 3600) * GPU_COST_PER_HOUR

    # Use server-side input/output token counts when available, fall back to estimates
    total_input_tokens = sum(
        r.get("server_input_tokens_est") or r.get("prompt_char_len", 0) / 4
        for r in steady
    )
    total_output_tokens = sum(
        r.get("server_output_tokens") or r.get("tokens_out", 0)
        for r in steady
    )
    total_tokens = total_input_tokens + total_output_tokens

    return {
        "gpu_cost_usd": round(gpu_cost, 4),
        "total_input_tokens_est": int(total_input_tokens),
        "total_output_tokens": int(total_output_tokens),
        "cost_per_1m_input_usd": round(gpu_cost / total_input_tokens * 1_000_000, 2) if total_input_tokens > 0 else 0,
        "cost_per_1m_output_usd": round(gpu_cost / total_output_tokens * 1_000_000, 2) if total_output_tokens > 0 else 0,
        "cost_per_1m_total_usd": round(gpu_cost / total_tokens * 1_000_000, 2) if total_tokens > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Strategy switching — patches the K8s deployment
# ---------------------------------------------------------------------------
def patch_worker_config(
    strategy: str,
    prefetch_count: int = 10,
    admission_threshold: float = 0.80,
    bottleneck_metric: str = "decode_throughput",
    decode_capacity: float = 500.0,
    gpu_compute_capacity: float = 300.0,
    kv_cache_capacity_bytes: int = 9_600_000_000,
    ema_alpha: float = None,
) -> bool:
    """Patch the vllm-worker deployment with updated admission config.

    WHY kubectl set env: faster than applying a full manifest. The deployment
    controller detects the env change and triggers a rolling update. The old pod
    terminates gracefully (SIGTERM → drain in-flight requests) and the new pod
    starts with the updated config.

    WHY all env vars at once: kubectl batches them into one Deployment spec
    update → one rolling update, not N separate rollouts.

    Returns True if successful.
    """
    print(f"\n  Patching worker: strategy={strategy}, threshold={admission_threshold:.2f}, "
          f"metric={bottleneck_metric}, prefetch={prefetch_count}")
    try:
        env_args = [
            f"ADMISSION_STRATEGY={strategy}",
            f"PREFETCH_COUNT={prefetch_count}",
            f"ADMISSION_THRESHOLD={admission_threshold}",
            f"BOTTLENECK_METRIC={bottleneck_metric}",
            f"DECODE_CAPACITY={decode_capacity}",
            f"GPU_COMPUTE_CAPACITY={gpu_compute_capacity}",
            f"KV_CACHE_CAPACITY_BYTES={kv_cache_capacity_bytes}",
        ]
        # WHY conditional: only set adaptive env vars when running adaptive strategy.
        # Other strategies ignore these, but setting them causes an unnecessary
        # rolling update if the value differs from what's currently deployed.
        if ema_alpha is not None:
            env_args.append(f"ADAPTIVE_EMA_ALPHA={ema_alpha}")
        result = subprocess.run(
            [
                "kubectl", "set", "env",
                "deployment/vllm-worker",
                "-n", "inference-system",
                "-c", "queue-worker",
            ] + env_args,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"  ERROR: kubectl set env failed: {result.stderr}")
            return False
        print(f"  Deployment patched. Forcing rollout restart...")

        # WHY rollout restart after set env: kubectl set env updates the
        # deployment spec, but with maxSurge=0 on a single GPU node, the old
        # pod may keep serving from a stale ReplicaSet. rollout restart forces
        # the pod to be recreated with the new env vars. Discovered in the
        # first 7B sweep where predictive/adaptive runs silently ran as reactive.
        result = subprocess.run(
            [
                "kubectl", "rollout", "restart",
                "deployment/vllm-worker",
                "-n", "inference-system",
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"  WARNING: rollout restart failed: {result.stderr}")

        # Wait for the rollout to complete (new pod with updated env)
        result = subprocess.run(
            [
                "kubectl", "rollout", "status",
                "deployment/vllm-worker",
                "-n", "inference-system",
                "--timeout=600s",
            ],
            capture_output=True, text=True, timeout=630,
        )
        if result.returncode != 0:
            print(f"  WARNING: Rollout may not have completed: {result.stderr}")
            return False

        print(f"  Config applied — rollout complete")
        return True

    except Exception as e:
        print(f"  ERROR patching worker config: {e}")
        return False


def switch_strategy(strategy: str) -> bool:
    """Legacy wrapper — calls patch_worker_config with default params."""
    return patch_worker_config(strategy=strategy)


def verify_queue_empty() -> bool:
    """Check that the inference_queue is empty before starting a new run.

    WHY: leftover messages from a previous run would contaminate the next run's
    metrics. Like draining an SQS queue between load test iterations.
    """
    try:
        result = subprocess.run(
            [
                "kubectl", "exec",
                "deployment/rabbitmq",
                "-n", "inference-system",
                "--",
                "rabbitmqctl", "list_queues",
                "name", "messages", "consumers",
            ],
            capture_output=True, text=True, timeout=15,
        )
        for line in result.stdout.strip().split("\n"):
            if "inference_queue" in line:
                parts = line.split()
                messages = int(parts[1]) if len(parts) > 1 else 0
                if messages > 0:
                    print(f"  WARNING: Queue has {messages} pending messages")
                    return False
                return True
        return True  # Queue doesn't exist yet — that's fine
    except Exception as e:
        print(f"  WARNING: Could not check queue: {e}")
        return True  # Proceed anyway


def wait_for_queue_drain(timeout: int = 60) -> bool:
    """Wait for the inference queue to fully drain."""
    start = time.time()
    while time.time() - start < timeout:
        if verify_queue_empty():
            return True
        time.sleep(5)
    return False


# ---------------------------------------------------------------------------
# KEDA pause/resume — prevent autoscaling interference during comparison
# ---------------------------------------------------------------------------
def pause_keda() -> bool:
    """Pause KEDA autoscaling by annotating the ScaledObject.

    WHY: KEDA watches RabbitMQ queue depth and scales vllm-worker between 1-2
    replicas. During the comparison test, we need exactly 1 replica with a known
    strategy. KEDA scaling mid-run would contaminate metrics.

    Uses the paused-replicas annotation (KEDA 2.8+) which freezes the replica
    count without deleting the ScaledObject. Like pausing an ASG scaling policy
    instead of deleting the ASG.
    """
    print("Pausing KEDA autoscaling...")
    try:
        result = subprocess.run(
            [
                "kubectl", "annotate", "scaledobject",
                "vllm-worker-scaler",
                "-n", "inference-system",
                "autoscaling.keda.sh/paused-replicas=1",
                "--overwrite",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            # KEDA may not be installed, or ScaledObject may not exist — that's OK
            print(f"  WARNING: Could not pause KEDA: {result.stderr.strip()}")
            print("  (KEDA may not be installed — proceeding without pausing)")
            return False
        print("  KEDA paused — replica count frozen at 1")
        return True
    except Exception as e:
        print(f"  WARNING: Could not pause KEDA: {e}")
        return False


def resume_keda() -> bool:
    """Resume KEDA autoscaling by removing the paused annotation."""
    print("Resuming KEDA autoscaling...")
    try:
        result = subprocess.run(
            [
                "kubectl", "annotate", "scaledobject",
                "vllm-worker-scaler",
                "-n", "inference-system",
                "autoscaling.keda.sh/paused-replicas-",
                "--overwrite",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            print(f"  WARNING: Could not resume KEDA: {result.stderr.strip()}")
            return False
        print("  KEDA resumed — autoscaling re-enabled")
        return True
    except Exception as e:
        print(f"  WARNING: Could not resume KEDA: {e}")
        return False


# ---------------------------------------------------------------------------
# Grafana screenshot capture
# ---------------------------------------------------------------------------
# All 14 dashboard panels — capture everything for a complete record.
# Panel IDs from phase2/k8s/monitoring/grafana-dashboard-configmap.yaml.
GRAFANA_PANELS = {
    # Row 1: GPU Hardware (DCGM)
    1: "gpu_compute_utilization",     # SM utilization % — is GPU saturated?
    2: "memory_bandwidth_utilization",# Memory BW % — decode is BW-bound on T4
    3: "vram_usage",                  # VRAM (MiB) — model + KV cache footprint
    4: "gpu_temperature",             # Temperature — thermal throttle risk
    # Row 2: Inference Engine (vLLM)
    5: "kv_cache_usage",              # KV Cache % — admission control trigger
    6: "requests_running_waiting",    # Running vs Waiting — concurrency pattern
    7: "throughput_tok_s",            # Throughput — token generation rate
    # Row 3: Latency & Traffic
    8: "ttft",                        # Time to First Token — p50 + p99
    9: "tbt",                         # Time Between Tokens — streaming quality
    10: "request_rate_errors",        # Request rate + error codes
    # Row 4: Queue & Scaling
    11: "queue_depth",                # RabbitMQ queue depth — backpressure signal
    12: "queue_ready_unacked",        # Ready vs Unacked — prefetch behavior
    13: "worker_replicas",            # Desired vs Ready replicas
    14: "gpu_nodes",                  # Karpenter-provisioned GPU node count
}


def capture_grafana_screenshots(
    strategy: str,
    workload: str,
    run_start_epoch_ms: int,
    run_end_epoch_ms: int,
    output_dir: str,
) -> dict:
    """Capture Grafana panel screenshots for a completed test run.

    Uses the Grafana render API to export panel PNGs. If the Image Renderer
    plugin isn't installed (common in default kube-prometheus-stack), falls back
    to saving time-windowed dashboard URLs the user can open manually.

    WHY capture after each run: GPU metrics are ephemeral. Prometheus default
    retention is 10d, but if the cluster is torn down, the data is lost. PNGs
    are permanent artifacts. Like saving CloudWatch dashboards as PDFs.

    Returns dict with panel names → file paths (or URLs on fallback).
    """
    if not GRAFANA_URL:
        return {"skipped": "No --grafana URL provided"}

    run_dir = os.path.join(output_dir, f"{strategy}_{workload}")
    os.makedirs(run_dir, exist_ok=True)

    # Grafana API auth — kube-prometheus-stack defaults
    auth = ("admin", "inference-lab")
    dashboard_uid = "inference-platform"

    results = {}
    render_available = True

    for panel_id, panel_name in GRAFANA_PANELS.items():
        filename = f"{panel_name}.png"
        filepath = os.path.join(run_dir, filename)

        if render_available:
            # Try the render API (requires Grafana Image Renderer plugin)
            render_url = (
                f"{GRAFANA_URL}/render/d/{dashboard_uid}/inference-platform"
                f"?panelId={panel_id}&width=1000&height=500"
                f"&from={run_start_epoch_ms}&to={run_end_epoch_ms}"
                f"&tz=UTC"
            )
            try:
                resp = requests.get(render_url, auth=auth, timeout=30)
                if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                    results[panel_name] = filepath
                    continue
                else:
                    # Render API not available — switch to URL fallback for all panels
                    render_available = False
            except Exception:
                render_available = False

        # Fallback: save a clickable Grafana URL with the exact time window
        # WHY time window in URL: Grafana's ?from=&to= params lock the dashboard
        # to the exact period of this test run. User opens the link and sees the
        # metrics from that specific run, not the current time.
        dashboard_url = (
            f"{GRAFANA_URL}/d/{dashboard_uid}/inference-platform"
            f"?from={run_start_epoch_ms}&to={run_end_epoch_ms}"
            f"&viewPanel={panel_id}"
        )
        results[panel_name] = dashboard_url

    # Save a manifest file with all panel URLs/paths for this run
    manifest_path = os.path.join(run_dir, "panels.json")
    manifest = {
        "strategy": strategy,
        "workload": workload,
        "time_range": {
            "from_epoch_ms": run_start_epoch_ms,
            "to_epoch_ms": run_end_epoch_ms,
            "from_iso": datetime.fromtimestamp(run_start_epoch_ms / 1000, tz=timezone.utc).isoformat(),
            "to_iso": datetime.fromtimestamp(run_end_epoch_ms / 1000, tz=timezone.utc).isoformat(),
        },
        "dashboard_url": (
            f"{GRAFANA_URL}/d/{dashboard_uid}/inference-platform"
            f"?from={run_start_epoch_ms}&to={run_end_epoch_ms}"
        ),
        "panels": results,
        "render_available": render_available,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if render_available:
        saved_count = sum(1 for v in results.values() if v.endswith(".png"))
        print(f"  Grafana: {saved_count} panel screenshots saved to {run_dir}/")
    else:
        print(f"  Grafana: Image Renderer not available — dashboard URLs saved to {manifest_path}")
        print(f"           Open: {manifest['dashboard_url']}")

    return results


# ---------------------------------------------------------------------------
# Incremental result saving — persist after each run
# ---------------------------------------------------------------------------
def save_incremental_results(all_results: dict, output_path: str) -> None:
    """Save results to disk after each run completes.

    WHY incremental: a 9-run comparison takes ~25 min. If the test crashes on
    run 7, the first 6 runs' data is lost without incremental saves. Like
    checkpointing a long-running batch job.

    Saves to the same output path — each write overwrites with the latest state.
    Raw TBT arrays are stripped to keep file size reasonable.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    save_results = {}
    for key, val in all_results.items():
        save_copy = json.loads(json.dumps(val))
        for r in save_copy.get("raw_results", []):
            r.pop("client_tbts", None)
        save_results[key] = save_copy

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)


# ---------------------------------------------------------------------------
# Pipeline readiness probe + strategy verification
# ---------------------------------------------------------------------------
def verify_pipeline_ready(expected_strategy: str, max_retries: int = 18) -> bool:
    """Send a test request through the full pipeline to verify readiness.

    WHY not just sleep(30s): a hardcoded sleep doesn't verify anything. This probe
    confirms vLLM is serving, queue worker is consuming, Redis pub/sub is flowing,
    AND the correct strategy is active. Like a synthetic health check in production.

    Returns True if the pipeline completed a request with the expected strategy.
    """
    for attempt in range(1, max_retries + 1):
        print(f"  Pipeline probe attempt {attempt}/{max_retries}...")
        try:
            # Submit a short request. WHY max_tokens=50 (not 5): with max_tokens=5,
            # the worker completes in ~150ms — before the stream client connects to
            # Redis pub/sub. The tokens are published and lost. 50 tokens takes ~1.5s,
            # giving the stream client time to subscribe.
            submit_resp = requests.post(
                f"{HOST}/generate",
                json={"prompt": "Count from 1 to 50", "max_tokens": 50, "temperature": 0.1},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if submit_resp.status_code != 202:
                print(f"    Submit failed: {submit_resp.status_code}")
                time.sleep(10)
                continue

            job_data = submit_resp.json()
            stream_url = job_data["stream_url"]

            # Stream the response
            stream_resp = requests.get(
                f"{HOST}{stream_url}",
                stream=True,
                headers={"Accept": "text/event-stream"},
                timeout=(10, 60),
            )

            got_tokens = False
            actual_strategy = None

            for line in stream_resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode()
                if not line_str.startswith("data: "):
                    continue
                try:
                    data = json.loads(line_str[6:])
                except json.JSONDecodeError:
                    continue

                if "token" in data:
                    got_tokens = True
                if data.get("status") == "complete":
                    actual_strategy = data.get("metrics", {}).get("strategy")
                    break
                if data.get("status") == "error":
                    print(f"    Probe error: {data.get('error')}")
                    break

            if got_tokens and actual_strategy == expected_strategy:
                print(f"    Pipeline ready — strategy '{actual_strategy}' confirmed")
                return True
            elif got_tokens and actual_strategy != expected_strategy:
                print(f"    Pipeline responded but strategy mismatch: "
                      f"expected '{expected_strategy}', got '{actual_strategy}'")
                # May be a stale pod still draining — wait and retry
                time.sleep(10)
            else:
                print(f"    No tokens received — pipeline not ready yet")
                time.sleep(10)

        except Exception as e:
            print(f"    Probe failed: {e}")
            time.sleep(10)

    print(f"  WARNING: Pipeline readiness probe failed after {max_retries} attempts")
    return False


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------
def print_run_summary(analysis: dict, strategy: str, workload: str):
    """Print a formatted summary of one run's steady-state metrics."""
    print(f"\n  {'='*70}")
    print(f"  Strategy: {strategy:12s}  |  Workload: {workload}")
    print(f"  {'='*70}")

    if analysis.get("steady_state_requests", 0) == 0:
        print("  No successful requests in steady-state window!")
        return

    print(f"  Requests: {analysis['steady_state_requests']} steady-state "
          f"({analysis['total_requests']} total, "
          f"{analysis['total_errors']} errors, "
          f"{analysis['completion_rate']:.0%} completion rate)")

    ttft = analysis["ttft"]
    print(f"  TTFT:     p50={ttft['p50']:.3f}s  p95={ttft['p95']:.3f}s  "
          f"p99={ttft['p99']:.3f}s  max={ttft['max']:.3f}s")

    pf = analysis.get("prefill_ms", {})
    if pf.get("p50") is not None:
        print(f"  Prefill:  p50={pf['p50']:.1f}ms  p95={pf['p95']:.1f}ms  "
              f"p99={pf['p99']:.1f}ms  (server-side, no queue)")

    tpot = analysis.get("tpot_ms", {})
    if tpot.get("p50") is not None:
        print(f"  TPOT:     p50={tpot['p50']:.2f}ms  p95={tpot['p95']:.2f}ms  "
              f"p99={tpot['p99']:.2f}ms  (decode only, excludes prefill)")

    qw = analysis["queue_wait_ms"]
    if qw["p50"] is not None:
        print(f"  QueueWait: p50={qw['p50']:.0f}ms  p95={qw['p95']:.0f}ms  "
              f"p99={qw['p99']:.0f}ms  max={qw['max']:.0f}ms")

    tu = analysis.get("tracker_utilization", {})
    if tu.get("p50") is not None:
        print(f"  Tracker:  p50={tu['p50']:.1f}%  p95={tu['p95']:.1f}%  "
              f"p99={tu['p99']:.1f}%  utilization")

    tp = analysis["throughput"]
    print(f"  Throughput: {tp['tok_per_sec_avg']:.1f} tok/s aggregate  "
          f"({tp['per_request_tok_per_sec_avg']:.1f} tok/s per-request avg)")

    print(f"  NACKs:    {analysis['nack_count']} total requeue events")

    # SLA compliance
    sla = analysis.get("sla", {})
    overall = sla.get("overall_compliance_pct", "N/A")
    sla_parts = []
    for pt in ["short", "medium", "long", "very_long"]:
        if pt in sla:
            s = sla[pt]
            met = "MET" if s["met"] else "MISS"
            sla_parts.append(f"{pt}={s['compliance_pct']:.0f}%({met})")
    if sla_parts:
        print(f"  SLA:      overall {overall}% | {' '.join(sla_parts)}")

    # Cost
    cost = analysis.get("cost", {})
    if cost.get("cost_per_1m_output_usd"):
        print(f"  Cost:     ${cost['cost_per_1m_output_usd']:.2f}/1M output tok  "
              f"${cost['cost_per_1m_total_usd']:.2f}/1M total tok")


def print_comparison_table(all_results: dict):
    """Print the final 3x3 comparison matrix."""
    print("\n" + "=" * 100)
    print("COMPARISON MATRIX — Steady-State Metrics")
    print("=" * 100)

    # Build workload list from results (preserves order of actual runs)
    workload_names = list(dict.fromkeys(
        v["workload"] for v in all_results.values()
    ))
    col_width = max(16, max((len(w) for w in workload_names), default=14))

    def print_metric_table(title, metric_path, fmt, unit=""):
        """Print one metric row across all workloads × strategies."""
        header = f"\n{title:>30s}"
        for w in workload_names:
            header += f"  {w:>{col_width}s}"
        print(header)
        print(f"  {'-' * (28 + (col_width + 2) * len(workload_names))}")
        for strategy in STRATEGIES:
            row = f"  {strategy:>28s}"
            for workload in workload_names:
                key = f"{strategy}_{workload}"
                analysis = all_results.get(key, {}).get("analysis", {})
                # Navigate nested dict path like "ttft.p50"
                val = analysis
                for part in metric_path.split("."):
                    if isinstance(val, dict):
                        val = val.get(part)
                    else:
                        val = None
                        break
                if val is not None and not isinstance(val, dict):
                    formatted = fmt.format(val)
                    row += f"  {formatted:>{col_width}s}"
                else:
                    row += f"  {'N/A':>{col_width}s}"
            print(row)

    print_metric_table("TTFT p95 (seconds)", "ttft.p95", "{:.3f}s")
    print_metric_table("Prefill p95 (ms)", "prefill_ms.p95", "{:.1f}ms")
    print_metric_table("TPOT p95 (ms)", "tpot_ms.p95", "{:.2f}ms")
    print_metric_table("Queue Wait p95 (ms)", "queue_wait_ms.p95", "{:.0f}ms")
    print_metric_table("Throughput (tok/s)", "throughput.tok_per_sec_avg", "{:.1f}")
    print_metric_table("NACK count", "nack_count", "{:d}")
    print_metric_table("Completion Rate", "completion_rate", "{:.0%}")
    print_metric_table("SLA Compliance %", "sla.overall_compliance_pct", "{:.0f}%")
    print_metric_table("$/1M input tokens", "cost.cost_per_1m_input_usd", "${:.2f}")
    print_metric_table("$/1M output tokens", "cost.cost_per_1m_output_usd", "${:.2f}")
    print_metric_table("Tracker util p50 (%)", "tracker_utilization.p50", "{:.1f}%")

    # Footnotes
    print(f"\n  {'─' * (28 + (col_width + 2) * len(workload_names))}")
    print("  NOTES:")
    print("  - TPOT = (total_ms - prefill_ms) / (output_tokens - 1). Decode-only, no prefill.")
    print("  - Predictive strategy estimates request cost before admitting.")
    print("    Higher NACK count = more cautious admission.")
    print("  - NACK counts are per-run deltas, not cumulative across workloads.")
    print("  - Tracker utilization shows local AdmissionTracker load (no remote polling).")
    print("  - SLA targets: short <5s, medium <15s, long <30s, very_long <90s P99 complete time.")
    print("  - Cost assumes g4dn.xlarge Spot at $0.22/hr. GPU always on during steady-state.")


# ---------------------------------------------------------------------------
# Baseline calibration — measure uncontested TPOT for SLA target
# ---------------------------------------------------------------------------
def run_baseline_calibration(num_requests: int = 10) -> float:
    """Run sequential single requests to measure uncontested TPOT.

    WHY: SLA target should be relative to hardware capability, not absolute.
    2× baseline TPOT P95 means "we tolerate 2× degradation from ideal."
    This makes the experiment portable across GPU types.

    Returns baseline TPOT in milliseconds.
    """
    print("\n" + "=" * 60)
    print("  BASELINE CALIBRATION — measuring uncontested TPOT")
    print("=" * 60)

    tpot_values = []
    for i in range(num_requests):
        results = [None]
        send_request(0, "medium", results, time.time())
        r = results[0]
        if r and "error" not in r and r.get("server_total_ms") and r.get("server_prefill_ms"):
            output_tokens = r.get("server_output_tokens", r.get("tokens_out", 0))
            if output_tokens > 1:
                tpot = (r["server_total_ms"] - r["server_prefill_ms"]) / (output_tokens - 1)
                tpot_values.append(tpot)
                print(f"  Request {i+1}/{num_requests}: TPOT={tpot:.2f}ms "
                      f"(prefill={r['server_prefill_ms']:.0f}ms, {output_tokens} tokens)")
        time.sleep(0.5)  # Brief pause between requests

    if not tpot_values:
        print("  WARNING: No valid TPOT measurements — using default 50ms")
        return 50.0

    tpot_values.sort()
    p95_idx = min(int(len(tpot_values) * 0.95), len(tpot_values) - 1)
    baseline_tpot = tpot_values[p95_idx]
    sla_target = baseline_tpot * 2

    print(f"\n  Baseline TPOT P95: {baseline_tpot:.2f}ms")
    print(f"  SLA target (2×):   {sla_target:.2f}ms")
    print(f"  Measurements: {len(tpot_values)} valid out of {num_requests} attempts")

    return baseline_tpot


# ---------------------------------------------------------------------------
# Pareto sweep — sweep aggressiveness per strategy
# ---------------------------------------------------------------------------
def run_pareto_sweep(
    args,
    bottleneck_metric: str,
    static_prefetch_values: list,
    reactive_thresholds: list,
    predictive_thresholds: list,
    adaptive_alphas: list = None,
    workload: str = "pareto_heavy",
    baseline_tpot_ms: float = None,
    kv_cache_capacity_bytes: int = 9_600_000_000,
) -> dict:
    """Run a full Pareto frontier sweep for one bottleneck metric.

    For each (strategy, parameter_value), patches the worker config and
    runs one load test. Returns dict of all results keyed by
    "{strategy}_{param_value}".

    Each data point has 7 metric dimensions for Pareto plotting.
    """
    all_results = {}
    run_number = 0

    # Build sweep plan
    sweep_plan = []
    for pf in static_prefetch_values:
        sweep_plan.append(("static", f"prefetch_{pf}", {"prefetch_count": pf}))
    for th in reactive_thresholds:
        sweep_plan.append(("reactive", f"threshold_{th:.2f}", {"admission_threshold": th}))
    for th in predictive_thresholds:
        sweep_plan.append(("predictive", f"threshold_{th:.2f}", {"admission_threshold": th}))
    for alpha in (adaptive_alphas or []):
        sweep_plan.append(("adaptive", f"alpha_{alpha:.2f}", {"ema_alpha": alpha}))

    total_runs = len(sweep_plan)
    total_time_est = total_runs * (args.duration + args.cooldown) / 60

    print("\n" + "=" * 100)
    print(f"  PARETO SWEEP — bottleneck_metric={bottleneck_metric}")
    print("=" * 100)
    print(f"  Workload:   {workload}")
    print(f"  Runs:       {total_runs} ({len(static_prefetch_values)} static + "
          f"{len(reactive_thresholds)} reactive + {len(predictive_thresholds)} predictive + "
          f"{len(adaptive_alphas or [])} adaptive)")
    print(f"  Est. time:  ~{total_time_est:.0f} minutes")
    if baseline_tpot_ms:
        print(f"  Baseline TPOT: {baseline_tpot_ms:.2f}ms, SLA target: {baseline_tpot_ms * 2:.2f}ms")
    print("=" * 100)

    for strategy, param_label, config_overrides in sweep_plan:
        run_number += 1
        run_key = f"{strategy}_{param_label}"

        print(f"\n{'#'*80}")
        print(f"  SWEEP {run_number}/{total_runs}: {strategy} — {param_label} — {bottleneck_metric}")
        print(f"{'#'*80}")

        # Patch worker config
        if not args.skip_switch:
            success = patch_worker_config(
                strategy=strategy,
                prefetch_count=config_overrides.get("prefetch_count", 10),
                admission_threshold=config_overrides.get("admission_threshold", 0.80),
                bottleneck_metric=bottleneck_metric,
                kv_cache_capacity_bytes=kv_cache_capacity_bytes,
                ema_alpha=config_overrides.get("ema_alpha"),
            )
            if not success:
                print(f"  SKIPPING {run_key} — could not patch worker")
                continue

            if not verify_pipeline_ready(expected_strategy=strategy):
                print(f"  WARNING: Pipeline probe failed — proceeding anyway")

        # Verify queue is empty
        print("  Checking queue is empty...")
        if not verify_queue_empty():
            wait_for_queue_drain(timeout=60)

        # Run the load test
        run_rate = args.rate or WORKLOADS[workload]["rate"]
        run_start_time = time.time()
        results = run_sustained_load(
            workload_key=workload,
            rate=run_rate,
            duration_sec=args.duration,
            run_start_time=run_start_time,
        )

        analysis = analyze_steady_state(
            results,
            warmup_sec=args.warmup,
            drain_sec=args.drain,
            duration_sec=float(args.duration),
        )

        print_run_summary(analysis, strategy, f"{param_label} ({bottleneck_metric})")

        run_end_time = time.time()

        all_results[run_key] = {
            "strategy": strategy,
            "param_label": param_label,
            "workload": workload,
            "bottleneck_metric": bottleneck_metric,
            "config": {
                "rate": run_rate,
                "duration": args.duration,
                "warmup": args.warmup,
                "drain": args.drain,
                **config_overrides,
            },
            "analysis": analysis,
            "raw_results": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline_tpot_ms": baseline_tpot_ms,
        }

        # Grafana screenshots
        grafana_results = capture_grafana_screenshots(
            strategy=strategy,
            workload=f"{bottleneck_metric}_{param_label}",
            run_start_epoch_ms=int(run_start_time * 1000),
            run_end_epoch_ms=int(run_end_time * 1000),
            output_dir=args.screenshots_dir,
        )
        all_results[run_key]["grafana"] = grafana_results

        # Save incrementally
        output_path = args.output.replace(".json", f"_{bottleneck_metric}.json")
        save_incremental_results(all_results, output_path)
        print(f"  Results saved to {output_path} ({run_number}/{total_runs} runs)")

        # Cooldown
        if run_number < total_runs:
            print(f"\n  Cooling down {args.cooldown}s...")
            time.sleep(args.cooldown)

    return all_results


# ---------------------------------------------------------------------------
# Main — orchestrates grid or sweep mode
# ---------------------------------------------------------------------------
def main():
    global HOST
    parser = argparse.ArgumentParser(
        description="Phase 3 — Admission Control Pareto Experiment"
    )
    parser.add_argument(
        "--host", default="http://localhost:8080",
        help="API gateway URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--rate", type=float, default=None,
        help="Override request rate for all workloads (default: use per-workload rate)",
    )
    parser.add_argument(
        "--duration", type=int, default=90,
        help="Duration per run in seconds (default: 90)",
    )
    parser.add_argument(
        "--warmup", type=float, default=20.0,
        help="Warm-up period to discard in seconds (default: 20)",
    )
    parser.add_argument(
        "--drain", type=float, default=10.0,
        help="Drain period to discard in seconds (default: 10)",
    )
    parser.add_argument(
        "--strategy", choices=STRATEGIES, default=None,
        help="Run only this strategy (default: all three)",
    )
    parser.add_argument(
        "--workload", choices=list(WORKLOADS.keys()), default=None,
        help="Run only this workload (default: all four)",
    )
    parser.add_argument(
        "--cooldown", type=int, default=30,
        help="Seconds between runs for queue drain + GPU cool (default: 30)",
    )
    parser.add_argument(
        "--skip-switch", action="store_true",
        help="Don't kubectl switch strategies (for local testing)",
    )
    parser.add_argument(
        "--output", default="phase3/tests/backpressure_results.json",
        help="Output file for raw results",
    )
    parser.add_argument(
        "--grafana", default="http://localhost:3000",
        help="Grafana URL for screenshot capture (default: http://localhost:3000)",
    )
    parser.add_argument(
        "--no-grafana", action="store_true",
        help="Skip Grafana screenshot capture",
    )
    parser.add_argument(
        "--screenshots-dir", default="phase3/screenshots",
        help="Directory for Grafana screenshots (default: phase3/screenshots)",
    )
    # Sweep mode arguments
    parser.add_argument(
        "--sweep", action="store_true",
        help="Enable Pareto frontier sweep mode",
    )
    parser.add_argument(
        "--sweep-workload", default="pareto_heavy",
        choices=list(WORKLOADS.keys()),
        help="Workload to use for sweep (default: pareto_heavy)",
    )
    parser.add_argument(
        "--sweep-static", default="1,2,3,5,8,10,15,20",
        help="Comma-separated prefetch_count values for static sweep",
    )
    parser.add_argument(
        "--sweep-reactive", default="0.50,0.60,0.70,0.80,0.90",
        help="Comma-separated threshold values for reactive sweep",
    )
    parser.add_argument(
        "--sweep-predictive", default="0.50,0.60,0.70,0.80,0.90",
        help="Comma-separated threshold values for predictive sweep",
    )
    parser.add_argument(
        "--sweep-adaptive", default="0.01,0.05,0.10,0.20,0.30",
        help="Comma-separated EMA alpha values for adaptive sweep (learning rate)",
    )
    parser.add_argument(
        "--bottleneck-metric", default="decode_throughput",
        choices=["decode_throughput", "gpu_compute", "kv_cache", "compound"],
        help="Bottleneck metric for sweep (default: decode_throughput)",
    )
    parser.add_argument(
        "--baseline-tpot-ms", type=float, default=None,
        help="Manual baseline TPOT in ms (skip calibration if provided)",
    )
    parser.add_argument(
        "--skip-calibration", action="store_true",
        help="Skip baseline calibration (use default or manual TPOT)",
    )
    parser.add_argument(
        "--kv-cache-capacity-bytes", type=int, default=9_600_000_000,
        help="KV cache capacity in bytes (default 9.6GB for 7B; use ~5.3GB for 14B)",
    )

    args = parser.parse_args()
    HOST = args.host.rstrip("/")
    global GRAFANA_URL
    GRAFANA_URL = "" if args.no_grafana else args.grafana.rstrip("/")

    # WHY set context explicitly: when multiple EKS clusters exist (phase3, phase4),
    # background processes or other terminals can change the default kubectl context.
    # This caused the first sweep to silently deploy to phase4 mid-run. Setting the
    # context at startup ensures all kubectl calls target the right cluster.
    print("Setting kubectl context to inference-phase3...")
    try:
        ctx_result = subprocess.run(
            ["kubectl", "config", "use-context",
             "arn:aws:eks:us-east-1:019167255542:cluster/inference-phase3"],
            capture_output=True, text=True, timeout=10,
        )
        if ctx_result.returncode != 0:
            # Try updating kubeconfig if context doesn't exist
            subprocess.run(
                ["aws", "eks", "update-kubeconfig",
                 "--name", "inference-phase3", "--region", "us-east-1"],
                capture_output=True, text=True, timeout=30,
            )
        current_ctx = subprocess.run(
            ["kubectl", "config", "current-context"],
            capture_output=True, text=True, timeout=10,
        )
        print(f"  Context: {current_ctx.stdout.strip()}")
        if "inference-phase3" not in current_ctx.stdout:
            print("  ERROR: Failed to set context to inference-phase3!")
            sys.exit(1)
    except Exception as e:
        print(f"  WARNING: Could not verify kubectl context: {e}")
    print()

    # Health check
    print("Checking API gateway health...")
    try:
        health = requests.get(f"{HOST}/health", timeout=5)
        if health.status_code == 200:
            print(f"  API gateway: {health.json()}")
        else:
            print(f"  WARNING: health returned {health.status_code}: {health.text}")
    except Exception as e:
        print(f"  ERROR: Cannot reach API gateway at {HOST}: {e}")
        print("  Is port-forward running? (bash phase3/scripts/port-forward.sh)")
        sys.exit(1)
    print()

    # Pause KEDA to prevent autoscaling interference.
    if not args.skip_switch:
        pause_keda()
    print()

    # ===== SWEEP MODE =====
    if args.sweep:
        # Baseline calibration
        baseline_tpot_ms = args.baseline_tpot_ms
        if not baseline_tpot_ms and not args.skip_calibration:
            baseline_tpot_ms = run_baseline_calibration()

        static_values = [int(x) for x in args.sweep_static.split(",")]
        reactive_thresholds = [float(x) for x in args.sweep_reactive.split(",")]
        predictive_thresholds = [float(x) for x in args.sweep_predictive.split(",")]
        adaptive_alphas = [float(x) for x in args.sweep_adaptive.split(",")]

        try:
            sweep_results = run_pareto_sweep(
                args=args,
                bottleneck_metric=args.bottleneck_metric,
                static_prefetch_values=static_values,
                reactive_thresholds=reactive_thresholds,
                predictive_thresholds=predictive_thresholds,
                adaptive_alphas=adaptive_alphas,
                workload=args.sweep_workload,
                baseline_tpot_ms=baseline_tpot_ms,
                kv_cache_capacity_bytes=args.kv_cache_capacity_bytes,
            )

            # Print sweep summary
            print("\n" + "=" * 100)
            print(f"  PARETO SWEEP COMPLETE — {args.bottleneck_metric}")
            print("=" * 100)
            for key, val in sweep_results.items():
                a = val.get("analysis", {})
                tpot = a.get("tpot_ms", {}).get("p95", "N/A")
                throughput = a.get("throughput", {}).get("tok_per_sec_avg", "N/A")
                nacks = a.get("nack_count", 0)
                print(f"  {key:40s}  throughput={throughput} tok/s  TPOT_p95={tpot}ms  NACKs={nacks}")

        finally:
            if not args.skip_switch:
                resume_keda()

        print(f"\nSweep results saved to {args.output.replace('.json', f'_{args.bottleneck_metric}.json')}")
        print("\nSweep complete!")
        return

    # ===== GRID MODE (legacy) =====
    strategies = [args.strategy] if args.strategy else STRATEGIES
    workloads = [args.workload] if args.workload else list(WORKLOADS.keys())

    total_runs = len(strategies) * len(workloads)
    total_time_est = total_runs * (args.duration + args.cooldown) / 60

    print("=" * 100)
    print("  Phase 3 — Admission Control Strategy Comparison")
    print("=" * 100)
    print(f"  Target:     {HOST}")
    print(f"  Rate:       {args.rate or 'per-workload'} req/s")
    print(f"  Duration:   {args.duration}s per run ({args.warmup}s warmup, {args.drain}s drain)")
    print(f"  Strategies: {strategies}")
    print(f"  Workloads:  {workloads}")
    for w in workloads:
        wd = WORKLOADS[w]
        r = args.rate or wd["rate"]
        print(f"    {w:20s}  {r} req/s  {wd['description']}")
    print(f"  Total runs: {total_runs}")
    print(f"  Est. time:  ~{total_time_est:.0f} minutes")
    print(f"  Output:     {args.output}")
    print(f"  Grafana:    {GRAFANA_URL or 'disabled (--no-grafana)'}")
    print(f"  Screenshots: {args.screenshots_dir}")
    print("=" * 100)
    print()

    all_results = {}
    run_number = 0

    try:
        for strategy in strategies:
            # Switch worker to this strategy
            if not args.skip_switch:
                success = switch_strategy(strategy)
                if not success:
                    print(f"  SKIPPING strategy {strategy} — could not switch worker")
                    continue

                if not verify_pipeline_ready(expected_strategy=strategy):
                    print(f"  WARNING: Pipeline probe failed for strategy '{strategy}'")
                    print(f"  Proceeding anyway — results may be from wrong strategy")

            for workload in workloads:
                run_number += 1
                run_key = f"{strategy}_{workload}"

                print(f"\n{'#'*80}")
                print(f"  RUN {run_number}/{total_runs}: strategy={strategy}  workload={workload}")
                print(f"{'#'*80}")

                # Verify queue is empty before starting
                print("  Checking queue is empty...")
                if not verify_queue_empty():
                    print(f"  Waiting for queue to drain (up to 60s)...")
                    if not wait_for_queue_drain(timeout=60):
                        print("  WARNING: Queue not empty, proceeding anyway")

                run_rate = args.rate or WORKLOADS[workload]["rate"]

                run_start_time = time.time()
                results = run_sustained_load(
                    workload_key=workload,
                    rate=run_rate,
                    duration_sec=args.duration,
                    run_start_time=run_start_time,
                )

                analysis = analyze_steady_state(
                    results,
                    warmup_sec=args.warmup,
                    drain_sec=args.drain,
                    duration_sec=float(args.duration),
                )

                print_run_summary(analysis, strategy, workload)

                run_end_time = time.time()

                all_results[run_key] = {
                    "strategy": strategy,
                    "workload": workload,
                    "config": {
                        "rate": run_rate,
                        "duration": args.duration,
                        "warmup": args.warmup,
                        "drain": args.drain,
                    },
                    "analysis": analysis,
                    "raw_results": results,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                grafana_results = capture_grafana_screenshots(
                    strategy=strategy,
                    workload=workload,
                    run_start_epoch_ms=int(run_start_time * 1000),
                    run_end_epoch_ms=int(run_end_time * 1000),
                    output_dir=args.screenshots_dir,
                )
                all_results[run_key]["grafana"] = grafana_results

                save_incremental_results(all_results, args.output)
                print(f"  Results saved to {args.output} ({run_number}/{total_runs} runs)")

                if run_number < total_runs:
                    print(f"\n  Cooling down {args.cooldown}s before next run "
                          f"(queue drain + GPU cool)...")
                    time.sleep(args.cooldown)

    finally:
        if not args.skip_switch:
            resume_keda()

    print_comparison_table(all_results)

    save_incremental_results(all_results, args.output)

    print(f"\nResults saved to {args.output}")
    if GRAFANA_URL:
        print(f"Grafana screenshots saved to {args.screenshots_dir}/")
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
