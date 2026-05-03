"""
Phase 4 — KEDA Scaling Policy Comparison Experiment

Compares four KEDA autoscaling policies under graduated load:
  A. Queue-Only (value=5)    — baseline from Phase 2
  B. Queue-Only (value=3)    — eager: scale sooner on queue depth alone
  C. Composite Conservative  — queue(5) + KV cache > 0.80
  D. Composite Aggressive    — queue(5) + KV cache > 0.65

TEST DESIGN — WHY GRADUATED LOAD WITH PARETO DISTRIBUTION:
  Real inference traffic follows a Pareto pattern: most requests are short,
  but a long tail of XL requests (1000+ tokens) dominate GPU time and KV
  cache usage.  A simple cycling pattern (S -> M -> L) misses this — it
  gives every policy the same predictable mix.  Pareto distribution (80%
  short, 15% medium, 4% long, 1% XL) surfaces the difference between
  queue-only policies (blind to memory) and composite policies (react to
  KV cache pressure from XL requests).

LOAD PHASES (per policy, 7 phases):
  | Phase      | Rate    | Duration | Workload     | Purpose                                          |
  |------------|---------|----------|--------------|--------------------------------------------------|
  | 1. Light   | 0.8/s   | 120s     | Pareto       | Baseline — should NOT trigger scale-up            |
  | 2. Moderate| 1.5/s   | 120s     | Pareto       | Edge — XL requests build queue                    |
  | 3. Heavy   | 2.5/s   | 300s     | Pareto       | Exceeds 1-worker capacity, forces scale-up        |
  | 4. Sustained| 3.0/s  | 420s     | Pareto       | Multi-worker steady state                         |
  | 5. KV Stress| 0.5/s  | 180s     | All-XL       | Forces KV cache to 80%, differentiates composite  |
  | 6. Burst   | 1.0/s   | 180s     | 90% S + XL   | Tests spike handling with XL bursts every 30s     |
  | 7. Cooldown| 0.0/s   | 480s     | None         | Observe scale-down (300s cooldown + observation)  |

SLA + COST TRACKING:
  Each prompt size has a P99 complete-time SLA.  After each phase we report
  SLA compliance % and cost-per-million-tokens so policies can be compared
  on both quality and efficiency.

GRAFANA SNAPSHOTS:
  After each phase, captures a dashboard screenshot via Grafana Image Renderer.
  Stored in phase4.5/tests/snapshots/policy-{a,b,c,d}_phase-{1..7}.png

Usage:
    # Start port-forward first
    bash phase3/scripts/port-forward.sh

    # Run all 4 policies
    python3 phase4.5/tests/scaling_policy_comparison.py --host http://localhost:8080

    # Run a single policy for debugging
    python3 phase4.5/tests/scaling_policy_comparison.py --host http://localhost:8080 \
        --policy policy-a

    # Scale durations down for a quick smoke test
    python3 phase4.5/tests/scaling_policy_comparison.py --host http://localhost:8080 \
        --duration-multiplier 0.5

    # Skip Grafana snapshots (e.g., if renderer not deployed)
    python3 phase4.5/tests/scaling_policy_comparison.py --host http://localhost:8080 \
        --no-snapshots
"""

import json
import time
import threading
import argparse
import subprocess
import statistics
import random
import sys
import os
from typing import Optional
from datetime import datetime, timezone

import requests

HOST = ""
GRAFANA_URL = ""
SNAPSHOT_DIR = ""

# ---------------------------------------------------------------------------
# KEDA policy definitions
# ---------------------------------------------------------------------------
POLICIES = {
    "policy-a": {
        "name": "Queue-Only (value=5)",
        "file": "phase4.5/k8s/autoscaling/policy-a-queue-only.yaml",
        "description": "Baseline: scale on queue depth alone, threshold=5 per replica",
    },
    "policy-b": {
        "name": "Queue-Only Eager (value=3)",
        "file": "phase4.5/k8s/autoscaling/policy-b-queue-eager.yaml",
        "description": "Eager: lower queue threshold=3, scale sooner",
    },
    "policy-c": {
        "name": "Composite Conservative (KV>0.80)",
        "file": "phase4.5/k8s/autoscaling/policy-c-composite-conservative.yaml",
        "description": "Queue(5) + Prometheus KV cache > 0.80",
    },
    "policy-d": {
        "name": "Composite Aggressive (KV>0.65)",
        "file": "phase4.5/k8s/autoscaling/policy-d-composite-aggressive.yaml",
        "description": "Queue(5) + Prometheus KV cache > 0.65",
    },
}

# ---------------------------------------------------------------------------
# Workload prompts — same as Phase 3 for consistency
# ---------------------------------------------------------------------------
PROMPTS = {
    "short": {
        "text": "What is 2+2?",
        "max_tokens": 20,
        "label": "S",
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
    },
    "very_long": {
        "text": (
            "You are a principal engineer at a large cloud provider designing the "
            "next-generation inference serving platform. The platform must serve "
            "multiple foundation models (7B to 70B parameters) across a heterogeneous "
            "GPU fleet (NVIDIA T4, A10G, A100, H100) with strict latency SLAs. Write "
            "a comprehensive technical design document that covers the following areas "
            "in detail:\n\n"
            "1. REQUEST ROUTING AND LOAD BALANCING: Design a multi-tier routing system "
            "that accounts for model placement, GPU memory availability, KV cache "
            "locality, and request priority. Explain how you would handle prefix caching "
            "across replicas, session affinity for multi-turn conversations, and graceful "
            "degradation when capacity is exhausted. Compare hash-based routing vs "
            "least-loaded vs power-of-two-choices for this workload.\n\n"
            "2. MEMORY MANAGEMENT AND KV CACHE: Design the KV cache management strategy "
            "including PagedAttention block allocation, cross-request prefix sharing, "
            "preemption policies (FIFO vs priority vs cost-based), and multi-tier caching "
            "(GPU VRAM \u2192 CPU RAM \u2192 NVMe \u2192 network). Calculate the memory budget for a "
            "70B model on 8xH100 with tensor parallelism, showing how many concurrent "
            "requests of various context lengths can be served simultaneously.\n\n"
            "3. SCHEDULING AND BATCHING: Design the continuous batching scheduler that "
            "handles heterogeneous request sizes (32 to 32768 context tokens). Explain "
            "chunked prefill, how to balance prefill vs decode throughput, and how to "
            "prevent long-context prefills from starving short decode steps. Include a "
            "mathematical model of throughput as a function of batch size, sequence "
            "length, and model size.\n\n"
            "4. AUTOSCALING AND CAPACITY PLANNING: Design the autoscaling system that "
            "handles GPU provisioning latency (5-10 minutes for new nodes). Cover "
            "predictive scaling based on traffic patterns, warm pools of pre-loaded "
            "models, bin-packing multiple small models on large GPUs, and Spot instance "
            "interruption handling. Define the metrics and thresholds for scaling "
            "decisions.\n\n"
            "5. OBSERVABILITY AND DEBUGGING: Design the metrics, logging, and tracing "
            "infrastructure for debugging inference latency issues. Cover GPU-level "
            "metrics (SM utilization, memory bandwidth, thermal), engine-level metrics "
            "(KV cache hit rate, preemption count, batch efficiency), and request-level "
            "metrics (TTFT, TBT, queue wait). Explain how to correlate a user-reported "
            "latency spike to a specific GPU, request batch, or model loading event.\n\n"
            "For each area, provide concrete numbers, formulas, and architectural "
            "diagrams. Reference real systems (vLLM, TensorRT-LLM, Triton, SGLang) "
            "where applicable. Justify every design choice with quantitative reasoning."
        ),
        "max_tokens": 2048,
        "label": "XL",
    },
}

# ---------------------------------------------------------------------------
# SLA definitions and cost parameters
# ---------------------------------------------------------------------------
SLAS = {
    "short":     {"complete_time_p99_sec": 5},
    "medium":    {"complete_time_p99_sec": 15},
    "long":      {"complete_time_p99_sec": 30},
    "very_long": {"complete_time_p99_sec": 90},
}
GPU_COST_PER_HOUR = 0.22

# ---------------------------------------------------------------------------
# Workload configurations — replaces simple cycling pattern
# ---------------------------------------------------------------------------
WORKLOAD_CONFIGS = {
    "pareto": {
        "distribution": {"short": 0.80, "medium": 0.15, "long": 0.04, "very_long": 0.01},
    },
    "kv_stress": {
        "pattern": ["very_long"],
    },
    "burst_outlier": {
        "distribution": {"short": 0.90, "medium": 0.05, "long": 0.04, "very_long": 0.01},
        "burst": {"type": "very_long", "count": 5, "interval_sec": 30},
    },
}

# ---------------------------------------------------------------------------
# Load phase definitions — graduated pressure with Pareto distribution
# ---------------------------------------------------------------------------
LOAD_PHASES = [
    {"name": "light", "rate": 0.8, "duration": 120, "warmup": 30, "drain": 15,
     "workload": "pareto",
     "description": "0.8 req/s Pareto — baseline, should NOT trigger scale-up"},
    {"name": "moderate", "rate": 1.5, "duration": 120, "warmup": 30, "drain": 15,
     "workload": "pareto",
     "description": "1.5 req/s Pareto — edge, XL requests build queue"},
    {"name": "heavy", "rate": 2.5, "duration": 300, "warmup": 30, "drain": 15,
     "workload": "pareto",
     "description": "2.5 req/s Pareto — exceeds 1-worker capacity, forces scale-up"},
    {"name": "sustained", "rate": 3.0, "duration": 420, "warmup": 60, "drain": 30,
     "workload": "pareto",
     "description": "3.0 req/s Pareto — multi-worker steady state"},
    {"name": "kv_stress", "rate": 0.5, "duration": 180, "warmup": 30, "drain": 15,
     "workload": "kv_stress",
     "description": "0.5 req/s all-XL — forces KV cache to 80%, differentiates composite policies"},
    {"name": "burst", "rate": 1.0, "duration": 180, "warmup": 30, "drain": 15,
     "workload": "burst_outlier",
     "description": "1.0 req/s 90% short + XL bursts every 30s — tests spike handling"},
    {"name": "cooldown", "rate": 0.0, "duration": 480, "warmup": 0, "drain": 0,
     "workload": None,
     "description": "0 req/s — observe scale-down (300s cooldown + observation)"},
]


# ---------------------------------------------------------------------------
# Prompt selector — picks prompt type based on workload config
# ---------------------------------------------------------------------------
def _select_prompt(workload_config: dict, request_index: int, rng: random.Random,
                   elapsed_sec: float) -> str:
    """Select a prompt type based on the workload configuration.

    Supports three modes:
      - burst: during burst windows, override with burst type
      - distribution: weighted random sampling (e.g., Pareto)
      - pattern: deterministic cycling through a list
    """
    # Check for burst override first
    burst = workload_config.get("burst")
    if burst:
        interval = burst["interval_sec"]
        count = burst["count"]
        time_in_cycle = elapsed_sec % interval
        if time_in_cycle < (count / 2.0):
            return burst["type"]

    # Weighted random distribution (Pareto, etc.)
    if "distribution" in workload_config:
        dist = workload_config["distribution"]
        types = list(dist.keys())
        weights = list(dist.values())
        return rng.choices(types, weights=weights, k=1)[0]

    # Deterministic pattern cycling
    pattern = workload_config["pattern"]
    return pattern[request_index % len(pattern)]


# ---------------------------------------------------------------------------
# Request sender — fires one request and collects all metrics
# ---------------------------------------------------------------------------
def send_request(
    request_id: int,
    prompt_key: str,
    results: list,
    run_start_time: float,
) -> None:
    """Send a single request through the queue pipeline, collect timing metrics."""
    prompt_config = PROMPTS[prompt_key]
    submit_time = time.perf_counter()
    submit_wall = time.time()

    token_times = []
    tokens = []
    error = None
    server_metrics = {}

    try:
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

        stream_resp = requests.get(
            f"{HOST}{stream_url}",
            stream=True,
            headers={"Accept": "text/event-stream"},
            timeout=(10, 600),  # XL requests can take very long at high queue depths
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

    if error or not token_times:
        results[request_id] = {
            "request_id": request_id,
            "prompt_type": prompt_key,
            "label": prompt_config["label"],
            "prompt_char_len": len(prompt_config["text"]),
            "max_tokens_requested": prompt_config["max_tokens"],
            "error": error or "no tokens received",
            "total_sec": round(request_end - submit_time, 3),
            "submit_offset_sec": round(submit_wall - run_start_time, 2),
        }
        return

    client_ttft = token_times[0] - submit_time
    tbts = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]

    results[request_id] = {
        "request_id": request_id,
        "prompt_type": prompt_key,
        "label": prompt_config["label"],
        "prompt_char_len": len(prompt_config["text"]),
        "max_tokens_requested": prompt_config["max_tokens"],
        "submit_offset_sec": round(submit_wall - run_start_time, 2),
        "client_ttft_sec": round(client_ttft, 4),
        "client_total_sec": round(request_end - submit_time, 3),
        "tokens_out": len(tokens),
        "client_tokens_per_sec": round(
            len(tokens) / (request_end - token_times[0]), 1
        ) if len(token_times) > 1 else 0,
        "server_queue_wait_ms": server_metrics.get("queue_wait_ms"),
        "server_ttft_ms": server_metrics.get("ttft_ms"),
        "server_total_ms": server_metrics.get("total_ms"),
        "server_tokens_per_sec": server_metrics.get("tokens_per_sec"),
        "server_strategy": server_metrics.get("strategy"),
        "server_nack_count": server_metrics.get("worker_nack_count"),
        "server_admit_count": server_metrics.get("worker_admit_count"),
    }


# ---------------------------------------------------------------------------
# Open-loop load generator
# ---------------------------------------------------------------------------
def run_load_phase(
    phase: dict,
    run_start_time: float,
) -> list:
    """Send requests at a fixed rate for a given duration (open-loop).

    WHY open-loop: closed-loop testing underestimates latency under load.
    Returns list of result dicts (one per request, including errors).
    """
    rate = phase["rate"]
    duration = phase["duration"]

    if rate == 0:
        # Cooldown phase — just wait, no requests
        print(f"    Cooldown: waiting {duration}s for scale-down...")
        time.sleep(duration)
        return []

    interval = 1.0 / rate
    total_requests = int(rate * duration)
    results = [None] * total_requests
    threads = []

    # Load workload config for prompt selection
    workload_name = phase.get("workload", "pareto")
    workload_config = WORKLOAD_CONFIGS[workload_name]
    rng = random.Random(42)

    print(f"    Sending {total_requests} requests at {rate} req/s for {duration}s...")
    print(f"    Workload: {workload_name}")

    phase_start_wall = time.time()

    for i in range(total_requests):
        elapsed = time.time() - phase_start_wall
        prompt_key = _select_prompt(workload_config, i, rng, elapsed)
        t = threading.Thread(
            target=send_request,
            args=(i, prompt_key, results, run_start_time),
        )
        threads.append(t)
        t.start()

        if (i + 1) % 20 == 0:
            elapsed = time.time() - run_start_time
            print(f"\r    [{elapsed:>5.0f}s] Sent {i + 1}/{total_requests}", end="", flush=True)

        if i < total_requests - 1:
            time.sleep(interval)

    print(f"\n    All {total_requests} requests submitted. Waiting for completions...")

    for t in threads:
        t.join(timeout=600)  # XL requests can take very long at high queue depths

    completed = [r for r in results if r is not None]
    errors = [r for r in completed if "error" in r]
    successes = [r for r in completed if "error" not in r]

    print(f"    Completed: {len(successes)} success, {len(errors)} errors, "
          f"{total_requests - len(completed)} timeouts")

    return completed


# ---------------------------------------------------------------------------
# SLA and cost computation
# ---------------------------------------------------------------------------
def _compute_sla(steady: list, all_errors: list) -> dict:
    """Compute per-prompt-type SLA compliance from steady-state results.

    SLA is defined as: % of requests whose client_total_sec <= the P99
    threshold for that prompt type.  Errors count as SLA misses.
    """
    sla_results = {}
    total_met = 0
    total_count = 0

    for prompt_type, sla_def in SLAS.items():
        threshold = sla_def["complete_time_p99_sec"]
        # Successes for this prompt type
        type_ok = [
            r for r in steady
            if r.get("prompt_type") == prompt_type
        ]
        # Errors for this prompt type
        type_err = [
            r for r in all_errors
            if r.get("prompt_type") == prompt_type
        ]
        type_total = len(type_ok) + len(type_err)
        if type_total == 0:
            sla_results[prompt_type] = {"count": 0, "compliance": None}
            continue

        met = sum(
            1 for r in type_ok
            if r.get("client_total_sec", float("inf")) <= threshold
        )
        compliance = met / type_total
        sla_results[prompt_type] = {
            "count": type_total,
            "met": met,
            "missed": type_total - met,
            "compliance": round(compliance, 4),
            "threshold_sec": threshold,
        }
        total_met += met
        total_count += type_total

    overall = round(total_met / total_count, 4) if total_count > 0 else None
    return {"per_type": sla_results, "overall_compliance": overall}


def _compute_cost(steady: list, steady_duration: float) -> dict:
    """Compute cost efficiency from steady-state results.

    Reports GPU cost per million output tokens, using wall-clock GPU-hours.
    """
    total_tokens = sum(r.get("tokens_out", 0) for r in steady)
    if total_tokens == 0 or steady_duration <= 0:
        return {"cost_per_million_tokens": None, "total_tokens": total_tokens}

    # GPU-hours consumed during the steady-state window
    gpu_hours = steady_duration / 3600.0
    gpu_cost = gpu_hours * GPU_COST_PER_HOUR

    cost_per_million = (gpu_cost / total_tokens) * 1_000_000

    return {
        "total_tokens": total_tokens,
        "gpu_hours": round(gpu_hours, 4),
        "gpu_cost_usd": round(gpu_cost, 4),
        "cost_per_million_tokens": round(cost_per_million, 2),
    }


# ---------------------------------------------------------------------------
# Steady-state analysis
# ---------------------------------------------------------------------------
def analyze_phase_results(
    results: list,
    phase: dict,
) -> dict:
    """Analyze metrics from the steady-state window of a load phase."""
    if not results:
        return {"phase": phase["name"], "note": "No requests (cooldown phase)"}

    warmup = phase["warmup"]
    drain = phase["drain"]
    duration = phase["duration"]
    steady_start = warmup
    steady_end = duration - drain

    steady = [
        r for r in results
        if "error" not in r
        and steady_start <= r["submit_offset_sec"] <= steady_end
    ]
    all_errors = [r for r in results if "error" in r]

    if not steady:
        return {
            "phase": phase["name"],
            "steady_state_requests": 0,
            "note": "No successful requests in steady-state window",
        }

    ttfts = sorted([r["client_ttft_sec"] for r in steady])
    queue_waits = sorted([
        r["server_queue_wait_ms"] for r in steady
        if r.get("server_queue_wait_ms") is not None
    ])

    total_tokens = sum(r.get("tokens_out", 0) for r in steady)
    steady_duration = steady_end - steady_start
    throughput = total_tokens / steady_duration if steady_duration > 0 else 0

    nack_counts = [r.get("server_nack_count", 0) for r in steady
                   if r.get("server_nack_count") is not None]
    total_nacks = (max(nack_counts) - min(nack_counts)) if nack_counts else 0

    def pct(data, p):
        if not data:
            return None
        idx = min(int(len(data) * p / 100), len(data) - 1)
        return round(data[idx], 4)

    # Compute SLA compliance and cost efficiency
    sla = _compute_sla(steady, all_errors)
    cost = _compute_cost(steady, steady_duration)

    return {
        "phase": phase["name"],
        "steady_state_requests": len(steady),
        "total_errors": len(all_errors),
        "completion_rate": round(
            len(steady) / (len(steady) + len(all_errors)), 3
        ) if (len(steady) + len(all_errors)) > 0 else 0,
        "ttft_p50": pct(ttfts, 50),
        "ttft_p99": pct(ttfts, 99),
        "queue_wait_p99_ms": round(pct(queue_waits, 99) or 0, 1),
        "throughput_tok_per_sec": round(throughput, 1),
        "nack_count": total_nacks,
        "sla": sla,
        "cost": cost,
    }


# ---------------------------------------------------------------------------
# Kubernetes helpers
# ---------------------------------------------------------------------------
def kubectl(cmd: str, timeout: int = 60) -> str:
    """Run a kubectl command and return stdout."""
    full_cmd = f"kubectl {cmd}"
    result = subprocess.run(
        full_cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        print(f"    kubectl error: {result.stderr.strip()}")
    return result.stdout.strip()


def get_replica_count() -> tuple[int, int]:
    """Get (desired, ready) replica count for vllm-worker."""
    output = kubectl(
        "get deployment vllm-worker -n inference-system "
        "-o jsonpath='{.spec.replicas} {.status.readyReplicas}'"
    )
    parts = output.strip("'").split()
    desired = int(parts[0]) if parts else 0
    ready = int(parts[1]) if len(parts) > 1 and parts[1] != "<none>" else 0
    return desired, ready


def get_node_count() -> int:
    """Count GPU nodes (labeled node-role=gpu)."""
    output = kubectl(
        "get nodes -l node-role=gpu --no-headers 2>/dev/null | wc -l"
    )
    try:
        return int(output.strip())
    except ValueError:
        return 0


def switch_policy(policy_key: str) -> bool:
    """Apply a KEDA ScaledObject policy and wait for reconciliation."""
    policy = POLICIES[policy_key]
    print(f"\n  Switching KEDA policy: {policy['name']}")
    print(f"    Applying {policy['file']}...")

    result = kubectl(f"apply -f {policy['file']}")
    if "configured" in result or "created" in result:
        print(f"    ScaledObject applied: {result}")
    else:
        print(f"    WARNING: unexpected output: {result}")

    # Wait for KEDA to reconcile (15s polling interval + processing)
    print("    Waiting 20s for KEDA to reconcile...")
    time.sleep(20)

    return True


def ensure_single_replica() -> bool:
    """Wait for cluster to settle at exactly 1 worker replica.

    WHY: each policy experiment must start from the same baseline (1 replica).
    If the previous policy's cooldown didn't finish, we wait.
    """
    print("    Waiting for cluster to settle at 1 replica...")
    max_wait = 600  # 10 min — accounts for 180s KEDA cooldown + 300s Karpenter consolidation
    start = time.time()

    while time.time() - start < max_wait:
        desired, ready = get_replica_count()
        if desired == 1 and ready == 1:
            print(f"    Cluster at 1 replica (took {time.time() - start:.0f}s)")
            return True
        elapsed = time.time() - start
        print(f"\r    [{elapsed:.0f}s] Replicas: desired={desired} ready={ready}", end="", flush=True)
        time.sleep(10)

    print(f"\n    WARNING: Cluster did not settle to 1 replica within {max_wait}s")
    return False


# ---------------------------------------------------------------------------
# Scaling event tracker — monitors replica changes during a load phase
# ---------------------------------------------------------------------------
class ScalingTracker:
    """Background thread that polls replica count and records scaling events.

    Captures:
    - When scale-up was triggered (desired > ready)
    - When new worker became ready (ready matched desired)
    - Scale-down timing during cooldown
    """

    def __init__(self, poll_interval: float = 5.0):
        self.poll_interval = poll_interval
        self.events = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _poll_loop(self):
        prev_desired = None
        prev_ready = None

        while not self._stop.is_set():
            try:
                desired, ready = get_replica_count()
                nodes = get_node_count()
                ts = time.time()

                if desired != prev_desired:
                    self.events.append({
                        "time": ts,
                        "event": "desired_changed",
                        "from": prev_desired,
                        "to": desired,
                        "ready": ready,
                        "nodes": nodes,
                    })

                if ready != prev_ready:
                    self.events.append({
                        "time": ts,
                        "event": "ready_changed",
                        "from": prev_ready,
                        "to": ready,
                        "desired": desired,
                        "nodes": nodes,
                    })

                prev_desired = desired
                prev_ready = ready

            except Exception as e:
                self.events.append({
                    "time": time.time(),
                    "event": "error",
                    "error": str(e),
                })

            self._stop.wait(self.poll_interval)

    def get_summary(self) -> dict:
        """Extract key scaling metrics from recorded events."""
        if not self.events:
            return {"scale_events": 0}

        # Find first scale-up trigger (desired went from 1 to 2+)
        scale_up_trigger = None
        scale_up_ready = None
        max_replicas = 1

        for ev in self.events:
            if ev["event"] == "desired_changed" and (ev.get("from") or 0) < (ev.get("to") or 0):
                if scale_up_trigger is None:
                    scale_up_trigger = ev["time"]

            if ev["event"] == "ready_changed":
                new_ready = ev.get("to") or 0
                if new_ready > 1 and scale_up_ready is None:
                    scale_up_ready = ev["time"]
                max_replicas = max(max_replicas, new_ready)

        # Find scale-down completion (ready went back to 1)
        scale_down_time = None
        for ev in reversed(self.events):
            if ev["event"] == "ready_changed" and ev.get("to") == 1 and (ev.get("from") or 0) > 1:
                scale_down_time = ev["time"]
                break

        scale_up_latency = None
        if scale_up_trigger and scale_up_ready:
            scale_up_latency = round(scale_up_ready - scale_up_trigger, 1)

        return {
            "scale_events": len([e for e in self.events if e["event"] in ("desired_changed", "ready_changed")]),
            "max_replicas_reached": max_replicas,
            "scale_up_trigger_time": scale_up_trigger,
            "scale_up_ready_time": scale_up_ready,
            "scale_up_latency_sec": scale_up_latency,
            "scale_down_time": scale_down_time,
            "events": self.events,
        }


# ---------------------------------------------------------------------------
# Grafana snapshot capture
# ---------------------------------------------------------------------------
def capture_grafana_snapshot(
    policy_key: str,
    phase_name: str,
    phase_start_epoch: float,
    phase_end_epoch: float,
) -> Optional[str]:
    """Capture a Grafana dashboard screenshot via the Image Renderer.

    Uses Grafana's /render API which returns a PNG image of the dashboard
    for the specified time range.
    """
    if not GRAFANA_URL or not SNAPSHOT_DIR:
        return None

    try:
        # Grafana render API expects millisecond timestamps
        from_ms = int(phase_start_epoch * 1000)
        to_ms = int(phase_end_epoch * 1000)

        render_url = (
            f"{GRAFANA_URL}/render/d/inference-platform/inference-platform"
            f"?from={from_ms}&to={to_ms}"
            f"&width=1920&height=1200"
            f"&timeout=30"
        )

        resp = requests.get(
            render_url,
            auth=("admin", "inference-lab"),
            timeout=60,
        )

        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
            filename = f"{policy_key}_phase-{phase_name}.png"
            filepath = os.path.join(SNAPSHOT_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(resp.content)
            print(f"    Snapshot saved: {filepath} ({len(resp.content)} bytes)")
            return filepath
        else:
            print(f"    Snapshot failed: HTTP {resp.status_code} ({resp.text[:100]})")
            return None

    except Exception as e:
        print(f"    Snapshot error: {e}")
        return None


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------
def print_comparison_table(all_results: dict):
    """Print a formatted comparison table across all policies and phases."""
    print("\n" + "=" * 120)
    print("KEDA SCALING POLICY COMPARISON — RESULTS")
    print("=" * 120)

    # Summary table: one row per policy, key metrics
    header = (
        f"{'Policy':<35} {'ScaleUp':<10} {'MaxR':<6} {'TTFT p50':<10} "
        f"{'TTFT p99':<10} {'Thru':<8} {'NACKs':<7} {'Success':<8} "
        f"{'SLA%':<7} {'$/1M tok':<10}"
    )
    print(f"\n{header}")
    print("-" * 120)

    for policy_key, policy_data in all_results.items():
        policy_name = POLICIES[policy_key]["name"]
        scaling = policy_data.get("scaling_summary", {})
        scale_up_lat = scaling.get("scale_up_latency_sec")
        max_r = scaling.get("max_replicas_reached", 1)

        # Aggregate metrics across heavy + sustained phases (where scaling matters)
        heavy_phases = [
            p for p in policy_data.get("phases", [])
            if p.get("phase") in ("heavy", "sustained") and p.get("steady_state_requests", 0) > 0
        ]

        if heavy_phases:
            avg_ttft_p50 = statistics.mean([p["ttft_p50"] for p in heavy_phases if p.get("ttft_p50")])
            avg_ttft_p99 = statistics.mean([p["ttft_p99"] for p in heavy_phases if p.get("ttft_p99")])
            total_throughput = statistics.mean([p["throughput_tok_per_sec"] for p in heavy_phases])
            total_nacks = sum(p.get("nack_count", 0) for p in heavy_phases)
            avg_success = statistics.mean([p.get("completion_rate", 0) for p in heavy_phases])
            # Aggregate SLA compliance across heavy phases
            sla_vals = [
                p["sla"]["overall_compliance"] for p in heavy_phases
                if p.get("sla") and p["sla"].get("overall_compliance") is not None
            ]
            avg_sla = statistics.mean(sla_vals) if sla_vals else 0
            # Aggregate cost across heavy phases
            cost_vals = [
                p["cost"]["cost_per_million_tokens"] for p in heavy_phases
                if p.get("cost") and p["cost"].get("cost_per_million_tokens") is not None
            ]
            avg_cost = statistics.mean(cost_vals) if cost_vals else 0
        else:
            avg_ttft_p50 = avg_ttft_p99 = total_throughput = total_nacks = avg_success = 0
            avg_sla = 0
            avg_cost = 0

        scale_str = f"{scale_up_lat:.0f}s" if scale_up_lat else "N/A"
        cost_str = f"${avg_cost:.2f}" if avg_cost else "N/A"
        print(
            f"{policy_name:<35} {scale_str:<10} {max_r:<6} "
            f"{avg_ttft_p50:>8.3f}s {avg_ttft_p99:>8.3f}s "
            f"{total_throughput:>6.1f} {total_nacks:>6} {avg_success:>6.1%} "
            f"{avg_sla:>5.1%} {cost_str:>9}"
        )

    # Detailed per-phase breakdown for each policy
    for policy_key, policy_data in all_results.items():
        policy_name = POLICIES[policy_key]["name"]
        print(f"\n--- {policy_name} ---")
        print(f"  {'Phase':<12} {'Reqs':<6} {'TTFT p50':<10} {'TTFT p99':<10} "
              f"{'QWait p99':<10} {'Thru':<8} {'NACKs':<7} {'Rate':<6} "
              f"{'SLA%':<7} {'$/1M tok':<10}")
        print(f"  {'-'*90}")

        for phase_result in policy_data.get("phases", []):
            phase_name = phase_result.get("phase", "?")
            n = phase_result.get("steady_state_requests", 0)
            if n == 0:
                print(f"  {phase_name:<12} {'--':>5} {'(no data)':<60}")
                continue

            ttft_p50 = phase_result.get("ttft_p50", 0)
            ttft_p99 = phase_result.get("ttft_p99", 0)
            qw_p99 = phase_result.get("queue_wait_p99_ms", 0)
            thru = phase_result.get("throughput_tok_per_sec", 0)
            nacks = phase_result.get("nack_count", 0)
            rate = phase_result.get("completion_rate", 0)

            # SLA compliance for this phase
            sla_data = phase_result.get("sla", {})
            sla_pct = sla_data.get("overall_compliance")
            sla_str = f"{sla_pct:>5.1%}" if sla_pct is not None else "  N/A"

            # Cost for this phase
            cost_data = phase_result.get("cost", {})
            cost_val = cost_data.get("cost_per_million_tokens")
            cost_str = f"${cost_val:.2f}" if cost_val is not None else "N/A"

            print(
                f"  {phase_name:<12} {n:>5} {ttft_p50:>8.3f}s {ttft_p99:>8.3f}s "
                f"{qw_p99:>8.1f}ms {thru:>6.1f} {nacks:>6} {rate:>5.1%} "
                f"{sla_str:>6} {cost_str:>9}"
            )

        scaling = policy_data.get("scaling_summary", {})
        lat = scaling.get("scale_up_latency_sec")
        print(f"  Scale-up latency: {lat:.0f}s" if lat else "  Scale-up: not triggered")
        print(f"  Max replicas: {scaling.get('max_replicas_reached', 1)}")


# ---------------------------------------------------------------------------
# Main experiment orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    policy_keys: list,
    skip_snapshots: bool = False,
    duration_multiplier: float = 1.0,
):
    """Run the full scaling policy comparison experiment."""
    all_results = {}
    experiment_start = datetime.now(timezone.utc).isoformat()

    # Apply duration multiplier to all phases
    for phase in LOAD_PHASES:
        phase["duration"] = int(phase["duration"] * duration_multiplier)

    print(f"\nPhase 4 — KEDA Scaling Policy Comparison")
    print(f"Start: {experiment_start}")
    print(f"Policies: {', '.join(policy_keys)}")
    print(f"Load phases: {len(LOAD_PHASES)} ({' -> '.join(p['name'] for p in LOAD_PHASES)})")
    if duration_multiplier != 1.0:
        print(f"Duration multiplier: {duration_multiplier}x")
    print(f"Estimated time: ~{len(policy_keys) * 30} minutes")

    for policy_idx, policy_key in enumerate(policy_keys):
        policy = POLICIES[policy_key]
        print(f"\n{'='*80}")
        print(f"[{policy_idx + 1}/{len(policy_keys)}] Policy: {policy['name']}")
        print(f"  {policy['description']}")
        print(f"{'='*80}")

        # Step 1: Apply the ScaledObject
        switch_policy(policy_key)

        # Step 2: Ensure we start from 1 replica
        if not ensure_single_replica():
            print(f"  SKIPPING {policy_key} — cluster did not settle")
            all_results[policy_key] = {"error": "cluster_not_settled"}
            continue

        # Step 3: Start scaling tracker
        tracker = ScalingTracker(poll_interval=5.0)
        tracker.start()

        # Step 4: Run graduated load phases
        phase_results = []
        policy_start = time.time()

        for phase_idx, phase in enumerate(LOAD_PHASES):
            print(f"\n  Phase {phase_idx + 1}/{len(LOAD_PHASES)}: {phase['name']}")
            print(f"    {phase['description']}")

            phase_start = time.time()

            # Run the load
            raw_results = run_load_phase(phase, run_start_time=phase_start)

            phase_end = time.time()

            # Analyze
            analysis = analyze_phase_results(raw_results, phase)
            phase_results.append(analysis)

            # Report replica status
            desired, ready = get_replica_count()
            print(f"    Replicas at end of phase: desired={desired} ready={ready}")

            # Capture Grafana snapshot
            if not skip_snapshots:
                capture_grafana_snapshot(
                    policy_key, phase["name"],
                    phase_start, phase_end,
                )

        # Step 5: Stop tracking
        tracker.stop()
        scaling_summary = tracker.get_summary()

        policy_duration = time.time() - policy_start
        print(f"\n  Policy {policy_key} complete ({policy_duration:.0f}s)")
        print(f"  Scale-up latency: {scaling_summary.get('scale_up_latency_sec', 'N/A')}s")
        print(f"  Max replicas: {scaling_summary.get('max_replicas_reached', 1)}")

        all_results[policy_key] = {
            "policy": policy["name"],
            "phases": phase_results,
            "scaling_summary": {
                k: v for k, v in scaling_summary.items()
                if k != "events"  # Don't include raw events in summary
            },
            "scaling_events": scaling_summary.get("events", []),
            "duration_sec": round(policy_duration, 1),
        }

    # Step 6: Print comparison table
    print_comparison_table(all_results)

    # Step 7: Save results to JSON
    output = {
        "experiment": "phase4_scaling_policy_comparison",
        "start_time": experiment_start,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "policies_tested": policy_keys,
        "load_phases": [p["name"] for p in LOAD_PHASES],
        "results": all_results,
    }

    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "scaling_policy_results.json",
    )
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    global HOST, GRAFANA_URL, SNAPSHOT_DIR

    parser = argparse.ArgumentParser(
        description="Phase 4 — KEDA Scaling Policy Comparison Experiment"
    )
    parser.add_argument(
        "--host", required=True,
        help="API gateway URL (e.g., http://localhost:8080)"
    )
    parser.add_argument(
        "--grafana", default="http://localhost:3000",
        help="Grafana URL for snapshot capture (default: http://localhost:3000)"
    )
    parser.add_argument(
        "--policy", choices=list(POLICIES.keys()),
        help="Run a single policy (default: run all 4)"
    )
    parser.add_argument(
        "--no-snapshots", action="store_true",
        help="Skip Grafana snapshot capture"
    )
    parser.add_argument(
        "--duration-multiplier", type=float, default=1.0,
        help="Scale all phase durations by this factor (default: 1.0)"
    )
    args = parser.parse_args()

    HOST = args.host.rstrip("/")
    GRAFANA_URL = args.grafana.rstrip("/")
    SNAPSHOT_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "snapshots",
    )
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    # Verify connectivity
    print(f"Testing gateway connectivity: {HOST}/health")
    try:
        resp = requests.get(f"{HOST}/health", timeout=5)
        print(f"  Gateway health: {resp.status_code}")
        if resp.status_code != 200:
            print(f"  WARNING: gateway returned {resp.status_code}")
    except Exception as e:
        print(f"  ERROR: cannot reach gateway: {e}")
        sys.exit(1)

    # Select policies to run
    if args.policy:
        policy_keys = [args.policy]
    else:
        policy_keys = list(POLICIES.keys())

    run_experiment(
        policy_keys,
        skip_snapshots=args.no_snapshots,
        duration_multiplier=args.duration_multiplier,
    )


if __name__ == "__main__":
    main()
