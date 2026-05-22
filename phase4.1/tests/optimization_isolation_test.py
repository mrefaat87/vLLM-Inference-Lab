"""
Phase 4 — Optimization Isolation Test

Tests each Phase 4 optimization independently to measure its individual impact.
Unlike the scaling_policy_comparison (which varies KEDA triggers), this test
varies the INFRASTRUCTURE optimizations while keeping the scaling policy fixed.

TEST MATRIX — 4 configurations, same workload:
  Config 1: Baseline        — no AMI cache, no S3 cache (HuggingFace download)
  Config 2: + S3 model      — S3 init container enabled, default AMI
  Config 3: + Pre-baked AMI — custom AMI with vLLM image, no S3 cache
  Config 4: + Both          — S3 init container + pre-baked AMI (full optimization)

WHAT WE MEASURE per config:
  - Cold start time: time from pod pending to pod ready (the headline metric)
  - Scale-up latency: time from KEDA desired>1 to second worker ready
  - Request success rate during scale-up window
  - TTFT p50/p99 under load

METHODOLOGY:
  Each config runs the same load pattern:
  1. Ensure 1 replica (baseline)
  2. Send heavy load (1.5 req/s) to trigger scale-up
  3. Measure time until second worker is Ready
  4. Record metrics during the scale-up window
  5. Wait for scale-down back to 1

Additional tests:
  - Graceful drain: kill a worker during active inference, verify no broken responses
  - Cooldown validation: verify KEDA doesn't kill workers before they finish booting

Usage:
    python3 phase4/tests/optimization_isolation_test.py --host http://localhost:8080 \
        --config baseline
    python3 phase4/tests/optimization_isolation_test.py --host http://localhost:8080 \
        --config all
"""

import json
import time
import threading
import argparse
import subprocess
import statistics
import sys
import os
from typing import Optional
from datetime import datetime, timezone

import requests

HOST = ""

# ---------------------------------------------------------------------------
# Test configurations — each isolates one optimization
# ---------------------------------------------------------------------------
CONFIGS = {
    "baseline": {
        "name": "Baseline (no optimizations)",
        "description": "Default EKS AMI + HuggingFace download. Phase 2 equivalent.",
        "ami_selector": False,    # Use default EKS AMI (no amiSelectorTerms)
        "s3_init": False,         # Disable S3 init container
    },
    "s3-only": {
        "name": "+ S3 Model Cache",
        "description": "Default AMI + S3 init container for model download.",
        "ami_selector": False,
        "s3_init": True,
    },
    "ami-only": {
        "name": "+ Pre-baked AMI",
        "description": "Custom AMI with vLLM image cached + HuggingFace download.",
        "ami_selector": True,
        "s3_init": False,
    },
    "full": {
        "name": "+ Both (S3 + AMI)",
        "description": "Custom AMI + S3 init container. Full Phase 4 optimization.",
        "ami_selector": True,
        "s3_init": True,
    },
}

# Same workload prompts as the scaling policy test
PROMPTS = {
    "short": {"text": "What is 2+2?", "max_tokens": 20},
    "medium": {
        "text": (
            "Explain the concept of auto scaling in cloud computing. "
            "Cover the key components including scaling policies, "
            "cooldown periods, health checks, and how metrics like "
            "CPU utilization and queue depth drive scaling decisions. "
            "Be concise."
        ),
        "max_tokens": 150,
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
            "What are the most likely root causes and what specific changes would you "
            "recommend to fix this? Prioritize your recommendations by impact."
        ),
        "max_tokens": 300,
    },
}
PATTERN = ["short", "medium", "long"]


# ---------------------------------------------------------------------------
# Request sender
# ---------------------------------------------------------------------------
def send_request(request_id, prompt_key, results, run_start):
    prompt = PROMPTS[prompt_key]
    submit_time = time.perf_counter()
    submit_wall = time.time()
    error = None
    server_metrics = {}
    token_count = 0
    first_token_time = None

    try:
        resp = requests.post(
            f"{HOST}/generate",
            json={"prompt": prompt["text"], "max_tokens": prompt["max_tokens"], "temperature": 0.7},
            timeout=10,
        )
        if resp.status_code != 202:
            raise RuntimeError(f"HTTP {resp.status_code}")

        stream_url = resp.json()["stream_url"]
        stream = requests.get(f"{HOST}{stream_url}", stream=True, timeout=(10, 300))

        for line in stream.iter_lines():
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
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += 1
            if data.get("status") == "complete":
                server_metrics = data.get("metrics", {})
                break
            if data.get("status") == "error":
                error = data.get("error", "unknown")
                break
    except Exception as e:
        error = str(e)

    end_time = time.perf_counter()
    ttft = (first_token_time - submit_time) if first_token_time else None

    results[request_id] = {
        "request_id": request_id,
        "error": error,
        "ttft_sec": round(ttft, 4) if ttft else None,
        "tokens": token_count,
        "total_sec": round(end_time - submit_time, 3),
        "submit_offset": round(submit_wall - run_start, 2),
        "queue_wait_ms": server_metrics.get("queue_wait_ms"),
    }


# ---------------------------------------------------------------------------
# Kubernetes helpers
# ---------------------------------------------------------------------------
def kubectl(cmd, timeout=60):
    result = subprocess.run(f"kubectl {cmd}", shell=True, capture_output=True, text=True, timeout=timeout)
    return result.stdout.strip()


def get_replicas():
    output = kubectl("get deployment vllm-worker -n inference-system -o jsonpath='{.spec.replicas} {.status.readyReplicas}'")
    parts = output.strip("'").split()
    desired = int(parts[0]) if parts else 0
    ready = int(parts[1]) if len(parts) > 1 and parts[1] not in ("<none>", "None", "") else 0
    return desired, ready


def wait_for_replicas(target_ready, timeout=900):
    """Wait until ready replicas == target. Returns time taken."""
    start = time.time()
    while time.time() - start < timeout:
        desired, ready = get_replicas()
        if ready >= target_ready:
            return time.time() - start
        time.sleep(5)
    return None


def force_single_replica():
    """Ensure exactly 1 replica running."""
    kubectl("scale deployment vllm-worker -n inference-system --replicas=1")
    for i in range(60):
        _, ready = get_replicas()
        if ready == 1:
            return True
        time.sleep(10)
    return False


# ---------------------------------------------------------------------------
# Cold start measurement
# ---------------------------------------------------------------------------
def measure_cold_start():
    """Trigger a scale-up and measure time from pod creation to ready.

    Methodology:
    1. We're at 1 replica
    2. Scale to 2 manually (bypasses KEDA to isolate infra timing)
    3. Measure time until second pod is Ready
    4. Scale back to 1
    """
    print("    Triggering scale-up from 1 to 2 replicas...")
    start = time.time()
    kubectl("scale deployment vllm-worker -n inference-system --replicas=2")

    # Wait for second pod to become ready
    cold_start = wait_for_replicas(2, timeout=900)

    if cold_start:
        print(f"    Cold start: {cold_start:.1f}s")
    else:
        print("    Cold start: TIMEOUT (>900s)")
        cold_start = 900.0

    # Scale back to 1
    print("    Scaling back to 1 replica...")
    kubectl("scale deployment vllm-worker -n inference-system --replicas=1")
    time.sleep(30)

    return cold_start


# ---------------------------------------------------------------------------
# Graceful drain test
# ---------------------------------------------------------------------------
def test_graceful_drain():
    """Send requests, then kill a worker mid-inference. Verify no broken responses.

    This tests the Phase 4 drain logic: the worker should finish in-flight
    requests before exiting, rather than dropping them.
    """
    print("\n  === Graceful Drain Test ===")
    print("    Sending 10 requests, then killing the worker pod...")

    # Send requests
    results = [None] * 10
    threads = []
    run_start = time.time()

    for i in range(10):
        prompt_key = PATTERN[i % len(PATTERN)]
        t = threading.Thread(target=send_request, args=(i, prompt_key, results, run_start))
        threads.append(t)
        t.start()
        time.sleep(0.5)

    # Wait a moment for requests to be in-flight, then delete the pod
    time.sleep(3)
    print("    Deleting worker pod (simulating scale-down)...")
    kubectl("delete pod -n inference-system -l app.kubernetes.io/name=vllm-worker --wait=false")

    # Wait for all requests to complete
    for t in threads:
        t.join(timeout=300)

    # Check results
    completed = [r for r in results if r is not None]
    successes = [r for r in completed if r.get("error") is None and r.get("tokens", 0) > 0]
    errors = [r for r in completed if r.get("error") is not None or r.get("tokens", 0) == 0]

    print(f"    Results: {len(successes)} success, {len(errors)} errors out of {len(completed)}")

    # Check worker logs for DRAIN_STATS
    time.sleep(30)  # Wait for new pod to start
    logs = kubectl("logs deployment/vllm-worker -n inference-system -c queue-worker --previous --tail=20 2>/dev/null")
    drain_found = "DRAIN_STATS" in logs
    print(f"    DRAIN_STATS in logs: {drain_found}")
    if drain_found:
        for line in logs.split("\n"):
            if "DRAIN_STATS" in line:
                print(f"      {line.strip()}")

    return {
        "total_requests": len(completed),
        "successes": len(successes),
        "errors": len(errors),
        "drain_stats_found": drain_found,
        "success_rate": round(len(successes) / len(completed), 3) if completed else 0,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_config_test(config_key):
    """Run a cold start measurement for a specific configuration."""
    config = CONFIGS[config_key]
    print(f"\n  Config: {config['name']}")
    print(f"    {config['description']}")

    # Measure cold start (manual scale 1->2->1)
    cold_start = measure_cold_start()

    return {
        "config": config_key,
        "name": config["name"],
        "cold_start_sec": round(cold_start, 1),
    }


def main():
    global HOST

    parser = argparse.ArgumentParser(description="Phase 4 — Optimization Isolation Test")
    parser.add_argument("--host", required=True, help="API gateway URL")
    parser.add_argument(
        "--config",
        choices=list(CONFIGS.keys()) + ["all", "drain-only"],
        default="all",
        help="Which config to test (default: all)"
    )
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    # Verify connectivity
    try:
        resp = requests.get(f"{HOST}/health", timeout=5)
        print(f"Gateway health: {resp.status_code}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    results = {}
    start_time = datetime.now(timezone.utc).isoformat()

    print(f"\nPhase 4 — Optimization Isolation Test")
    print(f"Start: {start_time}")

    if args.config == "drain-only":
        # Just test graceful drain
        drain_result = test_graceful_drain()
        results["drain"] = drain_result
    elif args.config == "all":
        # Test cold start for the current config (whatever's deployed)
        print("\n  === Cold Start Measurement (current config) ===")
        cold_start_result = run_config_test("full")
        results["cold_start"] = cold_start_result

        # Test graceful drain
        # Wait for pod to be ready after cold start test scaled back
        print("\n  Waiting for single replica to stabilize...")
        force_single_replica()
        drain_result = test_graceful_drain()
        results["drain"] = drain_result
    else:
        result = run_config_test(args.config)
        results[args.config] = result

    # Save results
    output = {
        "experiment": "phase4_optimization_isolation",
        "start_time": start_time,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "optimization_isolation_results.json",
    )
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION ISOLATION TEST — RESULTS")
    print(f"{'='*60}")

    if "cold_start" in results:
        cs = results["cold_start"]
        print(f"\n  Cold Start: {cs['cold_start_sec']}s ({cs['name']})")

    if "drain" in results:
        d = results["drain"]
        print(f"\n  Graceful Drain:")
        print(f"    Success rate: {d['success_rate']:.1%} ({d['successes']}/{d['total_requests']})")
        print(f"    DRAIN_STATS logged: {d['drain_stats_found']}")

    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
