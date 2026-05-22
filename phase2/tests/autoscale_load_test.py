"""
Phase 2 — Autoscale Load Test

Validates KEDA + Karpenter autoscaling by running 4 phases:

1. Baseline (N=5):    Verify single worker handles 5 concurrent (matches Phase 1)
2. Trigger (N=15):    Fire 15 concurrent → queue depth spikes → KEDA scales to 2 workers
3. Post-scale (N=15): With 2 workers, measure improved queue_wait_time
4. Scale-down:        Stop sending, verify KEDA scales back to 1

KEY COMPARISON: Phase 1 at N=15 had queue_wait max = 12s.
With 2 workers, target is queue_wait < 2s at N=15.

This test reuses the same prompts and send_request() pattern from Phase 1's
stage3_load_test.py for apples-to-apples comparison.

Usage:
    # Start port-forward first
    bash phase2/scripts/port-forward-phase2.sh

    # In another terminal
    python3 phase2/tests/autoscale_load_test.py --host http://localhost:8080

    # With custom wait time for scale-up
    python3 phase2/tests/autoscale_load_test.py --host http://localhost:8080 --scale-wait 300
"""

import json
import time
import threading
import argparse
import subprocess
import sys
from typing import Optional

import requests

HOST = ""

# Same prompts as Phase 1 — apples-to-apples comparison
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
}


def send_request(request_id: int, prompt_key: str, results: list, global_start: float):
    """Send a request through the queue pipeline and stream the response via SSE.

    Identical to Phase 1's send_request() — same two-phase pattern:
    POST /generate → GET /stream/{job_id} (SSE).
    """
    prompt_config = PROMPTS[prompt_key]
    submit_start = time.perf_counter()

    token_times = []
    tokens = []
    error = None
    queue_wait_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    total_ms: Optional[float] = None
    server_tokens_per_sec: Optional[float] = None

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

        # Step 2: Stream results via SSE
        stream_resp = requests.get(
            f"{HOST}{stream_url}",
            stream=True,
            headers={"Accept": "text/event-stream"},
            timeout=(10, 300),
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
                metrics = data.get("metrics", {})
                queue_wait_ms = metrics.get("queue_wait_ms")
                ttft_ms = metrics.get("ttft_ms")
                total_ms = metrics.get("total_ms")
                server_tokens_per_sec = metrics.get("tokens_per_sec")
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
            "error": error or "no tokens received",
            "total_sec": round(request_end - submit_start, 3),
        }
        return

    client_ttft = token_times[0] - submit_start
    tbts = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
    avg_tbt = sum(tbts) / len(tbts) if tbts else None

    results[request_id] = {
        "request_id": request_id,
        "prompt_type": prompt_key,
        "label": prompt_config["label"],
        "client_ttft_sec": round(client_ttft, 4),
        "client_avg_tbt_sec": round(avg_tbt, 5) if avg_tbt else None,
        "client_total_sec": round(request_end - submit_start, 3),
        "client_tokens_per_sec": round(
            len(tokens) / (request_end - token_times[0]), 1
        ) if len(token_times) > 1 else 0,
        "server_queue_wait_ms": round(queue_wait_ms, 1) if queue_wait_ms else None,
        "server_ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
        "server_total_ms": round(total_ms, 1) if total_ms else None,
        "server_tokens_per_sec": server_tokens_per_sec,
        "tokens_out": len(tokens),
        "abs_start": round(submit_start - global_start, 4),
        "abs_first_token": round(token_times[0] - global_start, 4),
        "abs_end": round(request_end - global_start, 4),
    }


def run_round(prompt_mix: list) -> dict:
    """Fire a batch of concurrent requests and aggregate results."""
    n = len(prompt_mix)
    results = [None] * n
    threads = []
    global_start = time.perf_counter()
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    for i, prompt_key in enumerate(prompt_mix):
        t = threading.Thread(
            target=send_request,
            args=(i, prompt_key, results, global_start),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    wall_end = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    successes = [r for r in results if r and "error" not in r]
    failures = [r for r in results if r and "error" in r]

    by_type = {}
    for r in successes:
        pt = r["prompt_type"]
        if pt not in by_type:
            by_type[pt] = {
                "client_ttfts": [], "server_ttfts_ms": [],
                "queue_waits_ms": [], "totals": [],
                "tok_rates": [], "tokens_out": [],
            }
        by_type[pt]["client_ttfts"].append(r["client_ttft_sec"])
        if r.get("server_ttft_ms"):
            by_type[pt]["server_ttfts_ms"].append(r["server_ttft_ms"])
        if r.get("server_queue_wait_ms") is not None:
            by_type[pt]["queue_waits_ms"].append(r["server_queue_wait_ms"])
        by_type[pt]["totals"].append(r["client_total_sec"])
        by_type[pt]["tok_rates"].append(r["client_tokens_per_sec"])
        by_type[pt]["tokens_out"].append(r["tokens_out"])

    type_summaries = {}
    for pt, data in by_type.items():
        ttfts = sorted(data["client_ttfts"])
        qwaits = sorted(data["queue_waits_ms"]) if data["queue_waits_ms"] else [0]
        type_summaries[pt] = {
            "count": len(ttfts),
            "client_ttft_p50": round(ttfts[len(ttfts) // 2], 4),
            "client_ttft_max": round(max(ttfts), 4),
            "queue_wait_p50_ms": round(qwaits[len(qwaits) // 2], 1),
            "queue_wait_max_ms": round(max(qwaits), 1),
            "total_avg": round(sum(data["totals"]) / len(data["totals"]), 3),
            "tok_s_avg": round(sum(data["tok_rates"]) / len(data["tok_rates"]), 1),
            "avg_tokens_out": round(
                sum(data["tokens_out"]) / len(data["tokens_out"]), 1
            ),
        }

    if successes:
        by_first_token = sorted(successes, key=lambda r: r["abs_first_token"])
        scheduling_order = "".join(
            r["label"] for r in by_first_token[:min(20, len(by_first_token))]
        )
    else:
        scheduling_order = "N/A"

    return {
        "concurrency": n,
        "wall_start": wall_start,
        "wall_end": wall_end,
        "prompt_mix": {k: prompt_mix.count(k) for k in set(prompt_mix)},
        "successes": len(successes),
        "failures": len(failures),
        "by_type": type_summaries,
        "scheduling_order_first20": scheduling_order,
        "all_requests": successes,
        "failed_requests": failures,
    }


def print_round_results(result: dict, label: str):
    """Print formatted results for a test round."""
    header = (
        f"  {'Type':<8} {'N':>3}  {'TTFT p50':>9}  {'TTFT max':>9}  "
        f"{'QWait p50':>10}  {'QWait max':>10}  "
        f"{'Total avg':>10}  {'Tok/s avg':>10}"
    )
    print(f"\n{label}")
    print(header)
    print(f"  {'-' * 90}")

    for pt in ["short", "medium", "long"]:
        if pt in result["by_type"]:
            ts = result["by_type"][pt]
            print(
                f"  {pt:<8} {ts['count']:>3}  "
                f"{ts['client_ttft_p50']:>9.4f}  "
                f"{ts['client_ttft_max']:>9.4f}  "
                f"{ts['queue_wait_p50_ms']:>9.1f}ms  "
                f"{ts['queue_wait_max_ms']:>9.1f}ms  "
                f"{ts['total_avg']:>10.3f}  "
                f"{ts['tok_s_avg']:>10.1f}"
            )

    print(
        f"  Scheduling order: {result['scheduling_order_first20']}"
    )
    if result["failures"]:
        print(f"  FAILURES: {result['failures']}")
        for f in result.get("failed_requests", []):
            print(f"    [{f['label']}] {f.get('error', 'unknown')}")


def sustained_load(duration_sec: int = 60, rps: float = 2.0):
    """Send sustained load for a given duration to keep queue depth above threshold.

    WHY sustained load: a single burst of 15 requests drains within ~20-30s.
    KEDA polls every 15s and needs to see queue_depth > 5 consistently to trigger
    scale-up. Sustained load at 2 req/s for 60s = 120 requests, ensuring the queue
    stays full long enough for KEDA to react.

    This is fire-and-forget — we don't collect detailed metrics here, just keep
    the queue saturated. The actual measurement happens in the post-scale N=15 round.
    """
    interval = 1.0 / rps
    end_time = time.time() + duration_sec
    request_count = 0
    errors = 0

    print(f"  Sending sustained load: ~{rps} req/s for {duration_sec}s...")

    while time.time() < end_time:
        # Pick a random prompt type — long prompts keep the GPU busy longer,
        # which builds up queue depth more effectively
        prompt_key = ["short", "medium", "long"][request_count % 3]
        prompt_config = PROMPTS[prompt_key]

        try:
            resp = requests.post(
                f"{HOST}/generate",
                json={
                    "prompt": prompt_config["text"],
                    "max_tokens": prompt_config["max_tokens"],
                    "temperature": 0.7,
                },
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if resp.status_code == 202:
                request_count += 1
            else:
                errors += 1
        except Exception:
            errors += 1

        elapsed = int(time.time() + duration_sec - end_time)
        desired, ready = get_worker_replicas()
        if request_count % 5 == 0:
            print(f"\r  [{elapsed:>3}s] Sent {request_count} requests, "
                  f"errors={errors}, workers desired={desired} ready={ready}", end="", flush=True)

        # Check if KEDA has already scaled up — we can stop early
        if desired >= 2:
            print(f"\n  KEDA triggered scale-up after {request_count} requests!")
            return request_count, errors, True

        time.sleep(interval)

    print(f"\n  Sustained load complete: {request_count} requests sent, {errors} errors")
    desired, ready = get_worker_replicas()
    return request_count, errors, desired >= 2


def get_worker_replicas() -> tuple[int, int]:
    """Get current desired and ready replicas for vllm-worker."""
    try:
        output = subprocess.check_output(
            ["kubectl", "get", "deployment", "vllm-worker",
             "-n", "inference-system",
             "-o", "jsonpath={.spec.replicas},{.status.readyReplicas}"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        parts = output.strip().split(",")
        desired = int(parts[0]) if parts[0] else 0
        ready = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        return desired, ready
    except Exception:
        return 0, 0


def wait_for_replicas(target_desired: int, target_ready: int, timeout: int = 600) -> bool:
    """Wait for vllm-worker to reach the target replica count."""
    start = time.time()
    while time.time() - start < timeout:
        desired, ready = get_worker_replicas()
        elapsed = int(time.time() - start)
        print(f"\r  [{elapsed:>3}s] vllm-worker: desired={desired} ready={ready}"
              f" (target: desired={target_desired} ready={target_ready})", end="", flush=True)

        if desired >= target_desired and ready >= target_ready:
            print()
            return True

        time.sleep(10)

    print()
    return False


def main():
    global HOST
    parser = argparse.ArgumentParser(
        description="Phase 2 autoscale load test — validates KEDA + Karpenter scaling"
    )
    parser.add_argument(
        "--host", default="http://localhost:8080",
        help="API gateway URL (default: http://localhost:8080 via port-forward)"
    )
    parser.add_argument(
        "--scale-wait", type=int, default=360,
        help="Max seconds to wait for scale-up (default: 360 = 6 min for Spot + model load)"
    )
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    print("=" * 120)
    print("  Phase 2 — Autoscale Load Test")
    print("  Validates KEDA scaling: queue_depth > 5 → scale vllm-worker 1 → 2")
    print(f"  Target: {HOST}")
    print("=" * 120)
    print()

    # ---- Health check ----
    print("Checking API gateway health...")
    try:
        health = requests.get(f"{HOST}/health", timeout=5)
        if health.status_code == 200:
            print(f"  API gateway: {health.json()}")
        else:
            print(f"  WARNING: health returned {health.status_code}: {health.text}")
    except Exception as e:
        print(f"  ERROR: Cannot reach API gateway at {HOST}: {e}")
        print("  Is port-forward running? (bash phase2/scripts/port-forward-phase2.sh)")
        sys.exit(1)

    desired, ready = get_worker_replicas()
    print(f"  vllm-worker: desired={desired} ready={ready}")
    print()

    all_results = {}

    # ==== Phase 1: Baseline (N=5) ====
    print("━" * 80)
    print("PHASE 1/4: Baseline — N=5 concurrent (should match Phase 1 results)")
    print("━" * 80)
    mix_5 = ["short", "medium", "long", "short", "medium"]
    result_baseline = run_round(mix_5)
    all_results["baseline_n5"] = result_baseline
    print_round_results(result_baseline, "  Baseline N=5 Results:")
    print()

    # ==== Phase 2: Trigger scale-up with sustained load ====
    print("━" * 80)
    print("PHASE 2/4: Trigger scale-up — sustained load to keep queue above threshold")
    print("━" * 80)
    print("  First, capture pre-scale N=15 metrics for comparison...")
    mix_15 = ["short"] * 5 + ["medium"] * 5 + ["long"] * 5
    result_trigger = run_round(mix_15)
    all_results["trigger_n15"] = result_trigger
    print_round_results(result_trigger, "  Pre-scale N=15 Results (single worker):")
    print()

    # Now send sustained load to trigger KEDA
    # WHY sustained load: a burst of 15 drains in ~20-30s. KEDA polls every 15s.
    # Sustained load at 2 req/s keeps queue_depth > 5 long enough for KEDA to see it.
    print("  Sending sustained load to trigger KEDA scale-up...")
    print("  (KEDA polls every 15s, needs queue_depth > 5 to trigger)")
    scale_up_start = time.time()
    req_count, err_count, keda_triggered = sustained_load(duration_sec=120, rps=3.0)
    all_results["sustained_load"] = {
        "requests_sent": req_count, "errors": err_count, "keda_triggered": keda_triggered
    }
    print()

    if not keda_triggered:
        print("  Waiting for KEDA to react (may take another polling cycle)...")

    # Wait for scale-up to complete (Spot node + model load)
    print("  Waiting for second worker to become ready...")
    print("  (Spot node ~60-90s + model download/load ~2-4 min)")

    scaled = wait_for_replicas(
        target_desired=2, target_ready=2, timeout=args.scale_wait
    )

    if scaled:
        scale_up_time = int(time.time() - scale_up_start)
        print(f"  Scale-up complete in {scale_up_time}s!")
        all_results["scale_up_seconds"] = scale_up_time
    else:
        print(f"  WARNING: Scale-up did not complete within {args.scale_wait}s.")
        print("  Check: kubectl get scaledobject -n inference-system")
        print("  Check: kubectl get pods -n inference-system")
        desired, ready = get_worker_replicas()
        print(f"  Current state: desired={desired} ready={ready}")

        if ready < 2:
            print("\n  Continuing with current state (results may not show full improvement)...")
        all_results["scale_up_seconds"] = args.scale_wait
        all_results["scale_up_timeout"] = True

    print()

    # ==== Phase 3: Post-scale validation (N=15) ====
    print("━" * 80)
    print("PHASE 3/4: Post-scale validation — N=15 with 2 workers")
    print("━" * 80)
    print("  Waiting 10s for second worker to fully warm up...")
    time.sleep(10)

    result_postscale = run_round(mix_15)
    all_results["postscale_n15"] = result_postscale
    print_round_results(result_postscale, "  Post-scale N=15 Results (2 workers):")
    print()

    # ==== Phase 4: Scale-down verification ====
    print("━" * 80)
    print("PHASE 4/4: Scale-down — wait for KEDA to scale back to 1")
    print("━" * 80)
    print("  Waiting for queue to drain and KEDA cooldownPeriod (60s) to pass...")
    print("  (Total wait: ~2 min = 60s cooldown + 60s Karpenter consolidation)")

    scaled_down = wait_for_replicas(
        target_desired=1, target_ready=1, timeout=300
    )

    if scaled_down:
        print("  Scale-down complete! Back to 1 replica.")
        all_results["scale_down_success"] = True
    else:
        print("  Scale-down did not complete within 5 min (may need more time).")
        all_results["scale_down_success"] = False

    print()

    # ==== Comparison ====
    print("=" * 120)
    print("COMPARISON — Single Worker vs Two Workers at N=15")
    print("=" * 120)
    print()

    # Queue wait comparison
    print("Queue Wait Time at N=15:")
    for label, key in [("1 worker (pre-scale)", "trigger_n15"),
                       ("2 workers (post-scale)", "postscale_n15")]:
        r = all_results.get(key, {})
        if not r:
            continue
        all_qwaits = [ts["queue_wait_max_ms"] for ts in r.get("by_type", {}).values()]
        max_qwait = max(all_qwaits) if all_qwaits else 0
        bar = "█" * max(1, int(max_qwait / 100))
        print(f"  {label:<30}: max queue_wait = {max_qwait:>8.1f}ms  {bar}")

    print()

    # Per-type comparison
    for pt in ["short", "medium", "long"]:
        print(f"\n{pt.upper()} prompt comparison:")
        for label, key in [("1 worker", "trigger_n15"), ("2 workers", "postscale_n15")]:
            r = all_results.get(key, {})
            if not r or pt not in r.get("by_type", {}):
                continue
            ts = r["by_type"][pt]
            print(
                f"  {label:<12}: TTFT p50={ts['client_ttft_p50']:.4f}s  "
                f"max={ts['client_ttft_max']:.4f}s  "
                f"qwait_max={ts['queue_wait_max_ms']:.0f}ms  "
                f"tok/s={ts['tok_s_avg']:.1f}"
            )

    print()

    # Summary
    trigger = all_results.get("trigger_n15", {})
    post = all_results.get("postscale_n15", {})

    trigger_max = max(
        (ts["queue_wait_max_ms"] for ts in trigger.get("by_type", {}).values()),
        default=0,
    )
    post_max = max(
        (ts["queue_wait_max_ms"] for ts in post.get("by_type", {}).values()),
        default=0,
    )

    if trigger_max > 0 and post_max > 0:
        improvement = ((trigger_max - post_max) / trigger_max) * 100
        print(f"RESULT: Queue wait max reduced from {trigger_max:.0f}ms → {post_max:.0f}ms "
              f"({improvement:.0f}% improvement)")
        if post_max < 2000:
            print("TARGET MET: queue_wait < 2s with 2 workers at N=15")
        else:
            print("TARGET MISSED: queue_wait still > 2s (may need more warm-up time)")
    else:
        print("INCOMPLETE: Could not compute comparison (check for errors above)")

    # Save results
    output_file = "phase2/tests/autoscale_results.json"
    save_results = {}
    for key, val in all_results.items():
        if isinstance(val, dict) and "all_requests" in val:
            save_results[key] = {
                k: v for k, v in val.items()
                if k not in ("all_requests", "failed_requests")
            }
            save_results[key]["request_details"] = val["all_requests"]
            save_results[key]["failures"] = val.get("failed_requests", [])
        else:
            save_results[key] = val

    with open(output_file, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nRaw results saved to {output_file}")


if __name__ == "__main__":
    main()
