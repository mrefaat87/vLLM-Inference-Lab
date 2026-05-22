"""
Stage 3 — Queue-Based Inference Load Test

Tests the EKS queue architecture (API gateway → RabbitMQ → worker → vLLM → Redis → SSE).
Measures queue overhead vs Stage 2's direct vLLM access.

Three targeted tests (not the full 28-test ramp — with a single worker, high
concurrency just builds a queue, and we already validated that in Stage 2):

1. Single request: end-to-end plumbing, baseline queue overhead
2. 5 concurrent, mixed sizes: queue ordering, per-size TTFT vs Stage 2
3. 15 concurrent, mixed sizes: queue builds up, queue_wait_time grows

NEW METRIC: queue_wait_ms — how long the job sat in RabbitMQ before a worker
picked it up. This is the ApproximateAgeOfOldestMessage equivalent.

Usage:
    python3 phase1/tests/stage3_load_test.py --host http://localhost:8080
    (requires kubectl port-forward running — see scripts/port_forward.sh)
"""

import json
import time
import threading
import argparse
import sys
from typing import Optional

import requests

HOST = ""

# Reuse the same PROMPTS dict from Stage 2 for apples-to-apples comparison.
# Short: ~15 input tokens, fast prefill — like a health check
# Medium: ~80 input tokens, moderate prefill — typical user query
# Long: ~300 input tokens, heavy prefill — context-heavy request
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

# Focused test rounds — validates queue architecture, not GPU limits
# WHY only 3 rounds: with a single worker, higher concurrency just grows
# the queue linearly. The interesting metric is queue_wait_time growth,
# not GPU saturation (we measured that in Stage 2).
TEST_ROUNDS = [
    # Round 1: single request — baseline plumbing test
    (1, ["short"]),
    # Round 2: 5 concurrent, mixed — same as Stage 2 baseline for comparison
    (5, ["short", "medium", "long", "short", "medium"]),
    # Round 3: 15 concurrent — queue builds up, queue_wait_time should grow
    (15, ["short"] * 5 + ["medium"] * 5 + ["long"] * 5),
]


def send_request(request_id: int, prompt_key: str, results: list, global_start: float):
    """Send a request through the queue pipeline and stream the response via SSE.

    WHY two HTTP calls (POST + GET): the queue architecture decouples submission
    from completion. POST /generate returns immediately with a job_id (like SQS
    SendMessage returning a MessageId). GET /stream/{job_id} subscribes to the
    result stream (like polling SQS ReceiveMessage, but push-based via SSE).
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
        # Step 1: Submit job to queue via API gateway
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
        job_id = job_data["job_id"]
        stream_url = job_data["stream_url"]

        submit_end = time.perf_counter()
        submit_latency = submit_end - submit_start

        # Step 2: Connect to SSE stream for results
        # WHY stream=True: SSE is a long-lived HTTP connection. We read chunks
        # as they arrive rather than waiting for the full response.
        stream_resp = requests.get(
            f"{HOST}{stream_url}",
            stream=True,
            headers={"Accept": "text/event-stream"},
            timeout=(10, 300),  # 10s connect, 300s read (long generations)
        )

        if stream_resp.status_code != 200:
            raise RuntimeError(
                f"Stream returned {stream_resp.status_code}: {stream_resp.text}"
            )

        # Parse SSE events — format: "data: {json}\n\n"
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

            # Token event: {"token": "..."}
            if "token" in data:
                token_times.append(time.perf_counter())
                tokens.append(data["token"])

            # Completion event: {"status": "complete", "metrics": {...}}
            # WHY parse server metrics: the worker calculates queue_wait_ms,
            # ttft_ms, etc. on the server side. More accurate than client-side
            # measurement because it removes network latency from the calculation.
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

    # Client-side metrics (includes network latency from laptop → K8s → back)
    client_ttft = token_times[0] - submit_start
    tbts = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
    avg_tbt = sum(tbts) / len(tbts) if tbts else None

    results[request_id] = {
        "request_id": request_id,
        "prompt_type": prompt_key,
        "label": prompt_config["label"],
        # Client-side measurements (includes network + queue overhead)
        "client_ttft_sec": round(client_ttft, 4),
        "client_avg_tbt_sec": round(avg_tbt, 5) if avg_tbt else None,
        "client_total_sec": round(request_end - submit_start, 3),
        "client_tokens_per_sec": round(
            len(tokens) / (request_end - token_times[0]), 1
        ) if len(token_times) > 1 else 0,
        # Server-side measurements (from worker, more accurate for GPU perf)
        "server_queue_wait_ms": round(queue_wait_ms, 1) if queue_wait_ms else None,
        "server_ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
        "server_total_ms": round(total_ms, 1) if total_ms else None,
        "server_tokens_per_sec": server_tokens_per_sec,
        "tokens_out": len(tokens),
        # Timeline for scheduling analysis
        "abs_start": round(submit_start - global_start, 4),
        "abs_first_token": round(token_times[0] - global_start, 4),
        "abs_end": round(request_end - global_start, 4),
    }


def run_round(prompt_mix: list) -> dict:
    """Fire a batch of concurrent requests through the queue pipeline."""
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

    # Group by prompt type
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

    # Scheduling order by first token time
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


def main():
    global HOST
    parser = argparse.ArgumentParser(
        description="Stage 3 queue-based inference load test"
    )
    parser.add_argument(
        "--host", default="http://localhost:8080",
        help="API gateway URL (default: http://localhost:8080 via port-forward)"
    )
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    print("Stage 3 — Queue-Based Inference Load Test")
    print(f"Target: {HOST} (API gateway → RabbitMQ → worker → vLLM → Redis → SSE)")
    print(f"Prompt sizes: Short (~15 tok), Medium (~80 tok), Long (~300 tok)")
    print(f"{'=' * 120}\n")

    # Verify connectivity before starting tests
    print("Checking API gateway health...")
    try:
        health = requests.get(f"{HOST}/health", timeout=5)
        if health.status_code == 200:
            print(f"  API gateway: {health.json()}")
        else:
            print(f"  WARNING: health check returned {health.status_code}: {health.text}")
            print("  Continuing anyway (worker might not be ready yet)")
    except Exception as e:
        print(f"  ERROR: Cannot reach API gateway at {HOST}: {e}")
        print("  Is port-forward running? (bash scripts/port_forward.sh)")
        sys.exit(1)

    print()

    all_results = []

    for i, (n, prompt_mix) in enumerate(TEST_ROUNDS):
        # Pause between rounds to let GPU cool down and queue drain
        if all_results:
            print("\n  Waiting 5s for queue to drain...\n")
            time.sleep(5)

        mix_summary = {k: prompt_mix.count(k) for k in set(prompt_mix)}
        print(f"--- Test {i + 1}/3: N={n} | Mix: {mix_summary} ---")

        result = run_round(prompt_mix)
        all_results.append(result)

        # Print per-type summary with queue_wait_time (the new metric)
        header = (
            f"  {'Type':<8} {'N':>3}  {'TTFT p50':>9}  {'TTFT max':>9}  "
            f"{'QWait p50':>10}  {'QWait max':>10}  "
            f"{'Total avg':>10}  {'Tok/s avg':>10}"
        )
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
            f"  Scheduling order (first 20 by TTFT): "
            f"{result['scheduling_order_first20']}"
        )
        if result["failures"]:
            print(f"  FAILURES: {result['failures']}")
            for f in result.get("failed_requests", []):
                print(f"    [{f['label']}] {f.get('error', 'unknown')}")

    # ---------- Overall Analysis ----------
    print(f"\n{'=' * 120}")
    print("ANALYSIS — Queue Architecture Overhead vs Stage 2 Direct Access")
    print(f"{'=' * 120}\n")

    # Queue wait time progression (the key new metric)
    print("Queue Wait Time Progression:")
    print("  (This is how long jobs sit in RabbitMQ before a worker picks them up)")
    print()
    for result in all_results:
        n = result["concurrency"]
        all_qwaits = []
        for ts in result["by_type"].values():
            all_qwaits.append(ts["queue_wait_max_ms"])
        max_qwait = max(all_qwaits) if all_qwaits else 0
        bar = "█" * max(1, int(max_qwait / 10))
        print(f"  N={n:>2}: max queue_wait = {max_qwait:>8.1f}ms  {bar}")

    print()

    # TTFT progression by type
    for pt in ["short", "medium", "long"]:
        print(f"\n{pt.upper()} prompt TTFT progression (client-side, includes queue wait):")
        for result in all_results:
            if pt in result["by_type"]:
                ts = result["by_type"][pt]
                bar = "█" * max(1, int(ts["client_ttft_max"] * 10))
                print(
                    f"  N={result['concurrency']:>2}: "
                    f"p50={ts['client_ttft_p50']:.4f}s  "
                    f"max={ts['client_ttft_max']:.4f}s  "
                    f"qwait_max={ts['queue_wait_max_ms']:.0f}ms  {bar}"
                )

    # Summary comparison guidance
    print(f"\n\n{'=' * 120}")
    print("COMPARISON GUIDANCE — Stage 2 (direct vLLM) vs Stage 3 (queue-based)")
    print(f"{'=' * 120}\n")
    print("Expected differences:")
    print("  1. queue_wait_ms: ~0ms at N=1, growing at N=15 (new metric, not in Stage 2)")
    print("  2. TTFT: slightly higher than Stage 2 due to queue + Redis + SSE hops")
    print("     Expected overhead: 5-20ms for N=1 (RabbitMQ enqueue + dequeue + Redis pub)")
    print("  3. TBT/throughput: should be identical — GPU doesn't care about the queue")
    print("  4. At N=15: queue_wait_time grows because single worker processes ~5 concurrent")
    print("     remaining 10 jobs wait in RabbitMQ → this is the scaling signal for Phase 3")
    print()
    print("Key insight: if queue_wait_time at N=15 is > 1s, that proves the need for")
    print("autoscaling (Phase 3). The queue is the backpressure mechanism that Stage 2 lacked.")

    # Save results
    output_file = "phase1/tests/stage3_load_results.json"
    slim_results = []
    for r in all_results:
        slim = {k: v for k, v in r.items() if k not in ("all_requests", "failed_requests")}
        slim["request_details"] = r["all_requests"]
        slim["failures"] = r.get("failed_requests", [])
        slim_results.append(slim)

    with open(output_file, "w") as f:
        json.dump(slim_results, f, indent=2)
    print(f"\nRaw results saved to {output_file}")


if __name__ == "__main__":
    main()
