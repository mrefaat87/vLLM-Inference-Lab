"""
Test 9: Prefix Caching — Explicit On vs Off

Measures the TTFT benefit when multiple requests share the same prompt prefix.
Prefix caching is ON by default in vLLM v0.17.1.

Think of it like a CDN: the first request computes the KV cache for the shared
prefix ("cache miss"), subsequent requests reuse it ("cache hit"). The shared
prefix is like static content — compute it once, serve from cache forever.

This script handles BOTH the ON and OFF phases:
  - Run with --phase on   (default server config, prefix caching enabled)
  - Run with --phase off  (server restarted with --no-prefix-caching)
  - Run with --phase both (runs ON, saves, prints instructions to restart)

Usage:
  python3 stage2_test9_prefix_caching.py --host http://<ip>:8000 --phase on
  python3 stage2_test9_prefix_caching.py --host http://<ip>:8000 --phase off
"""

import json
import time
import threading
import argparse
import requests

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# Long shared system prompt (~500 tokens) — identical across all requests
# This is the "static content" that prefix caching can reuse
SYSTEM_PROMPT = """You are an expert systems architect specializing in distributed systems,
cloud infrastructure, and high-availability design. You have deep experience with AWS,
Kubernetes, and large-scale data processing pipelines. When answering questions, always
consider scalability, fault tolerance, and cost optimization.

Background context for this conversation:
The organization operates a global SaaS platform serving 50 million monthly active users
across 12 AWS regions. The platform consists of 200+ microservices running on EKS clusters
with Karpenter for node provisioning. The data layer includes Aurora PostgreSQL for
transactional data, DynamoDB for session management, ElastiCache Redis for caching, and
S3 for object storage. The event streaming backbone uses Amazon MSK (Kafka) with
Schema Registry for event evolution. Observability is handled by Prometheus, Grafana,
and distributed tracing via OpenTelemetry and Jaeger.

Key constraints:
- P99 API latency must remain under 200ms for all user-facing endpoints
- System must maintain 99.99% availability (less than 52 minutes downtime per year)
- Data must be encrypted at rest and in transit (SOC2 Type II compliance)
- Cost optimization is critical — the infrastructure bill is $2.1M/month
- The team follows GitOps practices with ArgoCD for all deployments
- All changes require canary deployments with automated rollback on metric regression

Recent incidents that inform our architecture decisions:
1. A cascading failure in the payment service caused 45 minutes of downtime when a
   downstream dependency (fraud detection) became unresponsive and circuit breakers
   were not properly configured.
2. A Kafka consumer lag spike during Black Friday caused order processing delays of
   up to 15 minutes, triggering SLA violations with enterprise customers.
3. A DynamoDB hot partition issue in the session store caused authentication failures
   for 12% of users during a product launch event."""

# 10 different user questions — each shares the system prompt but has unique suffix
USER_QUESTIONS = [
    "How should we redesign the circuit breaker configuration to prevent cascading failures like the payment service incident?",
    "What Kafka consumer scaling strategy would prevent the Black Friday lag spike from recurring?",
    "How can we resolve the DynamoDB hot partition issue for our session store?",
    "Design a multi-region failover strategy that maintains our 99.99% SLA.",
    "What caching strategy should we implement to reduce our $2.1M monthly infrastructure cost?",
    "How should we implement canary deployments for database schema migrations?",
    "Design an event-driven architecture for real-time fraud detection with sub-100ms latency.",
    "What observability improvements would help us detect cascading failures before they impact users?",
    "How should we design our EKS cluster autoscaling to handle 10x traffic spikes during launches?",
    "What data partitioning strategy would improve our Aurora PostgreSQL query performance?",
]


def send_request(request_id: int, question: str, results: list, global_start: float):
    """Send one streaming request with the shared system prompt + unique question."""
    url = f"{HOST}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": 150,
        "stream": True,
        "temperature": 0.7,
    }

    token_times = []
    tokens = []
    error = None
    request_start = time.perf_counter()

    try:
        resp = requests.post(url, json=payload,
                             headers={"Content-Type": "application/json"},
                             stream=True, timeout=(60, 300))
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode()
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content", "")
            if text:
                token_times.append(time.perf_counter())
                tokens.append(text)
    except Exception as e:
        error = str(e)

    request_end = time.perf_counter()

    if error or not token_times:
        results[request_id] = {
            "request_id": request_id,
            "error": error or "no tokens received",
            "total_sec": round(request_end - request_start, 3),
        }
        return

    ttft = token_times[0] - request_start
    total = request_end - request_start
    decode_time = request_end - token_times[0]
    tok_s = len(tokens) / decode_time if decode_time > 0 else 0

    results[request_id] = {
        "request_id": request_id,
        "question_idx": request_id % len(USER_QUESTIONS),
        "ttft_sec": round(ttft, 4),
        "total_sec": round(total, 3),
        "tokens": len(tokens),
        "tokens_per_sec": round(tok_s, 1),
        "abs_start": round(request_start - global_start, 4),
        "abs_first_token": round(token_times[0] - global_start, 4),
        "abs_end": round(request_end - global_start, 4),
    }


def run_serial_test(n: int = 10, trials: int = 3) -> dict:
    """Send n requests serially. Request 1 is cold, rest should be warm."""
    all_trials = []

    for trial in range(trials):
        if trial > 0:
            time.sleep(3)

        print(f"\n  Trial {trial + 1}/{trials} (serial):")
        results = [None] * n
        global_start = time.perf_counter()

        for i in range(n):
            # Serial: send one, wait for completion, then send next
            send_request(i, USER_QUESTIONS[i % len(USER_QUESTIONS)],
                        results, global_start)
            r = results[i]
            if r and "error" not in r:
                label = "COLD" if i == 0 and trial == 0 else "warm"
                print(f"    Req {i+1:>2} [{label}]: TTFT={r['ttft_sec']:.4f}s | "
                      f"tok/s={r['tokens_per_sec']:.1f} | tokens={r['tokens']}")
            else:
                print(f"    Req {i+1:>2}: ERROR - {r.get('error', 'unknown') if r else 'None'}")

        successes = [r for r in results if r and "error" not in r]
        all_trials.append(successes)

    # Aggregate across trials
    if not all_trials or not all_trials[0]:
        return {"error": "all trials failed"}

    # Average cold TTFT (request 0 across trials)
    cold_ttfts = [t[0]["ttft_sec"] for t in all_trials if len(t) > 0]
    # Average warm TTFT (requests 1-9 across trials)
    warm_ttfts = [r["ttft_sec"] for t in all_trials for r in t[1:]]

    cold_avg = sum(cold_ttfts) / len(cold_ttfts) if cold_ttfts else 0
    warm_avg = sum(warm_ttfts) / len(warm_ttfts) if warm_ttfts else 0
    speedup = cold_avg / warm_avg if warm_avg > 0 else 0

    # Per-request TTFT averages across trials
    per_request_ttfts = {}
    for i in range(n):
        ttfts_for_i = [t[i]["ttft_sec"] for t in all_trials if len(t) > i]
        if ttfts_for_i:
            per_request_ttfts[f"req_{i+1}"] = round(sum(ttfts_for_i) / len(ttfts_for_i), 4)

    return {
        "cold_ttft_avg": round(cold_avg, 4),
        "warm_ttft_avg": round(warm_avg, 4),
        "speedup_ratio": round(speedup, 2),
        "per_request_ttft": per_request_ttfts,
        "trials": trials,
    }


def run_concurrent_test(n: int = 10, trials: int = 3) -> dict:
    """Send n requests concurrently. All share the prefix."""
    all_trials = []

    for trial in range(trials):
        if trial > 0:
            time.sleep(5)

        print(f"\n  Trial {trial + 1}/{trials} (concurrent, N={n}):")
        results = [None] * n
        threads = []
        global_start = time.perf_counter()
        wall_start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        for i in range(n):
            t = threading.Thread(target=send_request,
                                args=(i, USER_QUESTIONS[i % len(USER_QUESTIONS)],
                                      results, global_start))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        wall_end = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        wall_duration = time.perf_counter() - global_start

        successes = [r for r in results if r and "error" not in r]
        failures = [r for r in results if r and "error" in r]

        if successes:
            ttfts = sorted([r["ttft_sec"] for r in successes])
            totals = sorted([r["total_sec"] for r in successes])
            print(f"    OK={len(successes)} Fail={len(failures)} | "
                  f"TTFT p50={ttfts[len(ttfts)//2]:.4f}s | "
                  f"TTFT max={max(ttfts):.4f}s | "
                  f"Wall={wall_duration:.1f}s")

        all_trials.append({
            "wall_start": wall_start,
            "successes": len(successes),
            "failures": len(failures),
            "ttfts": [r["ttft_sec"] for r in successes] if successes else [],
            "totals": [r["total_sec"] for r in successes] if successes else [],
        })

    # Aggregate across trials
    all_ttfts = [t for trial in all_trials for t in trial["ttfts"]]
    all_totals = [t for trial in all_trials for t in trial["totals"]]

    if not all_ttfts:
        return {"error": "all concurrent trials failed"}

    all_ttfts.sort()
    all_totals.sort()

    return {
        "wall_start": all_trials[0]["wall_start"],
        "successes": sum(t["successes"] for t in all_trials),
        "failures": sum(t["failures"] for t in all_trials),
        "ttft_p50": round(all_ttfts[len(all_ttfts) // 2], 4),
        "ttft_p99": round(all_ttfts[int(len(all_ttfts) * 0.99)], 4),
        "ttft_max": round(max(all_ttfts), 4),
        "total_p50": round(all_totals[len(all_totals) // 2], 3),
        "total_p99": round(all_totals[int(len(all_totals) * 0.99)], 3),
        "total_max": round(max(all_totals), 3),
        "trials": trials,
    }


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Test 9: Prefix Caching On vs Off")
    parser.add_argument("--host", required=True)
    parser.add_argument("--phase", choices=["on", "off"], required=True,
                        help="Run with prefix caching ON or OFF")
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    phase_label = "Prefix Caching ON (default)" if args.phase == "on" else "Prefix Caching OFF"
    print(f"Test 9: Prefix Caching — {phase_label}")
    print(f"Target: {HOST} | Model: {MODEL}")
    print(f"Shared system prompt: ~500 tokens")
    print(f"10 unique user questions as suffixes")
    print(f"{'='*100}\n")

    # Serial test: measure cold vs warm TTFT
    print("SERIAL TEST (10 requests, one at a time)")
    print("  Request 1 = cold prefix (must compute full KV cache)")
    print("  Requests 2-10 = warm prefix (KV cache should be reused)")
    serial_result = run_serial_test(n=10, trials=3)

    print(f"\n  Summary: cold={serial_result.get('cold_ttft_avg', 'N/A')}s | "
          f"warm={serial_result.get('warm_ttft_avg', 'N/A')}s | "
          f"speedup={serial_result.get('speedup_ratio', 'N/A')}x")

    # Concurrent test
    print(f"\n\nCONCURRENT TEST (10 requests at once)")
    concurrent_result = run_concurrent_test(n=10, trials=3)

    # Save results
    result_key = "prefix_on" if args.phase == "on" else "prefix_off"
    output_file = f"/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_test9_{args.phase}_results.json"

    output = {
        result_key: {
            "serial": serial_result,
            "concurrent_10": concurrent_result,
        }
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")

    if args.phase == "on":
        print(f"\n{'='*100}")
        print("NEXT: Restart server with --no-prefix-caching, then run:")
        print(f"  python3 stage2_test9_prefix_caching.py --host {HOST} --phase off")


if __name__ == "__main__":
    main()
