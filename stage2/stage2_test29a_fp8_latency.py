"""
Test 29A: FP8 KV Cache — Latency Invariance

Replicates Test 25's open-loop load generation against an FP8 KV cache server.
Hypothesis: FP8 doesn't change per-request latency — it prevents the system
from drowning under load by doubling KV capacity.

AWS analogy: Memory overcommit doesn't make individual requests faster. It
prevents capacity exhaustion at higher concurrency. The latency improvement
at high RPS is indirect — avoiding preemption, not faster compute.

Compare results against stage2_test25_results.json (FP16 baseline).

Usage: python3 stage2_test29a_fp8_latency.py --host http://<ip>:8000
"""

import json
import time
import threading
import argparse
import requests

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# Same RPS levels as Test 25 + extended range to probe beyond FP16 crash point
# Why 24 and 28: FP16 crashed at 20 RPS — FP8 should survive, so we probe further
RPS_LEVELS = [1, 2, 4, 8, 12, 16, 20, 24, 28]
DURATION_SEC = 60  # Sustain each rate for 60 seconds

# Same prompt as Test 25 — must be identical for comparison
PROMPT = ("Explain the concept of consistent hashing and why it matters "
          "for distributed caching systems. Be concise.")


def send_request_and_record(results: list, lock: threading.Lock, global_start: float):
    """Send one streaming request, record metrics thread-safely."""
    url = f"{HOST}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 100,
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
                             stream=True, timeout=(30, 300))
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
            text = chunk.get("choices", [{}])[0].get("text", "")
            if text:
                token_times.append(time.perf_counter())
                tokens.append(text)
    except Exception as e:
        error = str(e)

    request_end = time.perf_counter()

    result = {}
    if error or not token_times:
        result = {
            "error": error or "no tokens received",
            "total_sec": round(request_end - request_start, 3),
            "abs_start": round(request_start - global_start, 4),
        }
    else:
        ttft = token_times[0] - request_start
        total = request_end - request_start
        decode_time = request_end - token_times[0]
        tok_s = len(tokens) / decode_time if decode_time > 0 else 0

        result = {
            "ttft_sec": round(ttft, 4),
            "total_sec": round(total, 3),
            "tokens": len(tokens),
            "tok_s": round(tok_s, 1),
            "abs_start": round(request_start - global_start, 4),
        }

    with lock:
        results.append(result)


def percentile(sorted_vals: list, p: float) -> float:
    """Compute percentile from a sorted list."""
    idx = int(len(sorted_vals) * p / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def run_at_rps(rps: int, duration: int = 60) -> dict:
    """Send requests at a fixed rate for the given duration."""
    results = []
    lock = threading.Lock()
    global_start = time.perf_counter()
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    # Calculate inter-request interval
    interval = 1.0 / rps
    total_requests = rps * duration
    threads = []

    print(f"  Sending {total_requests} requests at {rps} RPS for {duration}s...")

    for i in range(total_requests):
        # Schedule request at the right time (open-loop)
        target_time = global_start + i * interval
        now = time.perf_counter()
        if target_time > now:
            time.sleep(target_time - now)

        t = threading.Thread(target=send_request_and_record,
                           args=(results, lock, global_start))
        t.start()
        threads.append(t)

    # Wait for all requests to complete (with a generous timeout)
    for t in threads:
        t.join(timeout=120)

    wall_duration = time.perf_counter() - global_start

    # Analyze results
    successes = [r for r in results if "error" not in r]
    failures = [r for r in results if "error" in r]

    if not successes:
        return {
            "rps": rps,
            "duration_sec": duration,
            "total_requests": total_requests,
            "wall_start": wall_start,
            "successes": 0,
            "failures": len(failures),
            "error_sample": failures[0]["error"] if failures else "unknown",
        }

    ttfts = sorted([r["ttft_sec"] for r in successes])
    totals = sorted([r["total_sec"] for r in successes])
    tok_rates = [r["tok_s"] for r in successes]
    total_tokens = sum(r["tokens"] for r in successes)

    return {
        "rps": rps,
        "duration_sec": duration,
        "total_requests": total_requests,
        "wall_start": wall_start,
        "wall_duration_sec": round(wall_duration, 1),
        "successes": len(successes),
        "failures": len(failures),
        "ttft_p50": round(percentile(ttfts, 50), 4),
        "ttft_p90": round(percentile(ttfts, 90), 4),
        "ttft_p95": round(percentile(ttfts, 95), 4),
        "ttft_p99": round(percentile(ttfts, 99), 4),
        "ttft_p999": round(percentile(ttfts, 99.9), 4),
        "ttft_max": round(max(ttfts), 4),
        "total_p50": round(percentile(totals, 50), 3),
        "total_p90": round(percentile(totals, 90), 3),
        "total_p95": round(percentile(totals, 95), 3),
        "total_p99": round(percentile(totals, 99), 3),
        "total_p999": round(percentile(totals, 99.9), 3),
        "total_max": round(max(totals), 3),
        "system_tok_s": round(total_tokens / wall_duration, 1),
    }


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Test 29A: FP8 KV Cache — Latency Invariance")
    parser.add_argument("--host", required=True)
    parser.add_argument("--duration", type=int, default=60, help="Seconds per RPS level")
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    print(f"Test 29A: FP8 KV Cache — Latency Invariance")
    print(f"Target: {HOST} | Model: {MODEL}")
    print(f"KV Cache: fp8_e5m2 (compare against Test 25 fp16 baseline)")
    print(f"RPS levels: {RPS_LEVELS}")
    print(f"Duration per level: {args.duration}s")
    print(f"{'='*130}\n")

    all_results = []

    # Header
    print(f"{'RPS':>4}  {'OK':>5}  {'Fail':>5}  "
          f"{'TTFT p50':>9}  {'TTFT p90':>9}  {'TTFT p95':>9}  {'TTFT p99':>9}  {'TTFT max':>9}  "
          f"{'Total p50':>10}  {'Total p99':>10}  {'Sys tok/s':>10}")
    print("-" * 130)

    for rps in RPS_LEVELS:
        # Pause between levels to let queue drain
        if all_results:
            print(f"  (cooling down 10s...)")
            time.sleep(10)

        result = run_at_rps(rps, args.duration)
        all_results.append(result)

        if "error_sample" in result:
            print(f"{rps:>4}  {result['successes']:>5}  {result['failures']:>5}  FAILED")
        else:
            print(f"{rps:>4}  {result['successes']:>5}  {result['failures']:>5}  "
                  f"{result['ttft_p50']:>9.4f}  {result['ttft_p90']:>9.4f}  "
                  f"{result['ttft_p95']:>9.4f}  {result['ttft_p99']:>9.4f}  "
                  f"{result['ttft_max']:>9.4f}  "
                  f"{result['total_p50']:>10.3f}  {result['total_p99']:>10.3f}  "
                  f"{result['system_tok_s']:>10.1f}")

    # Analysis
    print(f"\n{'='*130}")
    print("ANALYSIS — P99/P50 Ratio (tail latency divergence)")
    print(f"{'='*130}\n")

    for r in all_results:
        if "ttft_p50" in r and r["ttft_p50"] > 0:
            ttft_ratio = r["ttft_p99"] / r["ttft_p50"]
            total_ratio = r["total_p99"] / r["total_p50"]
            print(f"  RPS={r['rps']:>3}: TTFT P99/P50={ttft_ratio:>5.2f}x | "
                  f"Total P99/P50={total_ratio:>5.2f}x | "
                  f"Sys={r['system_tok_s']:.0f} tok/s")

    # Find the "SLA-safe" RPS — where P99 TTFT exceeds 1 second
    for r in all_results:
        if "ttft_p99" in r and r["ttft_p99"] > 1.0:
            idx = all_results.index(r)
            prev_rps = all_results[idx - 1]['rps'] if idx > 0 else 0
            print(f"\n  SLA breach: TTFT P99 exceeds 1s at {r['rps']} RPS — "
                  f"FP8 SLA-safe capacity is {prev_rps} RPS")
            break
    else:
        print(f"\n  FP8 maintained TTFT P99 < 1s across ALL tested RPS levels!")

    # Save
    output_file = "stage2_test29a_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
