"""
Stage 2 — Test 1: Max Concurrency Test

Sends increasing numbers of identical concurrent requests to find the breaking point.
Same short prompt every time so we isolate concurrency behavior from prompt variance.

From our earlier tests, this prompt generates ~24-32 tokens per response.
vLLM reports max theoretical concurrency of 62 at 2048 context.
Let's find the practical limit.

Usage: python3 stage2_max_concurrency_test.py --host http://<ip>:8000
"""

import json
import time
import threading
import argparse
import requests
import sys

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
# Same prompt as all previous tests — generates ~24-32 tokens
PROMPT = "What is a load balancer? Answer in one sentence."

# We'll test these concurrency levels in order, stopping if we hit failures
CONCURRENCY_LEVELS = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80]


def send_request(request_id: int, results: list, global_start: float):
    """Send one streaming request, measure TTFT/TBT/total."""
    url = f"{HOST}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 50,
        "stream": True,
        "temperature": 0.7,
    }

    token_times = []
    tokens = []
    error = None
    request_start = time.perf_counter()

    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
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

    if error or not token_times:
        results[request_id] = {
            "request_id": request_id,
            "error": error or "no tokens received",
            "total_sec": round(request_end - request_start, 3),
        }
        return

    ttft = token_times[0] - request_start
    tbts = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
    avg_tbt = sum(tbts) / len(tbts) if tbts else None
    total = request_end - request_start
    tok_s = len(tokens) / (request_end - token_times[0]) if token_times else 0

    results[request_id] = {
        "request_id": request_id,
        "ttft_sec": round(ttft, 4),
        "avg_tbt_sec": round(avg_tbt, 5) if avg_tbt else None,
        "total_sec": round(total, 3),
        "tokens": len(tokens),
        "tokens_per_sec": round(tok_s, 1),
        # Absolute timestamps for log correlation
        "abs_start": round(request_start - global_start, 4),
        "abs_first_token": round(token_times[0] - global_start, 4),
        "abs_end": round(request_end - global_start, 4),
    }


def run_at_concurrency(n: int) -> dict:
    """Fire n concurrent requests and return aggregated metrics."""
    results = [None] * n
    threads = []
    global_start = time.perf_counter()
    # Record wall clock for log correlation
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    for i in range(n):
        t = threading.Thread(target=send_request, args=(i, results, global_start))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    wall_end = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    # Separate successes from failures
    successes = [r for r in results if r and "error" not in r]
    failures = [r for r in results if r and "error" in r]
    missing = n - len(successes) - len(failures)

    if not successes:
        return {
            "concurrency": n,
            "wall_start": wall_start,
            "wall_end": wall_end,
            "successes": 0,
            "failures": len(failures) + missing,
            "error_sample": failures[0]["error"] if failures else "all threads returned None",
        }

    ttfts = [r["ttft_sec"] for r in successes]
    totals = [r["total_sec"] for r in successes]
    tok_rates = [r["tokens_per_sec"] for r in successes]

    # Per-request throughput tells us if the GPU is saturated
    # Total throughput = sum of all concurrent tok/s = system throughput
    system_tok_s = sum(tok_rates)

    return {
        "concurrency": n,
        "wall_start": wall_start,
        "wall_end": wall_end,
        "successes": len(successes),
        "failures": len(failures) + missing,
        "ttft_p50": round(sorted(ttfts)[len(ttfts) // 2], 4),
        "ttft_p99": round(sorted(ttfts)[int(len(ttfts) * 0.99)], 4),
        "ttft_max": round(max(ttfts), 4),
        "total_p50": round(sorted(totals)[len(totals) // 2], 3),
        "total_p99": round(sorted(totals)[int(len(totals) * 0.99)], 3),
        "total_max": round(max(totals), 3),
        # Per-request tok/s — should decrease as batch size grows (shared bandwidth)
        "per_request_tok_s_avg": round(sum(tok_rates) / len(tok_rates), 1),
        "per_request_tok_s_min": round(min(tok_rates), 1),
        # System throughput — should increase then plateau as GPU saturates
        "system_tok_s": round(system_tok_s, 1),
        "avg_tokens_generated": round(sum(r["tokens"] for r in successes) / len(successes), 1),
    }


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Find max concurrency for vLLM")
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    print(f"Max Concurrency Test — {MODEL}")
    print(f"Target: {HOST}")
    print(f"KV cache max theoretical concurrency: 62 (from vLLM logs)")
    print(f"Testing levels: {CONCURRENCY_LEVELS}")
    print(f"{'='*100}\n")

    all_results = []

    # Header
    print(f"{'N':>4}  {'OK':>4}  {'Fail':>4}  {'TTFT p50':>9}  {'TTFT p99':>9}  "
          f"{'TTFT max':>9}  {'Total p50':>10}  {'Total max':>10}  "
          f"{'Req tok/s':>10}  {'Sys tok/s':>10}  {'Window':>20}")
    print("-" * 115)

    for n in CONCURRENCY_LEVELS:
        # Brief pause between rounds so vLLM scheduler resets
        if all_results:
            time.sleep(3)

        result = run_at_concurrency(n)
        all_results.append(result)

        if "error_sample" in result:
            print(f"{n:>4}  {result['successes']:>4}  {result['failures']:>4}  "
                  f"{'FAILED':>9}  {'':>9}  {'':>9}  {'':>10}  {'':>10}  "
                  f"{'':>10}  {'':>10}  {result['wall_start']}")
            print(f"      Error: {result['error_sample'][:80]}")
            # If all requests failed, we've hit the wall
            if result["successes"] == 0:
                print(f"\n*** All requests failed at N={n}. Stopping. ***")
                break
        else:
            print(f"{n:>4}  {result['successes']:>4}  {result['failures']:>4}  "
                  f"{result['ttft_p50']:>9.4f}  {result['ttft_p99']:>9.4f}  "
                  f"{result['ttft_max']:>9.4f}  {result['total_p50']:>10.3f}  "
                  f"{result['total_max']:>10.3f}  "
                  f"{result['per_request_tok_s_avg']:>10.1f}  "
                  f"{result['system_tok_s']:>10.1f}  "
                  f"{result['wall_start']}")

    # Summary insights
    print(f"\n{'='*100}")
    print("ANALYSIS")
    print(f"{'='*100}\n")

    successful = [r for r in all_results if "error_sample" not in r]
    if len(successful) >= 2:
        # Find where system throughput plateaus
        sys_toks = [(r["concurrency"], r["system_tok_s"]) for r in successful]
        peak = max(sys_toks, key=lambda x: x[1])
        print(f"Peak system throughput: {peak[1]} tok/s at N={peak[0]}")

        # Find where per-request tok/s starts degrading significantly
        base_tok_s = successful[0]["per_request_tok_s_avg"]
        for r in successful:
            degradation = (1 - r["per_request_tok_s_avg"] / base_tok_s) * 100
            if degradation > 50:
                print(f"Per-request tok/s dropped >50% at N={r['concurrency']} "
                      f"({r['per_request_tok_s_avg']} vs baseline {base_tok_s})")
                break

        # Find where TTFT exceeds 2s (user-noticeable threshold)
        for r in successful:
            if r["ttft_p99"] > 2.0:
                print(f"TTFT P99 exceeded 2s at N={r['concurrency']} ({r['ttft_p99']:.3f}s)")
                break

    # Dump JSON for graphing
    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_max_concurrency_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {output_file}")


if __name__ == "__main__":
    main()
