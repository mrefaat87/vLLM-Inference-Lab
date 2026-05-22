"""
Stage 2: vLLM Concurrent Load Test

Same test as Stage 1, but against vLLM's OpenAI-compatible API.
This lets us compare batching (vLLM) vs queuing (Ollama) directly.

Usage: python3 stage2_load_test.py --host <instance-ip>
"""

import json
import time
import threading
import argparse
import requests

# --- Configuration ---
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
CONCURRENT_REQUESTS = 5  # back to 5 — vLLM should handle these in parallel
PROMPT = "What is a load balancer? Answer in one sentence."

# Shared reference time so all threads measure from the same zero point
global_start = 0.0


def send_request(base_url: str, request_id: int) -> dict:
    """Send a single streaming request to vLLM and measure timing."""

    # vLLM serves an OpenAI-compatible API — same interface production systems use
    url = f"{base_url}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 50,
        "stream": True,  # SSE streaming for TTFT/TBT measurement
        "temperature": 0.7,
    }
    headers = {"Content-Type": "application/json"}

    token_times = []
    tokens = []
    request_start = time.perf_counter()

    # vLLM streams Server-Sent Events (SSE), each prefixed with "data: "
    resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=(10, 600))

    for line in resp.iter_lines():
        if not line:
            continue

        line_str = line.decode()

        # SSE format: "data: {json}" or "data: [DONE]"
        if not line_str.startswith("data: "):
            continue

        data_str = line_str[6:]  # strip "data: " prefix

        if data_str == "[DONE]":
            break

        chunk = json.loads(data_str)
        # OpenAI completions format nests the token in choices[0].text
        text = chunk.get("choices", [{}])[0].get("text", "")

        if text:
            token_times.append(time.perf_counter())
            tokens.append(text)

    request_end = time.perf_counter()

    # --- Calculate metrics (same as Stage 1 for direct comparison) ---
    ttft = token_times[0] - request_start if token_times else None

    tbts = []
    for i in range(1, len(token_times)):
        tbts.append(token_times[i] - token_times[i - 1])
    avg_tbt = sum(tbts) / len(tbts) if tbts else None

    total_duration = request_end - request_start
    token_count = len(tokens)
    tokens_per_sec = token_count / (request_end - token_times[0]) if token_times else 0

    return {
        "request_id": request_id,
        "ttft_sec": round(ttft, 3) if ttft else None,
        "avg_tbt_sec": round(avg_tbt, 4) if avg_tbt else None,
        "total_sec": round(total_duration, 3),
        "tokens": token_count,
        "tokens_per_sec": round(tokens_per_sec, 1),
        "start_time": round(request_start - global_start, 3),
        "first_token_time": round(token_times[0] - global_start, 3) if token_times else None,
        "end_time": round(request_end - global_start, 3),
    }


def run_load_test(base_url: str) -> list:
    """Fire all requests concurrently using threads."""
    global global_start
    global_start = time.perf_counter()

    results = [None] * CONCURRENT_REQUESTS
    threads = []

    def worker(req_id):
        try:
            results[req_id] = send_request(base_url, req_id)
        except Exception as e:
            print(f"Request {req_id} failed: {e}")

    for i in range(CONCURRENT_REQUESTS):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return results


def print_results(results, engine_name="vLLM"):
    """Print formatted results table — same format as Stage 1 for comparison."""
    # Filter out failed requests
    results = [r for r in results if r is not None]

    if not results:
        print("All requests failed!")
        return

    print(f"\n{'='*80}")
    print(f"{engine_name} Load Test: {CONCURRENT_REQUESTS} concurrent requests")
    print(f"Model: {MODEL}")
    print(f"{'='*80}\n")

    print(f"{'Req':>3}  {'Start':>7}  {'1st Token':>9}  {'End':>7}  "
          f"{'TTFT':>7}  {'Avg TBT':>8}  {'Total':>7}  {'Tok/s':>6}  {'Tokens':>6}")
    print(f"{'-'*3}  {'-'*7}  {'-'*9}  {'-'*7}  "
          f"{'-'*7}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*6}")

    for r in sorted(results, key=lambda x: x["first_token_time"] or 999):
        print(f"{r['request_id']:>3}  "
              f"{r['start_time']:>7.3f}  "
              f"{r['first_token_time']:>9.3f}  "
              f"{r['end_time']:>7.3f}  "
              f"{r['ttft_sec']:>7.3f}  "
              f"{r['avg_tbt_sec']:>8.4f}  "
              f"{r['total_sec']:>7.3f}  "
              f"{r['tokens_per_sec']:>6.1f}  "
              f"{r['tokens']:>6}")

    print(f"\n--- Summary ---")
    ttfts = [r["ttft_sec"] for r in results if r["ttft_sec"]]
    totals = [r["total_sec"] for r in results]
    print(f"TTFT  - min: {min(ttfts):.3f}s, max: {max(ttfts):.3f}s, spread: {max(ttfts)-min(ttfts):.3f}s")
    print(f"Total - min: {min(totals):.3f}s, max: {max(totals):.3f}s, spread: {max(totals)-min(totals):.3f}s")

    first_tokens = sorted([r["first_token_time"] for r in results if r["first_token_time"]])
    gaps = [first_tokens[i] - first_tokens[i-1] for i in range(1, len(first_tokens))]
    if gaps:
        print(f"\nGap between consecutive first-tokens: {['%.3fs' % g for g in gaps]}")
        print(f"Average gap: {sum(gaps)/len(gaps):.3f}s")

    # Compare with Stage 1 expectations
    avg_total = sum(totals) / len(totals)
    if gaps:
        avg_gap = sum(gaps) / len(gaps)
        ratio = avg_gap / avg_total if avg_total > 0 else 0
        print(f"\nGap/Total ratio: {ratio:.2f}")
        print(f"  ~1.0 = sequential queuing (Ollama behavior)")
        print(f"  ~0.0 = true parallel batching (vLLM behavior)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM concurrent load test")
    parser.add_argument("--host", required=True, help="vLLM server address (e.g., http://1.2.3.4:8000)")
    args = parser.parse_args()

    base_url = args.host.rstrip("/")
    print(f"Starting vLLM load test...")
    print(f"Target: {base_url}")
    print(f"Sending {CONCURRENT_REQUESTS} concurrent requests to {MODEL}...")

    results = run_load_test(base_url)
    print_results(results)
