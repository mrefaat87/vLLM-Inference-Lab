"""
Stage 1: Ollama Concurrent Load Test

Sends N concurrent requests to Ollama and measures:
- TTFT (Time to First Token) per request
- TBT (Time Between Tokens) per request
- Total duration per request
- Whether requests run in parallel or queue sequentially

Usage: python3 stage1_load_test.py
"""

import asyncio
import aiohttp
import json
import time

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
CONCURRENT_REQUESTS = 5
# Same prompt for all requests so we can compare apples-to-apples
PROMPT = "Explain what a load balancer does in exactly 3 sentences."


async def send_request(session: aiohttp.ClientSession, request_id: int) -> dict:
    """Send a single streaming request to Ollama and measure timing."""

    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": True,  # streaming lets us measure TTFT and TBT
    }

    token_times = []  # timestamps when each token arrived
    tokens = []       # the actual token strings
    request_start = time.perf_counter()

    async with session.post(OLLAMA_URL, json=payload) as resp:
        # Each line from Ollama is a JSON object with one token
        async for line in resp.content:
            chunk = json.loads(line.decode())

            if chunk.get("response"):
                token_times.append(time.perf_counter())
                tokens.append(chunk["response"])

            # Ollama signals completion with done=true
            if chunk.get("done"):
                break

    request_end = time.perf_counter()

    # --- Calculate metrics ---
    # TTFT: time from request start to first token received
    ttft = token_times[0] - request_start if token_times else None

    # TBT: average time between consecutive tokens (decode speed)
    tbts = []
    for i in range(1, len(token_times)):
        tbts.append(token_times[i] - token_times[i - 1])
    avg_tbt = sum(tbts) / len(tbts) if tbts else None

    total_duration = request_end - request_start
    token_count = len(tokens)
    # Tokens per second: only counting decode phase (excludes prefill)
    tokens_per_sec = token_count / (request_end - token_times[0]) if token_times else 0

    return {
        "request_id": request_id,
        "ttft_sec": round(ttft, 3) if ttft else None,
        "avg_tbt_sec": round(avg_tbt, 4) if avg_tbt else None,
        "total_sec": round(total_duration, 3),
        "tokens": token_count,
        "tokens_per_sec": round(tokens_per_sec, 1),
        "start_time": round(request_start - global_start, 3),  # relative to test start
        "first_token_time": round(token_times[0] - global_start, 3) if token_times else None,
        "end_time": round(request_end - global_start, 3),
    }


async def run_load_test():
    """Fire all requests concurrently and collect results."""
    global global_start
    global_start = time.perf_counter()

    # Timeout set high because queued requests may wait a long time
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Launch all requests at the same instant
        tasks = [send_request(session, i) for i in range(CONCURRENT_REQUESTS)]
        results = await asyncio.gather(*tasks)

    return results


def print_results(results: list[dict]):
    """Print a formatted table of results."""
    print(f"\n{'='*80}")
    print(f"Ollama Load Test: {CONCURRENT_REQUESTS} concurrent requests")
    print(f"Model: {MODEL}")
    print(f"{'='*80}\n")

    # Header
    print(f"{'Req':>3}  {'Start':>7}  {'1st Token':>9}  {'End':>7}  "
          f"{'TTFT':>7}  {'Avg TBT':>8}  {'Total':>7}  {'Tok/s':>6}  {'Tokens':>6}")
    print(f"{'-'*3}  {'-'*7}  {'-'*9}  {'-'*7}  "
          f"{'-'*7}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*6}")

    # Sort by when each request's first token arrived — reveals queuing order
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

    # Summary — the key insight is in the timing gaps
    print(f"\n--- Summary ---")
    ttfts = [r["ttft_sec"] for r in results if r["ttft_sec"]]
    totals = [r["total_sec"] for r in results]
    print(f"TTFT  — min: {min(ttfts):.3f}s, max: {max(ttfts):.3f}s, spread: {max(ttfts)-min(ttfts):.3f}s")
    print(f"Total — min: {min(totals):.3f}s, max: {max(totals):.3f}s, spread: {max(totals)-min(totals):.3f}s")

    # The spread tells the story: if requests ran in parallel, spread would be small.
    # If they queued, spread ≈ (N-1) * avg_request_duration.
    first_tokens = sorted([r["first_token_time"] for r in results if r["first_token_time"]])
    gaps = [first_tokens[i] - first_tokens[i-1] for i in range(1, len(first_tokens))]
    if gaps:
        print(f"\nGap between consecutive first-tokens: {['%.3fs' % g for g in gaps]}")
        print(f"Average gap: {sum(gaps)/len(gaps):.3f}s")
        print(f"\nIf this gap ≈ single request duration, requests are queuing (not batching).")


if __name__ == "__main__":
    print("Starting Ollama load test...")
    print(f"Sending {CONCURRENT_REQUESTS} concurrent requests to {MODEL}...")
    results = asyncio.run(run_load_test())
    print_results(results)
