"""
Test 8: KV Cache Quantization (FP16 vs FP8)

FP8 KV cache stores key-value cache in 8-bit format, halving memory per token.
This should ~double max concurrency at the same VRAM budget.

This script runs the FP8 phase. Compare results with cliff_results.json (FP16 baseline).

The T4 lacks native FP8 compute (needs sm_89+), so vLLM casts FP8→FP16 for
attention computation. The benefit is purely from memory savings.

Usage: python3 stage2_test8_kv_quantization.py --host http://<ip>:8000
"""

import json
import time
import threading
import argparse
import requests

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# Same cliff test as Test 10 — long prompts to stress KV cache
CONCURRENCY_LEVELS = [20, 40, 60, 80, 100, 120]

# Long prompt for KV pressure
PROMPT = (
    "Write a comprehensive essay about the history and evolution of distributed "
    "computing systems, from mainframes to cloud computing. Cover the key milestones: "
    "time-sharing systems in the 1960s, the ARPANET, client-server architecture, "
    "the rise of the web, grid computing, virtualization, cloud computing with AWS, "
    "containers and Kubernetes, serverless computing, and the current era of AI "
    "inference serving. For each era, explain the key technical innovations, the "
    "problems they solved, and how they influenced what came next. "
    "Discuss the fundamental tradeoffs in distributed systems: consistency vs "
    "availability (CAP theorem), latency vs throughput, cost vs performance, "
    "and simplicity vs flexibility. Include specific examples of systems and "
    "their architectures: Google's MapReduce and GFS, Amazon's Dynamo, "
    "Facebook's TAO, Netflix's microservices, and modern LLM serving systems. "
    "End with your predictions for the future of distributed computing."
)


def send_request(request_id: int, results: list, global_start: float):
    """Send one streaming request."""
    url = f"{HOST}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 1500,
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
                             stream=True, timeout=(60, 600))
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
    total = request_end - request_start
    decode_time = request_end - token_times[0]
    tok_s = len(tokens) / decode_time if decode_time > 0 else 0

    results[request_id] = {
        "request_id": request_id,
        "ttft_sec": round(ttft, 4),
        "total_sec": round(total, 3),
        "tokens": len(tokens),
        "tokens_per_sec": round(tok_s, 1),
    }


def run_at_concurrency(n: int) -> dict:
    """Fire n concurrent requests."""
    results = [None] * n
    threads = []
    global_start = time.perf_counter()
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    for i in range(n):
        t = threading.Thread(target=send_request, args=(i, results, global_start))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    wall_duration = time.perf_counter() - global_start
    successes = [r for r in results if r and "error" not in r]
    failures = [r for r in results if r and "error" in r]

    if not successes:
        return {
            "concurrency": n,
            "wall_start": wall_start,
            "successes": 0,
            "failures": len(failures) + (n - len(failures)),
            "wall_duration_sec": round(wall_duration, 1),
        }

    ttfts = sorted([r["ttft_sec"] for r in successes])
    totals = sorted([r["total_sec"] for r in successes])
    total_tokens = sum(r["tokens"] for r in successes)

    return {
        "concurrency": n,
        "wall_start": wall_start,
        "wall_duration_sec": round(wall_duration, 1),
        "successes": len(successes),
        "failures": len(failures),
        "total_tokens_generated": total_tokens,
        "system_tok_s": round(total_tokens / wall_duration, 1),
        "per_request_tok_s_avg": round(sum(r["tokens_per_sec"] for r in successes) / len(successes), 1),
        "ttft_p50": round(ttfts[len(ttfts) // 2], 4),
        "ttft_p99": round(ttfts[int(len(ttfts) * 0.99)], 4),
        "ttft_max": round(max(ttfts), 4),
        "total_p50": round(totals[len(totals) // 2], 3),
        "total_p99": round(totals[int(len(totals) * 0.99)], 3),
        "total_max": round(max(totals), 3),
        "avg_tokens_generated": round(total_tokens / len(successes), 0),
    }


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Test 8: FP8 KV Cache cliff test")
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    print(f"Test 8: KV Cache Quantization — FP8 Cliff Test")
    print(f"Target: {HOST} | Model: {MODEL}")
    print(f"FP8 KV cache: 255,696 tokens, max concurrency ~125")
    print(f"FP16 KV baseline: 127,840 tokens, max concurrency ~62")
    print(f"Concurrency levels: {CONCURRENCY_LEVELS}")
    print(f"{'='*100}\n")

    all_results = []

    print(f"{'N':>4}  {'OK':>4}  {'Fail':>4}  {'Sys tok/s':>10}  "
          f"{'Req tok/s':>10}  {'TTFT p50':>9}  {'Total p50':>10}  {'Wall sec':>9}")
    print("-" * 80)

    for n in CONCURRENCY_LEVELS:
        if all_results:
            time.sleep(5)

        result = run_at_concurrency(n)
        all_results.append(result)

        if result["successes"] > 0:
            print(f"{n:>4}  {result['successes']:>4}  {result['failures']:>4}  "
                  f"{result['system_tok_s']:>10.1f}  "
                  f"{result['per_request_tok_s_avg']:>10.1f}  "
                  f"{result['ttft_p50']:>9.4f}  "
                  f"{result['total_p50']:>10.1f}  "
                  f"{result['wall_duration_sec']:>9.1f}")
        else:
            print(f"{n:>4}  {0:>4}  {result['failures']:>4}  FAILED")
            if result["successes"] == 0:
                print(f"\n*** All failed at N={n}. Stopping. ***")
                break

    # Save
    output = {
        "fp8_kv": {
            "kv_cache_tokens": 255696,
            "max_concurrency_theoretical": 124.85,
            "cliff_results": all_results,
        },
        "fp16_kv_reference": {
            "kv_cache_tokens": 127840,
            "max_concurrency_theoretical": 62.42,
            "note": "See stage2_cliff_results.json for FP16 cliff data",
        },
    }

    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_test8_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
