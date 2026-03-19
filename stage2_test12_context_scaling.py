"""
Test 12: Context Window Scaling (max_model_len effect)

Tests how max_model_len affects concurrency capacity. Smaller context windows
mean less KV cache per request, allowing more concurrent requests.

This script runs a single config at a time. Call with --max-model-len to specify
which config is currently running on the server.

Usage: python3 stage2_test12_context_scaling.py --host http://<ip>:8000 --max-model-len 512
"""

import json
import time
import threading
import argparse
import requests
import os

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# Use short prompts + short output so we stay within even 512 context
PROMPT = "What is a load balancer? Answer in one sentence."


def send_request(request_id: int, results: list, global_start: float):
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
        if resp.status_code != 200:
            error = f"http_{resp.status_code}: {resp.text[:100]}"
        else:
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
        results[request_id] = {"request_id": request_id, "error": error or "no tokens",
                               "total_sec": round(request_end - request_start, 3)}
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
        return {"concurrency": n, "successes": 0, "failures": len(failures) + (n - len(failures)),
                "wall_start": wall_start, "wall_duration_sec": round(wall_duration, 1)}

    ttfts = sorted([r["ttft_sec"] for r in successes])
    totals = sorted([r["total_sec"] for r in successes])
    total_tokens = sum(r["tokens"] for r in successes)

    return {
        "concurrency": n, "wall_start": wall_start,
        "wall_duration_sec": round(wall_duration, 1),
        "successes": len(successes), "failures": len(failures),
        "system_tok_s": round(total_tokens / wall_duration, 1),
        "ttft_p50": round(ttfts[len(ttfts) // 2], 4),
        "total_p50": round(totals[len(totals) // 2], 3),
    }


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Test 12: Context Window Scaling")
    parser.add_argument("--host", required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-concurrency", type=int, default=0,
                        help="Reported max concurrency from server logs")
    args = parser.parse_args()
    HOST = args.host.rstrip("/")
    ctx = args.max_model_len

    # Concurrency levels to test — scale based on expected capacity
    levels = [10, 20, 40, 60, 80, 100, 120, 150, 200, 250]

    print(f"Test 12: Context Window Scaling — max_model_len={ctx}")
    print(f"Target: {HOST} | Model: {MODEL}")
    print(f"Reported max concurrency: {args.max_concurrency}")
    print(f"{'='*80}\n")

    all_results = []

    print(f"{'N':>5}  {'OK':>4}  {'Fail':>4}  {'Sys tok/s':>10}  {'TTFT p50':>9}  {'Total p50':>10}")
    print("-" * 50)

    for n in levels:
        if all_results:
            time.sleep(3)

        result = run_at_concurrency(n)
        all_results.append(result)

        if result["successes"] > 0:
            print(f"{n:>5}  {result['successes']:>4}  {result['failures']:>4}  "
                  f"{result['system_tok_s']:>10.1f}  {result['ttft_p50']:>9.4f}  "
                  f"{result['total_p50']:>10.3f}")
        else:
            print(f"{n:>5}  {0:>4}  {result['failures']:>4}  FAILED")
            break

    # Load existing results and append
    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_test12_results.json"
    existing = []
    if os.path.exists(output_file):
        with open(output_file) as f:
            existing = json.load(f)

    entry = {
        "max_model_len": ctx,
        "max_concurrency_reported": args.max_concurrency,
        "cliff_results": all_results,
    }

    # Replace if same max_model_len exists, otherwise append
    existing = [e for e in existing if e["max_model_len"] != ctx]
    existing.append(entry)
    existing.sort(key=lambda e: e["max_model_len"])

    with open(output_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults appended to {output_file}")


if __name__ == "__main__":
    main()
