"""
Test 30B: FP8 KV Cache — Throughput Scaling

Replicates Test 5's closed-loop concurrency ramp against an FP8 KV cache server.
Extended concurrency range to [1..128] to find the new ceiling beyond FP16's ~62.

Hypothesis: FP8 throughput matches FP16 up to N=64 (memory isn't the bottleneck
yet), then keeps climbing where FP16 would OOM. New plateau around N=96-112
(compute-bound, not memory-bound).

AWS analogy: Same fleet, 2x instance capacity. Throughput per instance is
identical at low utilization. The win is at high utilization where FP16 would
have triggered scaling events.

Compare results against stage2_test5_results.json (FP16 baseline).

Usage: python3 stage2_test30b_fp8_throughput.py --host http://<ip>:8000
"""

import json
import time
import threading
import argparse
import requests

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# Extended range: Test 5 stopped at 64; FP8 should survive to ~128
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128]
TRIALS = 3  # Median across 3 trials reduces noise

# Same prompt generation as Test 5 — ~200 input tokens
FILLER_SENTENCES = [
    "The rapid advancement of technology has transformed every aspect of modern life.",
    "From artificial intelligence to quantum computing, new innovations emerge daily.",
    "Cloud computing enables organizations to scale their infrastructure dynamically.",
    "Machine learning models require significant computational resources for training.",
    "Distributed systems must handle network partitions and maintain consistency.",
    "Load balancers distribute incoming traffic across multiple server instances.",
    "Containerization with Docker simplified application deployment and scaling.",
    "Kubernetes orchestrates container workloads across clusters of machines.",
    "Monitoring and observability are critical for maintaining system reliability.",
    "Auto scaling adjusts capacity based on real-time demand metrics.",
    "Database sharding improves query performance by distributing data horizontally.",
    "Content delivery networks cache static assets closer to end users.",
    "Microservices architecture decomposes monolithic applications into smaller services.",
    "API gateways provide a unified entry point for client applications.",
    "Message queues decouple producers and consumers for asynchronous processing.",
]


def generate_prompt(target_tokens: int = 200) -> str:
    """Generate a prompt of approximately target_tokens length."""
    base = "Summarize the following text in about 200 words:\n\n"
    words_needed = int((target_tokens - 20) / 1.3)
    words = []
    while len(words) < words_needed:
        for sentence in FILLER_SENTENCES:
            words.extend(sentence.split())
            if len(words) >= words_needed:
                break
    return base + " ".join(words[:words_needed])


# Pre-generate the prompt once — all requests use the same prompt
PROMPT = generate_prompt(200)


def send_request(request_id: int, results: list, global_start: float):
    """Send one streaming request, measure TTFT and throughput."""
    url = f"{HOST}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 200,
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
        "abs_start": round(request_start - global_start, 4),
        "abs_first_token": round(token_times[0] - global_start, 4),
        "abs_end": round(request_end - global_start, 4),
    }


def run_at_concurrency(n: int) -> dict:
    """Fire n concurrent requests and return aggregated metrics."""
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

    wall_end = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    wall_duration = time.perf_counter() - global_start

    successes = [r for r in results if r and "error" not in r]
    failures = [r for r in results if r and "error" in r]
    missing = n - len(successes) - len(failures)

    if not successes:
        return {
            "concurrency": n,
            "wall_start": wall_start,
            "wall_end": wall_end,
            "wall_duration_sec": round(wall_duration, 2),
            "successes": 0,
            "failures": len(failures) + missing,
            "error_sample": failures[0]["error"] if failures else "all threads returned None",
        }

    ttfts = sorted([r["ttft_sec"] for r in successes])
    totals = sorted([r["total_sec"] for r in successes])
    tok_rates = [r["tokens_per_sec"] for r in successes]
    total_tokens = sum(r["tokens"] for r in successes)

    # System throughput = total tokens generated / wall-clock time for the round
    system_tok_s = total_tokens / wall_duration

    return {
        "concurrency": n,
        "wall_start": wall_start,
        "wall_end": wall_end,
        "wall_duration_sec": round(wall_duration, 2),
        "successes": len(successes),
        "failures": len(failures) + missing,
        "total_tokens_generated": total_tokens,
        "system_tok_s": round(system_tok_s, 1),
        "per_request_tok_s_avg": round(sum(tok_rates) / len(tok_rates), 1),
        "per_request_tok_s_min": round(min(tok_rates), 1),
        "ttft_p50": round(ttfts[len(ttfts) // 2], 4),
        "ttft_p99": round(ttfts[int(len(ttfts) * 0.99)], 4),
        "ttft_max": round(max(ttfts), 4),
        "total_p50": round(totals[len(totals) // 2], 3),
        "total_p99": round(totals[int(len(totals) * 0.99)], 3),
        "total_max": round(max(totals), 3),
        "avg_tokens_generated": round(total_tokens / len(successes), 1),
    }


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Test 30B: FP8 KV Cache — Throughput Scaling")
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    print(f"Test 30B: FP8 KV Cache — Throughput Scaling")
    print(f"Target: {HOST} | Model: {MODEL}")
    print(f"KV Cache: fp8_e4m3 (A10G Ampere) (compare against Test 29B T4 fp8_e4m3 (A10G Ampere) + Test 5 fp16 baseline)")
    print(f"Concurrency levels: {CONCURRENCY_LEVELS}")
    print(f"Prompt: ~200 input tokens, max_tokens=200")
    print(f"Trials per level: {TRIALS} (using median)")
    print(f"{'='*120}\n")

    all_results = []

    # Header
    print(f"{'N':>4}  {'Trial':>5}  {'OK':>4}  {'Fail':>4}  "
          f"{'Sys tok/s':>10}  {'Req tok/s':>10}  "
          f"{'TTFT p50':>9}  {'TTFT max':>9}  "
          f"{'Total p50':>10}  {'Wall sec':>9}")
    print("-" * 120)

    for n in CONCURRENCY_LEVELS:
        trial_results = []

        for trial in range(TRIALS):
            # Pause between trials/levels so scheduler resets
            if trial > 0 or all_results:
                time.sleep(5)

            result = run_at_concurrency(n)
            result["trial"] = trial + 1
            trial_results.append(result)

            if "error_sample" in result:
                print(f"{n:>4}  {trial+1:>5}  {result['successes']:>4}  "
                      f"{result['failures']:>4}  {'FAILED':>10}  {'':>10}  "
                      f"{'':>9}  {'':>9}  {'':>10}  "
                      f"{result['wall_duration_sec']:>9.1f}")
            else:
                print(f"{n:>4}  {trial+1:>5}  {result['successes']:>4}  "
                      f"{result['failures']:>4}  "
                      f"{result['system_tok_s']:>10.1f}  "
                      f"{result['per_request_tok_s_avg']:>10.1f}  "
                      f"{result['ttft_p50']:>9.4f}  "
                      f"{result['ttft_max']:>9.4f}  "
                      f"{result['total_p50']:>10.3f}  "
                      f"{result['wall_duration_sec']:>9.1f}")

        all_results.extend(trial_results)

    # Analysis
    print(f"\n{'='*120}")
    print("ANALYSIS — Median values across trials")
    print(f"{'='*120}\n")

    # Group by concurrency, pick median trial by system_tok_s
    for n in CONCURRENCY_LEVELS:
        trials = [r for r in all_results if r["concurrency"] == n and "system_tok_s" in r]
        if not trials:
            continue
        trials.sort(key=lambda r: r["system_tok_s"])
        median = trials[len(trials) // 2]
        print(f"N={n:>3}: system={median['system_tok_s']:>7.1f} tok/s | "
              f"per-req={median['per_request_tok_s_avg']:>6.1f} tok/s | "
              f"TTFT p50={median['ttft_p50']:.4f}s | "
              f"wall={median['wall_duration_sec']:.1f}s")

    # Find throughput knee and peak
    medians = []
    for n in CONCURRENCY_LEVELS:
        trials = [r for r in all_results if r["concurrency"] == n and "system_tok_s" in r]
        if trials:
            trials.sort(key=lambda r: r["system_tok_s"])
            medians.append(trials[len(trials) // 2])

    if len(medians) >= 2:
        peak = max(medians, key=lambda r: r["system_tok_s"])
        single = medians[0]  # N=1 baseline
        print(f"\nPeak system throughput: {peak['system_tok_s']} tok/s at N={peak['concurrency']}")
        print(f"Single-request baseline: {single['system_tok_s']} tok/s")
        print(f"Throughput multiplier: {peak['system_tok_s'] / single['system_tok_s']:.1f}x")

        # Efficiency: how much of the ideal linear scaling do we get?
        if single["per_request_tok_s_avg"] > 0:
            for m in medians:
                ideal = single["per_request_tok_s_avg"] * m["concurrency"]
                efficiency = m["system_tok_s"] / ideal * 100
                print(f"  N={m['concurrency']:>3}: efficiency={efficiency:>5.1f}% "
                      f"(actual {m['system_tok_s']:.0f} vs ideal {ideal:.0f} tok/s)")

    # Save results
    output_file = "stage2_test30b_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
