"""
Stage 2 — Test 2: Variable Prompt Size + Gradual Ramp

Sends requests with 3 different prompt sizes to observe:
- How vLLM schedules prefill vs decode across different-sized requests
- How chunked prefill interleaves with ongoing decode
- Where queuing/preemption kicks in as concurrency grows

We ramp from 5 to 60 concurrent requests with a mix of short/medium/long prompts.

Usage: python3 stage2_variable_load_test.py --host http://<ip>:8000
"""

import json
import time
import threading
import argparse
import requests
import sys

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# Three prompt sizes to create scheduling pressure:
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

# Ramp schedule: (concurrency, prompt_mix)
# prompt_mix is a list of prompt keys — determines what each thread sends
RAMP_SCHEDULE = [
    # Warm up with uniform small batches
    (5,  ["short"] * 5),
    (5,  ["medium"] * 5),
    (5,  ["long"] * 5),
    # Mixed loads — this is where scheduling gets interesting
    (10, ["short"] * 4 + ["medium"] * 3 + ["long"] * 3),
    (20, ["short"] * 8 + ["medium"] * 6 + ["long"] * 6),
    (30, ["short"] * 10 + ["medium"] * 10 + ["long"] * 10),
    (40, ["short"] * 15 + ["medium"] * 13 + ["long"] * 12),
    (50, ["short"] * 20 + ["medium"] * 15 + ["long"] * 15),
    (60, ["short"] * 20 + ["medium"] * 20 + ["long"] * 20),
]


def send_request(request_id: int, prompt_key: str, results: list, global_start: float):
    """Send one streaming request with a specific prompt size."""
    prompt_config = PROMPTS[prompt_key]
    url = f"{HOST}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": prompt_config["text"],
        "max_tokens": prompt_config["max_tokens"],
        "stream": True,
        "temperature": 0.7,
    }

    token_times = []
    tokens = []
    error = None
    request_start = time.perf_counter()

    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
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
            "prompt_type": prompt_key,
            "label": prompt_config["label"],
            "error": error or "no tokens",
            "total_sec": round(request_end - request_start, 3),
        }
        return

    ttft = token_times[0] - request_start
    tbts = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
    avg_tbt = sum(tbts) / len(tbts) if tbts else None

    results[request_id] = {
        "request_id": request_id,
        "prompt_type": prompt_key,
        "label": prompt_config["label"],
        "ttft_sec": round(ttft, 4),
        "avg_tbt_sec": round(avg_tbt, 5) if avg_tbt else None,
        "total_sec": round(request_end - request_start, 3),
        "tokens_out": len(tokens),
        "tokens_per_sec": round(len(tokens) / (request_end - token_times[0]), 1) if token_times else 0,
        "abs_start": round(request_start - global_start, 4),
        "abs_first_token": round(token_times[0] - global_start, 4),
        "abs_end": round(request_end - global_start, 4),
    }


def run_round(prompt_mix: list) -> dict:
    """Fire a mixed batch of concurrent requests."""
    n = len(prompt_mix)
    results = [None] * n
    threads = []
    global_start = time.perf_counter()
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    for i, prompt_key in enumerate(prompt_mix):
        t = threading.Thread(target=send_request, args=(i, prompt_key, results, global_start))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    wall_end = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    successes = [r for r in results if r and "error" not in r]
    failures = [r for r in results if r and "error" in r]

    # Group by prompt type for per-type analysis
    by_type = {}
    for r in successes:
        pt = r["prompt_type"]
        if pt not in by_type:
            by_type[pt] = {"ttfts": [], "totals": [], "tok_rates": [], "tokens_out": []}
        by_type[pt]["ttfts"].append(r["ttft_sec"])
        by_type[pt]["totals"].append(r["total_sec"])
        by_type[pt]["tok_rates"].append(r["tokens_per_sec"])
        by_type[pt]["tokens_out"].append(r["tokens_out"])

    type_summaries = {}
    for pt, data in by_type.items():
        ttfts = sorted(data["ttfts"])
        type_summaries[pt] = {
            "count": len(ttfts),
            "ttft_p50": round(ttfts[len(ttfts) // 2], 4),
            "ttft_max": round(max(ttfts), 4),
            "total_avg": round(sum(data["totals"]) / len(data["totals"]), 3),
            "tok_s_avg": round(sum(data["tok_rates"]) / len(data["tok_rates"]), 1),
            "avg_tokens_out": round(sum(data["tokens_out"]) / len(data["tokens_out"]), 1),
        }

    # Timeline: find if short requests finished before long ones started generating
    # This reveals scheduling priority
    if successes:
        # Sort by first token time to see scheduling order
        by_first_token = sorted(successes, key=lambda r: r["abs_first_token"])
        scheduling_order = "".join(r["label"] for r in by_first_token[:min(20, len(by_first_token))])
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
        "all_requests": successes,  # keep full detail for analysis
    }


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Variable load test for vLLM")
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    print(f"Variable Load Test — {MODEL}")
    print(f"Target: {HOST}")
    print(f"Prompt sizes: Short (~15 tok), Medium (~80 tok), Long (~300 tok)")
    print(f"{'='*120}\n")

    all_results = []

    for n, prompt_mix in RAMP_SCHEDULE:
        # Pause between rounds to let scheduler drain
        if all_results:
            time.sleep(5)

        mix_summary = {k: prompt_mix.count(k) for k in set(prompt_mix)}
        print(f"\n--- Round: N={n} | Mix: {mix_summary} ---")

        result = run_round(prompt_mix)
        all_results.append(result)

        # Print per-type summary for this round
        print(f"  {'Type':<8} {'N':>3}  {'TTFT p50':>9}  {'TTFT max':>9}  "
              f"{'Total avg':>10}  {'Tok/s avg':>10}  {'Avg out':>8}")
        print(f"  {'-'*70}")

        for pt in ["short", "medium", "long"]:
            if pt in result["by_type"]:
                ts = result["by_type"][pt]
                print(f"  {pt:<8} {ts['count']:>3}  {ts['ttft_p50']:>9.4f}  "
                      f"{ts['ttft_max']:>9.4f}  {ts['total_avg']:>10.3f}  "
                      f"{ts['tok_s_avg']:>10.1f}  {ts['avg_tokens_out']:>8.1f}")

        print(f"  Scheduling order (first 20 by TTFT): {result['scheduling_order_first20']}")
        print(f"  Failures: {result['failures']}")

    # Overall analysis
    print(f"\n{'='*120}")
    print("ANALYSIS — How vLLM Schedules Under Mixed Load")
    print(f"{'='*120}\n")

    # Track TTFT growth by prompt type across rounds
    for pt in ["short", "medium", "long"]:
        print(f"\n{pt.upper()} prompt TTFT progression:")
        for result in all_results:
            if pt in result["by_type"]:
                ts = result["by_type"][pt]
                bar = "█" * int(ts["ttft_max"] * 10)
                print(f"  N={result['concurrency']:>3} ({ts['count']:>2}x): "
                      f"p50={ts['ttft_p50']:.4f}s  max={ts['ttft_max']:.4f}s  {bar}")

    # Check if short requests get prioritized over long ones
    print(f"\n\nScheduling fairness (order of first-token delivery):")
    for result in all_results:
        if result["concurrency"] >= 10:
            print(f"  N={result['concurrency']:>3}: {result['scheduling_order_first20']}")

    # Dump all results for graphing
    # Strip full request lists for the JSON to keep it manageable
    slim_results = []
    for r in all_results:
        slim = {k: v for k, v in r.items() if k != "all_requests"}
        slim["request_details"] = r["all_requests"]
        slim_results.append(slim)

    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_variable_load_results.json"
    with open(output_file, "w") as f:
        json.dump(slim_results, f, indent=2)
    print(f"\n\nRaw results saved to {output_file}")


if __name__ == "__main__":
    main()
