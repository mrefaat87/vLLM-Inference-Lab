"""
Test 21: Temperature/Sampling Effects on Speed

Do different sampling parameters affect generation speed? Theoretically no —
the GPU does the same matmul regardless of temperature. But greedy (temp=0)
might enable CUDA graph optimizations that skip the sampling kernel entirely.

This is like asking: does the Auto Scaling group's health check interval
affect the actual request latency? It shouldn't, but implementation
details might surprise you.

Usage: python3 stage2_test21_sampling_speed.py --host http://<ip>:8000
"""

import json
import time
import argparse
import requests

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
REPEATS = 10  # 10 requests per config, serial

# Same prompt for all configs — controls for content variation
PROMPT = (
    "Explain how distributed systems handle network partitions. "
    "Cover the CAP theorem, eventual consistency, and conflict resolution strategies. "
    "Include examples from real-world systems like DynamoDB and Cassandra."
)

# Sampling configurations to test
SAMPLING_CONFIGS = [
    {"label": "greedy (temp=0)", "temperature": 0.0, "top_p": 1.0},
    {"label": "low temp (0.3)", "temperature": 0.3, "top_p": 0.9},
    {"label": "medium temp (0.7)", "temperature": 0.7, "top_p": 0.9},
    {"label": "high temp (1.0)", "temperature": 1.0, "top_p": 1.0},
    {"label": "top_k=50", "temperature": 1.0, "top_p": 0.9, "top_k": 50},
    {"label": "hot (temp=1.5)", "temperature": 1.5, "top_p": 0.95},
]


def send_single_request(sampling_params: dict) -> dict:
    """Send one streaming request with given sampling params."""
    url = f"{HOST}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 200,
        "stream": True,
    }
    # Add sampling params (skip top_k if not present — vLLM uses -1 for disabled)
    payload["temperature"] = sampling_params.get("temperature", 0.7)
    payload["top_p"] = sampling_params.get("top_p", 1.0)
    if "top_k" in sampling_params:
        # vLLM uses the extra_body or direct parameter for top_k
        payload["top_k"] = sampling_params["top_k"]

    token_times = []
    tokens = []
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
        return {"error": str(e)}

    request_end = time.perf_counter()
    if not token_times:
        return {"error": "no tokens received"}

    ttft = token_times[0] - request_start
    decode_time = request_end - token_times[0]
    total = request_end - request_start
    tok_s = len(tokens) / decode_time if decode_time > 0 else 0

    # Per-token timing for TBT analysis
    tbts = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
    avg_tbt = sum(tbts) / len(tbts) if tbts else 0

    return {
        "ttft_sec": round(ttft, 4),
        "total_sec": round(total, 4),
        "tokens_out": len(tokens),
        "tok_s": round(tok_s, 1),
        "avg_tbt_ms": round(avg_tbt * 1000, 2),
    }


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Test 21: Sampling speed comparison")
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    print(f"Test 21: Temperature/Sampling Effects on Speed")
    print(f"Target: {HOST} | Model: {MODEL}")
    print(f"Requests per config: {REPEATS} (serial)")
    print(f"{'='*100}\n")

    all_results = []

    for config in SAMPLING_CONFIGS:
        label = config["label"]
        print(f"\n--- {label} ---")

        trial_results = []
        for i in range(REPEATS):
            # Small pause between requests to avoid prefix cache effects
            if i > 0:
                time.sleep(1)
            r = send_single_request(config)
            if "error" in r:
                print(f"  Request {i+1}: ERROR - {r['error']}")
            else:
                trial_results.append(r)
                print(f"  Request {i+1}: {r['tok_s']:>5.1f} tok/s | "
                      f"TTFT={r['ttft_sec']:.4f}s | "
                      f"TBT={r['avg_tbt_ms']:.2f}ms | "
                      f"tokens={r['tokens_out']}")

        if not trial_results:
            print(f"  ALL FAILED for {label}")
            continue

        # Aggregate
        avg_tok_s = sum(r["tok_s"] for r in trial_results) / len(trial_results)
        avg_ttft = sum(r["ttft_sec"] for r in trial_results) / len(trial_results)
        avg_tbt = sum(r["avg_tbt_ms"] for r in trial_results) / len(trial_results)
        avg_tokens = sum(r["tokens_out"] for r in trial_results) / len(trial_results)

        entry = {
            "label": label,
            "temperature": config.get("temperature"),
            "top_p": config.get("top_p"),
            "top_k": config.get("top_k", -1),
            "ttft_avg": round(avg_ttft, 4),
            "tok_s_avg": round(avg_tok_s, 1),
            "avg_tbt_ms": round(avg_tbt, 2),
            "avg_tokens_generated": round(avg_tokens, 1),
            "wall_start": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "successes": len(trial_results),
            "failures": REPEATS - len(trial_results),
        }
        all_results.append(entry)

        print(f"  AVG: {avg_tok_s:.1f} tok/s | TTFT={avg_ttft:.4f}s | "
              f"TBT={avg_tbt:.2f}ms | tokens={avg_tokens:.0f}")

    # Analysis
    print(f"\n{'='*100}")
    print("ANALYSIS")
    print(f"{'='*100}\n")

    if len(all_results) >= 2:
        tok_rates = [(r["label"], r["tok_s_avg"]) for r in all_results]
        fastest = max(tok_rates, key=lambda x: x[1])
        slowest = min(tok_rates, key=lambda x: x[1])
        spread = (fastest[1] - slowest[1]) / fastest[1] * 100

        print(f"Fastest:  {fastest[0]} at {fastest[1]} tok/s")
        print(f"Slowest:  {slowest[0]} at {slowest[1]} tok/s")
        print(f"Spread:   {spread:.1f}%")

        if spread < 5:
            print(f"\n→ Sampling has <5% impact on speed. GPU compute dominates.")
        elif spread < 15:
            print(f"\n→ Moderate impact ({spread:.1f}%). Greedy may benefit from CUDA graph optimizations.")
        else:
            print(f"\n→ Significant impact ({spread:.1f}%). Investigate CUDA graph / sampling kernel differences.")

    # Save
    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_test21_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
