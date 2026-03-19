"""
Test 16: Chunked Prefill — Direct Interleaving Measurement

Measures whether chunked prefill protects ongoing decode requests from
TBT stalls when a long-prefill "bomb" arrives.

1. Start a background decode request (short prompt, 500 output tokens)
2. Measure its per-token TBT
3. After ~1s, inject a "prefill bomb" (1500-token prompt, 10 output tokens)
4. Continue measuring background TBT — does it spike?

With chunked prefill: TBT should stay stable (prefill is interleaved)
Without: TBT should spike dramatically during the bomb's prefill

Usage: python3 stage2_test16_chunked_prefill.py --host http://<ip>:8000 --phase on
"""

import json
import time
import threading
import argparse
import requests

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
TRIALS = 5

# Short prompt for the background decode request
BG_PROMPT = "Count from 1 to 100, writing each number on a new line."

# Long prompt for the prefill bomb (~1500 tokens)
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
    "Circuit breakers prevent cascading failures in distributed systems.",
    "Service mesh technologies like Istio manage inter-service communication.",
    "Infrastructure as code enables reproducible and version-controlled deployments.",
    "Continuous integration pipelines automate building and testing of software.",
    "Blue-green deployments minimize downtime during application updates.",
]


def generate_bomb_prompt():
    """Generate a ~1500-token prompt."""
    base = "Summarize the following text in exactly 10 words:\n\n"
    words_needed = int(1400 / 1.3)
    words = []
    while len(words) < words_needed:
        for s in FILLER_SENTENCES:
            words.extend(s.split())
            if len(words) >= words_needed:
                break
    return base + " ".join(words[:words_needed])


BOMB_PROMPT = generate_bomb_prompt()


def stream_with_tbt(prompt: str, max_tokens: int, result: dict, global_start: float):
    """Stream a request and record per-token timestamps."""
    url = f"{HOST}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }

    token_times = []
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
    except Exception as e:
        result["error"] = str(e)

    result["token_times"] = token_times
    result["request_start"] = request_start
    result["request_end"] = time.perf_counter()
    result["abs_start"] = request_start - global_start


def run_one_trial() -> dict:
    """Run one trial: background decode + delayed bomb."""
    bg_result = {}
    bomb_result = {}
    global_start = time.perf_counter()

    # Start background decode
    bg_thread = threading.Thread(target=stream_with_tbt,
                                args=(BG_PROMPT, 300, bg_result, global_start))
    bg_thread.start()

    # Wait 1s for background to start generating
    time.sleep(1.0)

    # Inject the prefill bomb
    bomb_inject_time = time.perf_counter() - global_start
    bomb_thread = threading.Thread(target=stream_with_tbt,
                                  args=(BOMB_PROMPT, 10, bomb_result, global_start))
    bomb_thread.start()

    bg_thread.join()
    bomb_thread.join()

    # Analyze background TBT
    bg_times = bg_result.get("token_times", [])
    if len(bg_times) < 10:
        return {"error": "background produced too few tokens"}

    # Compute per-token TBT in ms
    tbts = [(bg_times[i] - bg_times[i-1]) * 1000 for i in range(1, len(bg_times))]
    # Relative times from start
    rel_times = [t - global_start for t in bg_times]

    # Split TBT into before/during/after bomb
    before = [tbts[i] for i in range(len(tbts)) if rel_times[i+1] < bomb_inject_time]
    during = [tbts[i] for i in range(len(tbts))
              if bomb_inject_time <= rel_times[i+1] <= bomb_inject_time + 2.0]
    after = [tbts[i] for i in range(len(tbts)) if rel_times[i+1] > bomb_inject_time + 2.0]

    before_avg = sum(before) / len(before) if before else 0
    during_avg = sum(during) / len(during) if during else 0
    after_avg = sum(after) / len(after) if after else 0
    spike_ratio = during_avg / before_avg if before_avg > 0 else 0

    bomb_ttft = None
    bomb_times = bomb_result.get("token_times", [])
    if bomb_times and "request_start" in bomb_result:
        bomb_ttft = bomb_times[0] - bomb_result["request_start"]

    return {
        "bg_tokens": len(bg_times),
        "bg_tbt_before_ms": round(before_avg, 2),
        "bg_tbt_during_ms": round(during_avg, 2),
        "bg_tbt_after_ms": round(after_avg, 2),
        "tbt_spike_ratio": round(spike_ratio, 2),
        "bomb_ttft_sec": round(bomb_ttft, 4) if bomb_ttft else None,
        "bomb_inject_time": round(bomb_inject_time, 4),
        "n_before": len(before),
        "n_during": len(during),
        "n_after": len(after),
    }


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Test 16: Chunked Prefill")
    parser.add_argument("--host", required=True)
    parser.add_argument("--phase", choices=["on", "off"], required=True)
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    label = "Chunked Prefill ON (default)" if args.phase == "on" else "Chunked Prefill OFF"
    print(f"Test 16: {label}")
    print(f"Target: {HOST} | Model: {MODEL}")
    print(f"Trials: {TRIALS}")
    print(f"{'='*80}\n")

    all_trials = []
    for trial in range(TRIALS):
        if trial > 0:
            time.sleep(5)
        print(f"Trial {trial + 1}/{TRIALS}...")
        result = run_one_trial()
        all_trials.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Before: {result['bg_tbt_before_ms']:.2f}ms ({result['n_before']} tokens) | "
                  f"During: {result['bg_tbt_during_ms']:.2f}ms ({result['n_during']} tokens) | "
                  f"After: {result['bg_tbt_after_ms']:.2f}ms ({result['n_after']} tokens) | "
                  f"Spike: {result['tbt_spike_ratio']:.2f}x | "
                  f"Bomb TTFT: {result.get('bomb_ttft_sec', 'N/A')}s")

    # Aggregate
    valid = [t for t in all_trials if "error" not in t]
    if valid:
        avg_before = sum(t["bg_tbt_before_ms"] for t in valid) / len(valid)
        avg_during = sum(t["bg_tbt_during_ms"] for t in valid) / len(valid)
        avg_after = sum(t["bg_tbt_after_ms"] for t in valid) / len(valid)
        avg_spike = sum(t["tbt_spike_ratio"] for t in valid) / len(valid)

        print(f"\nAVERAGE: before={avg_before:.2f}ms | during={avg_during:.2f}ms | "
              f"after={avg_after:.2f}ms | spike={avg_spike:.2f}x")

    # Save
    output_file = f"/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_test16_{args.phase}_results.json"
    output = {
        f"chunked_{args.phase}": {
            "trials": all_trials,
            "avg_tbt_before_ms": round(avg_before, 2) if valid else None,
            "avg_tbt_during_ms": round(avg_during, 2) if valid else None,
            "avg_tbt_after_ms": round(avg_after, 2) if valid else None,
            "avg_spike_ratio": round(avg_spike, 2) if valid else None,
        }
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
