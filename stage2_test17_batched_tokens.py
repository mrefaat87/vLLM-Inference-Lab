"""
Test 17: max_num_batched_tokens — Prefill Budget Control

Tests how --max-num-batched-tokens affects the prefill/decode tradeoff.
Lower values protect decode latency, higher values speed up prefill at
the cost of decode stalls.

Usage: python3 stage2_test17_batched_tokens.py --host http://<ip>:8000 --budget 512
"""

import json
import time
import threading
import argparse
import requests
import os

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

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
]


def generate_prompt(target_tokens=1500):
    base = "Summarize the following text in exactly 10 words:\n\n"
    words_needed = int(target_tokens / 1.3)
    words = []
    while len(words) < words_needed:
        for s in FILLER_SENTENCES:
            words.extend(s.split())
            if len(words) >= words_needed:
                break
    return base + " ".join(words[:words_needed])


LARGE_PROMPT = generate_prompt(1500)
SHORT_PROMPT = "What is a load balancer? Answer in one sentence."


def send_single(prompt, max_tokens):
    """Send one streaming request, return TTFT and token count."""
    url = f"{HOST}/v1/completions"
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
               "stream": True, "temperature": 0.7}

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
        return {"error": str(e)}

    if not token_times:
        return {"error": "no tokens"}

    ttft = token_times[0] - request_start
    total = time.perf_counter() - request_start
    return {"ttft_sec": round(ttft, 4), "total_sec": round(total, 3),
            "tokens": len(token_times)}


def run_concurrent_test(n=20):
    """Run n concurrent requests with mixed sizes."""
    results = [None] * n
    threads = []
    global_start = time.perf_counter()

    def worker(idx):
        if idx < 15:
            results[idx] = send_single(SHORT_PROMPT, 50)
        else:
            results[idx] = send_single(LARGE_PROMPT, 100)

    for i in range(n):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    wall = time.perf_counter() - global_start
    successes = [r for r in results if r and "error" not in r]
    if not successes:
        return {"error": "all failed"}

    ttfts = sorted([r["ttft_sec"] for r in successes])
    totals = sorted([r["total_sec"] for r in successes])
    return {
        "successes": len(successes),
        "failures": n - len(successes),
        "ttft_p50": round(ttfts[len(ttfts) // 2], 4),
        "ttft_p99": round(ttfts[int(len(ttfts) * 0.99)], 4),
        "total_p50": round(totals[len(totals) // 2], 3),
        "total_p99": round(totals[int(len(totals) * 0.99)], 3),
        "wall_sec": round(wall, 1),
    }


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Test 17: max_num_batched_tokens")
    parser.add_argument("--host", required=True)
    parser.add_argument("--budget", type=int, required=True,
                        help="max_num_batched_tokens value")
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    print(f"Test 17: max_num_batched_tokens = {args.budget}")
    print(f"{'='*60}\n")

    # Test 1: Isolated large prompt TTFT
    print("Isolated large prompt TTFT (3 trials):")
    ttfts = []
    for i in range(3):
        if i > 0:
            time.sleep(2)
        r = send_single(LARGE_PROMPT, 10)
        if "error" not in r:
            ttfts.append(r["ttft_sec"])
            print(f"  Trial {i+1}: TTFT={r['ttft_sec']:.4f}s")
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0

    # Test 2: TBT spike test (same as test 16)
    print("\nTBT spike test (background decode + prefill bomb, 3 trials):")
    spike_results = []
    for i in range(3):
        if i > 0:
            time.sleep(3)
        # Simple version: just measure background request TBT
        bg_result = {}
        bomb_result = {}
        global_start = time.perf_counter()

        def bg_worker():
            bg_result.update(send_single("Count from 1 to 200.", 200))

        def bomb_worker():
            time.sleep(1.0)
            bomb_result.update(send_single(LARGE_PROMPT, 10))

        t1 = threading.Thread(target=bg_worker)
        t2 = threading.Thread(target=bomb_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        if "error" not in bg_result and "error" not in bomb_result:
            print(f"  Trial {i+1}: BG total={bg_result['total_sec']:.3f}s | "
                  f"Bomb TTFT={bomb_result['ttft_sec']:.4f}s")
            spike_results.append({
                "bg_total": bg_result["total_sec"],
                "bomb_ttft": bomb_result["ttft_sec"],
            })

    # Test 3: Concurrent mixed load
    print("\nConcurrent mixed load (20 requests, 3 trials):")
    conc_results = []
    for i in range(3):
        if i > 0:
            time.sleep(5)
        r = run_concurrent_test(20)
        if "error" not in r:
            conc_results.append(r)
            print(f"  Trial {i+1}: TTFT p50={r['ttft_p50']:.4f}s p99={r['ttft_p99']:.4f}s | "
                  f"Total p50={r['total_p50']:.3f}s")

    # Save
    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_test17_results.json"
    existing = []
    if os.path.exists(output_file):
        with open(output_file) as f:
            existing = json.load(f)

    entry = {
        "max_num_batched_tokens": args.budget,
        "isolated_large_prompt_ttft_sec": round(avg_ttft, 4),
        "spike_results": spike_results,
        "concurrent_results": conc_results,
    }

    existing = [e for e in existing if e["max_num_batched_tokens"] != args.budget]
    existing.append(entry)
    existing.sort(key=lambda e: e["max_num_batched_tokens"])

    with open(output_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
