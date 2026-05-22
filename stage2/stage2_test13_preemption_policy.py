"""
Test 13: Preemption Policy — Recompute vs Swap

When KV cache is full, vLLM must preempt running requests to make room.
Two strategies:
  - Recompute: discard evicted KV, recompute from prompt when resumed
  - Swap: move evicted KV to CPU RAM, swap back when resumed

Uses GPTQ Int8 model (smaller KV cache = easier to trigger preemption)
with unique prompts (no prefix sharing) and no prefix caching.

Requires --preemption-mode flag which only works on V0 engine.

Usage:
  python3 stage2_test13_preemption_policy.py --host http://<ip>:8000 --mode recompute
  python3 stage2_test13_preemption_policy.py --host http://<ip>:8000 --mode swap
"""

import json
import time
import threading
import argparse
import requests
import os

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"

# 42 unique prompts to prevent prefix cache sharing, exceed ~39 max concurrency
ESSAY_TOPICS = [
    "the history of computer networking from ARPANET to 5G",
    "how operating systems manage virtual memory and page tables",
    "the evolution of database systems from hierarchical to distributed",
    "machine learning model training at scale using data parallelism",
    "the design of fault-tolerant distributed consensus algorithms",
    "how content delivery networks optimize global content distribution",
    "the architecture of modern web browsers and rendering engines",
    "cryptographic protocols that secure internet communications",
    "the principles of compiler design and code optimization",
    "how search engines index and rank billions of web pages",
    "the evolution of programming languages from Assembly to Rust",
    "containerization technology from chroot to Kubernetes",
    "the mathematics behind public key cryptography and RSA",
    "how GPUs accelerate parallel computation for deep learning",
    "the design of real-time operating systems for embedded devices",
    "distributed file systems from NFS to cloud object storage",
    "the architecture of modern CPU pipelines and branch prediction",
    "how message queues enable asynchronous microservice communication",
    "the principles of network security and intrusion detection",
    "the evolution of version control from CVS to Git",
    "how load balancers distribute traffic across server fleets",
    "the design of recommendation systems at Netflix scale",
    "memory management techniques in garbage-collected languages",
    "how DNS resolution works from browser to authoritative server",
    "the architecture of stream processing systems like Apache Kafka",
    "RAID storage configurations and their reliability tradeoffs",
    "how TLS handshakes establish secure encrypted connections",
    "the design of auto-scaling systems for cloud infrastructure",
    "functional programming paradigms and their practical applications",
    "how graph databases model and query connected data",
    "the principles of chaos engineering and resilience testing",
    "microprocessor fabrication from silicon wafer to finished chip",
    "the design of rate limiting algorithms for API protection",
    "how video codecs compress and decompress media streams",
    "the architecture of serverless computing platforms like Lambda",
    "cache coherence protocols in multi-core processor systems",
    "how observability platforms correlate metrics logs and traces",
    "the design of consensus algorithms like Raft and Paxos",
    "quantum computing fundamentals and current hardware limitations",
    "how edge computing reduces latency for real-time applications",
    "the principles of zero-trust security architecture",
    "how inference serving systems optimize GPU utilization",
]


def build_prompt(topic: str) -> str:
    return (f"Write a detailed technical essay about {topic}. "
            f"Cover the fundamental principles, key innovations, real-world examples, "
            f"and future directions. Include specific technical details and comparisons. "
            f"The essay should be comprehensive and educational, suitable for a senior "
            f"engineer who wants to deeply understand the topic.")


def send_request(request_id: int, prompt: str, results: list, global_start: float):
    url = f"{HOST}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": prompt,
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
        if resp.status_code != 200:
            error = f"http_{resp.status_code}"
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
        results[request_id] = {
            "request_id": request_id,
            "error": error or "no tokens",
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


def run_overload_test(n: int = 42) -> dict:
    """Send n concurrent requests with unique prompts to trigger preemption."""
    prompts = [build_prompt(ESSAY_TOPICS[i % len(ESSAY_TOPICS)]) for i in range(n)]
    results = [None] * n
    threads = []
    global_start = time.perf_counter()
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    for i in range(n):
        t = threading.Thread(target=send_request,
                           args=(i, prompts[i], results, global_start))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=600)

    wall_duration = time.perf_counter() - global_start

    successes = [r for r in results if r and "error" not in r]
    failures = [r for r in results if r and "error" in r]

    if not successes:
        return {
            "concurrency": n, "wall_start": wall_start,
            "wall_duration_sec": round(wall_duration, 1),
            "successes": 0, "failures": len(failures) + (n - len(failures)),
            "error_sample": failures[0]["error"] if failures else "all None",
        }

    ttfts = sorted([r["ttft_sec"] for r in successes])
    totals = sorted([r["total_sec"] for r in successes])
    tok_rates = [r["tokens_per_sec"] for r in successes]

    def pct(vals, p):
        idx = min(int(len(vals) * p / 100), len(vals) - 1)
        return vals[idx]

    return {
        "concurrency": n, "wall_start": wall_start,
        "wall_duration_sec": round(wall_duration, 1),
        "successes": len(successes), "failures": len(failures),
        "ttft_p50": round(pct(ttfts, 50), 4),
        "ttft_p99": round(pct(ttfts, 99), 4),
        "ttft_max": round(max(ttfts), 4),
        "total_p50": round(pct(totals, 50), 3),
        "total_p99": round(pct(totals, 99), 3),
        "total_max": round(max(totals), 3),
        "avg_tok_per_sec": round(sum(tok_rates) / len(tok_rates), 1),
        "system_tok_s": round(sum(r["tokens"] for r in successes) / wall_duration, 1),
    }


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Test 13: Preemption Policy")
    parser.add_argument("--host", required=True)
    parser.add_argument("--mode", choices=["recompute", "swap"], required=True)
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    print(f"Test 13: Preemption Policy — {args.mode}")
    print(f"Target: {HOST} | Model: {MODEL}")
    print(f"Sending 42 concurrent unique long prompts (max ~39 concurrent)")
    print(f"{'='*80}\n")

    result = run_overload_test(42)

    if "error_sample" in result:
        print(f"ALL FAILED: {result['error_sample']}")
    else:
        print(f"OK: {result['successes']}/42 | Fail: {result['failures']}")
        print(f"TTFT p50={result['ttft_p50']:.4f}s p99={result['ttft_p99']:.4f}s max={result['ttft_max']:.4f}s")
        print(f"Total p50={result['total_p50']:.1f}s p99={result['total_p99']:.1f}s max={result['total_max']:.1f}s")
        print(f"Avg tok/s={result['avg_tok_per_sec']:.1f} | System tok/s={result['system_tok_s']:.1f}")
        print(f"Wall duration: {result['wall_duration_sec']:.1f}s")

    # Save — merge with existing
    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_test13_results.json"
    existing = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            existing = json.load(f)

    existing[args.mode] = result
    with open(output_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
