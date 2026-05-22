"""
Test 28: Full Production Simulation

Combines all learnings into a realistic production traffic pattern:
  - 40% short queries (20-50 input, 50-100 output)
  - 30% medium queries (100-200 input, 100-300 output)
  - 15% long queries (300-500 input, 200-500 output)
  - 10% JSON output (guided decoding)
  - 5% multi-turn (3 turns per conversation)

Traffic pattern: Poisson process at 6 RPS avg, with 2 burst periods (15 RPS).
Duration: 10 minutes.

This is the "final exam" — the production readiness assessment. Can we define
SLAs and meet them? Like an ASG launch template validation: run your full
traffic mix and prove the fleet handles it within SLA bounds.

Usage: python3 stage2_test28_production_sim.py --host http://<ip>:8000
"""

import json
import time
import threading
import argparse
import requests
import random
import math

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

DURATION_MIN = 10
BASE_RPS = 6
BURST_RPS = 12

# Burst windows (seconds from start)
BURST_WINDOWS = [(180, 210), (420, 450)]

# Prompt templates by type
SHORT_PROMPTS = [
    "What is a load balancer?",
    "Define eventual consistency in 2 sentences.",
    "What is the CAP theorem?",
    "Explain DNS resolution briefly.",
    "What is a CDN?",
]

MEDIUM_PROMPTS = [
    ("Explain how auto-scaling works in cloud computing. Cover target tracking, "
     "step scaling, and scheduled scaling. Include when you'd use each approach."),
    ("Describe the differences between synchronous and asynchronous communication "
     "patterns in microservices. Include examples of message brokers and RPC frameworks."),
    ("How does a distributed cache like Redis work? Cover data structures, "
     "eviction policies, and cluster mode."),
]

LONG_PROMPTS = [
    ("You are designing a payment processing system for an e-commerce platform "
     "that handles 10 million transactions per day. The system must support "
     "multiple payment methods (credit card, bank transfer, digital wallets), "
     "handle currency conversion, comply with PCI DSS, and maintain an SLA of "
     "99.99% availability. Describe the complete architecture including: "
     "API gateway design, transaction processing pipeline, fraud detection "
     "integration, database choices, event sourcing for audit trail, "
     "retry and idempotency handling, and monitoring strategy."),
    ("Design a real-time recommendation engine for a streaming platform with "
     "100 million users. The system needs to generate personalized content "
     "recommendations within 100ms, handle cold start problems for new users, "
     "support A/B testing of different recommendation algorithms, and process "
     "user interaction events in real-time. Describe the ML pipeline, "
     "feature store, serving infrastructure, and data architecture."),
]

MULTI_TURN_TOPICS = [
    [
        "How should we design a URL shortener?",
        "What database would you choose for this?",
        "How do we handle the redirect with minimal latency?",
    ],
    [
        "Our API latency spiked 3x. Where do we investigate?",
        "Database queries are slow. What's the diagnosis approach?",
        "We found a missing index. How to add it safely in production?",
    ],
]

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["name", "age", "skills", "summary"],
}


def send_completions_request(prompt: str, max_tokens: int, req_type: str,
                              results: list, lock: threading.Lock,
                              global_start: float):
    """Send a completions request."""
    url = f"{HOST}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
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
    _record_result(request_start, request_end, token_times, tokens, error,
                   req_type, results, lock, global_start)


def send_chat_request(messages: list, max_tokens: int, req_type: str,
                      results: list, lock: threading.Lock,
                      global_start: float, response_format: dict = None):
    """Send a chat completions request."""
    url = f"{HOST}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }
    if response_format:
        payload["response_format"] = response_format

    token_times = []
    tokens = []
    error = None
    request_start = time.perf_counter()

    try:
        resp = requests.post(url, json=payload,
                             headers={"Content-Type": "application/json"},
                             stream=True, timeout=(60, 300))
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
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    token_times.append(time.perf_counter())
                    tokens.append(text)
    except Exception as e:
        error = str(e)

    request_end = time.perf_counter()

    # For JSON requests, check validity
    extra = {}
    if req_type == "json" and tokens:
        full_text = "".join(tokens)
        try:
            json.loads(full_text)
            extra["schema_valid"] = True
        except (json.JSONDecodeError, TypeError):
            extra["schema_valid"] = False

    _record_result(request_start, request_end, token_times, tokens, error,
                   req_type, results, lock, global_start, extra)


def _record_result(request_start, request_end, token_times, tokens, error,
                   req_type, results, lock, global_start, extra=None):
    """Record a request result."""
    if error or not token_times:
        result = {
            "type": req_type,
            "error": error or "no tokens",
            "total_sec": round(request_end - request_start, 3),
            "abs_start": round(request_start - global_start, 2),
        }
    else:
        ttft = token_times[0] - request_start
        total = request_end - request_start
        decode_time = request_end - token_times[0]
        tok_s = len(tokens) / decode_time if decode_time > 0 else 0
        result = {
            "type": req_type,
            "ttft_sec": round(ttft, 4),
            "total_sec": round(total, 3),
            "tokens": len(tokens),
            "tok_s": round(tok_s, 1),
            "abs_start": round(request_start - global_start, 2),
        }
    if extra:
        result.update(extra)

    with lock:
        results.append(result)


def send_multi_turn(topic_turns: list, results: list, lock: threading.Lock,
                    global_start: float):
    """Run a multi-turn conversation, recording each turn."""
    messages = [{"role": "system", "content": "You are a helpful engineer. Be concise."}]
    for turn in topic_turns:
        messages.append({"role": "user", "content": turn})
        # Blocking call for multi-turn (sequential within conversation)
        url = f"{HOST}/v1/chat/completions"
        payload = {
            "model": MODEL,
            "messages": messages,
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
                                 stream=True, timeout=(60, 300))
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
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        token_times.append(time.perf_counter())
                        tokens.append(text)
        except Exception as e:
            error = str(e)

        request_end = time.perf_counter()
        _record_result(request_start, request_end, token_times, tokens, error,
                       "multi_turn", results, lock, global_start)

        if tokens:
            messages.append({"role": "assistant", "content": "".join(tokens)})

        time.sleep(0.5)  # Simulate user reading


def generate_request(results: list, lock: threading.Lock, global_start: float):
    """Generate one random request based on the traffic mix."""
    roll = random.random()

    if roll < 0.40:
        # Short query
        prompt = random.choice(SHORT_PROMPTS)
        max_tokens = random.randint(50, 100)
        send_completions_request(prompt, max_tokens, "short", results, lock, global_start)

    elif roll < 0.70:
        # Medium query
        prompt = random.choice(MEDIUM_PROMPTS)
        max_tokens = random.randint(100, 200)
        send_completions_request(prompt, max_tokens, "medium", results, lock, global_start)

    elif roll < 0.85:
        # Long query
        prompt = random.choice(LONG_PROMPTS)
        max_tokens = random.randint(200, 300)
        send_completions_request(prompt, max_tokens, "long", results, lock, global_start)

    elif roll < 0.95:
        # JSON output
        messages = [{"role": "user",
                     "content": "Generate a profile for a software engineer. "
                                "Respond with JSON: name, age, skills array, summary."}]
        rf = {
            "type": "json_schema",
            "json_schema": {"name": "person", "strict": True, "schema": JSON_SCHEMA},
        }
        send_chat_request(messages, 200, "json", results, lock, global_start,
                         response_format=rf)
    else:
        # Multi-turn
        topic = random.choice(MULTI_TURN_TOPICS)
        send_multi_turn(topic, results, lock, global_start)


def get_rps_at_time(elapsed_sec: float) -> float:
    """Return target RPS based on elapsed time, with burst windows."""
    for burst_start, burst_end in BURST_WINDOWS:
        if burst_start <= elapsed_sec <= burst_end:
            return BURST_RPS
    return BASE_RPS


def percentile(sorted_vals: list, p: float) -> float:
    """Compute percentile."""
    if not sorted_vals:
        return 0
    idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
    return sorted_vals[idx]


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Test 28: Full Production Simulation")
    parser.add_argument("--host", required=True)
    parser.add_argument("--duration", type=int, default=10, help="Duration in minutes")
    args = parser.parse_args()
    HOST = args.host.rstrip("/")
    duration_sec = args.duration * 60

    print(f"Test 28: Full Production Simulation")
    print(f"Target: {HOST} | Model: {MODEL}")
    print(f"Duration: {args.duration} min | Base RPS: {BASE_RPS} | Burst RPS: {BURST_RPS}")
    print(f"Burst windows: {BURST_WINDOWS}")
    print(f"Traffic mix: 40% short, 30% medium, 15% long, 10% JSON, 5% multi-turn")
    print(f"{'='*120}\n")

    all_results = []
    lock = threading.Lock()
    global_start = time.perf_counter()
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    # Use Poisson process: inter-arrival times are exponentially distributed
    elapsed = 0
    request_count = 0

    while elapsed < duration_sec:
        current_rps = get_rps_at_time(elapsed)
        # Exponential inter-arrival time for Poisson process
        inter_arrival = random.expovariate(current_rps)

        target_time = global_start + elapsed + inter_arrival
        now = time.perf_counter()
        if target_time > now:
            time.sleep(target_time - now)

        elapsed = time.perf_counter() - global_start
        if elapsed >= duration_sec:
            break

        t = threading.Thread(target=generate_request,
                           args=(all_results, lock, global_start))
        t.start()
        request_count += 1

        # Progress every 60 seconds
        if request_count % (BASE_RPS * 60) == 0:
            minute = int(elapsed / 60)
            with lock:
                ok = sum(1 for r in all_results if "error" not in r)
                fail = sum(1 for r in all_results if "error" in r)
            current = "BURST" if any(s <= elapsed <= e for s, e in BURST_WINDOWS) else "normal"
            print(f"  [{minute}m] {request_count} sent | {ok} OK / {fail} fail | "
                  f"mode={current}")

    # Wait for stragglers
    print(f"\n  Waiting for in-flight requests to complete...")
    time.sleep(30)

    test_duration = time.perf_counter() - global_start
    wall_end = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    # Analysis
    print(f"\n{'='*120}")
    print("RESULTS")
    print(f"{'='*120}\n")

    successes = [r for r in all_results if "error" not in r]
    failures = [r for r in all_results if "error" in r]
    total = len(all_results)

    print(f"Total: {total} requests | {len(successes)} OK | {len(failures)} failures "
          f"({len(failures)/total*100:.1f}%) | Duration: {test_duration/60:.1f} min\n")

    # Overall metrics
    if successes:
        ttfts = sorted([r["ttft_sec"] for r in successes])
        totals = sorted([r["total_sec"] for r in successes])
        total_tokens = sum(r.get("tokens", 0) for r in successes)

        overall = {
            "ttft_p50": round(percentile(ttfts, 50), 4),
            "ttft_p99": round(percentile(ttfts, 99), 4),
            "ttft_max": round(max(ttfts), 4),
            "total_p50": round(percentile(totals, 50), 3),
            "total_p99": round(percentile(totals, 99), 3),
            "total_max": round(max(totals), 3),
            "system_tok_s_avg": round(total_tokens / test_duration, 1),
            "failure_rate_pct": round(len(failures) / total * 100, 2),
        }

        print(f"Overall: TTFT p50={overall['ttft_p50']:.4f}s p99={overall['ttft_p99']:.4f}s | "
              f"Total p50={overall['total_p50']:.3f}s p99={overall['total_p99']:.3f}s | "
              f"Sys={overall['system_tok_s_avg']:.0f} tok/s")

    # Per-type breakdown
    print(f"\n{'Type':<12} {'Count':>6} {'TTFT p50':>10} {'TTFT p99':>10} "
          f"{'Total p50':>11} {'Total p99':>11}")
    print("-" * 65)

    by_type = {}
    for req_type in ["short", "medium", "long", "json", "multi_turn"]:
        type_results = [r for r in successes if r["type"] == req_type]
        type_failures = [r for r in failures if r.get("type") == req_type]

        if type_results:
            t_ttfts = sorted([r["ttft_sec"] for r in type_results])
            t_totals = sorted([r["total_sec"] for r in type_results])

            by_type[req_type] = {
                "count": len(type_results),
                "failures": len(type_failures),
                "ttft_p50": round(percentile(t_ttfts, 50), 4),
                "ttft_p99": round(percentile(t_ttfts, 99), 4),
                "total_p50": round(percentile(t_totals, 50), 3),
                "total_p99": round(percentile(t_totals, 99), 3),
            }

            if req_type == "json":
                valid = sum(1 for r in type_results if r.get("schema_valid"))
                by_type[req_type]["schema_valid_pct"] = round(valid / len(type_results) * 100, 1)

            bt = by_type[req_type]
            print(f"{req_type:<12} {bt['count']:>6} {bt['ttft_p50']:>10.4f} "
                  f"{bt['ttft_p99']:>10.4f} {bt['total_p50']:>11.3f} "
                  f"{bt['total_p99']:>11.3f}")

    # SLA compliance
    print(f"\nSLA Compliance:")
    if successes:
        ttft_under_1s = sum(1 for r in successes if r["ttft_sec"] < 1.0) / len(successes) * 100
        total_under_5s = sum(1 for r in successes if r["total_sec"] < 5.0) / len(successes) * 100
        total_under_10s = sum(1 for r in successes if r["total_sec"] < 10.0) / len(successes) * 100
        print(f"  TTFT < 1s:  {ttft_under_1s:.1f}%")
        print(f"  Total < 5s: {total_under_5s:.1f}%")
        print(f"  Total < 10s: {total_under_10s:.1f}%")

    # Burst period analysis
    print(f"\nBurst Period Impact:")
    for burst_start, burst_end in BURST_WINDOWS:
        burst_results = [r for r in successes
                        if burst_start <= r.get("abs_start", 0) <= burst_end]
        burst_failures = [r for r in failures
                         if burst_start <= r.get("abs_start", 0) <= burst_end]
        if burst_results:
            b_ttfts = sorted([r["ttft_sec"] for r in burst_results])
            print(f"  [{burst_start}-{burst_end}s]: {len(burst_results)} OK / "
                  f"{len(burst_failures)} fail | "
                  f"TTFT p99={percentile(b_ttfts, 99):.4f}s")

    # Save
    output = {
        "duration_minutes": args.duration,
        "wall_start": wall_start,
        "wall_end": wall_end,
        "total_requests": total,
        "successes": len(successes),
        "failures": len(failures),
        "overall": overall if successes else {},
        "by_type": by_type,
    }

    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_test28_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
