"""
Test 24: Multi-Turn Conversation Simulation

Simulates real chat conversations where each turn includes all previous messages.
Input grows with each turn. With prefix caching, subsequent turns should see
faster TTFT because the prefix (all previous turns) is cached.

Think of it like session affinity in Auto Scaling: if the same user always hits
the same instance, that instance's warm caches benefit subsequent requests.
Here, vLLM's prefix cache acts as the "warm instance" — previous conversation
context is already computed.

Usage: python3 stage2_test24_multi_turn.py --host http://<ip>:8000
"""

import json
import time
import threading
import argparse
import requests

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# Conversation starters — 5 different topics for 5 conversations
CONVERSATIONS = [
    {
        "topic": "system_design",
        "turns": [
            "How would you design a URL shortener service?",
            "What database would you choose and why?",
            "How would you handle the redirect with minimal latency?",
            "What caching strategy would you use?",
            "How would you handle analytics and click tracking?",
        ],
    },
    {
        "topic": "debugging",
        "turns": [
            "Our API latency has increased 3x in the past week. Where do we start investigating?",
            "We found the database queries are slow. How do we diagnose further?",
            "It's a missing index on a frequently joined table. How should we add it safely in production?",
            "The index creation is taking too long on the 500GB table. What options do we have?",
        ],
    },
    {
        "topic": "scaling",
        "turns": [
            "Our application handles 1000 RPS. How do we scale to 100,000 RPS?",
            "We chose horizontal scaling. How do we handle session state?",
            "What about database scaling? We're hitting connection limits.",
            "How do we implement read replicas without stale data issues?",
        ],
    },
    {
        "topic": "security",
        "turns": [
            "What are the top security concerns for a REST API serving financial data?",
            "How should we implement authentication and authorization?",
            "What about rate limiting and DDoS protection?",
            "How do we handle API key rotation without downtime?",
        ],
    },
    {
        "topic": "observability",
        "turns": [
            "We need to improve observability for our microservices. Where do we start?",
            "How should we implement distributed tracing across 50 services?",
            "What metrics should we track for each service?",
            "How do we set up alerting that reduces false positives?",
        ],
    },
]


def run_single_conversation(conv: dict, conv_id: int) -> list:
    """Run one multi-turn conversation, measuring each turn's metrics."""
    url = f"{HOST}/v1/chat/completions"
    messages = [{"role": "system", "content": "You are a helpful senior software engineer. Keep responses concise (2-3 paragraphs max)."}]

    turn_results = []

    for turn_idx, user_msg in enumerate(conv["turns"]):
        # Add user message
        messages.append({"role": "user", "content": user_msg})

        payload = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 150,  # Keep responses short to stay within 2048 context
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
                error = f"HTTP {resp.status_code}: {resp.text[:100]}"
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

        if error or not token_times:
            turn_results.append({
                "turn": turn_idx + 1,
                "error": error or "no tokens received",
                "total_sec": round(request_end - request_start, 3),
            })
            # Still add assistant response placeholder to keep conversation going
            messages.append({"role": "assistant", "content": "(error)"})
            continue

        ttft = token_times[0] - request_start
        total = request_end - request_start
        decode_time = request_end - token_times[0]
        tok_s = len(tokens) / decode_time if decode_time > 0 else 0
        assistant_text = "".join(tokens)

        # Approximate input tokens (rough: 1.3 tokens per word)
        total_input_words = sum(len(m["content"].split()) for m in messages)
        approx_input_tokens = int(total_input_words * 1.3)

        turn_result = {
            "turn": turn_idx + 1,
            "input_tokens_approx": approx_input_tokens,
            "output_tokens": len(tokens),
            "ttft_sec": round(ttft, 4),
            "total_sec": round(total, 3),
            "tok_s": round(tok_s, 1),
            "cumulative_context_tokens": approx_input_tokens + len(tokens),
        }
        turn_results.append(turn_result)

        # Add assistant response to conversation history for next turn
        messages.append({"role": "assistant", "content": assistant_text})

        print(f"    Turn {turn_idx+1}: TTFT={ttft:.4f}s | tok/s={tok_s:.1f} | "
              f"in~{approx_input_tokens} out={len(tokens)} | "
              f"ctx~{approx_input_tokens + len(tokens)}")

        # Brief pause between turns (simulating user reading)
        time.sleep(1)

    return turn_results


def run_concurrent_conversations(n_conv: int = 5, trials: int = 1) -> dict:
    """Run n conversations concurrently."""
    all_results = [None] * n_conv
    threads = []
    global_start = time.perf_counter()
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    def worker(idx):
        all_results[idx] = run_single_conversation(CONVERSATIONS[idx % len(CONVERSATIONS)], idx)

    for i in range(n_conv):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    wall_duration = time.perf_counter() - global_start

    # Aggregate by turn number
    turn_stats = {}
    for conv_results in all_results:
        if not conv_results:
            continue
        for tr in conv_results:
            turn_num = tr["turn"]
            if turn_num not in turn_stats:
                turn_stats[turn_num] = {"ttfts": [], "tok_ss": []}
            if "ttft_sec" in tr:
                turn_stats[turn_num]["ttfts"].append(tr["ttft_sec"])
                turn_stats[turn_num]["tok_ss"].append(tr["tok_s"])

    return {
        "wall_start": wall_start,
        "wall_duration_sec": round(wall_duration, 1),
        "conversations": n_conv,
        "turn_stats": {
            f"turn_{k}": {
                "ttft_p50": round(sorted(v["ttfts"])[len(v["ttfts"]) // 2], 4) if v["ttfts"] else None,
                "tok_s_avg": round(sum(v["tok_ss"]) / len(v["tok_ss"]), 1) if v["tok_ss"] else None,
                "count": len(v["ttfts"]),
            }
            for k, v in sorted(turn_stats.items())
        },
    }


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Test 24: Multi-Turn Conversation")
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    print(f"Test 24: Multi-Turn Conversation Simulation")
    print(f"Target: {HOST} | Model: {MODEL}")
    print(f"5 conversations, 4-5 turns each")
    print(f"{'='*100}\n")

    # Single conversation tests (serial)
    print("SINGLE CONVERSATIONS (serial, one at a time)")
    single_results = []

    for i, conv in enumerate(CONVERSATIONS):
        print(f"\n  Conversation {i+1}: {conv['topic']} ({len(conv['turns'])} turns)")
        turns = run_single_conversation(conv, i)
        single_results.append({
            "topic": conv["topic"],
            "turns": turns,
        })

        time.sleep(3)

    # Concurrent conversations
    print(f"\n\n{'='*100}")
    print("CONCURRENT CONVERSATIONS (5 simultaneous)")
    print(f"{'='*100}")
    concurrent_result = run_concurrent_conversations(n_conv=5)

    # Analysis
    print(f"\n\n{'='*100}")
    print("ANALYSIS")
    print(f"{'='*100}\n")

    # Aggregate TTFT by turn number across all single conversations
    turn_ttfts = {}
    turn_contexts = {}
    for conv in single_results:
        for tr in conv["turns"]:
            turn = tr["turn"]
            if "ttft_sec" in tr:
                turn_ttfts.setdefault(turn, []).append(tr["ttft_sec"])
                turn_contexts.setdefault(turn, []).append(tr.get("cumulative_context_tokens", 0))

    print("TTFT by turn number (avg across 5 conversations):")
    for turn in sorted(turn_ttfts.keys()):
        avg_ttft = sum(turn_ttfts[turn]) / len(turn_ttfts[turn])
        avg_ctx = sum(turn_contexts.get(turn, [0])) / len(turn_contexts.get(turn, [1]))
        print(f"  Turn {turn}: TTFT={avg_ttft:.4f}s | "
              f"context~{avg_ctx:.0f} tokens")

    if len(turn_ttfts) >= 2:
        turn1_avg = sum(turn_ttfts[1]) / len(turn_ttfts[1])
        last_turn = max(turn_ttfts.keys())
        last_avg = sum(turn_ttfts[last_turn]) / len(turn_ttfts[last_turn])
        print(f"\n  Turn 1 → Turn {last_turn} TTFT change: "
              f"{turn1_avg:.4f}s → {last_avg:.4f}s "
              f"({last_avg/turn1_avg:.2f}x)")
        if last_avg < turn1_avg:
            print(f"  → TTFT DECREASED — prefix caching is reusing previous turns!")
        else:
            print(f"  → TTFT increased — growing context outweighs cache benefit")

    # Save
    output = {
        "single_conversations": single_results,
        "concurrent_5_conversations": concurrent_result,
    }
    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_test24_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
