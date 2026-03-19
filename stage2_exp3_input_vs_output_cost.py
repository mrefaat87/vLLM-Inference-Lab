"""
Experiment 3: Why Input Tokens Are Cheaper Than Output Tokens

This experiment demonstrates the cost asymmetry between input and output tokens
by measuring the actual GPU time consumed by each.

The key insight: input tokens are processed in PARALLEL during prefill (matrix-matrix
multiply), while output tokens are generated SEQUENTIALLY during decode (matrix-vector
multiply, one at a time). This means:
  - 1000 input tokens ≈ same GPU time as 100 input tokens (parallel batch)
  - 1000 output tokens ≈ 10x the GPU time of 100 output tokens (sequential)

We prove this by comparing:
  Test 1: "Equivalent" requests — same total tokens, different input/output split
          - 900 input + 100 output vs 100 input + 900 output
  Test 2: Cost per token — measure GPU-seconds per input token vs per output token
  Test 3: Throughput comparison — tokens/sec for prefill vs decode

Usage: python3 stage2_exp3_input_vs_output_cost.py --host http://<ip>:8000
"""

import json
import time
import argparse
import requests

HOST = ""
MODEL = ""


def generate_padding(target_tokens: int) -> str:
    """Generate filler text of approximately target_tokens length."""
    sentences = [
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
    words_needed = int(target_tokens / 1.3)
    words = []
    while len(words) < words_needed:
        for s in sentences:
            words.extend(s.split())
            if len(words) >= words_needed:
                break
    return " ".join(words[:words_needed])


def send_request(prompt: str, max_tokens: int) -> dict:
    """Send one streaming request, measure prefill and decode phases separately."""
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
        return {"error": str(e)}

    request_end = time.perf_counter()
    if not token_times:
        return {"error": "no tokens"}

    prefill_time = token_times[0] - request_start  # Time to first token = prefill
    decode_time = request_end - token_times[0]      # Time after first token = decode
    tokens_out = len(token_times)

    return {
        "prefill_sec": round(prefill_time, 4),
        "decode_sec": round(decode_time, 4),
        "total_sec": round(request_end - request_start, 4),
        "tokens_out": tokens_out,
        "decode_tok_s": round(tokens_out / decode_time, 1) if decode_time > 0 else 0,
    }


def avg_results(results: list) -> dict:
    """Average multiple trial results."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "all failed"}
    return {
        "prefill_sec": round(sum(r["prefill_sec"] for r in valid) / len(valid), 4),
        "decode_sec": round(sum(r["decode_sec"] for r in valid) / len(valid), 4),
        "total_sec": round(sum(r["total_sec"] for r in valid) / len(valid), 4),
        "tokens_out": round(sum(r["tokens_out"] for r in valid) / len(valid), 1),
        "decode_tok_s": round(sum(r["decode_tok_s"] for r in valid) / len(valid), 1),
        "trials": len(valid),
    }


def main():
    global HOST, MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    args = parser.parse_args()
    HOST = args.host.rstrip("/")
    MODEL = args.model

    print(f"Experiment 3: Why Input Tokens Cost Less Than Output Tokens")
    print(f"Target: {HOST} | Model: {MODEL}")
    print(f"{'='*80}\n")

    all_results = {}

    # ---- Test 1: Same total tokens, different split ----
    print("TEST 1: Same total tokens (~1000), different input/output split")
    print("  If input and output cost the same, these should take equal time.")
    print(f"  {'-'*70}")

    configs = [
        ("Heavy input",  900, 100),   # 900 input + 100 output
        ("Balanced",     500, 500),   # 500 input + 500 output
        ("Heavy output", 100, 900),   # 100 input + 900 output
    ]

    test1_results = []
    for label, input_tok, output_tok in configs:
        prompt = generate_padding(input_tok) + "\n\nContinue this text with new ideas:"
        trials = [send_request(prompt, output_tok) for _ in range(3)]
        avg = avg_results(trials)
        test1_results.append({"label": label, "input": input_tok, "output": output_tok, **avg})

        if "error" not in avg:
            print(f"  {label:<15} (in:{input_tok:>4} + out:{output_tok:>4}) | "
                  f"Prefill: {avg['prefill_sec']:.3f}s | Decode: {avg['decode_sec']:.3f}s | "
                  f"Total: {avg['total_sec']:.3f}s | Out tok/s: {avg['decode_tok_s']}")
        time.sleep(2)

    all_results["test1_split_comparison"] = test1_results

    # ---- Test 2: GPU-seconds per token type ----
    print(f"\nTEST 2: GPU cost per token — input vs output")
    print("  Measuring GPU-seconds consumed per token for each phase.")
    print(f"  {'-'*70}")

    # Vary input, fixed output
    input_sizes = [50, 200, 500, 1000]
    test2a_results = []
    for in_tok in input_sizes:
        prompt = generate_padding(in_tok) + "\n\nSummarize briefly:"
        trials = [send_request(prompt, 50) for _ in range(3)]
        avg = avg_results(trials)
        # GPU-seconds per input token = prefill_time / input_tokens
        cost_per_input = avg["prefill_sec"] / in_tok if "error" not in avg else 0
        entry = {"input_tokens": in_tok, "cost_per_input_token_ms": round(cost_per_input * 1000, 4), **avg}
        test2a_results.append(entry)
        if "error" not in avg:
            print(f"  {in_tok:>5} input tokens → prefill: {avg['prefill_sec']:.4f}s → "
                  f"{cost_per_input*1000:.4f} ms/input_token")
        time.sleep(1)

    # Vary output, fixed input
    output_sizes = [50, 200, 500, 1000]
    test2b_results = []
    for out_tok in output_sizes:
        prompt = "Write a detailed essay about cloud computing infrastructure:\n\n"
        trials = [send_request(prompt, out_tok) for _ in range(3)]
        avg = avg_results(trials)
        # GPU-seconds per output token = decode_time / output_tokens
        cost_per_output = avg["decode_sec"] / avg["tokens_out"] if "error" not in avg and avg["tokens_out"] > 0 else 0
        entry = {"output_tokens": out_tok, "cost_per_output_token_ms": round(cost_per_output * 1000, 4), **avg}
        test2b_results.append(entry)
        if "error" not in avg:
            print(f"  {avg['tokens_out']:>5.0f} output tokens → decode: {avg['decode_sec']:.3f}s → "
                  f"{cost_per_output*1000:.4f} ms/output_token")
        time.sleep(1)

    all_results["test2a_cost_per_input_token"] = test2a_results
    all_results["test2b_cost_per_output_token"] = test2b_results

    # ---- Test 3: Throughput comparison ----
    print(f"\nTEST 3: Throughput — prefill tok/s vs decode tok/s")
    print("  Prefill processes many tokens at once (parallel). Decode generates one at a time.")
    print(f"  {'-'*70}")

    # Large input, small output — measures prefill throughput
    prompt_large = generate_padding(1000) + "\n\nSummarize in one word:"
    trials = [send_request(prompt_large, 10) for _ in range(3)]
    avg_prefill = avg_results(trials)
    # Prefill throughput = input tokens / prefill time
    prefill_throughput = 1000 / avg_prefill["prefill_sec"] if "error" not in avg_prefill else 0

    # Small input, large output — measures decode throughput
    prompt_small = "Write a very long detailed essay about distributed systems:\n\n"
    trials = [send_request(prompt_small, 1000) for _ in range(3)]
    avg_decode = avg_results(trials)
    decode_throughput = avg_decode["decode_tok_s"] if "error" not in avg_decode else 0

    all_results["test3_throughput"] = {
        "prefill_throughput_tok_s": round(prefill_throughput, 1),
        "decode_throughput_tok_s": decode_throughput,
        "prefill_result": avg_prefill,
        "decode_result": avg_decode,
    }

    print(f"  Prefill throughput: {prefill_throughput:.1f} tok/s (1000 input tokens in {avg_prefill.get('prefill_sec', '?')}s)")
    print(f"  Decode throughput:  {decode_throughput} tok/s (sequential generation)")
    print(f"  Ratio: prefill is {prefill_throughput/decode_throughput:.1f}x faster per token" if decode_throughput > 0 else "")

    # ---- Summary ----
    print(f"\n{'='*80}")
    print("WHY INPUT TOKENS ARE CHEAPER THAN OUTPUT TOKENS")
    print(f"{'='*80}\n")

    if test2a_results and test2b_results:
        avg_input_cost = sum(r["cost_per_input_token_ms"] for r in test2a_results) / len(test2a_results)
        avg_output_cost = sum(r["cost_per_output_token_ms"] for r in test2b_results) / len(test2b_results)
        ratio = avg_output_cost / avg_input_cost if avg_input_cost > 0 else 0

        print(f"  Average GPU cost per INPUT token:  {avg_input_cost:.4f} ms")
        print(f"  Average GPU cost per OUTPUT token: {avg_output_cost:.4f} ms")
        print(f"  Output tokens cost {ratio:.1f}x more GPU time than input tokens")
        print()
        print(f"  This is why API providers charge more for output:")
        print(f"  - Input tokens are processed in PARALLEL (matrix-matrix multiply)")
        print(f"    → Doubling input barely increases prefill time")
        print(f"  - Output tokens are generated SEQUENTIALLY (matrix-vector multiply)")
        print(f"    → Doubling output doubles decode time")
        print(f"  - The GPU sits mostly idle during decode (memory-bandwidth bound)")
        print(f"    → Each output token consumes a full weight-read cycle")

    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_exp3_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    main()
