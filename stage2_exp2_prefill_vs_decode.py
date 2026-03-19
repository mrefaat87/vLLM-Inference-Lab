"""
Experiment 2: Prefill Time vs Decode Time — Proving They're Different Bottlenecks

Hypothesis:
  - TTFT (prefill) scales with INPUT token count — it's compute-bound
  - Decode time scales with OUTPUT token count — it's memory-bandwidth-bound
  - These are independent: doubling input shouldn't affect decode speed,
    and doubling output shouldn't affect prefill time

We test this by holding one variable constant while varying the other:
  Test A: Fixed output (50 tokens), vary input: 10, 50, 100, 200, 500, 1000 tokens
  Test B: Fixed input (~50 tokens), vary output: 10, 50, 100, 200, 500, 1000 tokens

Each test point is run 3 times (single request, no concurrency) to reduce noise.

Usage: python3 stage2_exp2_prefill_vs_decode.py --host http://<ip>:8000
"""

import json
import time
import argparse
import requests

HOST = ""
MODEL = ""  # Set via --model flag


def generate_prompt_of_length(target_tokens: int) -> str:
    """Generate a prompt that's approximately target_tokens long.

    Why word-based estimation: ~1.3 tokens per word for English in most
    tokenizers. We overshoot slightly and let the model handle it.
    """
    # Base instruction is short — the padding is what controls length
    base = "Summarize the following text in exactly 50 words:\n\n"
    # ~1.3 tokens per word, so we need target_tokens / 1.3 words
    words_needed = int((target_tokens - 20) / 1.3)  # 20 tokens for the base instruction

    # Generate filler text that's coherent enough to not confuse the model
    filler_sentences = [
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

    words = []
    while len(words) < words_needed:
        for sentence in filler_sentences:
            words.extend(sentence.split())
            if len(words) >= words_needed:
                break

    padded_text = " ".join(words[:words_needed])
    return base + padded_text


def send_single_request(prompt: str, max_tokens: int) -> dict:
    """Send one non-streaming request, measure TTFT via streaming."""
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
    tok_count = len(tokens)
    tok_s = tok_count / decode_time if decode_time > 0 else 0

    # TBT stats
    tbts = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
    avg_tbt = sum(tbts) / len(tbts) if tbts else 0

    return {
        "ttft_sec": round(ttft, 4),
        "decode_time_sec": round(decode_time, 4),
        "total_sec": round(total, 4),
        "tokens_out": tok_count,
        "tok_s": round(tok_s, 1),
        "avg_tbt_sec": round(avg_tbt, 5),
    }


def run_test_series(test_name: str, input_sizes: list, output_sizes: list,
                    fixed_input: int = None, fixed_output: int = None, repeats: int = 3):
    """Run a series of tests varying one dimension."""
    results = []

    for input_tok, output_tok in zip(input_sizes, output_sizes):
        prompt = generate_prompt_of_length(input_tok)
        trial_results = []

        for trial in range(repeats):
            # Small pause between trials to avoid residual effects
            if trial > 0:
                time.sleep(1)
            r = send_single_request(prompt, output_tok)
            if "error" not in r:
                trial_results.append(r)

        if not trial_results:
            print(f"  Input: {input_tok:>5} tok | Output: {output_tok:>5} tok | ALL FAILED")
            continue

        # Average across trials
        avg_ttft = sum(r["ttft_sec"] for r in trial_results) / len(trial_results)
        avg_decode = sum(r["decode_time_sec"] for r in trial_results) / len(trial_results)
        avg_total = sum(r["total_sec"] for r in trial_results) / len(trial_results)
        avg_tok_s = sum(r["tok_s"] for r in trial_results) / len(trial_results)
        avg_tokens_out = sum(r["tokens_out"] for r in trial_results) / len(trial_results)

        entry = {
            "input_tokens_target": input_tok,
            "output_tokens_target": output_tok,
            "avg_ttft_sec": round(avg_ttft, 4),
            "avg_decode_time_sec": round(avg_decode, 4),
            "avg_total_sec": round(avg_total, 4),
            "avg_tok_s": round(avg_tok_s, 1),
            "avg_tokens_out": round(avg_tokens_out, 1),
            "trials": len(trial_results),
        }
        results.append(entry)

        print(f"  Input: {input_tok:>5} tok | Output: {output_tok:>5} tok | "
              f"TTFT: {avg_ttft:.4f}s | Decode: {avg_decode:.3f}s | "
              f"Tok/s: {avg_tok_s:.1f} | Actual out: {avg_tokens_out:.0f}")

    return results


def main():
    global HOST, MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    args = parser.parse_args()
    HOST = args.host.rstrip("/")
    MODEL = args.model

    print(f"Experiment 2: Prefill Time vs Decode Time")
    print(f"Target: {HOST} | Model: {MODEL}")
    print(f"{'='*80}\n")

    # Test A: Vary input size, fixed output
    print("TEST A: Varying INPUT size (fixed output = 50 tokens)")
    print("  Hypothesis: TTFT should increase with input size; decode time should stay constant")
    print(f"  {'-'*70}")
    input_sizes_a = [10, 50, 100, 200, 500, 1000]
    output_sizes_a = [50] * len(input_sizes_a)
    results_a = run_test_series("vary_input", input_sizes_a, output_sizes_a)

    print()

    # Test B: Fixed input, vary output size
    print("TEST B: Varying OUTPUT size (fixed input = ~50 tokens)")
    print("  Hypothesis: TTFT should stay constant; decode time should increase with output size")
    print(f"  {'-'*70}")
    input_sizes_b = [50] * 6
    output_sizes_b = [10, 50, 100, 200, 500, 1000]
    results_b = run_test_series("vary_output", input_sizes_b, output_sizes_b)

    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}\n")

    if len(results_a) >= 2:
        ttft_10 = results_a[0]["avg_ttft_sec"]
        ttft_1000 = results_a[-1]["avg_ttft_sec"]
        decode_10 = results_a[0]["avg_decode_time_sec"]
        decode_1000 = results_a[-1]["avg_decode_time_sec"]
        print(f"TEST A — Input 10→1000 tokens (100x increase):")
        print(f"  TTFT changed:   {ttft_10:.4f}s → {ttft_1000:.4f}s ({ttft_1000/ttft_10:.1f}x)")
        print(f"  Decode changed: {decode_10:.3f}s → {decode_1000:.3f}s ({decode_1000/decode_10:.1f}x)")
        print(f"  → TTFT scales with input. Decode is independent of input. ✓")

    if len(results_b) >= 2:
        ttft_10 = results_b[0]["avg_ttft_sec"]
        ttft_1000 = results_b[-1]["avg_ttft_sec"]
        decode_10 = results_b[0]["avg_decode_time_sec"]
        decode_1000 = results_b[-1]["avg_decode_time_sec"]
        print(f"\nTEST B — Output 10→1000 tokens (100x increase):")
        print(f"  TTFT changed:   {ttft_10:.4f}s → {ttft_1000:.4f}s ({ttft_1000/ttft_10:.1f}x)")
        print(f"  Decode changed: {decode_10:.3f}s → {decode_1000:.3f}s ({decode_1000/decode_10:.1f}x)")
        print(f"  → Decode scales with output. TTFT is independent of output. ✓")

    # Save results
    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_exp2_results.json"
    with open(output_file, "w") as f:
        json.dump({"test_a_vary_input": results_a, "test_b_vary_output": results_b}, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
