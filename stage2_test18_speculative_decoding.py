"""
Test 18: Speculative Decoding — Draft Model Speedup
Test 19: Speculative Decoding — Varying num_speculative_tokens

Speculative decoding uses a small "draft" model to propose tokens, then the
target model verifies them in one forward pass. If predictions match, you
get multiple tokens per main-model step.

Test 18: Baseline (no spec) vs speculative (k=5) across 3 prompt types
Test 19: Sweep k=1,2,3,5,7 with factual prompt

Usage:
  python3 stage2_test18_speculative_decoding.py --host http://<ip>:8000 --phase baseline
  python3 stage2_test18_speculative_decoding.py --host http://<ip>:8000 --phase speculative
  python3 stage2_test18_speculative_decoding.py --host http://<ip>:8000 --phase sweep
"""

import json
import time
import threading
import argparse
import requests
import os

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

PROMPTS = {
    "factual": "List the planets in the solar system in order from the sun, with their approximate diameters in kilometers.",
    "creative": "Write a short poem about a robot learning to dream. Use vivid imagery and unexpected metaphors.",
    "code": "Write a Python function that implements binary search on a sorted array. Include docstring and type hints.",
}

REPEATS = 5


def send_single(prompt: str, max_tokens: int = 200) -> dict:
    url = f"{HOST}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,  # Greedy for reproducibility
    }

    token_times = []
    tokens = []
    request_start = time.perf_counter()

    try:
        resp = requests.post(url, json=payload,
                             headers={"Content-Type": "application/json"},
                             stream=True, timeout=(60, 300))
        if resp.status_code != 200:
            return {"error": f"http_{resp.status_code}: {resp.text[:100]}"}
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

    if not token_times:
        return {"error": "no tokens"}

    ttft = token_times[0] - request_start
    total = time.perf_counter() - request_start
    decode_time = time.perf_counter() - token_times[0]
    tok_s = len(tokens) / decode_time if decode_time > 0 else 0

    return {
        "ttft_sec": round(ttft, 4),
        "total_sec": round(total, 3),
        "tokens": len(tokens),
        "tok_s": round(tok_s, 1),
    }


def run_prompt_type_test(prompt_type: str) -> dict:
    """Run REPEATS requests for a given prompt type."""
    prompt = PROMPTS[prompt_type]
    results = []

    for i in range(REPEATS):
        if i > 0:
            time.sleep(1)
        r = send_single(prompt)
        if "error" not in r:
            results.append(r)
            print(f"    [{prompt_type}] {i+1}/{REPEATS}: {r['tok_s']:.1f} tok/s | "
                  f"TTFT={r['ttft_sec']:.4f}s | tokens={r['tokens']}")
        else:
            print(f"    [{prompt_type}] {i+1}/{REPEATS}: ERROR - {r['error'][:60]}")

    if not results:
        return {"error": "all failed"}

    return {
        "ttft_avg": round(sum(r["ttft_sec"] for r in results) / len(results), 4),
        "total_avg": round(sum(r["total_sec"] for r in results) / len(results), 3),
        "tok_s_avg": round(sum(r["tok_s"] for r in results) / len(results), 1),
        "trials": len(results),
    }


def run_concurrent_test(n: int = 5) -> dict:
    """Run n concurrent requests."""
    results = [None] * n
    threads = []
    global_start = time.perf_counter()

    def worker(idx):
        results[idx] = send_single(PROMPTS["factual"])

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
    tok_rates = [r["tok_s"] for r in successes]
    total_tokens = sum(r["tokens"] for r in successes)

    return {
        "successes": len(successes),
        "failures": n - len(successes),
        "ttft_p50": round(ttfts[len(ttfts) // 2], 4),
        "per_request_tok_s_avg": round(sum(tok_rates) / len(tok_rates), 1),
        "system_tok_s": round(total_tokens / wall, 1),
    }


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Tests 18-19: Speculative Decoding")
    parser.add_argument("--host", required=True)
    parser.add_argument("--phase", choices=["baseline", "speculative", "sweep"], required=True)
    parser.add_argument("--num-spec-tokens", type=int, default=5,
                        help="For sweep phase: specific k value to test")
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_test18_results.json"
    existing = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            existing = json.load(f)

    if args.phase == "baseline":
        print(f"Test 18: Speculative Decoding — BASELINE (no draft model)")
        print(f"{'='*80}\n")

        single_results = {}
        for pt in ["factual", "creative", "code"]:
            print(f"\n  Prompt type: {pt}")
            single_results[pt] = run_prompt_type_test(pt)

        print(f"\n  Concurrent test (N=5):")
        conc = run_concurrent_test(5)
        if "error" not in conc:
            print(f"    OK={conc['successes']} | sys_tok/s={conc['system_tok_s']} | "
                  f"per_req={conc['per_request_tok_s_avg']}")

        existing["baseline"] = {
            "single_request": single_results,
            "concurrent_5": conc,
        }

    elif args.phase == "speculative":
        print(f"Test 18: Speculative Decoding — WITH DRAFT MODEL (k=5)")
        print(f"{'='*80}\n")

        single_results = {}
        for pt in ["factual", "creative", "code"]:
            print(f"\n  Prompt type: {pt}")
            single_results[pt] = run_prompt_type_test(pt)

        print(f"\n  Concurrent test (N=5):")
        conc = run_concurrent_test(5)
        if "error" not in conc:
            print(f"    OK={conc['successes']} | sys_tok/s={conc['system_tok_s']} | "
                  f"per_req={conc['per_request_tok_s_avg']}")

        existing["speculative"] = {
            "draft_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "num_speculative_tokens": 5,
            "single_request": single_results,
            "concurrent_5": conc,
        }

    elif args.phase == "sweep":
        k = args.num_spec_tokens
        print(f"Test 19: Speculative Decoding — k={k}")
        print(f"{'='*80}\n")

        result = run_prompt_type_test("factual")

        sweep = existing.get("sweep", [])
        sweep = [s for s in sweep if s.get("num_speculative_tokens") != k]
        sweep.append({"num_speculative_tokens": k, **result})
        sweep.sort(key=lambda s: s["num_speculative_tokens"])
        existing["sweep"] = sweep

    with open(output_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
