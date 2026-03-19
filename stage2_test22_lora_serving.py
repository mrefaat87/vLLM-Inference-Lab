"""
Test 22: LoRA Adapter Serving — Multi-Tenant

Tests serving multiple LoRA adapters from a single base model.
Different requests target different adapters via the model field.

Note: LoRA with AWQ-quantized base models may not be supported.
This script detects and reports compatibility issues.

Usage: python3 stage2_test22_lora_serving.py --host http://<ip>:8000
"""

import json
import time
import threading
import argparse
import requests

HOST = ""
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"


def check_models():
    """List available models to verify LoRA adapters loaded."""
    try:
        resp = requests.get(f"{HOST}/v1/models", timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def send_request(model_name: str, prompt: str, max_tokens: int = 100) -> dict:
    url = f"{HOST}/v1/completions"
    payload = {
        "model": model_name,
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
        if resp.status_code != 200:
            return {"error": f"http_{resp.status_code}: {resp.text[:200]}",
                    "total_sec": round(time.perf_counter() - request_start, 3)}
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
        return {"error": str(e),
                "total_sec": round(time.perf_counter() - request_start, 3)}

    if not token_times:
        return {"error": "no tokens",
                "total_sec": round(time.perf_counter() - request_start, 3)}

    ttft = token_times[0] - request_start
    total = time.perf_counter() - request_start
    decode_time = time.perf_counter() - token_times[0]
    tok_s = len(tokens) / decode_time if decode_time > 0 else 0

    return {
        "model": model_name,
        "ttft_sec": round(ttft, 4),
        "total_sec": round(total, 3),
        "tokens": len(tokens),
        "tok_s": round(tok_s, 1),
    }


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Test 22: LoRA Adapter Serving")
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    print(f"Test 22: LoRA Adapter Serving")
    print(f"Target: {HOST}")
    print(f"{'='*80}\n")

    # Step 1: List models
    print("Step 1: Checking available models...")
    models = check_models()
    if "error" in models:
        print(f"  ERROR: {models['error']}")
        return

    model_ids = [m["id"] for m in models.get("data", [])]
    print(f"  Available models: {model_ids}")

    # Identify LoRA adapters (anything that's not the base model)
    lora_models = [m for m in model_ids if m != BASE_MODEL]
    print(f"  Base model: {BASE_MODEL}")
    print(f"  LoRA adapters: {lora_models if lora_models else 'NONE'}")

    if not lora_models:
        print("\n  No LoRA adapters found. The --enable-lora flag may have failed,")
        print("  or LoRA serving is not compatible with AWQ quantized models.")
        # Still test the base model for comparison
        lora_models = []

    # Step 2: Serial requests to each model
    print(f"\nStep 2: Serial requests (5 per model)...")
    all_models = [BASE_MODEL] + lora_models
    serial_results = {}

    for model_name in all_models:
        print(f"\n  Model: {model_name}")
        model_results = []
        for i in range(5):
            if i > 0:
                time.sleep(1)
            r = send_request(model_name, "Explain what a hash table is in 2 sentences.")
            if "error" not in r:
                model_results.append(r)
                print(f"    {i+1}/5: {r['tok_s']:.1f} tok/s | TTFT={r['ttft_sec']:.4f}s")
            else:
                print(f"    {i+1}/5: ERROR - {r['error'][:60]}")

        if model_results:
            serial_results[model_name] = {
                "ttft_avg": round(sum(r["ttft_sec"] for r in model_results) / len(model_results), 4),
                "tok_s_avg": round(sum(r["tok_s"] for r in model_results) / len(model_results), 1),
                "total_avg": round(sum(r["total_sec"] for r in model_results) / len(model_results), 3),
                "trials": len(model_results),
            }

    # Step 3: Mixed concurrent requests
    if lora_models:
        print(f"\nStep 3: Mixed concurrent requests (10 total)...")
        results = [None] * 10
        threads = []

        def worker(idx):
            # Mix: 5 base, rest split among LoRA adapters
            if idx < 5:
                model = BASE_MODEL
            else:
                model = lora_models[(idx - 5) % len(lora_models)]
            results[idx] = send_request(model, "What is cloud computing? Answer briefly.")

        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        successes = [r for r in results if r and "error" not in r]
        print(f"  OK: {len(successes)}/10")

        if successes:
            # Group by model
            by_model = {}
            for r in successes:
                m = r["model"]
                by_model.setdefault(m, []).append(r["tok_s"])

            concurrent_results = {}
            for m, rates in by_model.items():
                concurrent_results[m] = round(sum(rates) / len(rates), 1)
                print(f"    {m}: avg tok/s = {concurrent_results[m]}")

    # Save
    output = {
        "adapters_loaded": model_ids,
        "lora_adapters": lora_models,
        "serial_results": serial_results,
        "lora_compatible_with_awq": len(lora_models) > 0,
    }

    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_test22_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
