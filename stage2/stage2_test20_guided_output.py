"""
Test 20: Guided/Structured Output (JSON Schema)

Compares three modes:
  1. Unguided: prompt says "respond in JSON" but no enforcement
  2. JSON mode: response_format={"type": "json_object"}
  3. Strict schema: response_format with full JSON schema

Guided decoding constrains token sampling to only produce valid JSON matching
the schema. This is like schema validation at the API Gateway — guaranteeing
every response matches the contract, eliminating retry logic and parse errors.

Usage: python3 stage2_test20_guided_output.py --host http://<ip>:8000
"""

import json
import time
import threading
import argparse
import requests

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# JSON schema for structured output
PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["name", "age", "skills", "summary"],
}

# Same prompt for all modes — only the enforcement mechanism differs
BASE_PROMPT = "Generate a profile for a senior software engineer who specializes in distributed systems."


def send_chat_request(mode: str, request_id: int = 0) -> dict:
    """Send one chat completion with the specified output mode.

    mode: 'unguided', 'json_mode', or 'strict_schema'
    """
    url = f"{HOST}/v1/chat/completions"

    messages = [{"role": "user", "content": BASE_PROMPT}]

    if mode == "unguided":
        # Just ask nicely for JSON — no enforcement
        messages[0]["content"] += " Respond in valid JSON format with keys: name, age, skills (array), summary."

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 300,
        "stream": True,
        "temperature": 0.7,
    }

    # Add response format based on mode
    if mode == "json_mode":
        payload["response_format"] = {"type": "json_object"}
        messages[0]["content"] += " Respond in JSON with keys: name, age, skills (array), summary."
    elif mode == "strict_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "person_profile",
                "strict": True,
                "schema": PERSON_SCHEMA,
            },
        }

    token_times = []
    tokens = []
    error = None
    request_start = time.perf_counter()

    try:
        resp = requests.post(url, json=payload,
                             headers={"Content-Type": "application/json"},
                             stream=True, timeout=(60, 300))
        # Check for HTTP errors
        if resp.status_code != 200:
            error_body = resp.text
            return {
                "request_id": request_id,
                "error": f"HTTP {resp.status_code}: {error_body[:200]}",
                "total_sec": round(time.perf_counter() - request_start, 3),
            }

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
        return {
            "request_id": request_id,
            "error": error or "no tokens received",
            "total_sec": round(request_end - request_start, 3),
        }

    ttft = token_times[0] - request_start
    total = request_end - request_start
    decode_time = request_end - token_times[0]
    tok_s = len(tokens) / decode_time if decode_time > 0 else 0
    full_text = "".join(tokens)

    # Validate the output
    is_valid_json = False
    matches_schema = False
    try:
        parsed = json.loads(full_text)
        is_valid_json = True
        # Check if it has the required keys
        required = {"name", "age", "skills", "summary"}
        if isinstance(parsed, dict) and required.issubset(parsed.keys()):
            matches_schema = True
    except (json.JSONDecodeError, TypeError):
        pass

    return {
        "request_id": request_id,
        "ttft_sec": round(ttft, 4),
        "total_sec": round(total, 3),
        "tokens_out": len(tokens),
        "tok_s": round(tok_s, 1),
        "is_valid_json": is_valid_json,
        "matches_schema": matches_schema,
        "output_preview": full_text[:200],
    }


def run_serial_test(mode: str, n: int = 20) -> dict:
    """Run n serial requests in the given mode."""
    results = []
    for i in range(n):
        if i > 0:
            time.sleep(0.5)
        r = send_chat_request(mode, request_id=i)
        results.append(r)
        if "error" in r:
            print(f"    [{mode}] Req {i+1}: ERROR - {r['error'][:80]}")
        else:
            valid = "JSON-OK" if r["is_valid_json"] else "INVALID"
            schema = "SCHEMA-OK" if r["matches_schema"] else "NO-SCHEMA"
            print(f"    [{mode}] Req {i+1}: {r['tok_s']:>5.1f} tok/s | "
                  f"TTFT={r['ttft_sec']:.4f}s | tokens={r['tokens_out']} | "
                  f"{valid} {schema}")

    successes = [r for r in results if "error" not in r]
    if not successes:
        return {"mode": mode, "error": "all requests failed",
                "failures": len(results), "successes": 0}

    valid_json = sum(1 for r in successes if r["is_valid_json"])
    schema_match = sum(1 for r in successes if r["matches_schema"])

    return {
        "mode": mode,
        "valid_json_count": valid_json,
        "schema_match_count": schema_match,
        "ttft_avg": round(sum(r["ttft_sec"] for r in successes) / len(successes), 4),
        "total_avg": round(sum(r["total_sec"] for r in successes) / len(successes), 3),
        "tok_s_avg": round(sum(r["tok_s"] for r in successes) / len(successes), 1),
        "avg_tokens_generated": round(sum(r["tokens_out"] for r in successes) / len(successes), 1),
        "wall_start": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "successes": len(successes),
        "failures": len(results) - len(successes),
    }


def run_concurrent_test(mode: str, n: int = 10) -> dict:
    """Run n concurrent requests in the given mode."""
    results = [None] * n
    threads = []
    global_start = time.perf_counter()
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    def worker(idx):
        results[idx] = send_chat_request(mode, request_id=idx)

    for i in range(n):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    wall_duration = time.perf_counter() - global_start
    successes = [r for r in results if r and "error" not in r]
    failures = [r for r in results if r and "error" in r]

    if not successes:
        return {"mode": mode, "error": "all concurrent requests failed",
                "successes": 0, "failures": len(failures)}

    valid_json = sum(1 for r in successes if r["is_valid_json"])
    schema_match = sum(1 for r in successes if r["matches_schema"])
    ttfts = sorted([r["ttft_sec"] for r in successes])
    totals = sorted([r["total_sec"] for r in successes])
    tok_rates = [r["tok_s"] for r in successes]
    total_tokens = sum(r["tokens_out"] for r in successes)

    return {
        "mode": mode,
        "valid_json_count": valid_json,
        "schema_match_count": schema_match,
        "ttft_p50": round(ttfts[len(ttfts) // 2], 4),
        "ttft_p99": round(ttfts[int(len(ttfts) * 0.99)], 4),
        "total_p50": round(totals[len(totals) // 2], 3),
        "total_p99": round(totals[int(len(totals) * 0.99)], 3),
        "system_tok_s": round(total_tokens / wall_duration, 1),
        "wall_start": wall_start,
        "successes": len(successes),
        "failures": len(failures),
    }


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Test 20: Guided/Structured Output")
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    print(f"Test 20: Guided/Structured Output (JSON Schema)")
    print(f"Target: {HOST} | Model: {MODEL}")
    print(f"{'='*100}\n")

    all_results = {"serial": {}, "concurrent": {}}

    for mode in ["unguided", "json_mode", "strict_schema"]:
        print(f"\n{'='*60}")
        print(f"MODE: {mode}")
        print(f"{'='*60}")

        # Serial test (20 requests)
        print(f"\n  Serial (20 requests):")
        serial = run_serial_test(mode, n=20)
        all_results["serial"][mode] = serial
        print(f"\n  Summary: valid_json={serial.get('valid_json_count', 'N/A')}/20 | "
              f"schema_match={serial.get('schema_match_count', 'N/A')}/20 | "
              f"tok/s={serial.get('tok_s_avg', 'N/A')} | "
              f"TTFT={serial.get('ttft_avg', 'N/A')}s")

        time.sleep(3)

        # Concurrent test (10 requests)
        print(f"\n  Concurrent (10 requests):")
        concurrent = run_concurrent_test(mode, n=10)
        all_results["concurrent"][mode] = concurrent
        print(f"  Summary: valid_json={concurrent.get('valid_json_count', 'N/A')}/10 | "
              f"schema_match={concurrent.get('schema_match_count', 'N/A')}/10 | "
              f"sys_tok/s={concurrent.get('system_tok_s', 'N/A')}")

        time.sleep(3)

    # Analysis
    print(f"\n{'='*100}")
    print("ANALYSIS")
    print(f"{'='*100}\n")

    print(f"{'Mode':<20} {'Valid JSON':>12} {'Schema OK':>12} {'Tok/s':>8} {'TTFT':>8}")
    print("-" * 65)
    for mode in ["unguided", "json_mode", "strict_schema"]:
        s = all_results["serial"].get(mode, {})
        if "error" in s:
            print(f"{mode:<20} {'ERROR':>12}")
        else:
            print(f"{mode:<20} {s.get('valid_json_count', '?'):>8}/20   "
                  f"{s.get('schema_match_count', '?'):>8}/20   "
                  f"{s.get('tok_s_avg', 0):>7.1f} {s.get('ttft_avg', 0):>7.4f}s")

    # Save
    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_test20_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
