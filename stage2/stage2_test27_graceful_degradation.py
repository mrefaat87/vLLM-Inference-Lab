"""
Test 27: Graceful Degradation Under Overload

Ramps load from comfortable to extreme and monitors failure modes:
  Stage 1 (0-60s):   4 RPS (comfortable)
  Stage 2 (60-120s): 10 RPS (moderate)
  Stage 3 (120-180s): 20 RPS (heavy — likely over capacity)
  Stage 4 (180-240s): 40 RPS (overload)
  Stage 5 (240-300s): 4 RPS (recovery)

Key question: does the system recover in Stage 5 after being overloaded?
Like an ASG target tracking policy under a traffic spike — does the system
gracefully shed load and bounce back, or does it get "stuck"?

Usage: python3 stage2_test27_graceful_degradation.py --host http://<ip>:8000
"""

import json
import time
import threading
import argparse
import requests

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

PROMPT = ("Explain load balancing algorithms: round-robin, least connections, "
          "and consistent hashing. When would you use each?")

# Stage definitions: (name, rps, duration_sec)
STAGES = [
    ("comfortable", 4, 60),
    ("moderate", 8, 60),
    ("heavy", 14, 60),
    ("overload", 20, 60),
    ("recovery", 4, 60),
]


def send_request_timed(results: list, lock: threading.Lock, global_start: float):
    """Send one request, record metrics."""
    url = f"{HOST}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 100,
        "stream": True,
        "temperature": 0.7,
    }

    token_times = []
    tokens = []
    error = None
    error_type = None
    request_start = time.perf_counter()

    try:
        resp = requests.post(url, json=payload,
                             headers={"Content-Type": "application/json"},
                             stream=True, timeout=(10, 60))
        if resp.status_code == 429:
            error = "rate_limited"
            error_type = "429"
        elif resp.status_code >= 500:
            error = f"server_error_{resp.status_code}"
            error_type = str(resp.status_code)
        elif resp.status_code != 200:
            error = f"http_{resp.status_code}"
            error_type = str(resp.status_code)
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
    except requests.exceptions.Timeout:
        error = "timeout"
        error_type = "timeout"
    except requests.exceptions.ConnectionError:
        error = "connection_refused"
        error_type = "connection_refused"
    except Exception as e:
        error = str(e)
        error_type = "other"

    request_end = time.perf_counter()

    if error or not token_times:
        result = {
            "error": error or "no tokens received",
            "error_type": error_type or "no_tokens",
            "total_sec": round(request_end - request_start, 3),
            "abs_start": round(request_start - global_start, 4),
        }
    else:
        ttft = token_times[0] - request_start
        total = request_end - request_start
        decode_time = request_end - token_times[0]
        tok_s = len(tokens) / decode_time if decode_time > 0 else 0
        result = {
            "ttft_sec": round(ttft, 4),
            "total_sec": round(total, 3),
            "tokens": len(tokens),
            "tok_s": round(tok_s, 1),
            "abs_start": round(request_start - global_start, 4),
        }

    with lock:
        results.append(result)


def percentile(sorted_vals: list, p: float) -> float:
    """Compute percentile from sorted list."""
    if not sorted_vals:
        return 0
    idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
    return sorted_vals[idx]


def main():
    global HOST
    parser = argparse.ArgumentParser(description="Test 27: Graceful Degradation Under Overload")
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    print(f"Test 27: Graceful Degradation Under Overload")
    print(f"Target: {HOST} | Model: {MODEL}")
    print(f"Stages: {[(s[0], f'{s[1]} RPS', f'{s[2]}s') for s in STAGES]}")
    print(f"{'='*130}\n")

    all_results = {"stages": []}
    global_start = time.perf_counter()

    for stage_idx, (name, rps, duration) in enumerate(STAGES):
        stage_results = []
        lock = threading.Lock()
        stage_start = time.perf_counter()
        wall_start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        interval = 1.0 / rps
        total_requests = rps * duration
        threads = []

        print(f"Stage {stage_idx+1}: {name} ({rps} RPS for {duration}s, "
              f"{total_requests} requests)")

        for i in range(total_requests):
            target_time = stage_start + i * interval
            now = time.perf_counter()
            if target_time > now:
                time.sleep(target_time - now)

            t = threading.Thread(target=send_request_timed,
                               args=(stage_results, lock, global_start))
            t.start()
            threads.append(t)

        # Wait for all stage threads to complete
        for t in threads:
            t.join(timeout=180)

        stage_duration = time.perf_counter() - stage_start

        # Analyze stage results
        successes = [r for r in stage_results if "error" not in r]
        failures = [r for r in stage_results if "error" in r]

        # Count error types
        error_types = {}
        for f in failures:
            et = f.get("error_type", "unknown")
            error_types[et] = error_types.get(et, 0) + 1

        stage_data = {
            "stage": stage_idx + 1,
            "name": name,
            "rps": rps,
            "duration_sec": duration,
            "wall_start": wall_start,
            "total_requests": total_requests,
            "successes": len(successes),
            "failures": len(failures),
            "error_types": error_types,
            "wall_duration_sec": round(stage_duration, 1),
        }

        if successes:
            ttfts = sorted([r["ttft_sec"] for r in successes])
            totals = sorted([r["total_sec"] for r in successes])
            total_tokens = sum(r["tokens"] for r in successes)

            stage_data.update({
                "ttft_p50": round(percentile(ttfts, 50), 4),
                "ttft_p99": round(percentile(ttfts, 99), 4),
                "ttft_max": round(max(ttfts), 4),
                "total_p50": round(percentile(totals, 50), 3),
                "total_p99": round(percentile(totals, 99), 3),
                "total_max": round(max(totals), 3),
                "system_tok_s": round(total_tokens / stage_duration, 1),
            })

            success_rate = len(successes) / total_requests * 100
            print(f"  OK={len(successes)}/{total_requests} ({success_rate:.0f}%) | "
                  f"TTFT p50={stage_data['ttft_p50']:.4f} p99={stage_data['ttft_p99']:.4f} | "
                  f"Total p50={stage_data['total_p50']:.3f} p99={stage_data['total_p99']:.3f} | "
                  f"Sys={stage_data['system_tok_s']:.0f} tok/s")
            if error_types:
                print(f"  Errors: {error_types}")
        else:
            print(f"  ALL FAILED ({len(failures)} failures)")
            if error_types:
                print(f"  Errors: {error_types}")

        all_results["stages"].append(stage_data)

    # Analysis
    print(f"\n{'='*130}")
    print("ANALYSIS")
    print(f"{'='*130}\n")

    stages_with_data = [s for s in all_results["stages"] if "ttft_p50" in s]
    if len(stages_with_data) >= 2:
        comfortable = next((s for s in stages_with_data if s["name"] == "comfortable"), None)
        recovery = next((s for s in stages_with_data if s["name"] == "recovery"), None)

        if comfortable and recovery:
            ttft_diff = abs(recovery["ttft_p50"] - comfortable["ttft_p50"])
            print(f"Recovery check:")
            print(f"  Comfortable: TTFT p50={comfortable['ttft_p50']:.4f}s, "
                  f"p99={comfortable['ttft_p99']:.4f}s")
            print(f"  Recovery:    TTFT p50={recovery['ttft_p50']:.4f}s, "
                  f"p99={recovery['ttft_p99']:.4f}s")
            if ttft_diff < 0.05:
                print(f"  → System recovered to baseline within 1 minute!")
            else:
                print(f"  → Recovery incomplete — TTFT still {ttft_diff:.3f}s above baseline")

    # Save
    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_test27_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
