"""
Test 26: Soak Test (30+ min sustained)

Runs sustained load for 30 minutes to detect memory leaks, performance
degradation, CUDA OOM events, or stability issues that don't appear in
short tests.

Think of it like a burn-in test for new instances in an ASG: you run
steady traffic for an extended period before marking the instance healthy.
If metrics drift upward over time, something is leaking.

Usage: python3 stage2_test26_soak_test.py --host http://<ip>:8000
       python3 stage2_test26_soak_test.py --host http://<ip>:8000 --duration 15
"""

import json
import time
import threading
import argparse
import requests
import subprocess

HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
SSH_HOST = ""
SSH_KEY = ""

# Steady 4 RPS — well below saturation based on Test 25 findings
TARGET_RPS = 4
CHECKPOINT_INTERVAL_SEC = 60  # Record stats every minute

PROMPT = ("Describe the architecture of a distributed message queue. "
          "Cover producers, consumers, partitioning, and replication.")


def send_request_timed(results: list, lock: threading.Lock):
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
    request_start = time.perf_counter()

    try:
        resp = requests.post(url, json=payload,
                             headers={"Content-Type": "application/json"},
                             stream=True, timeout=(30, 120))
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
        result = {
            "error": error or "no tokens received",
            "total_sec": round(request_end - request_start, 3),
            "ts": time.time(),
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
            "ts": time.time(),
        }

    with lock:
        results.append(result)


def get_gpu_memory(ssh_host: str, ssh_key: str) -> int:
    """Query GPU memory usage via SSH."""
    if not ssh_host or not ssh_key:
        return -1
    try:
        cmd = ["ssh", "-i", ssh_key, "-o", "StrictHostKeyChecking=no",
               "-o", "ConnectTimeout=5", ssh_host,
               "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return int(result.stdout.strip())
    except Exception:
        return -1


def percentile(sorted_vals: list, p: float) -> float:
    """Compute percentile from sorted list."""
    if not sorted_vals:
        return 0
    idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
    return sorted_vals[idx]


def main():
    global HOST, SSH_HOST, SSH_KEY
    parser = argparse.ArgumentParser(description="Test 26: Soak Test")
    parser.add_argument("--host", required=True)
    parser.add_argument("--duration", type=int, default=30, help="Duration in minutes")
    parser.add_argument("--ssh-host", default="ubuntu@44.201.96.75")
    parser.add_argument("--ssh-key", default="/Users/mrefaat/.ssh/vllm-lab-key-2.pem")
    args = parser.parse_args()
    HOST = args.host.rstrip("/")
    SSH_HOST = args.ssh_host
    SSH_KEY = args.ssh_key
    duration_min = args.duration

    print(f"Test 26: Soak Test ({duration_min} minutes sustained)")
    print(f"Target: {HOST} | Model: {MODEL}")
    print(f"Target RPS: {TARGET_RPS} | Checkpoint interval: {CHECKPOINT_INTERVAL_SEC}s")
    print(f"{'='*120}\n")

    all_results = []
    lock = threading.Lock()
    checkpoints = []
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    test_start = time.perf_counter()
    total_duration_sec = duration_min * 60
    interval = 1.0 / TARGET_RPS
    total_requests = TARGET_RPS * total_duration_sec

    # Header
    print(f"{'Min':>4}  {'OK':>5}  {'Fail':>5}  "
          f"{'TTFT p50':>9}  {'TTFT p99':>9}  {'Total p50':>10}  {'Total p99':>10}  "
          f"{'Sys tok/s':>10}  {'GPU MB':>8}")
    print("-" * 100)

    # Track which results belong to which checkpoint window
    checkpoint_start_idx = 0
    last_checkpoint = time.perf_counter()
    threads = []

    for i in range(total_requests):
        # Schedule request
        target_time = test_start + i * interval
        now = time.perf_counter()
        if target_time > now:
            time.sleep(target_time - now)

        t = threading.Thread(target=send_request_timed,
                           args=(all_results, lock))
        t.start()
        threads.append(t)

        # Checkpoint every minute
        elapsed = time.perf_counter() - last_checkpoint
        if elapsed >= CHECKPOINT_INTERVAL_SEC:
            # Wait for outstanding threads from this window to complete
            time.sleep(3)

            with lock:
                window_results = all_results[checkpoint_start_idx:]
                checkpoint_start_idx = len(all_results)

            successes = [r for r in window_results if "error" not in r]
            failures = [r for r in window_results if "error" in r]
            minute = int((time.perf_counter() - test_start) / 60)
            gpu_mem = get_gpu_memory(SSH_HOST, SSH_KEY)

            checkpoint = {
                "minute": minute,
                "requests_this_window": len(window_results),
                "successes": len(successes),
                "failures": len(failures),
                "gpu_memory_used_mb": gpu_mem,
            }

            if successes:
                ttfts = sorted([r["ttft_sec"] for r in successes])
                totals = sorted([r["total_sec"] for r in successes])
                total_tokens = sum(r["tokens"] for r in successes)

                checkpoint.update({
                    "ttft_p50": round(percentile(ttfts, 50), 4),
                    "ttft_p99": round(percentile(ttfts, 99), 4),
                    "total_p50": round(percentile(totals, 50), 3),
                    "total_p99": round(percentile(totals, 99), 3),
                    "system_tok_s": round(total_tokens / CHECKPOINT_INTERVAL_SEC, 1),
                })

                print(f"{minute:>4}  {len(successes):>5}  {len(failures):>5}  "
                      f"{checkpoint['ttft_p50']:>9.4f}  {checkpoint['ttft_p99']:>9.4f}  "
                      f"{checkpoint['total_p50']:>10.3f}  {checkpoint['total_p99']:>10.3f}  "
                      f"{checkpoint['system_tok_s']:>10.1f}  "
                      f"{gpu_mem if gpu_mem > 0 else 'N/A':>8}")
            else:
                print(f"{minute:>4}  {0:>5}  {len(failures):>5}  "
                      f"{'FAIL':>9}")

            checkpoints.append(checkpoint)
            last_checkpoint = time.perf_counter()

    # Wait for remaining threads
    for t in threads:
        t.join(timeout=120)

    test_duration = time.perf_counter() - test_start

    # Final analysis
    print(f"\n{'='*120}")
    print("ANALYSIS")
    print(f"{'='*120}\n")

    total_successes = sum(1 for r in all_results if "error" not in r)
    total_failures = sum(1 for r in all_results if "error" in r)

    print(f"Total: {total_successes + total_failures} requests | "
          f"{total_successes} OK | {total_failures} failures | "
          f"Duration: {test_duration/60:.1f} min")

    # Degradation check: first 5 min vs last 5 min
    early_checkpoints = [c for c in checkpoints if c["minute"] <= 5 and "ttft_p99" in c]
    late_checkpoints = [c for c in checkpoints if c["minute"] >= duration_min - 5 and "ttft_p99" in c]

    degradation = {}
    if early_checkpoints and late_checkpoints:
        early_p99 = sum(c["ttft_p99"] for c in early_checkpoints) / len(early_checkpoints)
        late_p99 = sum(c["ttft_p99"] for c in late_checkpoints) / len(late_checkpoints)
        drift_pct = (late_p99 - early_p99) / early_p99 * 100

        # Memory leak check
        early_mem = [c["gpu_memory_used_mb"] for c in early_checkpoints if c["gpu_memory_used_mb"] > 0]
        late_mem = [c["gpu_memory_used_mb"] for c in late_checkpoints if c["gpu_memory_used_mb"] > 0]

        degradation = {
            "first_5min_ttft_p99": round(early_p99, 4),
            "last_5min_ttft_p99": round(late_p99, 4),
            "drift_pct": round(drift_pct, 1),
            "memory_leak_detected": False,
        }

        if early_mem and late_mem:
            mem_drift = sum(late_mem) / len(late_mem) - sum(early_mem) / len(early_mem)
            degradation["gpu_memory_drift_mb"] = round(mem_drift, 0)
            degradation["memory_leak_detected"] = mem_drift > 100

        print(f"First 5min TTFT p99: {early_p99:.4f}s")
        print(f"Last 5min TTFT p99:  {late_p99:.4f}s")
        print(f"Drift: {drift_pct:+.1f}%")
        if abs(drift_pct) < 10:
            print("→ No significant performance degradation detected")
        else:
            print(f"→ Performance drift detected ({drift_pct:+.1f}%)")

    # Save
    output = {
        "duration_minutes": duration_min,
        "target_rps": TARGET_RPS,
        "total_requests": total_successes + total_failures,
        "wall_start": wall_start,
        "wall_end": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "successes": total_successes,
        "failures": total_failures,
        "checkpoints": checkpoints,
        "degradation_check": degradation,
    }

    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_test26_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
