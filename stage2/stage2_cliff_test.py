"""
Stage 2 — Cliff Test: Push vLLM to its KV cache breaking point.

Uses a long prompt (~500 input tokens) + max_tokens=1500 to fill ~2000 tokens
per request, approaching the 2048 max_model_len. At 127,840 total KV cache tokens,
that means ~62 concurrent requests should exhaust the cache.

Tests concurrency levels: [20, 30, 40, 50, 55, 60, 65, 70]
After each round, captures vLLM server logs for Running/Waiting/KV cache stats.

Usage: python3 stage2_cliff_test.py --host http://<ip>:8000 --ssh-host ubuntu@<ip>
"""

import json
import time
import threading
import argparse
import requests
import subprocess
import sys

HOST = ""
SSH_HOST = ""
SSH_KEY = "~/.ssh/vllm-lab-key-2.pem"
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# Long prompt (~500 tokens) that encourages verbose output
PROMPT = """You are a world-renowned historian and storytelling expert. I need you to write an extremely detailed, comprehensive essay about the history of computing, from the earliest mechanical calculators to modern artificial intelligence. Your essay must cover ALL of the following topics in great depth:

1. The Antikythera mechanism and early mechanical computing devices from ancient civilizations
2. Charles Babbage's Analytical Engine and Ada Lovelace's contributions to programming theory
3. Alan Turing's theoretical foundations - the Turing machine, computability theory, and the Enigma codebreaking effort during World War II
4. The development of ENIAC, UNIVAC, and other first-generation electronic computers in the 1940s and 1950s
5. The invention of the transistor at Bell Labs and how it revolutionized computer hardware miniaturization
6. The integrated circuit era - Jack Kilby, Robert Noyce, and the birth of Silicon Valley
7. The personal computer revolution - Altair 8800, Apple I and II, IBM PC, and the home computing explosion
8. The rise of the Internet from ARPANET through the World Wide Web, dot-com boom, and Web 2.0
9. The mobile computing revolution - smartphones, tablets, and the app economy
10. Modern artificial intelligence - deep learning breakthroughs, large language models, and the current AI landscape

For each topic, provide specific dates, names of key people involved, technical details about the innovations, and explain WHY each development was historically significant. Use vivid storytelling to bring each era to life. Include surprising facts and lesser-known details that most people would not know about.

Begin your essay now with an engaging introduction that sets the stage for this remarkable journey through computing history:"""

CONCURRENCY_LEVELS = [20, 30, 40, 50, 55, 60, 65, 70]


def send_request(request_id: int, results: list, global_start: float):
    """Send one streaming request, measure TTFT/total/tok_s."""
    url = f"{HOST}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 1500,
        "stream": True,
        "temperature": 0.7,
    }

    token_times = []
    tokens = []
    error = None
    request_start = time.perf_counter()

    try:
        resp = requests.post(
            url, json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=(60, 600),  # 60s connect, 600s read
        )

        if resp.status_code != 200:
            error = f"HTTP {resp.status_code}: {resp.text[:200]}"
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
            "error": error or "no tokens received",
            "total_sec": round(request_end - request_start, 3),
        }
        return

    ttft = token_times[0] - request_start
    total = request_end - request_start
    tok_s = len(tokens) / (request_end - token_times[0]) if len(token_times) > 1 else 0

    results[request_id] = {
        "request_id": request_id,
        "ttft_sec": round(ttft, 4),
        "total_sec": round(total, 3),
        "tokens": len(tokens),
        "tokens_per_sec": round(tok_s, 1),
        "abs_start": round(request_start - global_start, 4),
        "abs_first_token": round(token_times[0] - global_start, 4),
        "abs_end": round(request_end - global_start, 4),
    }


def fetch_server_logs(since_seconds=30):
    """SSH into the server and grab recent vLLM logs."""
    try:
        cmd = [
            "ssh", "-i", SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            SSH_HOST,
            f"sudo docker logs vllm-server --since {since_seconds}s 2>&1 | grep -E '(Running:|Waiting:|KV cache|preempt|Swapped|error|Error|OOM)' | tail -30"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return result.stdout.strip()
    except Exception as e:
        return f"Failed to fetch logs: {e}"


def run_at_concurrency(n: int) -> dict:
    """Fire n concurrent requests and return aggregated metrics."""
    results = [None] * n
    threads = []
    global_start = time.perf_counter()
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    wall_start_unix = time.time()

    for i in range(n):
        t = threading.Thread(target=send_request, args=(i, results, global_start))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    wall_end = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    elapsed = time.time() - wall_start_unix

    # Fetch server logs covering this round (add buffer)
    log_window = int(elapsed) + 15
    server_logs = fetch_server_logs(since_seconds=log_window)

    # Separate successes from failures
    successes = [r for r in results if r and "error" not in r]
    failures = [r for r in results if r and "error" in r]
    missing = n - len(successes) - len(failures)

    if not successes:
        return {
            "concurrency": n,
            "wall_start": wall_start,
            "wall_end": wall_end,
            "successes": 0,
            "failures": len(failures) + missing,
            "error_sample": failures[0]["error"] if failures else "all threads returned None",
            "server_logs": server_logs,
        }

    ttfts = [r["ttft_sec"] for r in successes]
    totals = [r["total_sec"] for r in successes]
    tok_rates = [r["tokens_per_sec"] for r in successes]
    token_counts = [r["tokens"] for r in successes]
    system_tok_s = sum(tok_rates)

    return {
        "concurrency": n,
        "wall_start": wall_start,
        "wall_end": wall_end,
        "successes": len(successes),
        "failures": len(failures) + missing,
        "ttft_p50": round(sorted(ttfts)[len(ttfts) // 2], 4),
        "ttft_p99": round(sorted(ttfts)[int(len(ttfts) * 0.99)], 4),
        "ttft_max": round(max(ttfts), 4),
        "total_p50": round(sorted(totals)[len(totals) // 2], 3),
        "total_p99": round(sorted(totals)[int(len(totals) * 0.99)], 3),
        "total_max": round(max(totals), 3),
        "per_request_tok_s_avg": round(sum(tok_rates) / len(tok_rates), 1),
        "per_request_tok_s_min": round(min(tok_rates), 1),
        "system_tok_s": round(system_tok_s, 1),
        "avg_tokens_generated": round(sum(token_counts) / len(token_counts), 1),
        "min_tokens_generated": min(token_counts),
        "max_tokens_generated": max(token_counts),
        "server_logs": server_logs,
    }


def main():
    global HOST, SSH_HOST, SSH_KEY
    parser = argparse.ArgumentParser(description="KV Cache Cliff Test for vLLM")
    parser.add_argument("--host", required=True, help="vLLM API endpoint")
    parser.add_argument("--ssh-host", required=True, help="SSH user@host for log capture")
    parser.add_argument("--ssh-key", default="~/.ssh/vllm-lab-key-2.pem", help="SSH key path")
    args = parser.parse_args()
    HOST = args.host.rstrip("/")
    SSH_HOST = args.ssh_host
    SSH_KEY = args.ssh_key

    print(f"KV Cache Cliff Test — {MODEL}")
    print(f"Target: {HOST}")
    print(f"SSH:    {SSH_HOST}")
    print(f"Prompt: ~500 input tokens + max_tokens=1500 = ~2000 total per request")
    print(f"KV cache: 127,840 tokens / 2048 per req = ~62 max concurrent")
    print(f"Testing levels: {CONCURRENCY_LEVELS}")
    print(f"{'='*130}\n")

    all_results = []

    # Header
    print(f"{'N':>4}  {'OK':>4}  {'Fail':>4}  {'TTFT p50':>9}  {'TTFT max':>9}  "
          f"{'Total p50':>10}  {'Total max':>10}  "
          f"{'Req tok/s':>10}  {'Sys tok/s':>10}  {'Avg toks':>9}  {'Window':>20}")
    print("-" * 130)

    for n in CONCURRENCY_LEVELS:
        # Let vLLM fully drain between rounds
        if all_results:
            print(f"\n  ... sleeping 10s to let vLLM drain ...\n")
            time.sleep(10)

        print(f"  >> Starting round N={n} at {time.strftime('%H:%M:%S', time.gmtime())} UTC")
        result = run_at_concurrency(n)
        all_results.append(result)

        if "error_sample" in result:
            print(f"{n:>4}  {result['successes']:>4}  {result['failures']:>4}  "
                  f"{'FAILED':>9}  {'':>9}  {'':>10}  {'':>10}  "
                  f"{'':>10}  {'':>10}  {'':>9}  {result['wall_start']}")
            print(f"      Error: {result['error_sample'][:120]}")
        else:
            print(f"{n:>4}  {result['successes']:>4}  {result['failures']:>4}  "
                  f"{result['ttft_p50']:>9.4f}  {result['ttft_max']:>9.4f}  "
                  f"{result['total_p50']:>10.3f}  {result['total_max']:>10.3f}  "
                  f"{result['per_request_tok_s_avg']:>10.1f}  "
                  f"{result['system_tok_s']:>10.1f}  "
                  f"{result['avg_tokens_generated']:>9.1f}  "
                  f"{result['wall_start']}")

        # Print server logs for this round
        if result.get("server_logs"):
            print(f"\n  --- vLLM Server Logs (N={n}) ---")
            for log_line in result["server_logs"].split("\n"):
                print(f"  | {log_line}")
            print(f"  --- end server logs ---\n")
        else:
            print(f"  (no relevant server logs captured for N={n})\n")

    # Summary
    print(f"\n{'='*130}")
    print("CLIFF TEST ANALYSIS")
    print(f"{'='*130}\n")

    successful = [r for r in all_results if "error_sample" not in r]
    failed_rounds = [r for r in all_results if "error_sample" in r]

    if successful:
        sys_toks = [(r["concurrency"], r["system_tok_s"]) for r in successful]
        peak = max(sys_toks, key=lambda x: x[1])
        print(f"Peak system throughput: {peak[1]} tok/s at N={peak[0]}")

        # Look for KV cache pressure signals
        for r in successful:
            if r.get("failures", 0) > 0:
                print(f"First partial failures at N={r['concurrency']}: "
                      f"{r['failures']} failures out of {r['concurrency']} requests")
                break

        # TTFT degradation
        if len(successful) >= 2:
            base_ttft = successful[0]["ttft_p50"]
            for r in successful[1:]:
                ratio = r["ttft_p50"] / base_ttft if base_ttft > 0 else 0
                if ratio > 3:
                    print(f"TTFT jumped {ratio:.1f}x at N={r['concurrency']} "
                          f"({r['ttft_p50']:.3f}s vs baseline {base_ttft:.3f}s)")
                    break

    if failed_rounds:
        print(f"\nFull failure rounds: {[r['concurrency'] for r in failed_rounds]}")

    # Check for preemption evidence in logs
    all_logs = "\n".join(r.get("server_logs", "") for r in all_results)
    if "preempt" in all_logs.lower():
        print("\n*** PREEMPTION DETECTED in server logs! ***")
    if "swap" in all_logs.lower():
        print("\n*** SWAPPING DETECTED in server logs! ***")

    # Save results
    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_cliff_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {output_file}")


if __name__ == "__main__":
    main()
