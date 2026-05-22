"""
Phase 4 — Throughput Diagnostic: Direct vLLM vs Full Pipeline

Isolates the 5x throughput gap between Stage 2 local vLLM (1131 tok/s at 60
concurrent) and Phase 4 EKS pipeline (~232 tok/s).

AWS analogy: when an ASG's target tracking policy shows high latency, step 1 is
"bypass the ALB and hit the instance directly."  If latency drops, the problem
is in the routing layer.  Same idea here — Test A bypasses the queue pipeline.

TEST DESIGN:
  Test A — Direct vLLM:  POST to /v1/completions (port-forwarded pod).
                          Bypasses RabbitMQ, worker, Redis, SSE gateway.
  Test B — Full Pipeline: POST /generate -> RabbitMQ -> worker -> Redis -> SSE.

Both tests: same prompt, same concurrency (60), same max_tokens (150).

PORT-FORWARD SETUP:
  kubectl port-forward pod/<vllm-pod-name> 8000:8000 -n inference-system
  kubectl port-forward svc/api-gateway 8080:8080 -n inference-system

Usage:
    python3 phase4/tests/throughput_diagnostic.py \\
        --vllm-host http://localhost:8000 \\
        --gateway-host http://localhost:8080 \\
        --concurrency 60 --max-tokens 150

    # Single test:  --test direct  or  --test pipeline
"""

import json
import time
import threading
import argparse
import statistics
import sys
import os
from datetime import datetime, timezone

import requests

# Medium complexity prompt — same as scaling_policy_comparison "medium".
# WHY this prompt: exercises real reasoning but stays short enough that 60
# concurrent requests fit in a single vLLM batch without KV cache eviction.
PROMPT = (
    "Explain the concept of auto scaling in cloud computing. "
    "Cover the key components including scaling policies, "
    "cooldown periods, health checks, and how metrics like "
    "CPU utilization and queue depth drive scaling decisions. "
    "Be concise."
)

# Must match the model deployed in EKS
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"


# ---------------------------------------------------------------------------
# Test A — Direct vLLM request sender
# WHY separate from pipeline: vLLM's /v1/completions SSE uses OpenAI format
# (choices[].text), not our custom {"token": ...} format.
# ---------------------------------------------------------------------------
def send_direct_request(request_id, vllm_host, max_tokens, results, run_start_time):
    """Send one request directly to vLLM — like hitting EC2 bypassing the ALB."""
    submit_time = time.perf_counter()
    submit_wall = time.time()
    token_count = 0
    first_token_time = None
    error = None

    try:
        # WHY stream=True: need TTFT, which requires seeing the first token arrive
        resp = requests.post(
            f"{vllm_host}/v1/completions",
            json={"model": MODEL_NAME, "prompt": PROMPT,
                  "max_tokens": max_tokens, "temperature": 0.7, "stream": True},
            headers={"Content-Type": "application/json"},
            stream=True, timeout=(10, 300),
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode()
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            # vLLM streams one token per SSE event in choices[0].text
            choices = data.get("choices", [])
            if choices and choices[0].get("text"):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += 1
    except Exception as e:
        error = str(e)

    end = time.perf_counter()
    ttft = (first_token_time - submit_time) if first_token_time else None
    gen_time = (end - first_token_time) if first_token_time else None

    results[request_id] = {
        "request_id": request_id, "error": error, "tokens": token_count,
        "ttft_sec": round(ttft, 4) if ttft else None,
        "total_sec": round(end - submit_time, 3),
        "per_request_tps": (round(token_count / gen_time, 1)
                            if gen_time and gen_time > 0 and token_count > 1 else 0.0),
        "submit_offset_sec": round(submit_wall - run_start_time, 2),
    }


# ---------------------------------------------------------------------------
# Test B — Full pipeline request sender
# WHY two-step (POST then GET): gateway enqueues to RabbitMQ and returns a
# stream_url.  Client GETs it to receive SSE tokens from Redis — full
# production path.
# ---------------------------------------------------------------------------
def send_pipeline_request(request_id, gateway_host, max_tokens, results, run_start_time):
    """Send one request through gateway -> queue -> worker -> SSE."""
    submit_time = time.perf_counter()
    submit_wall = time.time()
    token_count = 0
    first_token_time = None
    error = None

    try:
        # Step 1: submit — returns 202 with stream_url (like SQS receipt handle)
        submit_resp = requests.post(
            f"{gateway_host}/generate",
            json={"prompt": PROMPT, "max_tokens": max_tokens, "temperature": 0.7},
            headers={"Content-Type": "application/json"}, timeout=10,
        )
        if submit_resp.status_code != 202:
            raise RuntimeError(f"Expected 202, got {submit_resp.status_code}: {submit_resp.text[:200]}")

        stream_url = submit_resp.json()["stream_url"]

        # Step 2: SSE stream — tokens arrive via Redis pub/sub
        stream_resp = requests.get(
            f"{gateway_host}{stream_url}", stream=True,
            headers={"Accept": "text/event-stream"}, timeout=(10, 300),
        )
        if stream_resp.status_code != 200:
            raise RuntimeError(f"Stream {stream_resp.status_code}: {stream_resp.text[:200]}")

        for line in stream_resp.iter_lines():
            if not line:
                continue
            line_str = line.decode()
            if not line_str.startswith("data: "):
                continue
            try:
                data = json.loads(line_str[6:])
            except json.JSONDecodeError:
                continue
            if "token" in data:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += 1
            if data.get("status") == "complete":
                break
            if data.get("status") == "error":
                error = data.get("error", "unknown server error")
                break
    except Exception as e:
        error = str(e)

    end = time.perf_counter()
    ttft = (first_token_time - submit_time) if first_token_time else None
    gen_time = (end - first_token_time) if first_token_time else None

    results[request_id] = {
        "request_id": request_id, "error": error, "tokens": token_count,
        "ttft_sec": round(ttft, 4) if ttft else None,
        "total_sec": round(end - submit_time, 3),
        "per_request_tps": (round(token_count / gen_time, 1)
                            if gen_time and gen_time > 0 and token_count > 1 else 0.0),
        "submit_offset_sec": round(submit_wall - run_start_time, 2),
    }


# ---------------------------------------------------------------------------
# Test runner — fires N concurrent requests and collects results
# WHY threading (not asyncio): matches scaling_policy_comparison pattern.
# For 60 threads, overhead is negligible vs network RTT + GPU inference.
# ---------------------------------------------------------------------------
def run_test(name, concurrency, sender_fn, sender_kwargs):
    """Fire `concurrency` requests in parallel, return aggregate metrics."""
    print(f"\n  Running: {name} (concurrency={concurrency})")
    print(f"  Prompt: {len(PROMPT)} chars, max_tokens={sender_kwargs.get('max_tokens', '?')}")

    # WHY list not dict: each thread writes results[i], no lock needed
    results = [None] * concurrency
    threads = []
    run_start = time.time()
    wall_start = time.perf_counter()

    for i in range(concurrency):
        t = threading.Thread(
            target=sender_fn, args=(i,),
            kwargs={**sender_kwargs, "results": results, "run_start_time": run_start},
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=600)

    wall_time = time.perf_counter() - wall_start

    # Aggregate metrics
    completed = [r for r in results if r is not None]
    successes = [r for r in completed if r.get("error") is None and r.get("tokens", 0) > 0]
    errors = [r for r in completed if r.get("error") is not None or r.get("tokens", 0) == 0]
    total_tokens = sum(r["tokens"] for r in successes)

    # WHY wall clock (not sum of per-request times): measures total system
    # throughput under concurrent load — like total TPS through an ALB.
    aggregate_tps = round(total_tokens / wall_time, 1) if wall_time > 0 else 0.0

    ttfts = [r["ttft_sec"] for r in successes if r.get("ttft_sec") is not None]
    ttft_p50 = round(statistics.median(ttfts) * 1000, 1) if ttfts else None
    ttft_p99 = (round(sorted(ttfts)[int(len(ttfts) * 0.99)] * 1000, 1)
                if len(ttfts) >= 2 else ttft_p50)

    per_req_tps = [r["per_request_tps"] for r in successes if r.get("per_request_tps", 0) > 0]
    per_req_tps_median = round(statistics.median(per_req_tps), 1) if per_req_tps else 0.0

    print(f"  Completed: {len(successes)} success, {len(errors)} errors")
    print(f"  Total tokens: {total_tokens}, Wall time: {wall_time:.1f}s")
    print(f"  Aggregate throughput: {aggregate_tps} tok/s")
    if ttft_p50 is not None:
        print(f"  TTFT p50: {ttft_p50}ms, p99: {ttft_p99}ms")

    return {
        "test_name": name, "concurrency": concurrency,
        "total_requests": concurrency, "successes": len(successes),
        "errors": len(errors),
        "error_details": [r.get("error") for r in errors][:5],
        "total_tokens": total_tokens, "wall_time_sec": round(wall_time, 2),
        "aggregate_tps": aggregate_tps,
        "ttft_p50_ms": ttft_p50, "ttft_p99_ms": ttft_p99,
        "per_request_tps_median": per_req_tps_median,
        "per_request_results": [r for r in completed],
    }


# ---------------------------------------------------------------------------
# Hypothesis generator — auto-interprets the throughput ratio so you don't
# defer analysis to "I'll look at the JSON later"
# ---------------------------------------------------------------------------
def generate_hypothesis(direct_tps, pipeline_tps):
    """Generate a diagnosis based on the direct-to-pipeline throughput ratio."""
    if direct_tps == 0 or pipeline_tps == 0:
        return "Insufficient data — one or both tests produced no tokens."
    ratio = direct_tps / pipeline_tps
    if ratio < 1.2:
        return (f"Pipeline overhead is minimal ({ratio:.1f}x). Gap vs Stage 2 is in "
                "vLLM itself — likely GPU difference (T4 vs M4), model config, or "
                "max-num-seqs. Check vLLM startup flags.")
    elif ratio < 2.0:
        return (f"Moderate pipeline overhead ({ratio:.1f}x). Both vLLM config and "
                "queue worker batching efficiency need investigation.")
    elif ratio < 3.0:
        return (f"Significant pipeline overhead ({ratio:.1f}x). Likely causes: "
                "single-threaded queue worker, Redis pub/sub per-token latency, "
                "or SSE gateway buffering. Profile the worker's per-token loop.")
    else:
        return (f"Severe pipeline overhead ({ratio:.1f}x). Queue pipeline dominates — "
                "vLLM is mostly idle. Suspect: worker processes one request at a time "
                "or RabbitMQ prefetch=1 serializes requests.")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def print_comparison(direct, pipeline):
    """Print side-by-side comparison table."""
    d_tps, p_tps = direct["aggregate_tps"], pipeline["aggregate_tps"]
    overhead = f"{d_tps / p_tps:.1f}x" if p_tps > 0 else "N/A"
    fmt_ms = lambda v: f"{v}ms" if v is not None else "N/A"

    print(f"\n{'='*65}")
    print("=== Throughput Diagnostic ===")
    print(f"{'='*65}")
    print(f"{'':20s}{'Direct vLLM':>15s}{'Full Pipeline':>15s}{'Overhead':>15s}")
    print(f"{'-'*65}")
    print(f"{'Throughput:':20s}{f'{d_tps} tok/s':>15s}{f'{p_tps} tok/s':>15s}{overhead:>15s}")
    print(f"{'TTFT p50:':20s}{fmt_ms(direct['ttft_p50_ms']):>15s}{fmt_ms(pipeline['ttft_p50_ms']):>15s}")
    print(f"{'TTFT p99:':20s}{fmt_ms(direct['ttft_p99_ms']):>15s}{fmt_ms(pipeline['ttft_p99_ms']):>15s}")
    print(f"{'Concurrency:':20s}{direct['concurrency']:>15d}{pipeline['concurrency']:>15d}")
    print(f"{'Successes:':20s}{direct['successes']:>15d}{pipeline['successes']:>15d}")
    print(f"{'Errors:':20s}{direct['errors']:>15d}{pipeline['errors']:>15d}")
    print(f"{'Total tokens:':20s}{direct['total_tokens']:>15d}{pipeline['total_tokens']:>15d}")
    print(f"{'Wall time:':20s}{direct['wall_time_sec']:>14.1f}s{pipeline['wall_time_sec']:>14.1f}s")
    print(f"{'-'*65}")
    print(f"\nHypothesis: {generate_hypothesis(d_tps, p_tps)}")
    print(f"{'='*65}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global MODEL_NAME

    parser = argparse.ArgumentParser(
        description="Phase 4 — Throughput Diagnostic: Direct vLLM vs Full Pipeline")
    parser.add_argument("--vllm-host", default="http://localhost:8000",
                        help="vLLM direct endpoint (port-forwarded pod)")
    parser.add_argument("--gateway-host", default="http://localhost:8080",
                        help="API gateway endpoint (port-forwarded svc)")
    parser.add_argument("--concurrency", type=int, default=60,
                        help="Concurrent requests per test (default: 60, matches Stage 2)")
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Max tokens per request (default: 150)")
    parser.add_argument("--test", choices=["direct", "pipeline", "both"], default="both",
                        help="Which test to run (default: both)")
    parser.add_argument("--model", default=None,
                        help=f"vLLM model name (default: {MODEL_NAME})")
    args = parser.parse_args()

    if args.model:
        MODEL_NAME = args.model

    run_direct = args.test in ("direct", "both")
    run_pipeline = args.test in ("pipeline", "both")

    start_time = datetime.now(timezone.utc).isoformat()
    print(f"\nPhase 4 — Throughput Diagnostic")
    print(f"Start: {start_time}")
    print(f"Concurrency: {args.concurrency}, Max tokens: {args.max_tokens}")

    # Connectivity checks — fail fast instead of waiting for 60 threads to timeout
    if run_direct:
        print(f"\n  Checking vLLM: {args.vllm_host}")
        try:
            resp = requests.get(f"{args.vllm_host}/v1/models", timeout=5)
            model_ids = [m.get("id", "?") for m in resp.json().get("data", [])]
            print(f"  vLLM OK — models: {model_ids}")
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Hint: kubectl port-forward pod/<vllm-pod> 8000:8000 -n inference-system")
            if not run_pipeline:
                sys.exit(1)
            print(f"  Skipping direct test, running pipeline only.")
            run_direct = False

    if run_pipeline:
        print(f"\n  Checking gateway: {args.gateway_host}")
        try:
            resp = requests.get(f"{args.gateway_host}/health", timeout=5)
            print(f"  Gateway OK — status: {resp.status_code}")
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Hint: kubectl port-forward svc/api-gateway 8080:8080 -n inference-system")
            if not run_direct:
                sys.exit(1)
            print(f"  Skipping pipeline test, running direct only.")
            run_pipeline = False

    if not run_direct and not run_pipeline:
        print("\nERROR: Both endpoints unreachable. Nothing to test.")
        sys.exit(1)

    # WHY direct first: let vLLM cool down before pipeline test.  10s pause
    # lets KV cache drain and batch scheduler reset between tests.
    direct_result = pipeline_result = None

    if run_direct:
        direct_result = run_test(
            "Test A — Direct vLLM", args.concurrency,
            send_direct_request, {"vllm_host": args.vllm_host, "max_tokens": args.max_tokens})

    if run_direct and run_pipeline:
        print("\n  Cooldown: 10s between tests (let KV cache drain)...")
        time.sleep(10)

    if run_pipeline:
        pipeline_result = run_test(
            "Test B — Full Pipeline", args.concurrency,
            send_pipeline_request, {"gateway_host": args.gateway_host, "max_tokens": args.max_tokens})

    if direct_result and pipeline_result:
        print_comparison(direct_result, pipeline_result)

    # Save results — WHY raw per-request data: enables post-hoc plotting of
    # TTFT distributions, outlier identification, Grafana time correlation.
    output = {
        "experiment": "throughput_diagnostic",
        "start_time": start_time,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "config": {
            "concurrency": args.concurrency, "max_tokens": args.max_tokens,
            "prompt_chars": len(PROMPT), "model": MODEL_NAME,
            "vllm_host": args.vllm_host if run_direct else None,
            "gateway_host": args.gateway_host if run_pipeline else None,
        },
    }
    if direct_result:
        output["test_a_direct"] = direct_result
    if pipeline_result:
        output["test_b_pipeline"] = pipeline_result
    if direct_result and pipeline_result:
        d, p = direct_result["aggregate_tps"], pipeline_result["aggregate_tps"]
        output["comparison"] = {
            "direct_tps": d, "pipeline_tps": p,
            "overhead_ratio": round(d / p, 2) if p > 0 else None,
            "hypothesis": generate_hypothesis(d, p),
        }

    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "throughput_diagnostic_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
