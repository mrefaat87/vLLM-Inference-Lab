"""
Test 23: Cold Start and Model Loading Time

Measures time from `docker run` to first successful API response.
Tests local EBS loading (model already cached).

Usage: python3 stage2_test23_cold_start.py --host <ip>
"""

import json
import time
import subprocess
import argparse
import requests

SSH_KEY = "/Users/mrefaat/.ssh/vllm-lab-key-2.pem"
TRIALS = 3


def ssh_cmd(host, cmd):
    """Run SSH command and return output."""
    result = subprocess.run(
        ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", host, cmd],
        capture_output=True, text=True, timeout=300
    )
    return result.stdout.strip(), result.returncode


def measure_cold_start(host, trial):
    """Stop container, start fresh, time until healthy."""
    ssh_host = f"ubuntu@{host}"

    # Stop any existing container
    ssh_cmd(ssh_host, "docker rm -f vllm-server 2>/dev/null")
    time.sleep(2)

    # Start container and record time
    start_time = time.perf_counter()
    start_wall = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    ssh_cmd(ssh_host,
            "docker run -d --name vllm-server --gpus all -p 8000:8000 "
            "-v /home/ubuntu/hf-cache:/root/.cache/huggingface "
            "vllm/vllm-openai:v0.17.1 "
            "--model Qwen/Qwen2.5-7B-Instruct-AWQ "
            "--quantization awq_marlin --dtype half "
            "--gpu-memory-utilization 0.9 --max-model-len 2048")

    # Poll health endpoint
    health_time = None
    for i in range(120):  # 10 min max
        time.sleep(2)
        try:
            resp = requests.get(f"http://{host}:8000/health", timeout=3)
            if resp.status_code == 200:
                health_time = time.perf_counter()
                break
        except Exception:
            pass

    if not health_time:
        return {"trial": trial, "error": "server did not become healthy in 240s"}

    # Send first request
    first_req_start = time.perf_counter()
    try:
        resp = requests.post(
            f"http://{host}:8000/v1/completions",
            json={"model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                  "prompt": "Hello", "max_tokens": 5, "stream": False},
            timeout=30
        )
        first_resp_time = time.perf_counter()
        first_ttft = first_resp_time - first_req_start
    except Exception as e:
        return {"trial": trial, "error": f"first request failed: {e}"}

    docker_to_healthy = health_time - start_time
    docker_to_first_resp = first_resp_time - start_time

    return {
        "trial": trial,
        "wall_start": start_wall,
        "docker_to_healthy_sec": round(docker_to_healthy, 1),
        "docker_to_first_response_sec": round(docker_to_first_resp, 1),
        "first_request_ttft_sec": round(first_ttft, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Test 23: Cold Start Timing")
    parser.add_argument("--host", required=True, help="IP address of the instance")
    args = parser.parse_args()
    host = args.host

    print(f"Test 23: Cold Start and Model Loading Time")
    print(f"Instance: {host}")
    print(f"Model: Qwen/Qwen2.5-7B-Instruct-AWQ (cached on EBS)")
    print(f"Trials: {TRIALS}")
    print(f"{'='*60}\n")

    results = []
    for i in range(TRIALS):
        print(f"Trial {i+1}/{TRIALS}...")
        r = measure_cold_start(host, i + 1)
        results.append(r)

        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  Docker→Healthy: {r['docker_to_healthy_sec']:.1f}s")
            print(f"  Docker→First Response: {r['docker_to_first_response_sec']:.1f}s")
            print(f"  First Request TTFT: {r['first_request_ttft_sec']:.3f}s")

    valid = [r for r in results if "error" not in r]
    if valid:
        avg_healthy = sum(r["docker_to_healthy_sec"] for r in valid) / len(valid)
        avg_first = sum(r["docker_to_first_response_sec"] for r in valid) / len(valid)
        print(f"\nAVERAGE:")
        print(f"  Docker→Healthy: {avg_healthy:.1f}s")
        print(f"  Docker→First Response: {avg_first:.1f}s")

    output = {
        "local_ebs": {
            "trials": results,
            "avg_cold_start_sec": round(avg_healthy, 1) if valid else None,
        }
    }

    output_file = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_test23_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
