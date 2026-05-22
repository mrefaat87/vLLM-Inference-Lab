"""
Phase 4 — Cold Start Benchmark: 8 Cumulative Configs + Sleep Wake Test

Measures cold start latency broken down by the 5 pipeline stages across 8
deployment configurations, each adding one optimization cumulatively. Plus a
9th test measuring wake-from-sleep latency.

THE 5 COLD START STAGES:
  Stage 1: Node Provisioning  — Karpenter + Spot + kubelet registration
  Stage 2: Image Pull/Unpack  — containerd decompresses vLLM layers to overlayfs
  Stage 3: Model Loading      — weight tensors from disk/S3 to GPU VRAM
  Stage 4: Runtime Init       — CUDA context + JIT compilation + CUDA graph capture
  Stage 5: Readiness          — startup probe + warmup + readiness file

THE 8 CONFIGS (cumulative — each adds one optimization):
  1. naked          — v0.19.0 baseline, zero optimizations (~500-600s)
  2. +s3-init       — add S3 model init container (saves ~12s on model download)
  3. +prebaked-ami  — switch to pre-baked AMI with vLLM image cached (saves ~7s)
  4. +warmup        — add model warmup before traffic (same total, TTFT 30s→0.5s)
  5. +startup-probe — replace initialDelaySeconds:120 with startup probe (saves ~5s)
  6. +soci          — switch to SOCI AMI for lazy image loading (saves ~250s)
  7. +streamer      — RunAI Model Streamer for concurrent tensor loading (saves ~34s)
  8. +cache-hit     — 2nd cold start on same node, compilation cache warm (saves ~30s)

PLUS:
  9. wake-from-sleep — vLLM sleep mode wake latency (~0.3-0.9s)

OUTPUTS:
  cold_start_benchmark_results.json      — structured per-stage timing data
  cold_start_benchmark_dashboard.html    — visual comparison dashboard

Usage:
    # Full benchmark (all 8 configs + sleep test)
    python3 phase4/tests/cold_start_benchmark.py --host http://localhost:8080

    # Subset of configs
    python3 phase4/tests/cold_start_benchmark.py --host http://localhost:8080 --configs naked,+s3-init,+startup-probe

    # Skip sleep test
    python3 phase4/tests/cold_start_benchmark.py --host http://localhost:8080 --skip-sleep

    # Dry run (print YAMLs without applying)
    python3 phase4/tests/cold_start_benchmark.py --dry-run
"""

import json
import time
import argparse
import subprocess
import dataclasses
import re
import sys
import os
import tempfile
import statistics
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NAMESPACE = "inference-system"
DEPLOYMENT = "vllm-worker"
LABEL_SELECTOR = "app.kubernetes.io/name=vllm-worker"
GPU_NODE_LABEL = "node-role=gpu"
GPU_COST_PER_HOUR = 0.16  # g4dn.xlarge Spot

# ECR registry for custom images
ECR_REGISTRY = "019167255542.dkr.ecr.us-east-1.amazonaws.com"
ECR_VLLM_IMAGE = f"{ECR_REGISTRY}/inference-lab/vllm-openai:v0.19.0-slim"
ECR_STREAMER_IMAGE = f"{ECR_REGISTRY}/inference-lab/vllm-openai:v0.19.0-slim-streamer"
ECR_WORKER_IMAGE = f"{ECR_REGISTRY}/inference-lab/worker:phase4v3"
DOCKERHUB_VLLM_IMAGE = "vllm/vllm-openai:v0.19.0"

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "cold_start_benchmark_results.json")
DASHBOARD_FILE = os.path.join(os.path.dirname(__file__), "cold_start_benchmark_dashboard.html")

# Prompts for verification requests
PROMPTS = {
    "short": {"text": "What is 2+2?", "max_tokens": 20},
    "medium": {
        "text": "Explain how auto-scaling works in cloud computing in 3 sentences.",
        "max_tokens": 150,
    },
    "long": {
        "text": "Analyze the trade-offs between horizontal and vertical scaling in distributed systems. "
        "Cover latency, throughput, cost, and operational complexity.",
        "max_tokens": 300,
    },
}

# ---------------------------------------------------------------------------
# The 8 benchmark configurations (cumulative)
# ---------------------------------------------------------------------------
# Each config adds ONE optimization on top of the previous. The config dict
# maps to knobs in the deployment YAML template. See render_deployment_yaml()
# for how each knob translates to K8s manifest changes.
CONFIGS = OrderedDict([
    ("naked", {
        "ami": "default",
        "init_container": False,
        "warmup": False,
        "enforce_eager": True,
        "probes": "legacy",
        "soci": False,
        "streamer": False,
        "compile_cache": False,
        "sleep": False,
        "description": "v0.19.0 baseline: default AMI, HF download, initialDelaySeconds:120, --enforce-eager, no warmup",
        "expected_impact": "True baseline — expect ~500-600s",
        "stage_affected": "all",
    }),
    ("+s3-init", {
        "ami": "default",
        "init_container": True,
        "warmup": False,
        "enforce_eager": True,
        "probes": "legacy",
        "soci": False,
        "streamer": False,
        "compile_cache": False,
        "sleep": False,
        "description": "Add S3 model init container. Tiered: EBS hostPath → S3 → HuggingFace fallback",
        "expected_impact": "Saves ~12s on model download (S3 same-region vs HF cross-region)",
        "stage_affected": "Stage 3",
    }),
    ("+prebaked-ami", {
        "ami": "prebaked",
        "init_container": True,
        "warmup": False,
        "enforce_eager": True,
        "probes": "legacy",
        "soci": False,
        "streamer": False,
        "compile_cache": False,
        "sleep": False,
        "description": "Switch to pre-baked AMI with v0.19.0 image cached in containerd",
        "expected_impact": "Saves ~7s (layer unpack is bottleneck, not download)",
        "stage_affected": "Stage 2",
    }),
    ("+warmup", {
        "ami": "prebaked",
        "init_container": True,
        "warmup": True,
        "enforce_eager": True,
        "probes": "legacy",
        "soci": False,
        "streamer": False,
        "compile_cache": False,
        "sleep": False,
        "description": "Add model warmup (synthetic 50-token request before consuming queue)",
        "expected_impact": "Same cold start total, but first real request TTFT drops from ~30s to ~0.5s",
        "stage_affected": "Stage 4 / TTFT",
    }),
    ("+startup-probe", {
        "ami": "prebaked",
        "init_container": True,
        "warmup": True,
        "enforce_eager": True,
        "probes": "startup",
        "soci": False,
        "streamer": False,
        "compile_cache": False,
        "sleep": False,
        "description": "Replace initialDelaySeconds:120 with startup probe (httpGet /health every 5s)",
        "expected_impact": "Saves ~5s (initialDelaySeconds runs concurrently with model loading; probe interval 5s vs 10s)",
        "stage_affected": "Stage 5",
    }),
    ("+soci", {
        "ami": "soci",
        "init_container": True,
        "warmup": True,
        "enforce_eager": False,
        "probes": "startup",
        "soci": True,
        "streamer": False,
        "compile_cache": False,
        "sleep": False,
        "description": "SOCI AMI (lazy FUSE image loading from ECR). Also drops --enforce-eager for CUDA graphs",
        "expected_impact": "Saves ~250s on image pull (container starts before full decompression)",
        "stage_affected": "Stage 2",
    }),
    ("+streamer", {
        "ami": "soci",
        "init_container": True,
        "warmup": True,
        "enforce_eager": False,
        "probes": "startup",
        "soci": True,
        "streamer": True,
        "compile_cache": False,
        "sleep": False,
        "description": "RunAI Model Streamer: concurrent C++ tensor streaming to GPU (16 threads)",
        "expected_impact": "Saves ~34s on model loading (48s → 14s from disk)",
        "stage_affected": "Stage 3",
    }),
    ("+cache-hit", {
        "ami": "soci",
        "init_container": True,
        "warmup": True,
        "enforce_eager": False,
        "probes": "startup",
        "soci": True,
        "streamer": True,
        "compile_cache": True,
        "sleep": False,
        "description": "2nd cold start on same node — /mnt/compile-cache has compiled Triton kernels from +streamer run",
        "expected_impact": "Saves ~30s (Stage 4 → ~0s, compiled kernels reused)",
        "stage_affected": "Stage 4",
    }),
])

# Valid config names for CLI --configs
VALID_CONFIGS = set(CONFIGS.keys())

# Verified kubectl context — set by verify_phase4_context() at startup
_KUBE_CONTEXT = None


# ---------------------------------------------------------------------------
# Stack safety — always verify Phase 4 context
# ---------------------------------------------------------------------------
def verify_phase4_context() -> str:
    """Verify kubectl context contains 'inference-phase4'. Hard exit if wrong.

    WHY: the benchmark deletes GPU nodes and modifies deployments. Running
    against the wrong cluster (e.g., inference-phase3) would be destructive.
    """
    global _KUBE_CONTEXT
    result = subprocess.run(
        "kubectl config current-context",
        shell=True, capture_output=True, text=True, timeout=10,
    )
    context = result.stdout.strip()
    if "inference-phase4" not in context:
        print(f"\n  ABORT: Wrong kubectl context '{context}'.")
        print(f"  Expected context containing 'inference-phase4'.")
        print(f"  Switch with: aws eks update-kubeconfig --name inference-phase4 --region us-east-1")
        sys.exit(1)
    _KUBE_CONTEXT = context
    return context


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class StageTimings:
    """Timestamps for each cold start stage boundary."""

    scale_requested: float = 0.0
    pod_created: Optional[float] = None
    node_scheduled: Optional[float] = None
    image_pull_done: Optional[float] = None
    vllm_health_ready: Optional[float] = None
    warmup_complete: Optional[float] = None
    pod_ready: Optional[float] = None

    def stage_durations(self) -> dict:
        """Compute per-stage durations in seconds."""

        def _d(start, end):
            if start is not None and end is not None:
                return round(max(0, end - start), 1)
            return None

        return {
            "stage1_node_provision": _d(self.scale_requested, self.node_scheduled),
            "stage2_image_pull": _d(self.node_scheduled, self.image_pull_done),
            "stage3_model_loading": _d(self.image_pull_done, self.vllm_health_ready),
            "stage4_runtime_init": _d(self.vllm_health_ready, self.warmup_complete),
            "stage5_readiness": _d(self.warmup_complete, self.pod_ready),
            "total": _d(self.scale_requested, self.pod_ready),
        }


@dataclasses.dataclass
class ConfigResult:
    """Result from one benchmark config measurement."""

    config_name: str
    config_number: int
    description: str
    expected_impact: str
    stage_affected: str
    timings: Optional[StageTimings] = None
    stages: Optional[dict] = None
    wake_latency_ms: Optional[float] = None
    first_request: Optional[dict] = None
    gpu_hours: float = 0.0
    gpu_cost_usd: float = 0.0
    node_name: Optional[str] = None
    pod_name: Optional[str] = None
    node_reused: bool = False
    cold_start_type: str = "full"  # "full" or "warm-node"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Kubernetes helpers
# ---------------------------------------------------------------------------
def kubectl(cmd: str, timeout: int = 60) -> str:
    """Run kubectl command against the verified Phase 4 cluster.

    WHY --context: prevents accidental cross-cluster operations if the user
    switches kubectl context in another terminal during the benchmark run.
    """
    full_cmd = f"kubectl --context {_KUBE_CONTEXT} {cmd}"
    result = subprocess.run(
        full_cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )
    return result.stdout.strip()


def get_replicas() -> tuple:
    """Get (desired, ready) from deployment."""
    output = kubectl(
        f"get deployment {DEPLOYMENT} -n {NAMESPACE} "
        f"-o jsonpath='{{.spec.replicas}} {{.status.readyReplicas}}'"
    )
    parts = output.strip("'").split()
    desired = int(parts[0]) if parts else 0
    ready = int(parts[1]) if len(parts) > 1 and parts[1] not in ("<none>", "None", "") else 0
    return desired, ready


def wait_for_replicas(target_ready: int, timeout: int = 900) -> Optional[float]:
    """Wait until ready replicas == target. Returns seconds taken."""
    start = time.time()
    while time.time() - start < timeout:
        _, ready = get_replicas()
        if ready >= target_ready:
            return time.time() - start
        time.sleep(5)
    return None


def wait_for_zero(timeout: int = 300) -> bool:
    """Wait until 0 ready replicas."""
    start = time.time()
    while time.time() - start < timeout:
        _, ready = get_replicas()
        if ready == 0:
            return True
        time.sleep(5)
    return False


def get_newest_pod() -> Optional[str]:
    """Get name of newest pod matching the deployment."""
    output = kubectl(
        f"get pods -n {NAMESPACE} -l {LABEL_SELECTOR} "
        f"--sort-by=.metadata.creationTimestamp "
        f"-o jsonpath='{{.items[-1].metadata.name}}'"
    )
    name = output.strip("'").strip()
    return name if name else None


def get_pod_node(pod_name: str) -> Optional[str]:
    """Get the node a pod is running on."""
    output = kubectl(
        f"get pod {pod_name} -n {NAMESPACE} -o jsonpath='{{.spec.nodeName}}'"
    )
    return output.strip("'").strip() or None


def is_pod_ready(pod_name: str) -> bool:
    """Check if pod has Ready=True condition."""
    output = kubectl(
        f"get pod {pod_name} -n {NAMESPACE} "
        f"-o jsonpath='{{.status.conditions[?(@.type==\"Ready\")].status}}'"
    )
    return output.strip("'").strip() == "True"


def parse_k8s_timestamp(ts: Optional[str]) -> Optional[float]:
    """Parse K8s ISO 8601 timestamp to epoch seconds."""
    if ts is None:
        return None
    ts = ts.strip("'").strip()
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.timestamp()
    except ValueError:
        return None


def get_pod_ready_time(pod_name: str) -> Optional[float]:
    """Get epoch when pod Ready condition became True."""
    output = kubectl(
        f"get pod {pod_name} -n {NAMESPACE} "
        f"-o jsonpath='{{.status.conditions[?(@.type==\"Ready\")].lastTransitionTime}}'"
    )
    return parse_k8s_timestamp(output)


def get_pod_creation_time(pod_name: str) -> Optional[float]:
    """Get pod creation timestamp as epoch."""
    output = kubectl(
        f"get pod {pod_name} -n {NAMESPACE} "
        f"-o jsonpath='{{.metadata.creationTimestamp}}'"
    )
    return parse_k8s_timestamp(output)


def get_gpu_node_count() -> int:
    """Count GPU nodes currently in the cluster."""
    output = kubectl(f"get nodes -l {GPU_NODE_LABEL} --no-headers 2>/dev/null")
    if not output.strip():
        return 0
    return len([l for l in output.strip().split("\n") if l.strip()])


def delete_all_gpu_nodes() -> None:
    """Delete all GPU nodes and wait until they're gone.

    WHY: ensures Karpenter provisions a completely fresh node for each config,
    so cold start timing includes full node provisioning (Stage 1).
    """
    nodes = kubectl(f"get nodes -l {GPU_NODE_LABEL} -o name").strip()
    if not nodes:
        print("    No GPU nodes to delete")
        return

    for node in nodes.split("\n"):
        node = node.strip()
        if node:
            kubectl(f"delete {node} --wait=false", timeout=30)
            print(f"    Deleting {node}")

    # Wait for nodes to disappear (timeout 300s)
    for _ in range(60):
        remaining = kubectl(f"get nodes -l {GPU_NODE_LABEL} --no-headers 2>/dev/null")
        if not remaining.strip():
            print("    All GPU nodes removed")
            return
        time.sleep(5)
    print("    WARNING: Some GPU nodes may still exist after 300s timeout")


# ---------------------------------------------------------------------------
# K8s event extraction — per-stage timing from pod lifecycle events
# ---------------------------------------------------------------------------
def get_pod_events(pod_name: str) -> list:
    """Get K8s events for a pod as structured dicts."""
    output = kubectl(
        f"get events -n {NAMESPACE} "
        f"--field-selector involvedObject.name={pod_name} -o json",
        timeout=30,
    )
    if not output:
        return []
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []
    events = []
    for item in data.get("items", []):
        ts = parse_k8s_timestamp(item.get("lastTimestamp") or item.get("firstTimestamp", ""))
        events.append({
            "reason": item.get("reason", ""),
            "message": item.get("message", ""),
            "timestamp": ts,
            "component": item.get("source", {}).get("component", ""),
        })
    return events


def extract_schedule_time(events: list) -> Optional[float]:
    """Find Scheduled event timestamp — marks end of Stage 1."""
    for e in events:
        if e["reason"] == "Scheduled" and e["timestamp"]:
            return e["timestamp"]
    return None


def extract_image_pull_done(events: list) -> Optional[float]:
    """Find Pulled event for vllm-openai — marks end of Stage 2."""
    for e in events:
        if e["reason"] == "Pulled" and "vllm" in e.get("message", "").lower() and e["timestamp"]:
            return e["timestamp"]
    # Fallback: any Pulled event
    for e in events:
        if e["reason"] == "Pulled" and e["timestamp"]:
            return e["timestamp"]
    return None


# ---------------------------------------------------------------------------
# Worker log parsing — extract timing markers from queue-worker container
# ---------------------------------------------------------------------------
def get_worker_logs(pod_name: str, since_seconds: int = 600) -> str:
    """Get recent logs from queue-worker container."""
    return kubectl(
        f"logs {pod_name} -n {NAMESPACE} -c queue-worker --since={since_seconds}s",
        timeout=30,
    )


def parse_worker_log_markers(logs: str) -> dict:
    """Extract timing markers from worker logs.

    Parses log lines like:
      2026-04-06 10:23:45,678 worker INFO WARMUP: vLLM is ready (attempt 3)
      2026-04-06 10:24:15,123 worker INFO WARMUP: Complete -- 50 tokens in 29500ms
      2026-04-06 10:24:15,234 worker INFO READY: Created /tmp/worker-ready
      2026-04-06 10:29:15,567 worker INFO SLEEP: vLLM entered L1 sleep
      2026-04-06 10:34:16,789 worker INFO WAKE: vLLM woke from L1 sleep in 450ms
    """
    markers = {}
    for line in logs.strip().split("\n"):
        ts_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
        if not ts_match:
            continue
        try:
            dt = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S,%f")
            epoch = dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue

        if "WARMUP: vLLM is ready" in line:
            markers["vllm_health_ready"] = epoch
        elif "WARMUP: Complete" in line:
            markers["warmup_complete"] = epoch
            ms_match = re.search(r"in (\d+)ms", line)
            if ms_match:
                markers["warmup_duration_ms"] = int(ms_match.group(1))
        elif "READY: Created /tmp/worker-ready" in line:
            markers["readiness_file_created"] = epoch
        elif "SLEEP: vLLM entered L" in line:
            markers["sleep_entered"] = epoch
        elif "WAKE: vLLM woke from L" in line:
            markers["wake_time"] = epoch
            ms_match = re.search(r"in (\d+)ms", line)
            if ms_match:
                markers["wake_latency_ms"] = int(ms_match.group(1))
    return markers


# ---------------------------------------------------------------------------
# Stage timing assembly — the core measurement function
# ---------------------------------------------------------------------------
def collect_stage_timings(scale_requested: float, timeout: int = 900) -> StageTimings:
    """Poll until pod is ready, then extract all stage boundary timestamps.

    This polls every 5s tracking pod lifecycle, then retroactively extracts
    precise timestamps from K8s events and worker logs.
    """
    timings = StageTimings(scale_requested=scale_requested)
    pod_name = None
    start = time.time()

    # Phase 1: Poll until pod is ready
    while time.time() - start < timeout:
        pod = get_newest_pod()
        if pod and pod_name is None:
            pod_name = pod
            timings.pod_created = get_pod_creation_time(pod)
            print(f"    Pod created: {pod}")

        if pod_name and timings.node_scheduled is None:
            node = get_pod_node(pod_name)
            if node:
                timings.node_scheduled = time.time()
                print(f"    Scheduled on node: {node}")

        if pod_name and is_pod_ready(pod_name):
            timings.pod_ready = time.time()
            elapsed = timings.pod_ready - scale_requested
            print(f"    Pod ready! Total: {elapsed:.1f}s")
            break

        time.sleep(5)
    else:
        print(f"    TIMEOUT after {timeout}s — pod never became ready")
        return timings

    # Phase 2: Extract precise timestamps from K8s events
    if pod_name:
        events = get_pod_events(pod_name)
        scheduled = extract_schedule_time(events)
        if scheduled:
            timings.node_scheduled = scheduled
        pull_done = extract_image_pull_done(events)
        if pull_done:
            timings.image_pull_done = pull_done

        # Precise ready time from K8s condition
        precise_ready = get_pod_ready_time(pod_name)
        if precise_ready:
            timings.pod_ready = precise_ready

    # Phase 3: Worker log markers
    if pod_name:
        logs = get_worker_logs(pod_name)
        markers = parse_worker_log_markers(logs)
        timings.vllm_health_ready = markers.get("vllm_health_ready")
        timings.warmup_complete = markers.get("warmup_complete") or markers.get(
            "readiness_file_created"
        )

    return timings


# ---------------------------------------------------------------------------
# Request sender
# ---------------------------------------------------------------------------
def send_test_request(host: str, prompt_key: str = "short") -> dict:
    """Send a single request through the pipeline, return timing."""
    prompt_cfg = PROMPTS[prompt_key]
    payload = {
        "prompt": prompt_cfg["text"],
        "max_tokens": prompt_cfg["max_tokens"],
        "temperature": 0.1,
        "stream": True,
    }

    submit = time.perf_counter()
    ttft = None
    token_count = 0
    error = None

    try:
        resp = requests.post(f"{host}/v1/completions", json=payload, stream=True, timeout=120)
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if data.get("status") == "complete":
                break
            if data.get("status") == "error":
                error = data.get("error", "unknown")
                break

            # Token from vLLM SSE stream
            choices = data.get("choices", [])
            if choices and choices[0].get("text"):
                token_count += 1
                if ttft is None:
                    ttft = (time.perf_counter() - submit) * 1000

            # Token from Redis-based pipeline
            if "token" in data:
                token_count += 1
                if ttft is None:
                    ttft = (time.perf_counter() - submit) * 1000
    except Exception as e:
        error = str(e)

    total_ms = (time.perf_counter() - submit) * 1000
    return {
        "prompt_type": prompt_key,
        "ttft_ms": round(ttft, 1) if ttft else None,
        "total_ms": round(total_ms, 1),
        "tokens": token_count,
        "error": error,
    }


# ---------------------------------------------------------------------------
# GPU cost computation
# ---------------------------------------------------------------------------
def compute_gpu_cost(duration_sec: float) -> dict:
    """Compute GPU-hours and cost for a config run."""
    gpu_hours = duration_sec / 3600.0
    return {
        "gpu_hours": round(gpu_hours, 4),
        "gpu_cost_usd": round(gpu_hours * GPU_COST_PER_HOUR, 4),
    }


# ---------------------------------------------------------------------------
# YAML rendering — deployment, nodepool, and configmap templates
# ---------------------------------------------------------------------------
def _label_safe(name: str) -> str:
    """Sanitize config name for use as a K8s label value."""
    return name.lstrip("+").replace("+", "-")


def _vllm_image(config: dict) -> str:
    """Determine vLLM container image URI based on config knobs.

    WHY this logic: the slim ECR image (~4GB, sm_75 only) is used for all
    configs except 'naked' (which uses Docker Hub stock to show true baseline).
    The prebaked AMI pre-caches the slim ECR image, so configs using it must
    reference the same image URI for containerd to find it locally.
    """
    if config["streamer"]:
        return ECR_STREAMER_IMAGE
    # naked (ami=default, no init container) uses Docker Hub stock for true baseline
    if config["ami"] == "default" and not config["init_container"]:
        return DOCKERHUB_VLLM_IMAGE
    # All other configs use the slim ECR image (cached on prebaked/soci AMIs)
    return ECR_VLLM_IMAGE


def _vllm_args(config: dict) -> str:
    """Build vLLM command arguments based on config knobs."""
    args = [
        "--model", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--quantization", "awq",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
    ]
    if config["enforce_eager"]:
        args.append("--enforce-eager")
    if config["streamer"]:
        args.extend(["--load-format", "runai_streamer"])
    if config["sleep"]:
        args.append("--enable-sleep-mode")
    return json.dumps(args)


def render_deployment_yaml(config_name: str, config: dict) -> str:
    """Render Deployment YAML from config dict.

    WHY templated: 8 configs differ only in specific knobs (init container,
    probes, image, args). A template with conditionals is cleaner than 8
    separate YAML files with 90% identical content.
    """
    vllm_image = _vllm_image(config)
    vllm_args = _vllm_args(config)
    label_name = _label_safe(config_name)

    # Init container block — only when S3 model loading is enabled
    init_container_block = ""
    if config["init_container"]:
        init_container_block = """
      initContainers:
        - name: model-download
          image: amazon/aws-cli:2.15.0
          command: ["/bin/sh", "-c"]
          args:
            - |
              set -e
              MODEL_DIR="/model-cache/Qwen/Qwen2.5-7B-Instruct-AWQ"
              # Tier 1: check EBS snapshot hostPath
              if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR 2>/dev/null | head -1)" ]; then
                echo "MODEL_CACHE: Found model at $MODEL_DIR (EBS/hostPath). Skipping download."
                exit 0
              fi
              # Tier 2: S3 download
              echo "MODEL_CACHE: Downloading from S3..."
              mkdir -p "$MODEL_DIR"
              if aws s3 sync "s3://${MODEL_CACHE_BUCKET}/Qwen/Qwen2.5-7B-Instruct-AWQ/" "$MODEL_DIR/" --quiet 2>/dev/null; then
                echo "MODEL_CACHE: S3 download complete."
                exit 0
              fi
              # Tier 3: HuggingFace fallback
              echo "MODEL_CACHE: S3 failed. vLLM will download from HuggingFace on startup."
          env:
            - name: MODEL_CACHE_BUCKET
              value: "inference-lab-model-cache-019167255542"
            - name: AWS_DEFAULT_REGION
              value: "us-east-1"
          volumeMounts:
            - name: model-cache
              mountPath: /model-cache
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "1"
              memory: "1Gi"
"""

    # vLLM env vars
    vllm_env = """
            - name: TRANSFORMERS_CACHE
              value: "/root/.cache/huggingface"
            - name: HF_HOME
              value: "/root/.cache/huggingface"
            - name: TORCH_CUDA_ARCH_LIST
              value: "7.5"
"""
    if config["streamer"]:
        vllm_env += """            - name: RUNAI_STREAMER_CONCURRENCY
              value: "16"
"""
    if config["sleep"]:
        vllm_env += """            - name: VLLM_SERVER_DEV_MODE
              value: "1"
"""

    # Probes — legacy vs startup
    if config["probes"] == "legacy":
        # Legacy: fixed initialDelaySeconds:120, no startup probe
        vllm_probes = """\
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 10
            failureThreshold: 30
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 30
            failureThreshold: 3
"""
        worker_probes = """\
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 10
            failureThreshold: 30
"""
    else:
        # Startup probe: detects readiness instantly, no fixed delay
        vllm_probes = """\
          startupProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 5
            failureThreshold: 60
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 10
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 30
            failureThreshold: 3
"""
        worker_probes = """\
          readinessProbe:
            exec:
              command: ["test", "-f", "/tmp/worker-ready"]
            periodSeconds: 5
            failureThreshold: 90
"""

    yaml = f"""---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {DEPLOYMENT}
  namespace: {NAMESPACE}
  labels:
    app.kubernetes.io/name: {DEPLOYMENT}
    app.kubernetes.io/part-of: inference-lab
    benchmark-config: "{label_name}"
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: {DEPLOYMENT}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: {DEPLOYMENT}
        benchmark-config: "{label_name}"
    spec:
      terminationGracePeriodSeconds: 180
      serviceAccountName: vllm-worker
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
{init_container_block}
      containers:
        - name: vllm-server
          image: {vllm_image}
          args: {vllm_args}
          ports:
            - containerPort: 8000
              name: http
          env:
{vllm_env}
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface
            - name: compile-cache
              mountPath: /root/.cache/vllm
            - name: shm
              mountPath: /dev/shm
          lifecycle:
            preStop:
              exec:
                command: ["sleep", "15"]
{vllm_probes}
          resources:
            requests:
              cpu: "2"
              memory: "12Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "4"
              memory: "12Gi"
              nvidia.com/gpu: "1"

        - name: queue-worker
          image: {ECR_WORKER_IMAGE}
          env:
            - name: RABBITMQ_HOST
              value: "rabbitmq"
            - name: RABBITMQ_PORT
              value: "5672"
            - name: RABBITMQ_USER
              value: "inference"
            - name: RABBITMQ_PASS
              value: "inference123"
            - name: REDIS_HOST
              value: "redis"
            - name: REDIS_PORT
              value: "6379"
            - name: VLLM_HOST
              value: "localhost"
            - name: VLLM_PORT
              value: "8000"
            - name: VLLM_MODEL
              value: "Qwen/Qwen2.5-7B-Instruct-AWQ"
          envFrom:
            - configMapRef:
                name: scaling-config
{worker_probes}
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"

      volumes:
        - name: model-cache
          hostPath:
            path: /mnt/model-cache
            type: DirectoryOrCreate
        - name: compile-cache
          hostPath:
            path: /mnt/compile-cache
            type: DirectoryOrCreate
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 2Gi
"""
    return yaml


def render_nodepool_yaml(ami_type: str) -> str:
    """Render EC2NodeClass + NodePool YAML for the given AMI type.

    WHY 3 variants: configs 1-2 use default EKS AMI, configs 3-5 use pre-baked
    AMI, configs 6-8 use SOCI AMI. Karpenter discovers AMIs via tag filters.
    """
    # amiSelectorTerms — only set for custom AMIs
    if ami_type == "default":
        ami_selector = ""
    elif ami_type == "prebaked":
        ami_selector = """
  amiSelectorTerms:
    - tags:
        Name: "inference-lab-gpu-node"
"""
    elif ami_type == "soci":
        ami_selector = """
  amiSelectorTerms:
    - tags:
        Name: "inference-lab-gpu-node-soci"
"""
    else:
        raise ValueError(f"Unknown AMI type: {ami_type}")

    return f"""---
apiVersion: karpenter.k8s.aws/v1beta1
kind: EC2NodeClass
metadata:
  name: gpu-nodes
  labels:
    app.kubernetes.io/name: gpu-nodes
    app.kubernetes.io/part-of: inference-lab
spec:
  amiFamily: AL2
{ami_selector}
  role: "inference-phase4-karpenter-node"

  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "inference-phase4"
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "inference-phase4"

  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 100Gi
        volumeType: gp3
        encrypted: true
        deleteOnTermination: true

  tags:
    Project: inference-lab
---
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: gpu-pool
  labels:
    app.kubernetes.io/name: gpu-pool
    app.kubernetes.io/part-of: inference-lab
spec:
  template:
    metadata:
      labels:
        node-role: gpu
    spec:
      nodeClassRef:
        name: gpu-nodes
      requirements:
        - key: node.kubernetes.io/instance-type
          operator: In
          values:
            - g4dn.xlarge
            - g4dn.2xlarge
            - g5.xlarge
        - key: karpenter.sh/capacity-type
          operator: In
          values:
            - spot
            - on-demand
        - key: topology.kubernetes.io/zone
          operator: In
          values:
            - us-east-1a
            - us-east-1b
        - key: kubernetes.io/arch
          operator: In
          values:
            - amd64
      taints:
        - key: nvidia.com/gpu
          value: "true"
          effect: NoSchedule
  limits:
    cpu: "32"
    memory: 128Gi
  disruption:
    consolidationPolicy: WhenEmpty
    consolidateAfter: 300s
    expireAfter: 6h
    budgets:
      - nodes: "1"
"""


def render_configmap_yaml(config: dict) -> str:
    """Render scaling-config ConfigMap with warmup/sleep settings."""
    warmup = "true" if config["warmup"] else "false"
    sleep_enabled = "true" if config["sleep"] else "false"

    return f"""---
apiVersion: v1
kind: ConfigMap
metadata:
  name: scaling-config
  namespace: {NAMESPACE}
data:
  # Admission control
  ADMISSION_STRATEGY: "threshold"
  GPU_CACHE_THRESHOLD: "0.80"
  MAX_BATCH_SIZE: "8"
  PREFETCH_COUNT: "10"
  # Per-request strategy
  KV_BYTES_PER_TOKEN: "401408"
  CHARS_PER_TOKEN: "4"
  TOTAL_KV_CACHE_BYTES: "9600000000"
  # KEDA reference values
  KEDA_QUEUE_TRIGGER_VALUE: "5"
  KEDA_ACTIVATION_VALUE: "3"
  KEDA_MAX_REPLICAS: "4"
  KEDA_COOLDOWN_PERIOD: "300"
  KEDA_KV_CACHE_THRESHOLD: "0.75"
  # Graceful shutdown
  DRAIN_TIMEOUT: "170"
  # Model warmup — toggled per config
  WARMUP_ENABLED: "{warmup}"
  # vLLM sleep mode — toggled per config
  SLEEP_ENABLED: "{sleep_enabled}"
  SLEEP_IDLE_TIMEOUT: "300"
  SLEEP_LEVEL: "1"
"""


def apply_yaml(yaml_str: str, description: str = "") -> None:
    """Write YAML to temp file, kubectl apply, delete temp file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_str)
        f.flush()
        tmp_path = f.name
    try:
        result = subprocess.run(
            f"kubectl --context {_KUBE_CONTEXT} apply -f {tmp_path}",
            shell=True, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"    WARNING: kubectl apply failed for {description}: {result.stderr.strip()}")
        else:
            applied = result.stdout.strip()
            if applied:
                print(f"    Applied: {applied}")
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
def preflight(configs_to_run: list) -> None:
    """Verify cluster state before running the benchmark."""
    print("\n  PRE-FLIGHT CHECKS")
    print("  " + "-" * 40)

    # 1. Context already verified by verify_phase4_context()
    print(f"  [OK] kubectl context: {_KUBE_CONTEXT}")

    # 2. Cluster reachable
    nodes_output = kubectl("get nodes --no-headers")
    if not nodes_output:
        print("  [FAIL] Cluster not reachable (no nodes returned)")
        sys.exit(1)
    node_count = len(nodes_output.strip().split("\n"))
    print(f"  [OK] Cluster reachable ({node_count} nodes)")

    # 3. Deployment exists
    deploy_check = kubectl(f"get deployment {DEPLOYMENT} -n {NAMESPACE} -o name")
    if not deploy_check:
        print(f"  [FAIL] Deployment '{DEPLOYMENT}' not found in namespace '{NAMESPACE}'")
        sys.exit(1)
    print(f"  [OK] Deployment '{DEPLOYMENT}' exists")

    # 4. Check for required AMIs (warn if missing)
    config_names = [c for c in configs_to_run if c in CONFIGS]
    needs_prebaked = any(CONFIGS[c]["ami"] == "prebaked" for c in config_names if c in CONFIGS)
    needs_soci = any(CONFIGS[c]["ami"] == "soci" for c in config_names if c in CONFIGS)

    if needs_prebaked:
        ami_check = subprocess.run(
            'aws ec2 describe-images --filters "Name=tag:Name,Values=inference-lab-gpu-node" '
            '--query "Images[0].ImageId" --output text 2>/dev/null',
            shell=True, capture_output=True, text=True, timeout=15,
        )
        ami_id = ami_check.stdout.strip()
        if ami_id and ami_id != "None":
            print(f"  [OK] Pre-baked AMI found: {ami_id}")
        else:
            print("  [WARN] Pre-baked AMI not found. Configs using 'prebaked' AMI may fail.")
            print("         Build with: bash phase4/scripts/build-ami.sh")

    if needs_soci:
        ami_check = subprocess.run(
            'aws ec2 describe-images --filters "Name=tag:Name,Values=inference-lab-gpu-node-soci" '
            '--query "Images[0].ImageId" --output text 2>/dev/null',
            shell=True, capture_output=True, text=True, timeout=15,
        )
        ami_id = ami_check.stdout.strip()
        if ami_id and ami_id != "None":
            print(f"  [OK] SOCI AMI found: {ami_id}")
        else:
            print("  [WARN] SOCI AMI not found. Configs using 'soci' AMI may fail.")
            print("         Build with: bash phase4/scripts/build-ami-soci.sh")

    # 5. Cost estimate
    n_configs = len(config_names)
    est_minutes = n_configs * 10
    est_cost = (est_minutes / 60) * GPU_COST_PER_HOUR
    print(f"\n  Estimated: ~{est_minutes} min GPU time, ~${est_cost:.2f} at Spot rates")
    print(f"  Configs to run: {n_configs}")

    # 6. Cost guard
    if n_configs > 8:
        print("  [ABORT] Cost guard: more than 8 configs. Refusing to run.")
        sys.exit(1)

    print()


# ---------------------------------------------------------------------------
# Config runner — the core benchmark loop
# ---------------------------------------------------------------------------
def run_config(config_name: str, config: dict, config_number: int,
               prev_config: Optional[dict], host: str) -> ConfigResult:
    """Run one benchmark config: deploy, cold start, measure, verify."""
    print("\n" + "=" * 60)
    print(f"  CONFIG {config_number}/8: {config_name}")
    print(f"  {config['description']}")
    print("=" * 60)

    result = ConfigResult(
        config_name=config_name,
        config_number=config_number,
        description=config["description"],
        expected_impact=config["expected_impact"],
        stage_affected=config["stage_affected"],
    )

    try:
        # Step 1: Apply NodePool if AMI changed
        if prev_config is None or config["ami"] != prev_config["ami"]:
            print(f"  [1/6] Applying NodePool (AMI: {config['ami']})...")
            nodepool_yaml = render_nodepool_yaml(config["ami"])
            apply_yaml(nodepool_yaml, f"nodepool-{config['ami']}")
        else:
            print(f"  [1/6] NodePool unchanged (AMI: {config['ami']})")

        # Step 2: Apply ConfigMap
        print("  [2/6] Applying ConfigMap...")
        configmap_yaml = render_configmap_yaml(config)
        apply_yaml(configmap_yaml, "scaling-config")

        # Step 3: Apply Deployment
        print("  [3/6] Applying Deployment...")
        deployment_yaml = render_deployment_yaml(config_name, config)
        apply_yaml(deployment_yaml, f"deployment-{config_name}")

        # Step 4: Scale to 0 and clean up
        print("  [4/6] Scaling to 0 replicas...")
        kubectl(f"scale deployment/{DEPLOYMENT} -n {NAMESPACE} --replicas=0")
        wait_for_zero(timeout=300)

        # Step 5: Delete GPU nodes only when AMI changes between configs.
        # WHY: configs sharing the same AMI reuse the existing node — eliminates
        # infrastructure variance (±50-100s) so within-group comparisons are
        # apples-to-apples on the same instance. Only 3 provisions instead of 8.
        ami_changed = prev_config is None or config["ami"] != prev_config["ami"]

        if ami_changed:
            print("  [5/6] Deleting all GPU nodes (AMI changed — full cold start)...")
            delete_all_gpu_nodes()
            result.node_reused = False
            result.cold_start_type = "full"
        else:
            gpu_count = get_gpu_node_count()
            print(f"  [5/6] Reusing GPU node (same AMI '{config['ami']}', {gpu_count} node(s))")
            if gpu_count == 0:
                print("    WARNING: No GPU node found. Will provision fresh.")
                result.cold_start_type = "full"
            elif gpu_count > 1:
                print(f"    Deleting extras ({gpu_count} → 1)...")
                nodes = kubectl(f"get nodes -l {GPU_NODE_LABEL} -o name").strip().split("\n")
                for extra in nodes[1:]:
                    extra = extra.strip()
                    if extra:
                        kubectl(f"delete {extra} --wait=false", timeout=30)
                time.sleep(10)
                result.node_reused = True
                result.cold_start_type = "warm-node"
            else:
                result.node_reused = True
                result.cold_start_type = "warm-node"

        # Step 6: Scale to 1 — triggers cold start
        print("  [6/6] Scaling to 1 (triggering cold start)...")
        scale_time = time.time()
        kubectl(f"scale deployment/{DEPLOYMENT} -n {NAMESPACE} --replicas=1")

        # Collect stage timings
        timings = collect_stage_timings(scale_time, timeout=900)
        stages = timings.stage_durations()
        result.timings = timings
        result.stages = stages
        result.pod_name = get_newest_pod()
        if result.pod_name:
            result.node_name = get_pod_node(result.pod_name)

        # Verify inference works
        print("  Sending verification request...")
        result.first_request = send_test_request(host, "short")
        ttft = result.first_request.get("ttft_ms")
        print(f"    First request TTFT: {ttft}ms")

        # Cost
        total = stages.get("total") or 0
        cost = compute_gpu_cost(total)
        result.gpu_hours = cost["gpu_hours"]
        result.gpu_cost_usd = cost["gpu_cost_usd"]

    except Exception as e:
        result.error = str(e)
        print(f"    ERROR: {e}")

    _print_stages(result.stages)
    return result


def run_sleep_test(host: str, sleep_timeout: int = 360) -> ConfigResult:
    """Config 9: Wake from vLLM sleep mode.

    Keeps the pod from +cache-hit running, enables sleep mode, waits for
    vLLM to sleep (SLEEP_IDLE_TIMEOUT=300s), then sends a request to
    trigger wake. Measures wake latency from worker logs.
    """
    print("\n" + "=" * 60)
    print("  CONFIG 9/9: wake-from-sleep")
    print("  Keep pod running, wait for sleep, measure wake latency")
    print("=" * 60)

    result = ConfigResult(
        config_name="wake-from-sleep",
        config_number=9,
        description="vLLM sleep mode wake: GPU VRAM offloaded to CPU, CUDA graphs preserved",
        expected_impact="Wake in ~0.3-0.9s (L1: CPU→GPU weight copy)",
        stage_affected="Wake",
    )

    try:
        pod = get_newest_pod()
        if not pod:
            result.error = "No running pod found — run +cache-hit config first"
            print(f"    ERROR: {result.error}")
            return result
        result.pod_name = pod
        result.node_name = get_pod_node(pod)

        # Apply sleep-enabled config
        sleep_config = dict(CONFIGS["+cache-hit"])
        sleep_config["sleep"] = True
        print("  [1/4] Applying sleep-enabled ConfigMap + Deployment...")
        configmap_yaml = render_configmap_yaml(sleep_config)
        apply_yaml(configmap_yaml, "scaling-config-sleep")
        deployment_yaml = render_deployment_yaml("wake-from-sleep", sleep_config)
        apply_yaml(deployment_yaml, "deployment-sleep")

        # Wait for pod to restart with new config and become ready
        print("  [2/4] Waiting for pod to restart with sleep mode enabled...")
        time.sleep(10)
        wait_for_replicas(1, timeout=600)

        # Get the new pod name after restart
        pod = get_newest_pod()
        if pod:
            result.pod_name = pod

        # Wait for vLLM to enter sleep
        print(f"  [3/4] Waiting up to {sleep_timeout}s for vLLM to enter sleep...")
        sleep_detected = False
        start = time.time()
        while time.time() - start < sleep_timeout:
            logs = get_worker_logs(pod, since_seconds=30)
            if "SLEEP: vLLM entered L" in logs:
                sleep_detected = True
                elapsed = time.time() - start
                print(f"    Sleep detected after {elapsed:.0f}s")
                break
            remaining = sleep_timeout - (time.time() - start)
            print(f"    Waiting... ({remaining:.0f}s remaining)", end="\r")
            time.sleep(10)

        if not sleep_detected:
            result.error = f"vLLM did not enter sleep within {sleep_timeout}s"
            print(f"\n    {result.error}")
            return result

        # Trigger wake by sending a request
        print("\n  [4/4] Sending request to trigger wake...")
        pre_wake = time.time()
        result.first_request = send_test_request(host, "short")
        post_wake = time.time()

        # Parse wake latency from logs
        time.sleep(2)  # brief wait for log flush
        logs = get_worker_logs(pod, since_seconds=30)
        markers = parse_worker_log_markers(logs)
        result.wake_latency_ms = markers.get("wake_latency_ms")

        if result.wake_latency_ms:
            print(f"    Wake latency: {result.wake_latency_ms}ms")
        print(f"    First request TTFT: {result.first_request.get('ttft_ms')}ms")
        print(f"    First request total: {result.first_request.get('total_ms')}ms")

        total = post_wake - pre_wake
        cost = compute_gpu_cost(total)
        result.gpu_hours = cost["gpu_hours"]
        result.gpu_cost_usd = cost["gpu_cost_usd"]
        result.stages = {
            "wake_latency_ms": result.wake_latency_ms,
            "total": round(total, 2),
        }

    except Exception as e:
        result.error = str(e)
        print(f"    ERROR: {e}")

    return result


def _print_stages(stages: Optional[dict]) -> None:
    """Print stage breakdown to console."""
    if not stages:
        return
    print("\n  Stage Breakdown:")
    labels = {
        "stage1_node_provision": "  Stage 1 (Node provision):",
        "stage2_image_pull": "  Stage 2 (Image pull):    ",
        "stage3_model_loading": "  Stage 3 (Model loading): ",
        "stage4_runtime_init": "  Stage 4 (Runtime init):  ",
        "stage5_readiness": "  Stage 5 (Readiness):     ",
        "total": "  TOTAL:                   ",
    }
    for key, label in labels.items():
        val = stages.get(key)
        if val is not None:
            print(f"  {label} {val:.1f}s")
        elif key != "total" and key != "wake_latency_ms":
            print(f"  {label} N/A")


# ---------------------------------------------------------------------------
# Cleanup — always runs, even on error
# ---------------------------------------------------------------------------
def cleanup() -> None:
    """Scale down and delete GPU nodes. Called in finally block."""
    print("\n" + "=" * 60)
    print("  CLEANUP")
    print("=" * 60)
    try:
        kubectl(f"scale deployment/{DEPLOYMENT} -n {NAMESPACE} --replicas=0")
        print("  Scaled to 0 replicas")
    except Exception as e:
        print(f"  WARNING: Failed to scale down: {e}")

    try:
        delete_all_gpu_nodes()
    except Exception as e:
        print(f"  WARNING: Failed to delete GPU nodes: {e}")

    # Restore original deployment + nodepool from the live manifests
    original_manifests = os.path.join(os.path.dirname(__file__), "..", "k8s")
    nodepool_path = os.path.join(original_manifests, "karpenter", "gpu-nodepool.yaml")
    deployment_path = os.path.join(original_manifests, "worker", "deployment-patch.yaml")
    configmap_path = os.path.join(original_manifests, "scaling-config", "configmap.yaml")

    for path, desc in [
        (nodepool_path, "original nodepool"),
        (configmap_path, "original configmap"),
        (deployment_path, "original deployment"),
    ]:
        if os.path.exists(path):
            result = subprocess.run(
                f"kubectl --context {_KUBE_CONTEXT} apply -f {path}",
                shell=True, capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                print(f"  Restored {desc}")
            else:
                print(f"  WARNING: Failed to restore {desc}: {result.stderr.strip()}")

    print("  Cleanup complete")


# ---------------------------------------------------------------------------
# HTML Dashboard Generator
# ---------------------------------------------------------------------------
def generate_dashboard(results: dict) -> str:
    """Generate the HTML dashboard from benchmark results.

    Layout:
      1. Hero stats (5 cards)
      2. Stacked bar chart (9 bars — all from live measurements)
      3. Optimization cards (8 cards in 3 groups)
      4. Scenario results table (per-stage timing)
      5. Cost summary
      6. Key findings + methodology
    """
    configs = results.get("config_results", {})

    # Hero stats
    naked_total = None
    optimized_total = None
    cache_hit_total = None
    wake_ms = None

    if "naked" in configs and configs["naked"].get("stages"):
        naked_total = configs["naked"]["stages"].get("total")
    if "+streamer" in configs and configs["+streamer"].get("stages"):
        optimized_total = configs["+streamer"]["stages"].get("total")
    if "+cache-hit" in configs and configs["+cache-hit"].get("stages"):
        cache_hit_total = configs["+cache-hit"]["stages"].get("total")
    if "wake-from-sleep" in configs:
        wake_ms = configs["wake-from-sleep"].get("wake_latency_ms")

    improvement_pct = round((1 - optimized_total / naked_total) * 100) if (naked_total and optimized_total) else "—"
    naked_str = f"{naked_total:.0f}s" if naked_total else "—"
    optimized_str = f"{optimized_total:.0f}s" if optimized_total else "—"
    cache_str = f"{cache_hit_total:.0f}s" if cache_hit_total else "—"
    wake_str = f"{wake_ms}ms" if wake_ms else "—"

    # Build bar chart data from live measurements
    stage_colors = {
        "stage1": "#e3b341",  # Node provision — amber
        "stage2": "#f85149",  # Image pull — red
        "stage3": "#58a6ff",  # Model loading — blue
        "stage4": "#bc8cff",  # Runtime init — purple
        "stage5": "#7ee787",  # Readiness — green
    }

    bar_data = []
    for cname in list(CONFIGS.keys()):
        if cname in configs and configs[cname].get("stages"):
            s = configs[cname]["stages"]
            bar_data.append({
                "label": cname,
                "stages": {
                    "stage1": s.get("stage1_node_provision") or 0,
                    "stage2": s.get("stage2_image_pull") or 0,
                    "stage3": s.get("stage3_model_loading") or 0,
                    "stage4": s.get("stage4_runtime_init") or 0,
                    "stage5": s.get("stage5_readiness") or 0,
                },
                "total": s.get("total") or 0,
            })

    # Wake from sleep bar
    if wake_ms:
        bar_data.append({
            "label": "wake-from-sleep",
            "stages": {"stage1": 0, "stage2": 0, "stage3": 0, "stage4": 0, "stage5": wake_ms / 1000},
            "total": wake_ms / 1000,
        })

    max_total = max((b["total"] for b in bar_data), default=300)

    # Generate bar rows HTML
    bar_rows_html = ""
    for b in bar_data:
        fills = ""
        for stage_key, color in stage_colors.items():
            val = b["stages"].get(stage_key, 0)
            if val > 0 and max_total > 0:
                pct = (val / max_total) * 100
                # Only show text in segments wide enough
                label_text = f"{val:.0f}" if pct > 4 else ""
                fills += f'<div class="bar-fill" style="width:{pct:.1f}%;background:{color}" title="{stage_key}: {val:.1f}s">{label_text}</div>'
        bar_rows_html += f"""
        <div class="bar-row">
          <div class="bar-label">{b["label"]}</div>
          <div class="bar-track">{fills}</div>
          <div class="bar-value">{b["total"]:.1f}s</div>
        </div>"""

    # Scenario results table
    scenario_rows_html = ""
    all_config_names = list(CONFIGS.keys()) + ["wake-from-sleep"]
    for cname in all_config_names:
        if cname not in configs:
            continue
        sc = configs[cname]
        stages = sc.get("stages") or {}

        # First request info
        ttft = "—"
        if sc.get("first_request") and sc["first_request"].get("ttft_ms"):
            ttft = f'{sc["first_request"]["ttft_ms"]:.0f}ms'
        if sc.get("wake_latency_ms"):
            ttft = f'{sc["wake_latency_ms"]}ms wake'

        def _sv(k):
            v = stages.get(k)
            return f"{v:.1f}s" if v is not None else "—"

        total_v = stages.get("total")
        total_str = f"{total_v:.1f}s" if total_v is not None else "—"

        cst = sc.get("cold_start_type", "")
        if cst == "full":
            cst_badge = '<span style="background:#f8514930;color:#f85149;padding:2px 6px;border-radius:3px;font-size:10px;margin-left:6px">full</span>'
        elif cst == "warm-node":
            cst_badge = '<span style="background:#7ee78730;color:#7ee787;padding:2px 6px;border-radius:3px;font-size:10px;margin-left:6px">warm</span>'
        else:
            cst_badge = ""

        scenario_rows_html += f"""
        <tr>
          <td><strong>{cname}</strong>{cst_badge}</td>
          <td style="font-size:12px;color:#8b949e">{sc.get("stage_affected", "")}</td>
          <td class="mono">{_sv("stage1_node_provision")}</td>
          <td class="mono">{_sv("stage2_image_pull")}</td>
          <td class="mono">{_sv("stage3_model_loading")}</td>
          <td class="mono">{_sv("stage4_runtime_init")}</td>
          <td class="mono">{_sv("stage5_readiness")}</td>
          <td class="mono"><strong>{total_str}</strong></td>
          <td class="mono">{ttft}</td>
        </tr>"""

    # Cost rows
    cost_rows_html = ""
    total_cost = 0
    for cname in all_config_names:
        if cname not in configs:
            continue
        sc = configs[cname]
        cost = sc.get("gpu_cost_usd", 0)
        total_cost += cost
        duration = (sc.get("stages") or {}).get("total", 0) or 0
        cost_rows_html += f"""
        <tr>
          <td>{cname}</td>
          <td class="mono">{duration:.1f}s</td>
          <td class="mono">{sc.get("gpu_hours", 0):.4f}</td>
          <td class="mono">${cost:.4f}</td>
        </tr>"""
    cost_rows_html += f"""
        <tr style="border-top:2px solid #30363d">
          <td><strong>Total</strong></td><td></td><td></td>
          <td class="mono"><strong>${total_cost:.4f}</strong></td>
        </tr>"""

    # Optimization cards — grouped by phase
    optimizations = [
        # Phase 4v1 (Infrastructure)
        ("Phase 4v1 (Infrastructure)", "S3 Init Container", "Stage 3",
         _delta_str(configs, "naked", "+s3-init", "stage3_model_loading"),
         "S3 model cache replaces HuggingFace download. Init container checks EBS → S3 → HF fallback.", "#58a6ff"),
        ("Phase 4v1 (Infrastructure)", "Pre-baked AMI", "Stage 2",
         _delta_str(configs, "+s3-init", "+prebaked-ami", "stage2_image_pull"),
         "vLLM image cached in containerd on AMI. Only saves ~7s — layer unpack is the bottleneck, not download.", "#f85149"),

        # Phase 4v2 (Runtime)
        ("Phase 4v2 (Runtime)", "Model Warmup", "TTFT",
         _ttft_delta_str(configs, "+prebaked-ami", "+warmup"),
         "Synthetic 50-token request absorbs CUDA JIT compilation. Same cold start total, but first real request TTFT drops dramatically.", "#bc8cff"),

        # Phase 4v3 (Cold Start)
        ("Phase 4v3 (Cold Start)", "Startup Probe", "Stage 5",
         _delta_str(configs, "+warmup", "+startup-probe", "stage5_readiness"),
         "Replaces fixed initialDelaySeconds:120. Detects readiness the instant /health returns 200.", "#7ee787"),
        ("Phase 4v3 (Cold Start)", "SOCI Lazy Loading", "Stage 2",
         _delta_str(configs, "+startup-probe", "+soci", "stage2_image_pull"),
         "containerd FUSE mount — files fetched on demand from ECR. Eliminates decompression bottleneck.", "#f85149"),
        ("Phase 4v3 (Cold Start)", "RunAI Model Streamer", "Stage 3",
         _delta_str(configs, "+soci", "+streamer", "stage3_model_loading"),
         "Concurrent C++ threads stream tensors from disk to GPU. 3.3x faster than safetensors.", "#58a6ff"),
        ("Phase 4v3 (Cold Start)", "Compilation Cache", "Stage 4",
         _delta_str(configs, "+streamer", "+cache-hit", "stage4_runtime_init"),
         "hostPath volume persists compiled Triton kernels across pod restarts on same node.", "#bc8cff"),
        ("Phase 4v3 (Cold Start)", "vLLM Sleep Mode", "Wake",
         f"→ {wake_ms}ms" if wake_ms else "—",
         "Offloads weights to CPU RAM, preserves CUDA graphs. Wake copies weights back in 0.3-0.9s.", "#e3b341"),
    ]

    opt_cards_html = ""
    current_phase = ""
    for phase, name, stage, delta, desc, color in optimizations:
        if phase != current_phase:
            if current_phase:
                opt_cards_html += "</div>"
            opt_cards_html += f'<h3>{phase}</h3><div class="opt-grid">'
            current_phase = phase
        opt_cards_html += f"""
        <div class="opt-card">
          <div class="title">{name}</div>
          <div class="saved" style="color:{color}">{stage}: {delta}</div>
          <div class="body">{desc}</div>
        </div>"""
    opt_cards_html += "</div>"

    # Find biggest win and biggest disappointment
    biggest_win_name = "—"
    biggest_win_saved = 0
    biggest_disappointment = "Pre-baked AMI + FSR"

    config_list = list(CONFIGS.keys())
    for i in range(1, len(config_list)):
        prev_name = config_list[i - 1]
        curr_name = config_list[i]
        if prev_name in configs and curr_name in configs:
            prev_total = (configs[prev_name].get("stages") or {}).get("total")
            curr_total = (configs[curr_name].get("stages") or {}).get("total")
            if prev_total and curr_total:
                saved = prev_total - curr_total
                if saved > biggest_win_saved:
                    biggest_win_saved = saved
                    biggest_win_name = curr_name

    # Assemble HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phase 4 — Cold Start Benchmark: 8 Optimizations</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif;
  background: #0d1117; color: #c9d1d9;
  max-width: 960px; margin: 0 auto; padding: 32px 24px 80px;
  line-height: 1.75; font-size: 15px;
}}
h1 {{ font-size: 32px; font-weight: 700; color: #e6edf3; line-height: 1.25; margin-bottom: 8px; }}
h2 {{ font-size: 22px; font-weight: 600; color: #e6edf3; margin: 56px 0 16px; padding-bottom: 8px; border-bottom: 1px solid #21262d; }}
h3 {{ font-size: 17px; font-weight: 600; color: #e6edf3; margin: 32px 0 12px; }}
p {{ margin-bottom: 16px; }}
.subtitle {{ color: #8b949e; font-size: 13px; margin-bottom: 32px; }}
.lead {{ font-size: 17px; color: #e6edf3; line-height: 1.7; margin-bottom: 24px; }}
strong {{ color: #e6edf3; }}
.callout {{ padding: 16px 20px; border-radius: 6px; margin: 20px 0; font-size: 14px; line-height: 1.7; }}
.callout-blue {{ background: #0d1117; border-left: 3px solid #58a6ff; }}
.callout-blue strong {{ color: #58a6ff; }}
.callout-green {{ background: #0d1117; border-left: 3px solid #7ee787; }}
.callout-green strong {{ color: #7ee787; }}
.callout-yellow {{ background: #0d1117; border-left: 3px solid #e3b341; }}
.callout-yellow strong {{ color: #e3b341; }}
table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px; }}
th {{ text-align: left; padding: 8px 12px; border-bottom: 2px solid #30363d; color: #8b949e; font-weight: 600; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; }}
tr:hover td {{ background: #161b22; }}
.mono {{ font-family: 'SF Mono', 'Fira Code', monospace; }}
.stats-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 20px 0; }}
.stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center; }}
.stat-card .num {{ font-size: 32px; font-weight: 700; line-height: 1.2; font-family: 'SF Mono', 'Fira Code', monospace; }}
.stat-card .desc {{ font-size: 11px; color: #8b949e; margin-top: 4px; }}
.num.green {{ color: #7ee787; }}
.num.yellow {{ color: #e3b341; }}
.num.red {{ color: #f85149; }}
.num.blue {{ color: #58a6ff; }}
.bar-row {{ display: flex; align-items: center; margin-bottom: 6px; }}
.bar-label {{ width: 160px; font-size: 12px; color: #8b949e; flex-shrink: 0; font-family: 'SF Mono', 'Fira Code', monospace; }}
.bar-track {{ flex: 1; height: 26px; background: #21262d; border-radius: 4px; position: relative; overflow: hidden; display: flex; }}
.bar-fill {{ height: 100%; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 600; color: white; min-width: 1px; }}
.bar-value {{ font-size: 12px; color: #8b949e; width: 60px; text-align: right; flex-shrink: 0; margin-left: 8px; font-family: 'SF Mono', 'Fira Code', monospace; }}
.opt-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0; }}
@media (max-width: 700px) {{ .opt-grid {{ grid-template-columns: 1fr; }} }}
.opt-card {{ background: #161b22; border-radius: 8px; padding: 20px; border: 1px solid #30363d; }}
.opt-card .title {{ font-weight: 600; font-size: 15px; color: #e6edf3; margin-bottom: 8px; }}
.opt-card .saved {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; margin-bottom: 8px; }}
.opt-card .body {{ font-size: 13px; color: #8b949e; line-height: 1.6; }}
.legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; font-size: 12px; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; }}
.legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; }}
.footer {{ margin-top: 64px; padding-top: 24px; border-top: 1px solid #21262d; text-align: center; font-size: 12px; color: #484f58; }}
</style>
</head>
<body>

<h1>Cold Start Benchmark: 8 Optimizations</h1>
<p class="subtitle">Phase 4 — cumulative optimization progression. {results.get("start_time", "")}</p>

<p class="lead">Each config adds one optimization on top of the previous. The stacked bar chart shows which pipeline stage shrinks with each addition — from the v0.19.0 baseline (~{naked_str}) down to a compilation cache hit (~{cache_str}) and vLLM sleep mode wake ({wake_str}).</p>

<h2>Results at a Glance</h2>
<div class="stats-row">
  <div class="stat-card"><div class="num red">{naked_str}</div><div class="desc">Naked Baseline</div></div>
  <div class="stat-card"><div class="num green">{optimized_str}</div><div class="desc">Fully Optimized</div></div>
  <div class="stat-card"><div class="num green">{improvement_pct}%</div><div class="desc">Improvement</div></div>
  <div class="stat-card"><div class="num green">{cache_str}</div><div class="desc">Cache Hit (2nd start)</div></div>
  <div class="stat-card"><div class="num blue">{wake_str}</div><div class="desc">Wake from Sleep</div></div>
</div>

<h2>Optimization Progression</h2>
<p>Each bar shows the 5 cold start stages. Watch how each optimization shrinks a different segment.</p>

<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#e3b341"></div>Stage 1: Node Provision</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f85149"></div>Stage 2: Image Pull</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#58a6ff"></div>Stage 3: Model Loading</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#bc8cff"></div>Stage 4: Runtime Init</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#7ee787"></div>Stage 5: Readiness</div>
</div>

{bar_rows_html}

<div class="callout callout-green">
<strong>Biggest single win:</strong> {biggest_win_name} — saved {biggest_win_saved:.0f}s from the previous config.
</div>

<div class="callout callout-yellow">
<strong>Biggest disappointment:</strong> {biggest_disappointment} — layer decompression (I/O), not network download, was the real bottleneck. SOCI solved this properly by eliminating decompression entirely.
</div>

<h2>Per-Stage Results</h2>
<table>
  <thead>
    <tr>
      <th>Config</th><th>Stage</th>
      <th>Stage 1</th><th>Stage 2</th><th>Stage 3</th><th>Stage 4</th><th>Stage 5</th>
      <th>Total</th><th>1st TTFT</th>
    </tr>
  </thead>
  <tbody>{scenario_rows_html}</tbody>
</table>

<h2>Optimizations</h2>
{opt_cards_html}

<h2>Cost</h2>
<table>
  <thead><tr><th>Config</th><th>Duration</th><th>GPU-Hours</th><th>Cost</th></tr></thead>
  <tbody>{cost_rows_html}</tbody>
</table>

<h2>Methodology</h2>
<div class="callout callout-blue">
<strong>How measurements are taken:</strong> Configs are grouped by AMI — the first config in each group provisions a fresh GPU node (<strong>[full]</strong> cold start), while subsequent configs reuse the same node (<strong>[warm-node]</strong>). This eliminates infrastructure variance (±50-100s between Spot instances) so within-group comparisons are apples-to-apples on the same hardware. Cross-group comparisons (full cold starts) show AMI-level impact. K8s events provide Scheduled and Pulled timestamps (~1s precision). Worker logs provide WARMUP and READY timestamps (~1s). Wake-from-sleep keeps the pod running and measures wake latency.
</div>

<div class="callout callout-blue">
<strong>Timing precision:</strong> Stage boundaries are extracted retroactively from K8s events and worker logs after the pod reaches Ready state. kubectl polling runs every 5s during the wait. All timestamps are UTC epoch seconds with ~1s precision.
</div>

<div class="footer">
  Generated by <code>cold_start_benchmark.py</code> | Phase 4 — Inference Learning Lab
</div>

</body>
</html>"""
    return html


def _delta_str(configs: dict, before: str, after: str, stage_key: str) -> str:
    """Compute before→after string for a stage between two configs."""
    before_val = None
    after_val = None
    if before in configs and configs[before].get("stages"):
        before_val = configs[before]["stages"].get(stage_key)
    if after in configs and configs[after].get("stages"):
        after_val = configs[after]["stages"].get(stage_key)
    if before_val is not None and after_val is not None:
        saved = before_val - after_val
        return f"{before_val:.0f}s → {after_val:.0f}s (saved {saved:.0f}s)"
    return "—"


def _ttft_delta_str(configs: dict, before: str, after: str) -> str:
    """Compute TTFT before→after string between two configs."""
    before_ttft = None
    after_ttft = None
    if before in configs and configs[before].get("first_request"):
        before_ttft = configs[before]["first_request"].get("ttft_ms")
    if after in configs and configs[after].get("first_request"):
        after_ttft = configs[after]["first_request"].get("ttft_ms")
    if before_ttft is not None and after_ttft is not None:
        return f"{before_ttft:.0f}ms → {after_ttft:.0f}ms"
    return "—"


# ---------------------------------------------------------------------------
# Dry run — print YAMLs without applying
# ---------------------------------------------------------------------------
def dry_run() -> None:
    """Print all generated YAMLs for inspection without applying."""
    print("\n" + "=" * 60)
    print("  DRY RUN — Generated YAMLs")
    print("=" * 60)

    # NodePool variants
    for ami_type in ["default", "prebaked", "soci"]:
        print(f"\n{'─' * 60}")
        print(f"  NodePool: {ami_type}")
        print(f"{'─' * 60}")
        print(render_nodepool_yaml(ami_type))

    # Config deployments
    for config_name, config in CONFIGS.items():
        print(f"\n{'─' * 60}")
        print(f"  Deployment: {config_name}")
        print(f"{'─' * 60}")
        print(render_deployment_yaml(config_name, config))

        print(f"\n{'─' * 60}")
        print(f"  ConfigMap: {config_name}")
        print(f"{'─' * 60}")
        print(render_configmap_yaml(config))

    # Sleep config
    sleep_config = dict(CONFIGS["+cache-hit"])
    sleep_config["sleep"] = True
    print(f"\n{'─' * 60}")
    print(f"  Deployment: wake-from-sleep")
    print(f"{'─' * 60}")
    print(render_deployment_yaml("wake-from-sleep", sleep_config))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def run_benchmark(host: str, configs_to_run: list, sleep_timeout: int,
                  skip_sleep: bool) -> dict:
    """Run all requested configs and produce results."""
    print("\n" + "=" * 60)
    print("  PHASE 4 — COLD START BENCHMARK (8 Configs)")
    print("=" * 60)
    print(f"  Host: {host}")
    print(f"  Configs: {', '.join(configs_to_run)}")
    print(f"  Sleep test: {'skip' if skip_sleep else f'enabled (timeout {sleep_timeout}s)'}")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")

    preflight(configs_to_run)

    start_time = datetime.now(timezone.utc).isoformat()
    config_results = {}
    prev_config = None
    config_number = 0

    try:
        # Run each config sequentially
        for config_name in configs_to_run:
            if config_name not in CONFIGS:
                print(f"\n  Skipping unknown config: {config_name}")
                continue
            config = CONFIGS[config_name]
            config_number += 1

            result = run_config(config_name, config, config_number, prev_config, host)
            config_results[config_name] = dataclasses.asdict(result)
            prev_config = config

        # Sleep test (only if +cache-hit was run and sleep not skipped)
        if not skip_sleep and "+cache-hit" in configs_to_run:
            result = run_sleep_test(host, sleep_timeout)
            config_results["wake-from-sleep"] = dataclasses.asdict(result)
        elif not skip_sleep and "+cache-hit" not in configs_to_run:
            print("\n  Skipping sleep test (+cache-hit not in configs list)")

    finally:
        cleanup()

    end_time = datetime.now(timezone.utc).isoformat()

    results = {
        "experiment": "phase4_cold_start_benchmark_8configs",
        "start_time": start_time,
        "end_time": end_time,
        "cluster_info": {
            "context": _KUBE_CONTEXT,
            "vllm_version": "v0.19.0",
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "instance_type": "g4dn.xlarge",
            "gpu_cost_per_hour": GPU_COST_PER_HOUR,
        },
        "configs_run": configs_to_run,
        "config_results": config_results,
    }

    # Save results JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {RESULTS_FILE}")

    # Generate HTML dashboard
    html = generate_dashboard(results)
    with open(DASHBOARD_FILE, "w") as f:
        f.write(html)
    print(f"  Dashboard saved to: {DASHBOARD_FILE}")

    # Print summary
    print("\n" + "=" * 60)
    print("  BENCHMARK COMPLETE")
    print("=" * 60)
    for cname, sc in config_results.items():
        total = (sc.get("stages") or {}).get("total", "—")
        cst = sc.get("cold_start_type", "")
        print(f"  {cname}: {total} [{cst}]")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 — Cold Start Benchmark: 8 Cumulative Configs + Sleep Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full benchmark (all 8 configs + sleep test)
  python3 phase4/tests/cold_start_benchmark.py --host http://localhost:8080

  # Subset of configs
  python3 phase4/tests/cold_start_benchmark.py --host http://localhost:8080 --configs naked,+s3-init,+startup-probe

  # Skip sleep test
  python3 phase4/tests/cold_start_benchmark.py --host http://localhost:8080 --skip-sleep

  # Dry run (print YAMLs without applying)
  python3 phase4/tests/cold_start_benchmark.py --dry-run
        """,
    )
    parser.add_argument(
        "--host",
        help="API gateway URL (e.g., http://localhost:8080). Required unless --dry-run.",
    )
    parser.add_argument(
        "--configs",
        default=",".join(CONFIGS.keys()),
        help=f"Comma-separated configs to run (default: all 8). Valid: {', '.join(CONFIGS.keys())}",
    )
    parser.add_argument(
        "--skip-sleep",
        action="store_true",
        help="Skip the wake-from-sleep test (config 9)",
    )
    parser.add_argument(
        "--sleep-timeout",
        type=int,
        default=360,
        help="Max seconds to wait for vLLM sleep (default: 360)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated YAMLs without applying. No cluster access needed.",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    if not args.host:
        parser.error("--host is required unless --dry-run is set")

    # Verify Phase 4 context before any cluster operations
    verify_phase4_context()

    # Parse config list
    configs_to_run = [c.strip() for c in args.configs.split(",")]
    invalid = set(configs_to_run) - VALID_CONFIGS
    if invalid:
        print(f"Invalid configs: {invalid}")
        print(f"Valid configs: {', '.join(CONFIGS.keys())}")
        sys.exit(1)

    run_benchmark(args.host, configs_to_run, args.sleep_timeout, args.skip_sleep)


if __name__ == "__main__":
    main()
