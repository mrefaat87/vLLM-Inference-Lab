# Phase 3 Backpressure Experiment — Handover Notes (v2 Redesign)

## What Was Built

### Infrastructure
- **Separate EKS cluster** (`inference-phase3`) in its own VPC (`10.1.0.0/16`)
- Terraform stack: `phase3/terraform/` (shares ECR + S3 with Phase 1, everything else isolated)
- Scripts: `phase3/scripts/setup-cluster.sh`, `teardown-cluster.sh`, `port-forward.sh`, `deploy.sh`, `rollback.sh`
- **Cluster is scaled to 0 nodes** — only control plane running ($0.10/hr). Destroy fully with `teardown-cluster.sh`.
- **KNOWN ISSUE**: aws-auth ConfigMap must be manually patched after `setup-cluster.sh` to add Karpenter node role. The script's sed only catches `karpenter.sh/discovery` but the EC2NodeClass also has `role`, `securityGroupSelectorTerms`, and `subnetSelectorTerms` referencing the cluster name. Fixed mid-session with global `sed 's/inference-lab/inference-phase3/g'` — the setup script was updated with this fix.
- **KNOWN ISSUE**: KEDA ScaledObject scales worker to 0 when queue is empty between strategy switches. Must pause KEDA (`annotate scaledobject ... autoscaling.keda.sh/paused-replicas=1`) before experiments. The test script does this, but manual strategy switches between runs don't.
- **Deployment strategy**: `maxSurge=0, maxUnavailable=1` set on vllm-worker deployment so strategy switches replace in-place on same GPU node (no need for a second GPU).

### Worker Code
- `phase3/app/worker/worker.py` — 3 pluggable strategies via `ADMISSION_STRATEGY` env var
- `phase3/app/worker/Dockerfile` + `requirements.txt` — pushed to ECR as `worker:phase3`
- **Config**: `PREFETCH_COUNT=10`, `MAX_BATCH_SIZE=8`, `GPU_CACHE_THRESHOLD=0.80`

### Test Script
- `phase3/tests/backpressure_comparison.py` — the main experiment runner
- Supports: per-workload rates, Pareto random distributions (seeded), burst patterns, SLA measurement, cost-per-token tracking, incremental JSON saves, Grafana screenshot capture
- Run per-strategy with `--skip-switch` to avoid rollout timeout issues inside the script

### Results (3 experiment rounds)

**v1** (9 runs: 3 strategies × uniform_short/uniform_long/mixed at 2 req/s):
- `phase3/tests/backpressure_results.json`
- `phase3/screenshots/` — 126 PNGs from Grafana Image Renderer
- Finding: **zero differentiation** — all strategies identical. KV cache peaked at 11% (threshold 80%), running_requests peaked at 5 (threshold 8). Admission gates unreachable.

**v2** (15 runs: 3 strategies × 5 workload levels with varying rates):
- `phase3/tests/backpressure_results_v2_merged.json` (also `_static.json`, `_threshold.json`, `_per_request.json`)
- `phase3/screenshots_v2/` — URL manifests only (Grafana port-forward died)
- Finding: **some differentiation on error rates** but still 0 NACKs. Static won on completion rate. Threshold collapsed under spike patterns. Per-request matched static.

**v3** (12 runs: 3 strategies × pareto_light/pareto_heavy/kv_stress/burst_outlier):
- `phase3/tests/backpressure_results_v3_merged.json` (also per-strategy files)
- `phase3/screenshots_v3/` — URL manifests only
- Finding: **real differentiation**. Static: 100% SLA on 3/4 workloads. Threshold: collapsed at pareto_heavy (57% SLA, 2.8x cost) and failed completely at kv_stress + burst_outlier. Per-request: matched static but 0% SLA at burst_outlier.

### HTML Dashboards
- `phase3/tests/phase3_results_dashboard.html` — v2 results with full commentary
- `phase3/tests/phase3_results_v3_dashboard.html` — v3 results with SLA + cost + strategy verdicts

## Critical Discovery: The Implementation Was Flawed

Through discussion we identified **three fundamental flaws** in the experiment design:

### Flaw 1: Remote `/metrics` polling in the admission hot path
Both threshold and per-request strategies call `GET http://vllm:8000/metrics` before every admission decision. This:
- Adds 2-5ms per request (HTTP round-trip to vLLM sidecar)
- Returns **stale state** — multiple async coroutines read the same snapshot simultaneously
- Creates a **positive feedback loop** under load: polling delays → queue growth → more polling → cascade
- This is why threshold collapsed — not because its logic was wrong, but because the metrics collection method poisoned it

### Flaw 2: KV cache is the wrong metric for this hardware
Phase 2 Grafana showed the T4 bottleneck is **memory bandwidth** (94%) not KV cache (1.66%). Both threshold and per-request gate on KV cache usage — the metric that doesn't matter for a 7B AWQ model on T4 (9.6GB KV headroom, only ~25% used at peak). The admission gate should match the actual bottleneck:
- **This T4**: gate on decode throughput (remaining tokens in flight / GPU speed)
- **Larger model**: gate on KV cache (when model leaves <2GB for KV)
- **Mixed prefill/decode**: gate on pending prefill tokens (prefill blocks decode)

### Flaw 3: `MAX_BATCH_SIZE` check is logically broken
The check `running_requests >= MAX_BATCH_SIZE` reads from vLLM metrics. But if the check enforces the limit, `running_requests` can never reach `MAX_BATCH_SIZE` — it's a self-defeating gate. The only way it fires is through race conditions (multiple coroutines reading the same stale snapshot). Not a reliable concurrency limiter.

## What Needs to Happen Next

### Redesign: Local Admission Tracker

Replace remote `/metrics` polling with a local tracker in the worker process:

```python
class AdmissionTracker:
    def __init__(self, kv_budget_bytes, max_concurrent):
        self.running = {}  # job_id → {estimated_kv, remaining_tokens, ...}
        self.kv_budget_bytes = kv_budget_bytes
        self.max_concurrent = max_concurrent
    
    def try_admit(self, job_id, estimated_kv, max_tokens):
        # Instant, atomic, no HTTP call
        ...
    
    def update_progress(self, job_id, tokens_generated):
        # Called as tokens stream — remaining budget shrinks in real-time
        ...
    
    def release(self, job_id):
        # Called on ACK — frees budget
        ...
```

Three strategies become:
- **Static**: `prefetch_count=N`, no tracker (unchanged)
- **Threshold**: `tracker.running_count < N` — same as static but with accurate local count instead of RabbitMQ's prefetch
- **Per-request**: `tracker.kv_remaining > this_request_cost` — cost-aware gate using local KV budget

The key insight from our discussion: **the only real difference between threshold and per-request is whether the gate accounts for request size**. Same local state, same zero overhead — the variable is decision quality, not implementation overhead.

### Bigger Model to Create KV Scarcity

A 7B AWQ model uses ~4GB VRAM, leaving 9.6GB for KV. That's too much headroom — the gate never fires. Options:
- **13-14B AWQ** (~7-8GB VRAM) → leaves 1-2GB for KV → gate fires constantly
- **Qwen2.5-14B-AWQ-INT4** would be ideal — same family, same tokenizer, 2x model size
- Check if it fits on T4 with `--gpu-memory-utilization 0.90`

### Throughput-Based Gate (for T4-specific bottleneck)

Even with a bigger model, the T4's memory bandwidth may still be the primary bottleneck. Add a fourth strategy:

```python
# Decode throughput gate
total_remaining_tokens = sum(r.remaining_tokens for r in tracker.running.values())
estimated_drain_sec = total_remaining_tokens / GPU_TOK_PER_SEC
if estimated_drain_sec > SLA_TARGET_SEC:
    reject  # GPU is saturated, new request would degrade everyone
```

This gates on the actual bottleneck (decode throughput) rather than a proxy (KV cache or running count).

## File Inventory

```
phase3/
├── terraform/           # Separate EKS cluster (inference-phase3)
│   ├── *.tf             # 10 TF files (shares ECR+S3 via data sources)
│   ├── terraform.tfvars # cluster_name=inference-phase3, vpc_cidr=10.1.0.0/16
│   └── terraform.tfstate # Local state
├── app/worker/
│   ├── worker.py        # 3-strategy worker (NEEDS REDESIGN — local tracker)
│   ├── Dockerfile
│   └── requirements.txt
├── tests/
│   ├── backpressure_comparison.py  # Test runner (v3 workloads, SLA, cost)
│   ├── backpressure_results.json   # v1 results
│   ├── backpressure_results_v2_merged.json
│   ├── backpressure_results_v3_merged.json
│   ├── phase3_results_dashboard.html      # v2 HTML dashboard
│   └── phase3_results_v3_dashboard.html   # v3 HTML dashboard (latest)
├── screenshots/         # v1 Grafana PNGs (126 files)
├── screenshots_v2/      # v2 URL manifests only
├── screenshots_v3/      # v3 URL manifests only
├── k8s/
│   ├── worker-strategy-patch.yaml
│   └── grafana-image-renderer-values.yaml
└── scripts/
    ├── setup-cluster.sh    # Full provisioning (TF + K8s + Helm)
    ├── teardown-cluster.sh # Destroy with confirmation
    ├── deploy.sh           # Patch worker image + strategy
    ├── port-forward.sh     # API 8080, Grafana 3000, RabbitMQ 15672
    ├── build_and_push.sh   # Docker build + ECR push
    ├── rollback.sh         # Revert to Phase 1 worker
    └── resume-cluster.sh   # Scale from 0 (legacy, use setup-cluster.sh for Phase 3 cluster)
```

## Resume Instructions

```bash
# 1. Continue with the existing Phase 3 cluster (cheapest):
bash phase3/scripts/setup-cluster.sh  # or just scale nodegroup back to 2

# 2. Redesign the worker with local tracker
# Edit: phase3/app/worker/worker.py
# - Add AdmissionTracker class
# - Replace /metrics polling with tracker calls in threshold + per_request
# - Add decode-throughput strategy as 4th option

# 3. Rebuild and push
bash phase3/scripts/build_and_push.sh

# 4. Try a bigger model (check if 14B AWQ fits on T4 first)
# Edit vLLM args in the K8s deployment or deploy.sh

# 5. Re-run experiments
bash phase3/scripts/deploy.sh static
python3 phase3/tests/backpressure_comparison.py --host http://localhost:8080 --strategy static --skip-switch ...

# 6. Switch back to Phase 4 cluster when done
aws eks update-kubeconfig --name inference-lab --region us-east-1
```
