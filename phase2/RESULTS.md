# Phase 2: Observability + Autoscaling — Test Results

## Test Environment
- **EKS 1.29**, us-east-1, Karpenter v0.34.0
- **GPU**: g4dn.xlarge Spot (NVIDIA T4 16GB) — $0.16-0.22/hr
- **Model**: Qwen/Qwen2.5-7B-Instruct-AWQ (4-bit, ~4GB VRAM)
- **Config**: `--enforce-eager`, `--gpu-memory-utilization 0.85`, `prefetch_count=5`

---

## Test 1: Baseline — N=5 Concurrent (1 Worker)

Validates single-worker behavior matches Phase 1 results.

| Metric | Short | Medium | Long |
|--------|-------|--------|------|
| Count | 2 | 2 | 1 |
| Queue wait max | 2.6ms | 4.8ms | 4.9ms |
| Client TTFT p50 | 666ms | 636ms | 634ms |
| Throughput avg | 33.2 tok/s | 32.7 tok/s | 33.3 tok/s |

**Verdict**: Matches Phase 1 (queue wait <5ms, ~33 tok/s). Queue adds negligible overhead at N=5.

---

## Test 2: Queue Saturation — N=15 Concurrent (1 Worker)

Proves the scaling signal: 15 concurrent requests with only 5 prefetch slots = queue buildup.

| Metric | Short | Medium | Long |
|--------|-------|--------|------|
| Count | 5 | 5 | 5 |
| Queue wait p50 | 5.2ms | 846ms | 5,957ms |
| Queue wait max | **16,798ms** | **13,089ms** | **7,001ms** |
| Client TTFT p50 | 542ms | 1,619ms | 6,623ms |
| Client TTFT max | 17,111ms | 13,449ms | 7,669ms |
| Throughput avg | 32.7 tok/s | 27.9 tok/s | 28.7 tok/s |

**Verdict**: Max queue wait **16.8 seconds**. Requests 6-15 wait 1-17s for a worker slot. This is the autoscaling signal — same pattern as Phase 1 (12s max).

---

## Test 3: Autoscaled — N=15 Concurrent (2 Workers)

After KEDA scaled to 2 workers (2x GPU nodes), run the same N=15 test.

| Metric | Short | Medium | Long |
|--------|-------|--------|------|
| Count | 5 | 5 | 5 |
| Queue wait p50 | 6.0ms | 824ms | 5.6ms |
| Queue wait max | **824ms** | **1,722ms** | **824ms** |
| Client TTFT p50 | 595ms | 1,395ms | 713ms |
| Throughput avg | 33.3 tok/s | 30.7 tok/s | 31.5 tok/s |

**Verdict**: Max queue wait **1,722ms** (down from 16,798ms = **90% reduction**). Target of <2s met.

---

## Comparison Summary

| Metric | 1 Worker | 2 Workers | Improvement |
|--------|----------|-----------|-------------|
| Short queue_wait max | 16,798ms | 824ms | **95%** |
| Medium queue_wait max | 13,089ms | 1,722ms | **87%** |
| Long queue_wait max | 7,001ms | 824ms | **88%** |
| **Overall max queue_wait** | **16,798ms** | **1,722ms** | **90%** |
| Throughput per worker | ~33 tok/s | ~33 tok/s | Same |
| Total throughput | ~33 tok/s | ~66 tok/s | **2x** |

**Why it works**: With 1 worker (`prefetch_count=5`), only 5 requests process concurrently — the other 10 queue. With 2 workers (5 prefetch each = 10 slots), only 5 requests queue. Both workers drain the queue concurrently, halving wait times.

---

## GPU Scale-Up Delay Breakdown

The most important learning: GPU autoscaling has **~5 min total latency**.

| Phase | Duration | Cumulative |
|-------|----------|-----------|
| KEDA detects queue_depth > 5 | 0-15s | 15s |
| Karpenter selects cheapest Spot instance | ~15s | 30s |
| EC2 Spot instance launch + boot | ~60s | 90s |
| Node joins cluster (kubelet + CNI init) | ~10s | 100s |
| vLLM container image pull (~8GB) | ~90s | 190s |
| Model download from HuggingFace (4GB) | ~60s | 250s |
| Model load into GPU VRAM | ~50s | 300s |
| **Total: Pending → Ready** | | **~300s (5 min)** |

**70% of the delay is model-related** (image pull + download + GPU load). Infrastructure is only 30%.

**Production optimization paths**:
1. Pre-baked AMI with vLLM image → saves 90s
2. S3 model cache with init container → saves 60s
3. Warm standby node pool → reduces to ~10s
4. All combined → **300s → ~60s**

---

## Grafana Dashboard Metrics Under Load

Captured during N=20 sustained heavy inference:

| Panel | Value | Interpretation |
|-------|-------|---------------|
| GPU Compute Utilization | **95%** | SMs near-fully saturated (prefill + decode) |
| Memory Bandwidth Utilization | **94%** | Near T4's 320 GB/s limit — confirms decode is BW-bound |
| VRAM Usage | **~14GB / 16GB** | 4GB model + ~10GB KV cache + overhead |
| GPU Temperature | **46C** | Well below T4's 83C thermal throttle threshold |
| KV Cache Usage | **1.66%** | Low because AWQ 7B model uses only ~200MB KV per request |
| Requests Running | **5** | At prefetch_count limit (continuous batching saturated) |
| Throughput | **118 tok/s** | Peak batched throughput (vs 33 tok/s single-request) |
| TTFT p50 | **~150ms** | Acceptable for interactive use |
| TTFT p99 | **~1s** | Under queue pressure; GPU prefill queuing |
| TBT p50 | **~47ms** | Smooth streaming quality |
| Request Rate | **0.8 req/s** | Peak ingest rate during test |
| Worker Replicas | **Desired=2** | KEDA triggered scale-up from sustained load |

---

## Screenshots

| File | Description |
|------|-------------|
| `01-gpu-hardware-row.png` | Full dashboard at idle — all 4 rows visible |
| `02-dashboard-with-load.png` | Dashboard showing KV cache + throughput during N=8 burst |
| `03-dashboard-test-window.png` | Historical view covering all load test phases |
| `04-dashboard-n15-active.png` | N=15 burst active — TTFT spike, 37.6 tok/s |
| `05-dashboard-heavy-load.png` | Heavy load — 43C, KV 1.66%, 74.1 tok/s, 5 running |
| `06-dashboard-during-inference.png` | **Peak load — GPU 94%, MEM BW 92%, 118 tok/s, KEDA scaling** |
| `07-dashboard-post-burst.png` | Post-burst — GPU 95%, MEM BW 94%, 46C, Desired=2 Ready=1 |
