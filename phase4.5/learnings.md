# Phase 4.5 — NVIDIA Dynamo Hands-On Spike: Learnings

**Date:** 2026-05-02
**Cluster:** EKS `inference-phase4-5`, 4× g4dn.xlarge spot (T4 16 GB) in us-east-1a
**Duration:** ~1 day end-to-end execution (after a prior false start that hit AWS spot G/VT vCPU quota wall)
**Budget actual:** ~$10 spent (spot prices were higher than Phase 2-3 baseline; $0.235/hr × 4 nodes × ~5h)
**Goal:** Go through the motions of deploying and exercising NVIDIA Dynamo end-to-end on AWS commodity GPUs. Not a perf shootout.

## What actually got run

Three experiments, each with 3–4 workloads (~1000 total inference requests across all runs):

- **A — Baseline:** 4 vLLM workers, co-located prefill+decode, round-robin routing
- **B — KV-aware routing:** same as A except Frontend `--router-mode kv --router-kv-events`
- **C — Disaggregation:** 2 prefill + 2 decode workers, NIXL KV transfer over TCP (no EFA)
- **D (ModelExpress vs Phase 4.1 streamer)** — *skipped* per the plan's drop-priority list once A/B/C produced a clean disagg-without-EFA finding

Results are in `phase4.5/tests/exp-{A,B,C}-W{1,2,3,4}-results.json` (Phase 3-compatible JSON schema).

## Numbers that matter

| Experiment | Workload | TTFT p50 | TTFT p99 | TBT p50 | Throughput tok/s |
|---|---|---|---|---|---|
| A baseline | W1 random | 733 ms | 1003 ms | 230 ms | 166 |
| A baseline | W2 shared-prefix | 752 ms | 930 ms | 226 ms | 215 |
| A baseline | W3 bursty | 709 ms | 1009 ms | 225 ms | 71 |
| A baseline | W4 cold-start | 493 ms | 493 ms | 219 ms | 2 |
| B KV router | W1 | 735 ms | 1030 ms | 229 ms | 166 |
| B KV router | W2 | 757 ms | 932 ms | 233 ms | 227 |
| B KV router | W3 | 686 ms | **899 ms** | 224 ms | 71 |
| C disagg (TCP) | W1 | **1502 ms** | **1944 ms** | 238 ms | 166 |
| C disagg (TCP) | W2 | **1870 ms** | **2136 ms** | 238 ms | 217 |
| C disagg (TCP) | W3 | **1524 ms** | **1828 ms** | 240 ms | 71 |

(W3 had 16 errors in every run — its 8K-input long prompts exceed the 4096 max-model-len cap on T4. Workload-side issue, not Dynamo.)

## Findings

### 1. Disaggregation on AWS without EFA is a *deopt*, not an opt

This is the headline finding. **TTFT roughly doubles** when we split prefill and decode onto separate g4dn.xlarge nodes vs. running them co-located, across all three workload patterns. NIXL falls back to TCP for cross-node KV transfer when no RDMA fabric is available, and on g4dn.xlarge's commodity 25 Gbps networking the TCP transfer cost (estimated ~700–1000 ms per request from the delta) is bigger than any head-of-line blocking it's supposed to alleviate.

The interview talking point isn't "disagg is bad" — it's "**disagg's value is gated on RDMA-class networking, and most AWS GPU instances don't have it**." On bare-metal H100 clusters with InfiniBand, the same architecture would likely flip to a 2× *win*. We measured the cost of the missing fabric, which is exactly the AWS-specific story I wanted to be able to tell.

### 2. KV-aware routing on a 4-worker fleet is a near-noop for our workload mix

KV routing produced essentially identical TTFT to round-robin on W1 (random) and W2 (shared-prefix), with a modest p99 improvement on W3 (-110ms = 11% better). Three plausible reasons:

- 4 workers is too few — round-robin already lands the same request on the same worker fairly often, so the KV router has little optimization headroom.
- W2's 2KB shared prefix is small relative to total prefill cost (~700ms TTFT is dominated by user-message-specific prefill, not the cached prefix).
- vLLM's per-worker prefix caching (`enable_prefix_caching=True`, on by default) already captures most of the win once a worker has seen the prefix once.

Real win: bursty long-prompt workloads (W3 p99 dropped 11%). Real implication: **KV routing's payoff scales with fleet size and prefix-share-ratio**, neither of which were stressed in this 4-worker setup. A 32-worker fleet running RAG would likely show a much bigger delta.

### 3. NIXL works on AWS without specialized networking — just slowly

We were not sure ahead of time whether NIXL would even initialize without RDMA. It did. It picked the TCP transport automatically. KV transfers happened, and disaggregation produced correct outputs. **Dynamo is operationally portable to commodity GPU fleets**; it's just not a performance win there.

### 4. Operator quirks that ate the most time

- **Worker-suffix namespace mismatch (Frontend ↔ Worker discovery)**: the operator injects `DYN_NAMESPACE=dynamo-system-exp-X` on Frontend, but workers add a hash suffix (`DYN_NAMESPACE_WORKER_SUFFIX=<8 hex chars>`) and register under the full `dynamo-system-exp-X-<suffix>` Dynamo namespace. Frontend's `/v1/models` returned empty until we manually overrode `DYN_NAMESPACE` on Frontend to include the worker suffix. **The workaround is fragile** — every redeploy generates a new hash, requiring a re-patch. This feels like an operator bug or undocumented contract; would file an upstream issue if I were on the team.
- **Disagg requires explicit NIXL connector flag**: `--connector` flag is deprecated; new requirement is `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'`. Old example DGDs in the v1.0.2 source don't have this and crash on first prefill worker boot.
- **Stale ReplicaSet pile-up on rolling update**: when patching the DGD, the operator created a NEW deployment hash but didn't terminate the OLD one, leading to a 6-on-4 pod overcommit and Pending pods. Required `kubectl delete deploy <old-hash>` to break the deadlock.

### 5. Infrastructure surprises (AWS / EKS / Bitnami / chart drift)

- **AWS spot G/VT vCPU quota is per-account-per-region** and defaults to 32 vCPU on accounts that haven't requested an increase. 4× g5.8xlarge = 128 vCPU = blocked. We pivoted to g4dn.xlarge (16 vCPU total) to fit. If you don't pre-flight this, Karpenter's `MaxSpotInstanceCountExceeded` masquerades as a transient capacity event.
- **Bitnami images on Docker Hub are partially auth-walled / pruned post-2025**. The Dynamo platform chart depends on `bitnami/etcd:3.6.4` which 404s on fresh pulls. We replaced etcd/postgres/minio with plain manifests pointing at official images (`quay.io/coreos/etcd`, `postgres:alpine`, `minio/minio`).
- **Dynamo `main` chart references image tag (1.2.0) that doesn't exist on NGC**. Listed tags via NGC's unauth token endpoint; latest stable image is 1.0.2. Pinning to the v1.0.2 *git tag* of the chart aligned versions.
- **EKS 1.30 doesn't ship a `gp3` StorageClass by default** — only `gp2`. Created and set as default; otherwise the Bitnami values' `storageClass: gp3` fail with PVCs Pending.
- **Cluster-guard hook (`.claude/hooks/phase-cluster-guard.py`) didn't recognize helm's `--kube-context` flag** (only kubectl's `--context`). Patched the regex to recognize both — this was a reusable hook fix that survives the spike.
- **Networking blips during Terraform apply** (DNS lookup failures mid-create for `eks.us-east-1.amazonaws.com` and `iam.amazonaws.com`) left the EKS cluster ACTIVE but TF state inconsistent. Re-running `terraform apply` reconciled everything, except the orphan cluster created during the first failure had to be `aws eks delete-cluster`'d first because TF tried to create-not-update.

## What I'd say in the interview

1. **"I built and exercised Dynamo end-to-end on EKS with T4 GPUs in a few hours."** Schema-correct DGDs for co-located, KV-aware-routing, and disaggregated topologies; 4-worker fleet; Frontend, NATS, etcd, MinIO, Postgres deps; full Prometheus+Grafana observability with DCGM scraping.
2. **"I measured the cost of disaggregation without RDMA on AWS commodity instances: roughly 2× TTFT regression on T4 + TCP-NIXL."** Concrete numbers, three workload patterns, four T4 nodes. The result is what I'd predict from reading NIXL docs but it's empirically grounded, not handwave.
3. **"KV-aware routing's win scales with fleet size and prefix-share ratio. On a 4-worker fleet with our workload mix it was a near-noop, with a modest 11% p99 improvement on bursty long-prompt traffic."** I'd want to repeat this on a bigger fleet — it's the natural follow-up.
4. **"The operator's discovery-namespace contract has a sharp edge: Frontend's `DYN_NAMESPACE` doesn't include the worker hash suffix that workers append, so first-deploy `/v1/models` is empty until you patch."** I logged this as a finding; would file upstream if I worked there.
5. **"Multi-cluster spot quota contention: a stray Karpenter NodePool from an old phase was eating into our quota on a different cluster."** Pre-flight check: quota is per-account-per-region, not per-cluster. Tear down before sizing a new GPU fleet.
6. **"Bitnami's mid-2025 image-hosting change broke many Helm charts that hadn't pinned digests."** The Dynamo platform chart was one of many casualties; we switched to plain manifests with official images.
7. **"NIXL is portable to commodity AWS GPUs without specialized networking — it just falls back to TCP, which is slow but functional."** Means you can experiment with disagg architectures without paying for p4d/p5 hardware first.
8. **"vLLM's per-worker prefix caching covers most of the KV-routing win when prefixes are small."** This is actually a Dynamo-design implication: the router earns its keep on workloads where prefix is large *or* the fleet is too big for round-robin to randomly hit the warm worker.
9. **"Time-to-first-deployment-that-served-tokens was about 5 hours from `terraform apply` to first chat completion**, including all the operator quirks we ran into. Most of that was schema mismatches and namespace gotchas, not infrastructure provisioning. Production teams adopting Dynamo should expect a 2–3 day operability investment before benchmarks are meaningful."
10. **"Each round of running a different DGD took ~5 minutes for image pull + ~2 minutes for vLLM warmup, ~7 minutes between experiments."** That sets the operability budget for any A/B testing — which matters when planning iterative optimization work.

## Open / unresolved

- **NIXL EFA performance on AWS** — out of scope after the T4 pivot. Would need g5.8xlarge/g6.8xlarge (EFA-equipped) to measure properly. **Filed as candidate for a future Phase 4.6** — even half a day on EFA hardware would close the disagg-economics question.
- **KV router behavior at higher concurrency / fleet size** — measured at 4 workers and 1.5–2 req/s. Production-scale (32 workers, 50+ req/s) would likely show a larger router delta.
- **ModelExpress vs Phase 4.1 streamer head-to-head** — skipped (Experiment D). Would need a Dynamo ModelExpress server install + a parallel vanilla-vLLM Deployment for the streamer arm. Likely a half-day investment for a finding that's mostly about pull-time, not inference.
- **Speculative decoding on T4** — dropped during planning; 7B-AWQ + 0.5B + KV cache is too tight on 16 GB. A future spike on 24 GB+ GPUs (A10G, L4) would unlock E.

## Files

- `phase4.5/research.md` — pre-spike research on Dynamo architecture
- `phase4.5/design.md` — locked design (post-pivot to T4)
- `phase4.5/k8s/dgd-{A,B,C}-*.yaml` — DGD manifests for each experiment
- `phase4.5/k8s/dynamo-operator/values.yaml` — Helm values pinning to v1.0.2 platform chart
- `phase4.5/k8s/deps/{etcd,postgresql,minio}.yaml` — plain manifests bypassing broken Bitnami images
- `phase4.5/scripts/load-test.py` + `workloads/{w1-w4}_*.py` — load tester (Phase 3-compatible JSON schema)
- `phase4.5/tests/exp-*-results.json` — 10 result files (3 experiments × 3–4 workloads)

## One-line CV summary

> Built a Dynamo-on-EKS hands-on lab on AWS commodity T4 GPUs; measured prefill/decode disaggregation regressing TTFT by ~2× on TCP-fallback NIXL (vs. RDMA), and quantified KV-aware routing as a near-noop at 4-worker scale with modest p99 wins on bursty long-prompt traffic.
