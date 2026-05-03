# Phase 4.5 — Dynamo Spike Design

**Status:** Locked design (pre-implementation)
**Date:** 2026-05-01
**Companion docs:** `research.md` (architecture background), plan at `~/.claude/plans/joyful-dreaming-hummingbird.md`

## Why this phase exists

Anthropic Cloud Inference EM interview is timed before the original Phase 8 (where Dynamo was scheduled). Phase 4.5 is a 3-day, time-boxed spike inserted between Phase 4 (KEDA composite KV autoscaling) and Phase 5 (smart routing). Goal: interview-grade conceptual depth backed by real measurements on AWS, not a platform rebuild.

We copy the Phase 4 stack (now torn down) into a new isolated cluster `inference-phase4.5` rather than mutating Phase 4 in place — keeps Phase 4 as a reference baseline and lets Phase 5+ resume cleanly.

## A. Infrastructure

- **Cluster:** new EKS `inference-phase4.5`, copied from `phase4/terraform/`, renamed throughout
- **VPC:** new VPC, two AZs for control plane (EKS requires ≥2), GPU NodePool constrained to us-east-1a only
- **GPU fleet:** **4× g4dn.xlarge spot** (4× T4 16 GB, 4 vCPU each). T4 pivot 2026-05-02 night after EFA-equipped fleets hit spot quota walls and on-demand busted budget. Same instance type Phase 2-3 used; known-good. ~$0.16/hr/node × 4 × 72h ≈ **$46 budget**. No EFA — NIXL falls back to TCP for cross-node KV transfer. Trade-off accepted: we lose the EFA performance story but keep the spike on rails and within budget.
- **EFA / placement group:** dropped. EFA isn't critical for "going through the motions of trying Dynamo," and g4dn.xlarge doesn't support EFA anyway.
- **EFA:** device plugin installed on EFA-labeled NodePool
- **Storage:** model weights pre-staged to S3, surfaced via EFS for steady-state experiments; etcd/MinIO/PostgreSQL on ephemeral PVCs
- **Cluster guard:** PreToolUse hook on `.phase-cluster=inference-phase4.5` (don't bypass)

## B. Software

**Carried from Phase 4 (copy as-is):**
- Karpenter v0.37.0
- NVIDIA GPU Operator + DCGM
- Prometheus + Grafana
- ECR pull-through cache

**New for Dynamo:**
- etcd, NATS, PostgreSQL, MinIO (Helm)
- Dynamo Operator + CRDs (DynamoGraphDeployment, DynamoGraphDeploymentRequest)
- Dynamo vLLM runtime image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.x` — different from Phase 2–4 `vllm/vllm-openai`; bundles vLLM-as-library + Rust worker harness + NIXL/UCX
- ModelExpress (for experiment D)
- Speculative decoding config (for experiment E) — Qwen2.5-0.5B draft

**Installed but inactive:**
- KEDA — no ScaledObjects on Dynamo workloads (static replicas during spike)
- RabbitMQ — Dynamo uses NATS as a control-plane event bus, not a data-plane request queue; RabbitMQ stays for Phase 7 gateway work

**Out of scope:**
KVBM, SLA Planner + AIPerf, multi-node TP, TRT-LLM/SGLang backends, Grove, multi-modal, DGDR auto-profiling, multi-model serving, production hardening (mTLS/RBAC/netpol), KEDA replacement, AMI rebake, approximate KV routing.

## C. Cluster configuration

| Concern | Setting | Rationale |
|---|---|---|
| Pod autoscaling | Static replica counts (KEDA installed but inactive) | Without Dynamo Planner (out of scope), autoscaling adds noise to per-experiment signal |
| Admission control | None on cluster; rate-limit at load tester | Dynamo has no built-in admission; Phase 3 gateway is Phase 7 territory |
| Cold start (all experiments) | Pre-warmed nodes, pre-staged weights, pre-pulled image | Eliminate cold-start variance; weight-load method is the variable in D, not image pull |
| AMI strategy | No rebake; rely on ECR pull-through cache | Static replicas, no per-pod cold start matters; Phase 4.1 lessons say AMI rebake takes long for marginal gain |
| Observability | Prometheus + Grafana + DCGM (copied) + new Dynamo panels | Reuse |

## D. Models

- **Qwen2.5-7B-Instruct** — primary target (continuity with Phase 2–4 baselines)
- **Qwen2.5-0.5B-Instruct** — draft model for experiment E only

## E. Workloads

| ID | Name | Shape | What it stresses |
|---|---|---|---|
| W1 | Random short | 50–500 token in, ~100 token out, no shared prefix | Control — low cache reuse, balanced P/D |
| W2 | Shared-prefix RAG | 2KB fixed system prompt + 200–500 token user msg, ~200 token out | Cache reuse — KV routing's home turf |
| W3 | Bursty long-prompt mix | 80% W1 + 20% 8K-input × 100-output | Head-of-line blocking on TBT — disaggregation's home turf |
| W4 | Cold-start probe | Single request immediately after pod start | Weight loading time |

## F. Experiments

**Design rule:** A is the baseline and runs against all four workloads. Every later experiment changes exactly one variable from A and reuses the relevant subset of workloads, so the delta is attributable.

| ID | Variable changed from A | Workloads | Setup |
|---|---|---|---|
| **A — Baseline** | (none) | **W1, W2, W3, W4** | Co-located 4 vLLM workers, round-robin routing, no spec decode |
| B — KV-aware routing | Router mode: round-robin → KV-aware | W1, W2, W3 | Otherwise identical to A |
| C — Disaggregation | Topology: 4 co-located → 2 prefill + 2 decode workers, NIXL on | W1, W2, W3 | Round-robin routing kept (so this isolates topology from routing) |
| D — ModelExpress | Weight-load method: Phase 4.1 streamer → ModelExpress | W4 | Otherwise identical to A |
| E — Speculative decoding | Spec decode: off → on (Qwen2.5-0.5B draft + 7B target) | W1, W2 | Otherwise identical to A |

**Notes:**
- A vs B isolates router. A vs C isolates topology. B+C combined ("full Dynamo") is implied by deltas — not run separately unless time permits.
- W4 only appears in A and D — it's a cold-start measurement; routing/topology don't apply.
- Hypotheses are kept directional, not numeric. Results are evaluated against questions, not fabricated targets.

**Question each experiment answers:**

| Experiment | Question |
|---|---|
| A | Baseline TTFT/ITL/throughput on this hardware. Does Dynamo-without-features match Phase 2 vanilla vLLM ±noise? |
| B | Does KV-aware routing reduce TTFT on prefix-heavy workload (W2) without hurting random workload (W1)? By how much? |
| C | Does disaggregation stabilize ITL under bursty long-prompt arrivals (W3) vs co-located baseline? What's NIXL transfer cost on AWS? |
| D | Does ModelExpress beat Phase 4.1's 212s streamer winner on AWS, where NIXL falls back to UCX/EFA/TCP? |
| E | Does speculative decoding improve throughput at fixed TTFT? What's the acceptance rate per workload? |

**Acceptance criteria for each experiment to be considered *valid*** (not for hypothesis to be right):
- Load tester completes the planned ramp without saturating itself (load-tester CPU < 70%)
- All workers stay healthy (no OOM, no crashes) for the duration
- Metrics captured: TTFT p50/p99, ITL p50/p99, throughput, GPU util, plus experiment-specific (router cache hit rate / NIXL transfer histogram / acceptance rate)
- Result JSON written; never overwritten on re-run (per `feedback_sweep_data_preservation.md`)

## G. Observability

**Reused:** Phase 2 Grafana dashboards (GPU util, throughput, TTFT/ITL, KV cache util), Phase 3 latency exporter, DCGM.

**New panels:**
- Dynamo router cache hit rate
- Per-pool replica counts (prefill vs decode)
- NIXL transfer latency histogram
- Router decision distribution per worker
- ModelExpress weight transfer time
- Speculative decoding acceptance rate

**Per-experiment deliverable:** result JSON in Phase 3 format + dashboard snapshot + paragraph in `learnings.md`.

## H. Operational

- 3 working days hard cap
- Cost ceiling: $100 hard cap, $50 target (4× g4dn.xlarge spot × 72h ≈ $0.64/hr ≈ $46)
- Cluster created day 1, torn down end of day 3
- Sweep data: separate JSON per experiment×workload, never overwrite
- No unilateral scope changes (per `feedback_no_unilateral_scope_changes.md`)

## I. Repo layout

```
phase4.5/
├── research.md
├── design.md                 # this file
├── terraform/                # copied from phase4/, renamed
├── packer/                   # copied for reference; not rebaking
├── k8s/
│   ├── dynamo-operator/      # Helm values
│   ├── deps/                 # etcd, NATS, postgres, minio
│   ├── efa-device-plugin/
│   ├── monitoring/
│   │   └── dynamo-dashboard.json
│   ├── dgd-A-baseline.yaml
│   ├── dgd-B-kvrouter.yaml
│   ├── dgd-C-disagg.yaml
│   ├── dgd-D-modelexpress.yaml
│   └── dgd-E-specdecode.yaml
├── scripts/
│   ├── load-test.py          # adapted from phase3
│   ├── workloads/            # W1-W4 generators
│   └── run-experiment.sh
├── tests/
│   ├── exp-A-W1-results.json
│   ├── exp-A-W2-results.json
│   ├── ... (one JSON per experiment×workload)
└── learnings.md              # written at end
```

## J. Drop-priority on time pressure

If running long, drop in this order: ~~F (already dropped)~~ → E → D → C's optional sub-runs → B's full sweep. **Never drop A** — it's the baseline.
