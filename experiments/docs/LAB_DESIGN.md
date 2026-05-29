# Empirical Inference Lab — Design

## Mental model

The lab is a fleet-of-one Auto Scaling experiment harness. A single "run" is one
(engine, model, hardware, workload, arrival-rate) tuple driven against a real
vLLM/SGLang/TRT-LLM pod on a Karpenter-scaled spot GPU node, capturing per-request
latency the way a load test against an ASG captures per-instance request stats.
Its purpose is to *falsify the sizing calculator*: the calc produces an analytical
throughput-vs-batch curve (the "predicted capacity plan"), the lab drops a measured
operating point on top of that curve, and the gap between them is the unit of
learning. Calc says "this fleet should sustain ~12 rps at 50 ms TBT"; the lab
says "the real engine ceilings at 7.4 rps because the INT8 kernel is slow." Same
relationship as a capacity model vs. a load test on an ASG.

---

## Section 1 — The calc ↔ lab feedback loop

The calc is the *predictor*. The lab is the *verifier*. They are joined by two
load-bearing keys (`model_ref`, `hw_ref`) and the predicted curve travels INTO
the result JSON, not alongside it.

```
                       ┌─────────────────────────────────┐
                       │   Sizing Calculator (browser)   │
                       │   calc.mjs::compute(inputs)     │
                       │                                 │
                       │   ┌── inputs ────────┐          │
                       │   │ model_key        │          │
                       │   │ hw_key           │          │
                       │   │ precisions       │          │
                       │   │ isl, osl, ngpus  │          │
                       │   │ tbt_ms target    │          │
                       │   └──────────────────┘          │
                       │           │                     │
                       │           ▼                     │
                       │   ┌── prediction ────┐          │
                       │   │ b_crit  b_slo    │          │
                       │   │ b_kv    y_max    │          │
                       │   │ curve[batch→tps] │          │
                       │   └──────────────────┘          │
                       └───────┬─────────────────────────┘
                               │  [ COPY EXP RUN ] button
                               │  emits an `exp launch …` command
                               │  with model_ref / hw_ref / quant / n-gpu
                               ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                       exp launch (CLI)                           │
   │                                                                  │
   │   1. CalcBridge.predict(pre-flight)   ← advisory KV-fit check    │
   │   2. SweepRunner.run_one                                         │
   │        → start engine pod                                        │
   │        → calibrate arrival rate                                  │
   │        → measurement window                                      │
   │        → CalcBridge.predict(snapshot, with measured ISL/OSL)     │
   │   3. Write RunResult JSON  (prediction block baked in)           │
   │   4. Rebuild static portal + bridge dir                          │
   │   5. Open browser at #<run_id>                                   │
   └─────────────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │             Results Explorer (static HTML, no JS calc)           │
   │                                                                  │
   │   Predicted curve   ── from prediction.curve  ───────────▶       │
   │                                                  ◇ measured       │
   │   Measured point    ── from analysis (tps, ttft, rate) ─▶        │
   │                                                                  │
   │   Validation panel on the calc side reads the bridge dir         │
   │   and pulls the SAME run JSON to overlay there too.              │
   └──────────────────────────────────────────────────────────────────┘
```

### Why the calc's prediction is snapshotted INTO the result JSON

The naive alternative is: store only `(model_ref, hw_ref)` in the result, and
re-run `calc.mjs::compute(...)` in the browser at view time. We rejected that.

```
  re-compute at view time          snapshot into result
  ───────────────────────         ──────────────────────
  pred = f(calc_today)            pred = f(calc_at_run_time)
  drifts as calc evolves          frozen, audit-able
  needs Node/JS at view           pure JSON, any static host
  hard to A/B calc versions       calc_version is a field
  prediction.inputs can drift     prediction.inputs are the
   from what was measured           measured ISL/OSL p50
```

This is the same reason an ASG capacity plan is a versioned artifact, not a live
re-query of CloudWatch every time you open the dashboard. Each run is a
falsifiable hypothesis carrying its own prediction; we can re-examine a 6-month
old run and the prediction in it is what we predicted then, not what we'd
predict today.

### Why the join keys (`model_ref`, `hw_ref`) are load-bearing

These are the *consistent-hash keys* between the calc's data tables and the
lab's runs. `model_ref` joins `RunResult` to a row in
`calculators/sizing_calc/src/data/models.json`; `hw_ref` joins to
`hardware.json`. Without them:

- the calc cannot find "all measured runs against H100" to plot on its
  Validation tab,
- the portal cannot pull `price_per_hour_usd` to derive measured $/Mtok,
- two engines run at different vLLM versions on the same hardware can't be
  compared against the same predicted curve.

They behave like target-group ARNs: opaque IDs that bind two independently
maintained systems.

---

## Section 2 — Inside `exp launch`

`exp launch` is the one-shot path from "click [COPY EXP RUN] in the calc" to
"browser tab open on measured-vs-predicted overlay". `exp run` is the same path
minus the portal-rebuild and browser-open steps.

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                         exp launch  (one shot)                         │
 └────────────────────────────────────────────────────────────────────────┘
   │
   │  CLI flags from the calc's [COPY EXP RUN]:
   │   --engine vllm --workload chatbot --rate auto
   │   --model meta-llama/Llama-3-70B-Instruct-AWQ --quant awq --tp 4
   │   --instance g5.12xlarge --gpu A10G --n-gpu 4
   │   --model-ref llama-3-70b --hw-ref A10G --tbt-target-ms 50
   │
   ▼
 ┌──────────────────────┐
 │  _do_run             │   1. _parse_rate("auto") → None  (calibrate)
 │  (exp.py)            │   2. build EngineConfig / WorkloadConfig
 │                      │   3. CalcBridge.predict(pre-flight)  ── advisory
 │                      │      → prints b_crit/b_slo/b_kv, warnings
 │                      │   4. SweepRunner.run_one(...)
 └──────────┬───────────┘
            │
            ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  SweepRunner.run_one  (sweep.py)                                    │
 │                                                                     │
 │   ManifestStore.write(PLANNED)                                      │
 │            │                                                        │
 │            ▼                                                        │
 │   driver.start(cfg)   ─────────────▶  K8s lifecycle (Section 4)     │
 │            │                                                        │
 │            ▼                                                        │
 │   _await_ready  (≤ 300s)            ─ engine /health 200            │
 │            │                                                        │
 │            ▼                                                        │
 │   ManifestStore.mark_running                                        │
 │            │                                                        │
 │            ▼                                                        │
 │   ┌─── if --rate auto ────┐    ┌─── if --rate explicit ───┐         │
 │   │ _calibrate (Section 3)│    │ explicit_calibration(r)  │         │
 │   │ probe sweep + bisect  │    │ (records uniform shape   │         │
 │   │ → Calibration block   │    │  with empty probes list) │         │
 │   │ → workload_cfg.rate = │    │                          │         │
 │   │   selected_rate       │    │                          │         │
 │   └───────────┬───────────┘    └────────────┬─────────────┘         │
 │               └────────────────┬────────────┘                       │
 │                                ▼                                    │
 │   run_loop(workload.requests(duration_s), loop_cfg)                 │
 │            │                                                        │
 │            │   For each Request:                                    │
 │            │     await sleep(arrival_offset_s)                      │
 │            │     POST /v1/chat/completions  (stream=True)           │
 │            │     parse SSE → ttft, tbt[], end_to_end                │
 │            ▼                                                        │
 │   analyze(records)            ─ ttft/tbt/e2e percentiles, ISL p50,  │
 │            │                    OSL p50, throughput, error counts   │
 │            ▼                                                        │
 │   _snapshot_prediction        ─ CalcBridge.predict(...)             │
 │     uses measured ISL/OSL       with measured isl/osl + hw price    │
 │     not pre-flight defaults     → Prediction block                  │
 │            │                                                        │
 │            ▼                                                        │
 │   RunResult(engine, model, hardware, workload, analysis,            │
 │             roofline_link, prediction, calibration, raw_results)    │
 │            │                                                        │
 │            ▼                                                        │
 │   _write_result → results/runs/<run_id>.json                        │
 │   ManifestStore.mark_done                                           │
 │            │                                                        │
 │            ▼                                                        │
 │   driver.stop()  (finally:)   ─ kubectl delete + port-forward kill  │
 └────────────────────────────────────────┬────────────────────────────┘
                                          │
                                          ▼
 ┌──────────────────────┐
 │  _do_build_portal    │   experiments/_site/* rebuilt
 │                      │   calculators/sizing_calc/lab_runs/* mirrored
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │  static http server  │   serves repo root on 127.0.0.1:8765
 │  reuses if running   │   (so /calculators/… and /experiments/_site/…
 │                      │    resolve from one origin)
 └──────────┬───────────┘
            ▼
        webbrowser.open(url#run_id)
```

### Why pre-flight uses default ISL/OSL but the snapshot uses measured

Pre-flight runs BEFORE the engine starts, so the workload hasn't produced
anything yet. We feed it lab-chatbot medians (ISL=200, OSL=150) only to populate
the warnings panel ("KV won't fit", "compute-bound regime", etc.) — those
checks are insensitive to the exact ISL/OSL. The post-run snapshot uses the
ACTUAL median ISL/OSL the engine saw, so the predicted curve drawn on the
portal reflects the measured workload shape. Same idea as comparing actual
fleet utilization to the capacity plan you'd build for that observed traffic
mix, not the synthetic mix you assumed in planning.

---

## Section 3 — Auto-rate calibration

### Why auto-calibrate at all instead of trusting the calc's recommended rate

The calc's `recommended_batch / recommended_rate` is analytical: it assumes an
optimal kernel exists for every (model × quant × hw) combo. In practice:

- INT8 on T4 falls back to bitsandbytes runtime dequant → 5–10× slower than the
  analytical estimate.
- AWQ on a kernel that's missing the head-dim path → 2–3× slower.
- vLLM V1 engine flags toggle paged-attention kernels that change throughput by
  30–80% at the same batch.

The calc cannot know which slow path will trigger. Only the running engine knows
its real ceiling. ASG analogue: capacity plan says the instance should serve
1,000 rps based on benchmark CPU, but the actual deployment is hitting a
hot-lock at 600 — only a real load test discovers that.

### The probe sweep + bisection

```
  rate axis (rps), log-spaced
  ─────┬─────┬─────┬─────┬─────┬─────┬───────────────────▶
       1     2     4     8    16    32

  ┌────────── probe schedule (PROBE_SCHEDULE) ─────────┐
  │                                                    │
  │   probe(1)   ── 8s burst ──▶ records              │
  │   probe(2)   ── 8s burst ──▶ records              │
  │   probe(4)   ── 8s burst ──▶ records              │
  │   probe(8)   ── 8s burst ──▶ saturated! ──┐       │
  │                                            │       │
  │   bisect(6)  ── 8s burst ──▶ ok            │       │
  │                                            │       │
  │   last_safe = 6     first_sat = 8          │       │
  │   capacity_ceiling = 6                     │       │
  │   selected_rate    = 0.8 × 6 = 4.8 rps     │       │
  └────────────────────────────────────────────┘       │
                                                       │
  Total budget: DEFAULT_BUDGET_S = 60s                 │
  Per-probe:    PROBE_S         = 8s                   │
  Headroom:     HEADROOM        = 0.8                  │
```

### Why geometric (1,2,4,8,16,32) instead of linear

Linear would burn the budget at low rates that aren't saturating anything. The
ceiling is uncertain across two orders of magnitude (a 7B FP16 might do 30 rps,
a 70B AWQ might do 1.5 rps), so log-spaced probes find the right decade fast
and the bisection refines within it. Same logic as binary-searching for a
queue's break point instead of stepping load by 1% increments.

### Why two-of-three saturation, not any-of-three

```
  signal                              floor / trigger
  ──────────────────────────────────  ────────────────────────────
  1) completion lag                   completed/dispatched < 0.90
  2) TTFT p95                         > 2 × ttft_slo_ms
  3) achieved RPS                     < 0.85 × target_rate

  hard-fail short-circuit              error rate > 10%
```

Any single signal is noisy: cold caches inflate TTFT p95 on the first probe;
a stragglers-burst inflates lag without the engine actually being saturated;
achieved RPS can dip 20% on a slow nodeLocalDNS pause without the engine
caring. Two signals together is a much stronger statement — the engine is
simultaneously dropping completions AND latency-blown OR rate-shortfall. False
positives at any one signal are common; false positives at two simultaneously
are rare.

The hard-fail (>10% errors) is special-cased because at that error rate the
other signals are meaningless (e.g. success_rate would be 0, achieved_rps
would be 0, both auto-trip, but the underlying cause isn't saturation — it's
engine misconfig).

### Calibration timeline

```
   t=0     ─── engine.start(cfg)
                │
   t≈30s ──── /health 200                   (driver._await_ready exits)
                │
   t=30 ──── probe(1)   ── 8s
   t=38 ──── probe(2)   ── 8s
   t=46 ──── probe(4)   ── 8s
   t=54 ──── probe(8)   ── 8s  SATURATED
   t=62 ──── probe(6)   ── 8s  (bisection)
                │
   t=70 ──── measurement window starts at selected_rate=4.8 rps
                │
                │   workload.requests(duration_s)  e.g. 300s
                │
   t=370 ──── analyze + snapshot prediction + write JSON
```

Each probe's workload is built from a fresh `workload_factory(rate, seed)`
call so probe RNG/conversation state cannot bleed into the measurement window.
Probe seeds are offset by a large prime (`seed_offset=7919`) from the user's
seed to keep them deterministic but disjoint.

### Why a fresh `workload_factory(rate, seed)` per probe

Workload generators carry state (conversation IDs, follow-up turn schedules,
RAG document mixes). Reusing one across rate changes would either (a) mix
probe state into the measurement window — non-reproducible — or (b) require
in-place rate mutation on every workload type, which is a constraint we
don't want to push onto custom workloads. A factory is the cheapest contract.

---

## Section 4 — K8s deployment shape

A single engine pod, scheduled by Karpenter on a spot GPU node, reached via
`kubectl port-forward` to the driver's host. The driver owns the lifecycle for
exactly one run.

```
 ┌────────────────────────────── Driver host ─────────────────────────────┐
 │                                                                        │
 │   exp run / exp launch                                                 │
 │      │                                                                 │
 │      ▼                                                                 │
 │   SweepRunner ──▶ VLLMDriver(K8sEngineDriver)                          │
 │                       │                                                │
 │                       │  render eks/manifests/engines/vllm.yaml        │
 │                       │  substitute {{MODEL}} {{QUANTIZATION}}         │
 │                       │            {{TENSOR_PARALLEL}} {{N_GPU}}       │
 │                       │            {{INSTANCE_FAMILY}}                 │
 │                       │            {{INSTANCE_SIZE}} {{GPU_NAME}}      │
 │                       │            {{CPU_REQUEST}} {{MEMORY_*}}        │
 │                       │            {{AWS_ACCOUNT}}                     │
 │                       ▼                                                │
 │                  kubectl apply -f -                                    │
 │                       │                                                │
 │   driver.run_loop ◀── kubectl port-forward                             │
 │       POST /v1/chat       svc/vllm-engine 8000:8000                    │
 │       (OpenAI SSE)                │                                    │
 │                                   │                                    │
 └───────────────────────────────────│────────────────────────────────────┘
                                     │
                                     ▼  (kube API)
 ┌────────────────────────── EKS control plane ───────────────────────────┐
 │                                                                        │
 │  ┌── Deployment: vllm-engine ──┐                                       │
 │  │  affinity:                  │                                       │
 │  │    node-affinity:           │                                       │
 │  │      instance-family = g5   │                                       │
 │  │      instance-size = 12xl   │                                       │
 │  │      instance-gpu = a10g    │                                       │
 │  │  resources:                 │                                       │
 │  │    nvidia.com/gpu: 4        │                                       │
 │  │    cpu/memory request       │                                       │
 │  └──────────────┬──────────────┘                                       │
 │                 │ unschedulable                                        │
 │                 ▼                                                      │
 │  ┌── Karpenter NodePool ───────────────────────────────────────────┐   │
 │  │   provisions a g5.12xlarge spot instance                        │   │
 │  │   labels: instance-family=g5, instance-size=12xlarge,           │   │
 │  │           instance-gpu-name=a10g                                │   │
 │  │   ─────────── (Auto Scaling analogue:) ────────────────         │   │
 │  │   This is a self-driving ASG: the NodePool is the launch        │   │
 │  │   template + scaling policy in one, and the unschedulable       │   │
 │  │   pod is the scale-out trigger.                                 │   │
 │  └──────────────┬──────────────────────────────────────────────────┘   │
 │                 │                                                      │
 └─────────────────│──────────────────────────────────────────────────────┘
                   ▼
 ┌──────────────────────────── Spot g5.12xlarge ──────────────────────────┐
 │                                                                        │
 │   nvidia-device-plugin → exposes 4× A10G                               │
 │                                                                        │
 │   ┌─ Pod: vllm-engine ──────────────────────────────────────────────┐  │
 │   │                                                                 │  │
 │   │  initContainer: s3 sync weights → emptyDir                      │  │
 │   │   (cluster-local cache; AMI prebake is the warm-pool analogue)  │  │
 │   │                                                                 │  │
 │   │  container: vllm/vllm-openai:latest                             │  │
 │   │    args: --model, --tensor-parallel-size, --quantization, ...   │  │
 │   │    exposes :8000  /v1/chat/completions  (OpenAI compat)         │  │
 │   │                   /health                                       │  │
 │   │                   /metrics  (Prometheus)                        │  │
 │   └─────────────────────┬───────────────────────────────────────────┘  │
 │                         │                                              │
 │                         ▼                                              │
 │                   Service: vllm-engine :8000                           │
 │                                                                        │
 └────────────────────────────────────────────────────────────────────────┘
```

### What the K8sEngineDriver actually verifies after apply

1. `kubectl wait deployment ... --for=condition=Available` (retried up to
   `readiness_timeout_s` = 1200s — long because cold pod = weights download +
   model load + CUDA graph capture).
2. `_verify_realized_instance` reads `pod.spec.nodeName` then
   `node.metadata.labels["node.kubernetes.io/instance-type"]` and warns LOUDLY
   if Karpenter parked the pod on a bigger/different instance than requested.
   Without this the user only finds out on the AWS bill — same failure mode as
   an ASG silently honoring a launch-template override.
3. `port-forward` (held as a subprocess on the driver) so the driver-loop hits
   `http://127.0.0.1:8000/v1/chat/completions`. The forward is killed in
   `stop()`'s `finally:` block — leaking it means the next run can't bind 8000.

### Why one pod per run, not a scale-out

A run is a single (engine, workload, rate) tuple. Multiple replicas would (a)
fight for the same GPU quota, (b) confound the calibration probe (which rate
ceiling? the per-replica or the aggregate?), (c) require a load-balancer that
isn't relevant to the question we're asking. The lab measures *engine
throughput at a given operating point*, not *fleet throughput under routing*.
Routing/scaling is the *next* phase (Phase 5+), measured separately.

---

## Section 5 — Result JSON schema

The on-disk shape is the single source of truth. Portal chips, calc Validation
panel, and any external consumer all read this and only this.

```
RunResult                              v1.2.0
├── schema_version
├── run_id                              "20260527T193011Z-vllm-chatbot-a1b2c3"
├── started_at / finished_at
├── engine        : EngineConfig         (engine name, image, model, quant, tp,
│                                          instance, gpu, n_gpu, max_model_len)
├── model         : ModelSpec            (name, quant, tp)
├── hardware      : HardwareSpec         (instance, gpu, n_gpu)
├── workload      : WorkloadConfig       (name, rate_rps, duration_s, seed, …)
├── analysis      : Analysis             (see below)
├── roofline_link : RooflineLink         { model_ref, hw_ref }    ← join keys
├── prediction    : Prediction | null    ← snapshot of calc.compute(...)
├── calibration   : Calibration | null   ← probe sweep that picked rate
├── raw_results   : list[RequestRecord]
├── engine_metrics: dict                  (Prometheus scrape at end of run)
└── notes
```

Abbreviated `Analysis`:

```
Analysis
├── steady_state_requests / failed_requests
├── ttft_s        : Percentiles { p50, p95, p99, mean, n }
├── tbt_s         : Percentiles
├── e2e_s         : Percentiles
├── throughput    : { total_completion_tokens, total_prompt_tokens,
│                     tok_per_sec_avg, requests_per_sec_avg }
├── gpu_util_pct_avg
├── kv_used_frac_p95
├── isl_tokens_p50      ← fed back into Prediction.inputs.isl
└── osl_tokens_p50      ← fed back into Prediction.inputs.osl
```

Abbreviated `Prediction` (the falsifiable-hypothesis block):

```
Prediction
├── calc_version          ← git SHA of calc.mjs (so old runs can be re-played)
├── data_hash             ← sha256 of models.json + hardware.json
├── inputs : PredictionInputs
│   ├── model_key / hw_key         (== RunResult.roofline_link.model_ref/hw_ref)
│   ├── weight_prec / kv_prec / act_prec
│   ├── isl / osl                  (= Analysis.isl_tokens_p50 / osl_tokens_p50)
│   ├── ngpus
│   ├── tbt_ms                     (user's --tbt-target-ms)
│   └── price_per_hour_usd         (looked up from calc hardware.json)
├── b_crit / b_slo / b_kv          ← regime boundaries
├── recommended_batch / y_max
├── curve : list[{ batch, step_ms, tps, cost_per_mtok }]
│                                  ← what the portal draws
├── warnings : list[{ level, msg }]
└── unavailable_reason             ← set when bridge couldn't reach calc
```

Abbreviated `Calibration` (always present on v1.2.0+):

```
Calibration
├── method            "auto" | "explicit"
├── probes : list[ CalibrationProbe {
│       rate, success_rate, ttft_p95_ms, achieved_rps, saturated
│   } ]
├── selected_rate     ← what the measurement window actually ran at
└── capacity_ceiling  ← highest non-saturated rate observed
```

### Why both `selected_rate` and `capacity_ceiling`

`selected_rate` is what the measurement window ran at (0.8 × ceiling). Reading
just that loses the headroom you applied. `capacity_ceiling` is the engine's
unloaded-ASG-instance-count equivalent: "this is how much it can take before it
breaks." Without both, you can't tell whether a 4.8-rps measurement was at 80%
of a 6-rps ceiling (healthy headroom) or at 95% of a 5-rps ceiling (right on
the edge — small jitter would have tripped saturation).

### Why `Calibration` is always written, even for explicit `--rate`

Uniform shape downstream. The portal, the calc Validation panel, and any test
fixture only have to handle one schema: `method == "explicit"` means
`probes == []` and `selected_rate == capacity_ceiling == --rate`. Without
this, every reader has to branch on field presence.

---

## Section 6 — User-facing surfaces

Four CLI subcommands, plus the portal HTML.

```
 ┌──────────────────┬──────────────────────────────────────────────────┐
 │ exp run          │ Run one experiment. Produces                     │
 │                  │   results/runs/<run_id>.json                     │
 │                  │   results/manifests/<run_id>.json                │
 │                  │ No portal rebuild, no browser. Use in scripts.   │
 ├──────────────────┼──────────────────────────────────────────────────┤
 │ exp launch       │ run + build-portal + open browser at #run_id.    │
 │                  │ Designed for [COPY EXP RUN] paste-and-go from    │
 │                  │ the calc.                                        │
 ├──────────────────┼──────────────────────────────────────────────────┤
 │ exp serve        │ Long-running static server + watchdog-driven     │
 │                  │ rebuild-on-change. Run once per session in a     │
 │                  │ side terminal; subsequent `exp launch` calls     │
 │                  │ detect the port is in use and reuse it.          │
 ├──────────────────┼──────────────────────────────────────────────────┤
 │ exp build-portal │ Rebuild _site/ from results/. Plus an optional   │
 │                  │ --calc-bridge dir that mirrors the same JSONs    │
 │                  │ where the calc's Validation panel looks for them │
 │                  │ (default: ../calculators/sizing_calc/lab_runs).  │
 ├──────────────────┼──────────────────────────────────────────────────┤
 │ exp plan         │ Read calc prediction, emit a suggested run grid  │
 │                  │ of (batch_target, rate_rps, predicted_tps) rows  │
 │                  │ centered on b_crit. Currently advisory output.   │
 ├──────────────────┼──────────────────────────────────────────────────┤
 │ exp list         │ Print all manifests with status. Cheap.          │
 └──────────────────┴──────────────────────────────────────────────────┘
```

### Why the portal stays static (no calc.mjs at view time)

The portal is `_site/results_explorer.html` + `assets/runs.json` + the per-run
JSON blobs. No backend, no Node, no JS-side calc. Three reasons:

1. **Auditability.** A prediction shown on the portal today is byte-identical
   to the one written into the result JSON the day the run completed. There is
   no path by which "what the chart shows" can drift from "what was hypothesized."
2. **Deployability.** Any S3 + CloudFront / GitHub Pages can serve it. No
   `calc.mjs` to invoke in the browser, no version mismatch between the calc
   the portal loaded and the calc the run was launched against.
3. **Two-system independence.** The calc owns the prediction. The lab owns the
   measurement. The portal owns the overlay. None of them require the other
   to be running. The bridge dir (calc-side) and the result JSON (lab-side)
   are the only contract.

### Why `exp launch` serves the repo root, not `_site/`

The portal has a bench-switcher button that links to
`/calculators/sizing_calculator.html` (calc) and back. If the server were rooted
at `_site/`, the calc URL would 404 — calc HTML lives outside that subtree.
Rerooting at the repo root makes both `/calculators/sizing_calculator.html`
and `/experiments/_site/results_explorer.html` resolve from a single port. One
origin, no CORS, one bookmark.

### Why `exp launch` reuses an existing port-8765 server

If `exp serve` is already up in a side terminal, the user is iterating: run,
look, tweak, run, look, tweak. Spawning a second `http.server` on the same
port would either fail to bind or fight the existing one. Detecting "port
already in use" and assuming it's the user's serve session means the iteration
loop stays smooth — fresh JSONs land in the watched dir, the existing server
rebuilds, the browser auto-reloads on the next click.

---

## Section 7 — Concept map (AWS / inference)

For reference when reading the code:

```
   AWS / distributed-systems concept         Inference mechanism in this lab
   ─────────────────────────────────         ───────────────────────────────
   Capacity plan / sizing model              calc prediction (b_crit, curve)
   Load test against an ASG                  exp run measurement window
   Synthetic canary / warmup                 PROBE_S burst at probe rate
   ASG cooldown                              measurement starts AFTER calibration
   Launch template + scaling policy          Karpenter NodePool
   Unhealthy host → ELB drains it            engine /health fail → run aborts
   Tier-0 availability SLA                   ttft_slo_ms in calibration rule
   CloudWatch p99 latency                    Analysis.ttft_s.p99
   Warm instance pool                        KV cache
   OS virtual-memory paging                  PagedAttention
   Charter bus (wait until full)             static batching
   City bus (continuous boarding)            continuous batching
   Multi-AZ sharding                         tensor parallelism
   Consistent-hash keys                      (model_ref, hw_ref) join keys
   Versioned capacity plan artifact          Prediction block in RunResult
```

---

## Section 8 — Failure modes and degraded paths

These are intentional design choices, not bugs.

```
   Condition                              Lab behavior
   ─────────────────────────────────      ──────────────────────────────────
   Calc tree not present                  prediction = null, run completes,
                                          portal shows "no prediction" banner
   Node not in PATH                       grid-only lookup; if grid miss,
                                          prediction = null with reason
   Even probe(1.0) saturated              selected_rate = FALLBACK_RATE_RPS
                                          (0.5), capacity_ceiling = 1.0,
                                          measurement still runs
   Engine never becomes Available         TimeoutError after 1200s,
                                          manifest marked FAILED with error
   Karpenter parks on wrong instance      LOUD warning in stderr;
                                          run continues (data still useful)
   port-forward dies mid-run              run_loop sees ClientError;
                                          per-request error recorded; analysis
                                          surfaces it as failed_requests
   AWS STS unreachable                    AWS_ACCOUNT template var = "";
                                          template must guard (vllm.yaml does)
   watchdog not installed                 exp serve runs without auto-rebuild;
                                          warns user; static serve still works
```

The throughline: the lab degrades gracefully toward "still write a result; let
the analysis stage tell the user what happened" instead of aborting. An ASG
that bricks the whole fleet on a single unhealthy host is worse than one that
marks it unhealthy and keeps serving. Same pattern.

---

## Section 9 — What this lab is NOT

- **Not a benchmark suite.** No "this engine beats that engine at X." A run is
  an observation; comparisons across runs are the user's job (the portal
  helps, but does not editorialize).
- **Not a production load tester.** Single pod, single run, no traffic
  shaping, no SLO enforcement. Phase 7 will add a gateway in front.
- **Not a continuous integration target for the calc.** The calc is verified
  by unit tests and by the Validation panel's cross-check against archived
  runs — not by every PR running `exp launch` (that requires a real GPU).
- **Not a scheduler.** One run at a time, serial. Parallel runs would
  contaminate each other's GPU and corrupt the calibration ceiling.

The lab does one thing: take an analytical capacity hypothesis, drive a real
engine until it tells us whether the hypothesis was right, and write the
result down in a format the next person (including future-Mohamed) can
audit without re-running anything.
