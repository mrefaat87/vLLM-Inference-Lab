# Validation integration

How the sizing calculator and the empirical lab cross-reference each
other. The two flows are documented in `~/.claude/plans/i-m-building-a-web-lucky-cloud.md`
(diagrams 1–5); this file is the operator-facing tldr.

## Flow A: predictions seed every empirical run

```
exp run ... → calc_bridge → predictions/grid.json (preferred)
                       └─→ node compute_cli.mjs (off-grid fallback)
                       └─→ None (graceful degrade if neither is available)
                  ↓
RunResult JSON carries a `prediction` block with calc_version, data_hash,
b_crit, b_kv, and the predicted (batch, tps, step_ms) curve. The lab's
results explorer plots the predicted curve from this block — no calc.mjs
fetched at view time.
```

Manual loop:

```bash
# 1. Build the prediction grid (cheap; runs once per data-file change)
cd calculators/sizing_calc
node scripts/emit_predictions.mjs   # → predictions/grid.json (~5 MB)

# 2. Run a lab experiment (pre-flight consults the calc by default)
cd ../../experiments
exp run --engine mock --workload chatbot \
  --model-ref llama-3-70b --hw-ref A10G \
  --duration 30

# 3. Inspect the prediction snapshot baked into the result
jq '.prediction | {b_crit, b_kv, curve_pts: (.curve|length), calc_version}' \
  results/runs/*.json
```

## Flow B: lab runs overlay onto the calculator

```bash
# Emit the bridge dir at the same time the lab portal builds.
exp build-portal --calc-bridge ../calculators/sizing_calc/lab_runs/
```

The calculator's bundled `validation.mjs` fetches `./lab_runs/index.json`
and filters by the currently-selected (model.key, hw.key). Each matched
run projects onto the existing scope panels via Little's-Law batch
proxy:

```
batch_proxy ≈ analysis.throughput.requests_per_sec_avg × analysis.ttft_s.p50 + 1
y_throughput = analysis.throughput.tok_per_sec_avg
y_latency    = analysis.ttft_s.p50 (×1000 for ms)
```

The proxy is documented in the panel header — engines don't expose
batch directly, and queueing makes Little's Law an underestimate.

## Pre-flight modes

| Mode      | Default? | Behavior |
| --------- | -------- | -------- |
| `off`      |         | skip the calc entirely |
| `advisory`| ✓       | print `b_crit / b_kv / warnings`; never blocks |
| `strict`  |          | exit non-zero on error-level warnings |

```bash
exp run --preflight strict ...    # block on KV-fit failure
exp run --preflight off ...        # legacy mode (no calc consultation)
```

## Graceful degradation

The lab never fails because the calc is missing.

* No `predictions/grid.json` → live `node compute_cli.mjs`.
* No `node` on PATH → `prediction: null` in the result JSON, plus a
  `notes` line documenting the gap.
* Bridge dir missing or empty → calculator's Validation lookup
  surfaces an empty-state card.

Contract tests `test_calc_bridge.py::test_missing_node_returns_none`
and `validation.test.mjs::loadLabRuns: 404 → unavailable` enforce
these paths so a regression here breaks CI rather than the user's
session.

## Join keys

| Side | File | Key field |
| ---- | ---- | --------- |
| Calc | `src/data/models.json[].key`    | e.g. `llama-3-70b`, `deepseek-v3` |
| Calc | `src/data/hardware.json[].key`  | e.g. `A10G`, `H100-80GB` |
| Lab  | `RunResult.roofline_link.model_ref` | matches `models.json[].key` |
| Lab  | `RunResult.roofline_link.hw_ref`    | matches `hardware.json[].key` |

`experiments/tests/build/test_join_keys.py` enforces that the lab's
CLI defaults map to real calc keys — catches typo drift between the
two sides at build time.

## Bridge layout

```
calculators/sizing_calc/lab_runs/      ← written by `exp build-portal --calc-bridge`
├── index.json                          ← summaries (run_id, engine, model_ref, hw_ref, hrefs)
└── runs/<run_id>.json                  ← byte-identical copies of result JSONs (C2)
```

The lab portal *and* the calc both read this directory; the calc never
writes back. `.gitignore` excludes `lab_runs/*` except committed
fixtures so CI builds run on a clean clone.
