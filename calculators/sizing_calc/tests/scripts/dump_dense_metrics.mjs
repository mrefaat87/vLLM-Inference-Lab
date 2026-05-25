// Regression-baseline dumper. Iterates every (hardware × DENSE-model × precision)
// combo, calls compute(), and serialises a deterministic JSON snapshot of the
// resulting metrics + a sampled chart curve. The MoE-correctness refactor must
// produce a byte-identical snapshot for every dense combo (proves the rename is
// purely structural — zero numerical drift on the dense path).
//
// Usage:
//   node calculators/sizing_calc/tests/scripts/dump_dense_metrics.mjs > out.json
//
// Determinism notes:
// - Numbers serialise via toJSON / JSON.stringify with Number.prototype.toString,
//   which is platform-deterministic for double-precision floats produced by
//   identical pure-function evaluation.
// - Keys are written in insertion order (V8 preserves this for string keys).
// - Models filtered to dense only (skip MoE) — by definition the MoE row is the
//   one that *should* shift after the refactor.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { compute } from "../../src/calc.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const hardware = JSON.parse(readFileSync(join(here, "../../src/data/hardware.json"))).hardware;
const models   = JSON.parse(readFileSync(join(here, "../../src/data/models.json"))).models;

// Dense filter: anything that doesn't carry MoE-only fields. Robust to either
// the current schema (no n_experts → dense) or post-refactor schema (n_experts
// + params_b_active < params_b_total → MoE). Today nothing has n_experts;
// post-refactor only DSv3 will.
const isDense = (m) => !m.n_experts && (m.params_b_active === undefined || m.params_b_active === m.params_b_total || m.params_b_active === m.params_b);

// Precision pairs that exercise every code path in bCrit/stepTime:
//   BF16/BF16  — baseline FP16 compute path
//   FP8/FP8    — FP8 tensor core path (on capable hardware)
//   INT8/BF16  — asymmetric, FP16 compute path with halved param-bytes
//   INT4/FP16  — sub-byte weight path
const PREC_PAIRS = [
  ["BF16", "BF16"],
  ["FP8",  "FP8"],
  ["INT8", "BF16"],
  ["INT4", "FP16"],
];

// Fixed workload — held constant so any drift must come from the formula side.
const ISL = 2048;
const OSL = 256;
const TBT_MS = 50;
const TTFT_MS = 1500;
const KV_PREC = "INT8";
const NGPUS_VALUES = [1, 8];  // single GPU + NVLink island; spans the bKv branches

// Chart curve sample points: B at decade-ish powers. Keeps the snapshot small
// while still pinning the full step-time curve shape.
const CURVE_B_SAMPLES = [1, 2, 4, 8, 16, 32, 64, 128, 256];

// Round to suppress sub-eV float noise that would make diffs flaky across
// machines. 1e-9 relative precision is far tighter than any real engineering
// decision but loose enough that compiler-level reordering can't bite.
const r = (x) => {
  if (typeof x !== "number" || !Number.isFinite(x)) return x;
  if (x === 0) return 0;
  const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(x))) - 9);
  return Math.round(x / mag) * mag;
};

// Walk every numeric leaf recursively and round. Preserves structure.
const roundDeep = (v) => {
  if (Array.isArray(v)) return v.map(roundDeep);
  if (v && typeof v === "object") {
    const out = {};
    for (const k of Object.keys(v)) out[k] = roundDeep(v[k]);
    return out;
  }
  return r(v);
};

const snapshot = { combos: [] };

for (const hw of hardware) {
  for (const model of models) {
    if (!isDense(model)) continue;
    for (const [wp, ap] of PREC_PAIRS) {
      for (const ngpus of NGPUS_VALUES) {
        const out = compute({
          hw, model,
          isl: ISL, osl: OSL,
          weight_prec: wp, kv_prec: KV_PREC, act_prec: ap,
          tbt_ms: TBT_MS, ttft_ms: TTFT_MS,
          ngpus,
        });
        // Sample the chart curve at fixed B values rather than dumping the full
        // log-spaced curve — full curve includes implementation-detail spacing
        // (Math.ceil(B * 1.2)) that's fine to evolve without it counting as a
        // semantic regression. The sampled B points are what users actually
        // read off the chart.
        const curve_samples = CURVE_B_SAMPLES
          .map((B) => out.chart?.curve?.find((p) => p.B === B) || null)
          .filter(Boolean)
          .map((p) => ({ B: p.B, step_ms: p.step_ms, tokens_per_sec: p.tokens_per_sec }));
        snapshot.combos.push({
          hw: hw.key,
          model: model.key,
          weight_prec: wp,
          act_prec: ap,
          ngpus,
          metrics: out.metrics,
          curve_samples,
          warnings: out.warnings?.map((w) => ({ level: w.level, msg: w.msg })) || [],
        });
      }
    }
  }
}

process.stdout.write(JSON.stringify(roundDeep(snapshot), null, 2) + "\n");
