// Dense regression guard. Loads a snapshot of compute() output captured before
// the MoE-correctness refactor (params_b → params_b_total + params_b_active),
// re-runs compute() against the current code, and asserts the dense combos are
// byte-identical. The refactor changes formulas only for MoE models; the dense
// path must produce identical numbers because for dense models params_b_total
// equals params_b_active (and equals the old params_b).
//
// If this test fails, the refactor introduced an unintended numerical change
// on the dense path. Fail loudly — re-derive the baseline only if the drift is
// deliberate AND reviewed.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { compute } from "../src/calc.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const baseline = JSON.parse(readFileSync(join(here, "fixtures/baseline_dense_metrics.json")));
const hardware = JSON.parse(readFileSync(join(here, "../src/data/hardware.json"))).hardware;
const models   = JSON.parse(readFileSync(join(here, "../src/data/models.json"))).models;

const HW    = Object.fromEntries(hardware.map((h) => [h.key, h]));
const MODEL = Object.fromEntries(models.map((m) => [m.key, m]));

// Truly-dense model keys. Hardcoded rather than derived because the schema for
// "what's MoE" is part of what's changing; this list is the stable contract.
const DENSE_KEYS = new Set([
  "llama-3-8b",
  "llama-3-70b",
  "qwen2.5-7b",
  "qwen2.5-14b",
  "mistral-7b",
]);

// Same rounding as dump_dense_metrics.mjs so re-derived numbers compare on the
// same precision floor.
const r = (x) => {
  if (typeof x !== "number" || !Number.isFinite(x)) return x;
  if (x === 0) return 0;
  const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(x))) - 9);
  return Math.round(x / mag) * mag;
};
const roundDeep = (v) => {
  if (Array.isArray(v)) return v.map(roundDeep);
  if (v && typeof v === "object") {
    const out = {};
    for (const k of Object.keys(v)) out[k] = roundDeep(v[k]);
    return out;
  }
  return r(v);
};

const ISL = 2048, OSL = 256, TBT_MS = 50, TTFT_MS = 1500, KV_PREC = "INT8";
const CURVE_B_SAMPLES = [1, 2, 4, 8, 16, 32, 64, 128, 256];

test("dense regression: compute() output byte-identical to pre-refactor baseline", () => {
  let checked = 0;
  for (const baselineCombo of baseline.combos) {
    if (!DENSE_KEYS.has(baselineCombo.model)) continue;
    const hw = HW[baselineCombo.hw];
    const model = MODEL[baselineCombo.model];
    if (!hw || !model) {
      assert.fail(`Baseline references missing hw/model: ${baselineCombo.hw} / ${baselineCombo.model}`);
    }
    const fresh = compute({
      hw, model,
      isl: ISL, osl: OSL,
      weight_prec: baselineCombo.weight_prec,
      kv_prec: KV_PREC,
      act_prec: baselineCombo.act_prec,
      tbt_ms: TBT_MS, ttft_ms: TTFT_MS,
      ngpus: baselineCombo.ngpus,
    });
    const freshSampled = CURVE_B_SAMPLES
      .map((B) => fresh.chart?.curve?.find((p) => p.B === B) || null)
      .filter(Boolean)
      .map((p) => ({ B: p.B, step_ms: p.step_ms, tokens_per_sec: p.tokens_per_sec }));
    const freshSnapshot = {
      metrics: roundDeep(fresh.metrics),
      curve_samples: roundDeep(freshSampled),
      warnings: fresh.warnings?.map((w) => ({ level: w.level, msg: w.msg })) || [],
    };
    const baselineSnapshot = {
      metrics: baselineCombo.metrics,
      curve_samples: baselineCombo.curve_samples,
      warnings: baselineCombo.warnings,
    };
    assert.deepEqual(
      freshSnapshot,
      baselineSnapshot,
      `Dense regression: drift detected for ${baselineCombo.hw} / ${baselineCombo.model} / ${baselineCombo.weight_prec}/${baselineCombo.act_prec} / ngpus=${baselineCombo.ngpus}`,
    );
    checked++;
  }
  assert.ok(checked > 0, "No dense combos found in baseline — fixture is stale or empty");
  // 5 dense models × 7 hardware × 4 prec pairs × 2 ngpus = 280 combos. Pin the
  // count so a silent fixture truncation can't sneak a coverage regression past.
  assert.equal(checked, 280, `Expected 280 dense combos in baseline; got ${checked}`);
});
