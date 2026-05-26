#!/usr/bin/env node
// Sweep the (model × hw × precision × ngpus) grid of the calc's data
// files and write a single `predictions/grid.json` artifact the lab
// consumes as a lookup table. Run at calc-build time; lab reads only.
//
// Usage:
//   node scripts/emit_predictions.mjs            # → predictions/grid.json
//   node scripts/emit_predictions.mjs --out X    # explicit output path
//   node scripts/emit_predictions.mjs --slim     # subset (smoke test fixtures)
//
// Determinism: iteration order is fixed (lexicographic by key tuple) so
// re-running on the same data produces byte-identical JSON. Tests rely
// on this.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { compute } from "../src/calc.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");

// Grid axes. Kept small enough that the artifact stays under ~3 MB
// (gzip-friendly; the lab only loads it lazily). The grid is a primary
// lookup — off-grid inputs fall through to live compute_cli.mjs invocations.
const PRECISIONS = [
  { weight_prec: "BF16", kv_prec: "FP16", act_prec: "BF16" },
  { weight_prec: "FP8",  kv_prec: "FP16", act_prec: "FP8"  },
  { weight_prec: "INT4", kv_prec: "FP16", act_prec: "BF16" },
];
const NGPUS_VALUES = [1, 2, 4, 8];
const ISL_OSL_PAIRS = [
  { isl: 128,  osl: 128  },  // short chat
  { isl: 1024, osl: 256  },  // medium
  { isl: 4096, osl: 256  },  // long context / RAG
];
// Single TBT default; live calls cover other SLOs.
const TBT_MS_VALUES = [50];

const SLIM_PRECISIONS = [PRECISIONS[2]];   // INT4 only
const SLIM_NGPUS = [4];
const SLIM_ISL_OSL = [ISL_OSL_PAIRS[0]];
const SLIM_TBT = [50];

function sha256(s) {
  return createHash("sha256").update(s, "utf-8").digest("hex");
}

function calcVersion() {
  if (process.env.SIZING_CALC_VERSION) return process.env.SIZING_CALC_VERSION;
  try {
    const pkg = JSON.parse(readFileSync(resolve(ROOT, "package.json"), "utf-8"));
    return `pkg-${pkg.version || "0.0.0"}`;
  } catch {
    return "unknown";
  }
}

function nullableNum(v) {
  if (v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function downsample(points, target) {
  if (points.length <= target) return points;
  const out = [points[0]];
  // Geometric stride past the first point so we keep more density near
  // the ridge and less near the tail.
  const step = (points.length - 1) / (target - 1);
  for (let i = 1; i < target - 1; i++) {
    out.push(points[Math.round(i * step)]);
  }
  out.push(points[points.length - 1]);
  return out;
}

function buildPrediction(model, hw, prec, ngpus, isl, osl, tbt_ms, dataHash, calc_version) {
  const result = compute({
    hw, model, isl, osl, ngpus, tbt_ms,
    weight_prec: prec.weight_prec,
    kv_prec: prec.kv_prec,
    act_prec: prec.act_prec,
  });
  const metrics = result.metrics || {};
  // Downsample the curve to ≤ 24 points per row to keep grid.json under a
  // few MB total. Always preserve the first and last point so the overlay
  // covers the full batch range. Live compute_cli.mjs returns the full
  // curve when the lab needs more density.
  const fullCurve = (result.chart?.curve ?? []).map((p) => ({
    batch: Math.trunc(Number(p.B)),
    step_ms: nullableNum(p.step_ms) ?? 0,
    tps: nullableNum(p.tokens_per_sec) ?? 0,
    cost_per_mtok: nullableNum(p.cost_per_mtok),
  }));
  const curve = downsample(fullCurve, 24);
  const parallelism = metrics.parallelism || {};
  // Rows omit `calc_version` and `data_hash` — those live on the top-level
  // grid envelope so the per-row payload stays compact. The lab's
  // calc_bridge.py copies them back onto each row when materializing a
  // Prediction so the RunResult JSON remains self-describing.
  void calc_version; void dataHash;
  return {
    inputs: {
      model_key: model.key, hw_key: hw.key,
      weight_prec: prec.weight_prec, kv_prec: prec.kv_prec, act_prec: prec.act_prec,
      isl, osl, ngpus, tbt_ms,
      price_per_hour_usd: hw.price_per_hour_usd ?? null,
    },
    b_crit: nullableNum(metrics.b_crit),
    b_slo:  nullableNum(metrics.b_slo),
    b_kv:   nullableNum(metrics.b_kv),
    recommended_batch: nullableNum(metrics.recommended_batch),
    y_max: Number.isFinite(parallelism.y_max) ? Math.trunc(parallelism.y_max) : null,
    curve,
    warnings: Array.isArray(result.warnings) ? result.warnings : [],
    unavailable_reason: null,
  };
}

function main() {
  const args = process.argv.slice(2);
  const slim = args.includes("--slim");
  const outIdx = args.indexOf("--out");
  const outPath = outIdx >= 0
    ? resolve(args[outIdx + 1])
    : resolve(ROOT, "predictions/grid.json");

  const modelsText = readFileSync(resolve(ROOT, "src/data/models.json"), "utf-8");
  const hardwareText = readFileSync(resolve(ROOT, "src/data/hardware.json"), "utf-8");
  const models = JSON.parse(modelsText).models;
  const hardware = JSON.parse(hardwareText).hardware;
  const dataHash = sha256(modelsText + hardwareText);
  const calc_version = calcVersion();

  const precisions = slim ? SLIM_PRECISIONS : PRECISIONS;
  const ngpusList  = slim ? SLIM_NGPUS     : NGPUS_VALUES;
  const islosls    = slim ? SLIM_ISL_OSL   : ISL_OSL_PAIRS;
  const tbtList    = slim ? SLIM_TBT       : TBT_MS_VALUES;

  // Sort axes by (model_key, hw_key, prec, ngpus, isl, osl, tbt) so the
  // output is reproducible byte-for-byte.
  const sortedModels = [...models].sort((a, b) => a.key.localeCompare(b.key));
  const sortedHw = [...hardware].sort((a, b) => a.key.localeCompare(b.key));

  const rows = [];
  for (const model of sortedModels) {
    for (const hw of sortedHw) {
      for (const prec of precisions) {
        for (const ngpus of ngpusList) {
          for (const pair of islosls) {
            for (const tbt_ms of tbtList) {
              try {
                rows.push(
                  buildPrediction(model, hw, prec, ngpus, pair.isl, pair.osl, tbt_ms, dataHash, calc_version),
                );
              } catch (e) {
                rows.push({
                  inputs: {
                    model_key: model.key, hw_key: hw.key,
                    weight_prec: prec.weight_prec, kv_prec: prec.kv_prec, act_prec: prec.act_prec,
                    isl: pair.isl, osl: pair.osl, ngpus, tbt_ms,
                    price_per_hour_usd: hw.price_per_hour_usd ?? null,
                  },
                  b_crit: null, b_slo: null, b_kv: null,
                  recommended_batch: null, y_max: null,
                  curve: [], warnings: [],
                  unavailable_reason: `compute() threw: ${e.message}`,
                });
              }
            }
          }
        }
      }
    }
  }

  mkdirSync(dirname(outPath), { recursive: true });
  const grid = {
    calc_version,
    data_hash: dataHash,
    generated_at: process.env.SIZING_CALC_FROZEN_TIMESTAMP || new Date().toISOString(),
    rows,
  };
  // Compact JSON: this is a generated lookup table, not a hand-edited file.
  // Slim fixture goldens use indent=2 separately (see --pretty flag).
  const pretty = args.includes("--pretty");
  writeFileSync(outPath, JSON.stringify(grid, null, pretty ? 2 : 0));
  process.stderr.write(`wrote ${rows.length} rows → ${outPath}\n`);
}

main();
