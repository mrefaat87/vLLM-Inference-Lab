#!/usr/bin/env node
// Headless bridge: read a single JSON payload from stdin, return the
// calc's compute() output as a Prediction object on stdout. Used by the
// empirical lab to consult the calculator for off-grid inputs without
// embedding a JS runtime in Python.
//
// Input shape (one JSON object on stdin):
//   {
//     "model_key":   "Llama-3-70B",
//     "hw_key":      "A10G",
//     "weight_prec": "INT4",
//     "kv_prec":     "FP16",
//     "act_prec":    "BF16",
//     "isl": 200, "osl": 150, "ngpus": 4, "tbt_ms": 50,
//     "price_per_hour_usd": null
//   }
//
// Output shape: see calculators/sizing_calc/predictions/schema.json#/$defs/prediction.

import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { compute } from "../src/calc.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");

function readDataFiles() {
  const modelsText = readFileSync(resolve(ROOT, "src/data/models.json"), "utf-8");
  const hardwareText = readFileSync(resolve(ROOT, "src/data/hardware.json"), "utf-8");
  // Files are wrapped: { _schema: {...}, models: [...] } / { _schema, hardware }.
  // Hash the bytes-on-disk (not the unwrapped slice) so provenance survives
  // any future top-level metadata additions.
  return {
    models: JSON.parse(modelsText).models,
    hardware: JSON.parse(hardwareText).hardware,
    dataHash: sha256(modelsText + hardwareText),
  };
}

function sha256(s) {
  return createHash("sha256").update(s, "utf-8").digest("hex");
}

function calcVersion() {
  // Prefer SIZING_CALC_VERSION env var (set by CI to a git SHA); fall back
  // to package.json version.
  if (process.env.SIZING_CALC_VERSION) return process.env.SIZING_CALC_VERSION;
  try {
    const pkg = JSON.parse(readFileSync(resolve(ROOT, "package.json"), "utf-8"));
    return `pkg-${pkg.version || "0.0.0"}`;
  } catch {
    return "unknown";
  }
}

function findByKey(arr, key) {
  return arr.find((row) => row.key === key);
}

function readStdin() {
  return readFileSync(0, "utf-8");  // fd 0 = stdin
}

function predictionFromCompute(inputObj, result, calc_version, data_hash) {
  // calc.mjs::compute() returns:
  //   { metrics: { b_crit, b_slo, b_kv, recommended_batch, parallelism, ... },
  //     chart:   { curve: [{B, step_ms, tokens_per_sec, cost_per_mtok}], bcrit, bslo, bkv },
  //     sweep, warnings }
  // We pluck the smallest subset that survives long-term — anything UI-specific
  // is left out so a future calc refactor doesn't break the bridge.
  const metrics = result.metrics || {};
  const curveRaw = result.chart?.curve ?? [];
  const curve = curveRaw.map((p) => ({
    batch: Math.trunc(Number(p.B)),
    step_ms: nullableNum(p.step_ms) ?? 0,
    tps: nullableNum(p.tokens_per_sec) ?? 0,
    cost_per_mtok: nullableNum(p.cost_per_mtok),
  }));
  const parallelism = metrics.parallelism || {};

  return {
    calc_version,
    data_hash,
    inputs: {
      model_key: inputObj.model_key,
      hw_key: inputObj.hw_key,
      weight_prec: inputObj.weight_prec,
      kv_prec: inputObj.kv_prec,
      act_prec: inputObj.act_prec,
      isl: inputObj.isl,
      osl: inputObj.osl,
      ngpus: inputObj.ngpus,
      tbt_ms: inputObj.tbt_ms,
      price_per_hour_usd: inputObj.price_per_hour_usd ?? null,
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

function nullableNum(v) {
  if (v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function main() {
  let input;
  try {
    input = JSON.parse(readStdin());
  } catch (e) {
    process.stderr.write(`compute_cli: stdin JSON parse error: ${e.message}\n`);
    process.exit(2);
  }
  const { models, hardware, dataHash } = readDataFiles();
  const model = findByKey(models, input.model_key);
  const hw = findByKey(hardware, input.hw_key);
  if (!model) {
    process.stderr.write(`compute_cli: unknown model_key '${input.model_key}'\n`);
    process.exit(3);
  }
  if (!hw) {
    process.stderr.write(`compute_cli: unknown hw_key '${input.hw_key}'\n`);
    process.exit(3);
  }
  const result = compute({
    hw,
    model,
    isl: input.isl,
    osl: input.osl,
    weight_prec: input.weight_prec,
    kv_prec: input.kv_prec,
    act_prec: input.act_prec,
    tbt_ms: input.tbt_ms,
    ngpus: input.ngpus,
    price_per_hour_usd: input.price_per_hour_usd ?? undefined,
  });
  const prediction = predictionFromCompute(input, result, calcVersion(), dataHash);
  process.stdout.write(JSON.stringify(prediction));
}

main();
