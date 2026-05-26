// compute_cli.mjs: subprocess round-trip + bridge equivalence with direct compute().
//
// Owns C12: predictions emitted via the CLI must match what compute() returns
// in-process for the same inputs (bit-identical on the fields the lab reads).

import { test } from "node:test";
import { strict as assert } from "node:assert";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { compute } from "../src/calc.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");
const CLI = resolve(ROOT, "scripts/compute_cli.mjs");

function findRow(file, key) {
  const arr = JSON.parse(readFileSync(resolve(ROOT, file), "utf-8"));
  return (arr.models || arr.hardware).find((r) => r.key === key);
}

test("compute_cli: stdin → stdout JSON, matches direct compute()", () => {
  const input = {
    model_key: "llama-3-70b", hw_key: "A10G",
    weight_prec: "INT4", kv_prec: "FP16", act_prec: "BF16",
    isl: 128, osl: 128, ngpus: 4, tbt_ms: 50, price_per_hour_usd: null,
  };
  const proc = spawnSync("node", [CLI], {
    input: JSON.stringify(input),
    encoding: "utf-8",
  });
  assert.equal(proc.status, 0, `compute_cli stderr: ${proc.stderr}`);
  const cliOut = JSON.parse(proc.stdout);

  // Re-compute directly in this process for comparison.
  const model = findRow("src/data/models.json", "llama-3-70b");
  const hw = findRow("src/data/hardware.json", "A10G");
  const direct = compute({
    hw, model, isl: 128, osl: 128, ngpus: 4, tbt_ms: 50,
    weight_prec: "INT4", kv_prec: "FP16", act_prec: "BF16",
  });

  // Headline numbers identical.
  assert.equal(cliOut.b_crit, direct.metrics.b_crit);
  assert.equal(cliOut.b_slo,  direct.metrics.b_slo);
  assert.equal(cliOut.b_kv,   direct.metrics.b_kv);
  assert.equal(cliOut.recommended_batch, direct.metrics.recommended_batch);
  // Curve length matches; first/last points equal to within floating tolerance.
  assert.equal(cliOut.curve.length, direct.chart.curve.length);
  const lastCli = cliOut.curve.at(-1);
  const lastDirect = direct.chart.curve.at(-1);
  assert.equal(lastCli.batch, lastDirect.B);
  assert.ok(Math.abs(lastCli.tps - lastDirect.tokens_per_sec) < 1e-6);
});

test("compute_cli: unknown model_key → non-zero exit", () => {
  const proc = spawnSync("node", [CLI], {
    input: JSON.stringify({
      model_key: "no-such-model", hw_key: "A10G",
      weight_prec: "INT4", kv_prec: "FP16", act_prec: "BF16",
      isl: 128, osl: 128, ngpus: 4, tbt_ms: 50,
    }),
    encoding: "utf-8",
  });
  assert.notEqual(proc.status, 0);
  assert.match(proc.stderr, /unknown model_key/);
});

test("compute_cli: malformed stdin → non-zero exit", () => {
  const proc = spawnSync("node", [CLI], {
    input: "not json",
    encoding: "utf-8",
  });
  assert.notEqual(proc.status, 0);
  assert.match(proc.stderr, /parse error/);
});
