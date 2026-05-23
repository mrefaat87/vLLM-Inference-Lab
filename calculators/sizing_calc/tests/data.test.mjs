// Validates the curated data tables. Catches typos in contributor PRs that
// change a number or add a row — the unit tests in calc.test.mjs assume these
// shapes, so they should fail noisily here first.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const hardware = JSON.parse(readFileSync(join(here, "../src/data/hardware.json"))).hardware;
const models   = JSON.parse(readFileSync(join(here, "../src/data/models.json"))).models;

const isPosNum = (v) => typeof v === "number" && v > 0 && Number.isFinite(v);
const isPosInt = (v) => Number.isInteger(v) && v > 0;

test("hardware: required fields present + plausibility ranges", () => {
  for (const h of hardware) {
    assert.ok(typeof h.key === "string" && h.key.length > 0, `bad key: ${JSON.stringify(h)}`);
    assert.ok(typeof h.label === "string");
    assert.ok(isPosInt(h.hbm_gb) && h.hbm_gb <= 256, `${h.key} hbm_gb out of [1,256]: ${h.hbm_gb}`);
    assert.ok(isPosNum(h.hbm_bw_gbs) && h.hbm_bw_gbs >= 100 && h.hbm_bw_gbs <= 10000,
      `${h.key} hbm_bw_gbs out of [100,10000]: ${h.hbm_bw_gbs}`);
    assert.ok(isPosNum(h.fp16_tflops) && h.fp16_tflops <= 5000, `${h.key} fp16_tflops out of range`);
    assert.ok(h.fp8_tflops === null || isPosNum(h.fp8_tflops),
      `${h.key} fp8_tflops must be null or positive number`);
    assert.equal(typeof h.nvlink, "boolean", `${h.key} nvlink must be boolean`);
    // FP8 should be ≥ FP16 (the whole point) when present.
    if (h.fp8_tflops) {
      assert.ok(h.fp8_tflops >= h.fp16_tflops,
        `${h.key} fp8_tflops (${h.fp8_tflops}) < fp16_tflops (${h.fp16_tflops})`);
    }
  }
});

test("hardware: keys are unique", () => {
  const keys = hardware.map((h) => h.key);
  assert.equal(new Set(keys).size, keys.length, "duplicate hardware keys");
});

test("models: required fields present + plausibility ranges", () => {
  for (const m of models) {
    assert.ok(typeof m.key === "string" && m.key.length > 0);
    assert.ok(typeof m.label === "string");
    assert.ok(isPosNum(m.params_b) && m.params_b <= 2000, `${m.key} params_b out of range`);
    assert.ok(isPosInt(m.n_layers) && m.n_layers <= 200, `${m.key} n_layers out of range`);
    assert.ok(isPosInt(m.d_model) && m.d_model <= 20000, `${m.key} d_model out of range`);
    assert.ok(isPosInt(m.n_heads) && m.n_heads <= 256, `${m.key} n_heads out of range`);
    assert.ok(isPosInt(m.n_kv_heads), `${m.key} n_kv_heads must be positive integer`);
    assert.ok(m.n_kv_heads <= m.n_heads, `${m.key} n_kv_heads > n_heads (invalid GQA)`);
    assert.ok(isPosInt(m.head_dim) && m.head_dim <= 512, `${m.key} head_dim out of range`);
    assert.ok(isPosInt(m.max_context) && m.max_context <= 2_000_000, `${m.key} max_context out of range`);
    assert.ok(["MHA", "GQA", "MQA", "MLA"].includes(m.attn_type),
      `${m.key} attn_type must be MHA|GQA|MQA|MLA, got ${m.attn_type}`);
  }
});

test("models: keys are unique", () => {
  const keys = models.map((m) => m.key);
  assert.equal(new Set(keys).size, keys.length, "duplicate model keys");
});

test("models: GQA models have n_kv_heads strictly < n_heads", () => {
  for (const m of models.filter((m) => m.attn_type === "GQA")) {
    assert.ok(m.n_kv_heads < m.n_heads,
      `${m.key} declared GQA but n_kv_heads (${m.n_kv_heads}) is not < n_heads (${m.n_heads})`);
  }
});
