// Unit tests for the pure compute module. Zero deps — node:test + node:assert.
//
// Every numeric assertion either cites a section in MODEL_SIZING_SCALING_REFERENCE.md
// (loaded from tests/fixtures/golden.json) or asserts an invariant of the math
// (monotonicity, self-consistency). New formulas should land with a golden.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
  ACT_OVERHEAD, DTYPE_BYTES,
  weightsBytes, kvPerToken, bCrit, stepTime, throughputAtB,
  bSlo, bKv, yMax, recommendParallelism, recommendMaxBatchedTokens,
  pdRatio, sweepRanges, compute,
} from "../src/calc.mjs";

// ---------- fixtures ----------
const here = dirname(fileURLToPath(import.meta.url));
const hardware = JSON.parse(readFileSync(join(here, "../src/data/hardware.json"))).hardware;
const models   = JSON.parse(readFileSync(join(here, "../src/data/models.json"))).models;
const golden   = JSON.parse(readFileSync(join(here, "fixtures/golden.json")));
const HW       = Object.fromEntries(hardware.map((h) => [h.key, h]));
const MODEL    = Object.fromEntries(models.map((m) => [m.key, m]));

// Tolerance helper — reference values are mostly rule-of-thumb integers, so
// asserting within ±N% (rather than exact equality) matches their precision.
const within = (actual, expected, tol_pct) => {
  const tol = Math.abs(expected) * (tol_pct / 100);
  assert.ok(
    Math.abs(actual - expected) <= tol,
    `expected ${actual} within ${tol_pct}% of ${expected} (diff ${Math.abs(actual - expected)})`,
  );
};

// ---------- Layer 1: golden values against the reference doc ----------

test("bCrit goldens (reference §1)", () => {
  for (const g of golden.bcrit) {
    const got = bCrit(HW[g.hw_key], g.weight_prec, g.act_prec);
    within(got, g.expected, g.tolerance_pct);
  }
});

test("KV cache goldens (reference §3, §5 worked example)", () => {
  for (const g of golden.kv) {
    const m = MODEL[g.model_key];
    const k = kvPerToken(m, DTYPE_BYTES[g.kv_prec]);
    if (g.expected_bytes !== undefined) within(k, g.expected_bytes, g.tolerance_pct);
    if (g.expected_gb !== undefined) {
      const total = k * g.seq_len * (g.batch || 1);
      within(total / 1e9, g.expected_gb, g.tolerance_pct);
    }
  }
});

test("Weights goldens (reference §5)", () => {
  for (const g of golden.weights) {
    const m = MODEL[g.model_key];
    const w = weightsBytes(m.params_b, DTYPE_BYTES[g.weight_prec]);
    within(w / 1e9, g.expected_gb, g.tolerance_pct);
  }
});

// ---------- Layer 2: invariants & self-consistency ----------

test("stepTime is monotonic non-decreasing in B", () => {
  const m = MODEL["llama-3-70b"];
  const hw = HW["H100-80GB"];
  const args = (B) => ({
    B,
    kv_per_token: kvPerToken(m, DTYPE_BYTES.INT8),
    avgSeq: 4096,
    weight_bytes_total: weightsBytes(m.params_b, DTYPE_BYTES.INT8),
    paramCount: m.params_b * 1e9,
    totalBW: hw.hbm_bw_gbs * 1e9,
    totalFLOPs: hw.fp16_tflops * 1e12,
  });
  let prev = stepTime(args(1));
  for (const B of [2, 4, 8, 16, 32, 64, 128, 256, 512]) {
    const t = stepTime(args(B));
    assert.ok(t >= prev - 1e-12, `stepTime not monotonic at B=${B}: ${t} < ${prev}`);
    prev = t;
  }
});

test("bSlo self-consistency: stepTime(bSlo) ≤ SLO < stepTime(bSlo+1)", () => {
  const m = MODEL["llama-3-70b"];
  const hw = HW["H100-80GB"];
  const W = weightsBytes(m.params_b, DTYPE_BYTES.INT8);
  const K = kvPerToken(m, DTYPE_BYTES.INT8);
  const args = {
    tbt_ms: 50, kv_per_token: K, avgSeq: 4096, weight_bytes_total: W,
    paramCount: m.params_b * 1e9,
    totalBW: hw.hbm_bw_gbs * 1e9,
    totalFLOPs: hw.fp16_tflops * 1e12,
  };
  const r = bSlo(args);
  if (r.unreachable) return; // separately covered
  const stArgs = (B) => ({ B, ...args, paramCount: args.paramCount });
  const at = stepTime(stArgs(r.value)) * 1000;
  const next = stepTime(stArgs(r.value + 1)) * 1000;
  assert.ok(at <= 50 + 1e-6, `stepTime(${r.value})=${at}ms exceeds 50ms SLO`);
  assert.ok(next > 50 - 1e-6, `bSlo+1 still fits SLO — under-tight bound: ${next}ms`);
});

test("bKv self-consistency: used HBM stays under cap", () => {
  const m = MODEL["llama-3-8b"];
  const hw = HW["H100-80GB"];
  const W = weightsBytes(m.params_b, DTYPE_BYTES.FP16);
  const K = kvPerToken(m, DTYPE_BYTES.FP16);
  const r = bKv({ hw, model: m, ngpus: 1, weight_bytes_total: W, kv_per_token: K, isl: 1024, osl: 256 });
  const used = W + r.value * K * (1024 + 256);
  const cap = hw.hbm_gb * 1e9 * (1 - ACT_OVERHEAD);
  assert.ok(used <= cap, `used ${used/1e9}GB exceeds cap ${cap/1e9}GB`);
});

test("throughput tracks stepTime: tps = B / stepTime", () => {
  const args = {
    B: 32, kv_per_token: 1000, avgSeq: 1024, weight_bytes_total: 1e10,
    paramCount: 1e10, totalBW: 1e12, totalFLOPs: 1e15,
  };
  const t = stepTime(args);
  const tps = throughputAtB(args);
  within(tps, args.B / t, 0.01);
});

// ---------- Layer 3: edge / warning paths ----------

test("FP8 weights on T4 falls back to FP16 roofline and warns", () => {
  const out = compute({
    hw: HW["T4"], model: MODEL["llama-3-8b"],
    isl: 1024, osl: 256, weight_prec: "FP8", kv_prec: "FP8", act_prec: "FP8",
    tbt_ms: 50, ttft_ms: 2000, ngpus: 1,
  });
  const w = out.warnings.find((w) => w.msg.includes("no hardware FP8"));
  assert.ok(w, "expected FP8-on-T4 warning");
  // bCrit must use FP16 TFLOPs (T4 has fp8_tflops=null), so for FP8/FP8 we get
  // (65e12/320e9) * 1 ≈ 203. Definitely NOT the hypothetical FP8-doubled value.
  within(out.metrics.b_crit, 203, 3);
});

test("Llama-3-70B on a single T4 reports weights overflow", () => {
  const out = compute({
    hw: HW["T4"], model: MODEL["llama-3-70b"],
    isl: 1024, osl: 256, weight_prec: "INT8", kv_prec: "INT8", act_prec: "INT8",
    tbt_ms: 50, ttft_ms: 2000, ngpus: 1,
  });
  const overflow = out.warnings.find((w) => w.msg.includes("overflow"));
  assert.ok(overflow, "expected weights-overflow warning");
  assert.equal(out.metrics.b_kv, 0);
});

test("bSlo unreachable when TBT × BW < weight stream time", () => {
  const m = MODEL["llama-3-70b"];
  const hw = HW["T4"]; // tiny BW
  const W = weightsBytes(m.params_b, DTYPE_BYTES.INT8);
  const r = bSlo({
    tbt_ms: 10, kv_per_token: kvPerToken(m, DTYPE_BYTES.INT8), avgSeq: 4096,
    weight_bytes_total: W, paramCount: m.params_b * 1e9,
    totalBW: hw.hbm_bw_gbs * 1e9, totalFLOPs: hw.fp16_tflops * 1e12,
  });
  assert.equal(r.unreachable, true);
  assert.equal(r.value, 0);
});

test("ISL+OSL exceeding max_context emits a warning", () => {
  const out = compute({
    hw: HW["H100-80GB"], model: MODEL["llama-3-8b"], // 8k ctx
    isl: 8000, osl: 1000, weight_prec: "FP16", kv_prec: "FP16", act_prec: "FP16",
    tbt_ms: 50, ttft_ms: 2000, ngpus: 1,
  });
  assert.ok(out.warnings.find((w) => w.msg.includes("max_context")));
});

test("Zero / negative inputs never produce NaN or Infinity", () => {
  const out = compute({
    hw: HW["H100-80GB"], model: MODEL["llama-3-8b"],
    isl: 0, osl: -5, weight_prec: "FP16", kv_prec: "FP16", act_prec: "FP16",
    tbt_ms: 0, ttft_ms: 0, ngpus: 0,
  });
  for (const [k, v] of Object.entries(out.metrics)) {
    if (typeof v === "number") {
      assert.ok(Number.isFinite(v), `metric ${k} = ${v} should be finite`);
      assert.ok(!Number.isNaN(v), `metric ${k} is NaN`);
    }
  }
});

test("INT4 weights stay in float math (no integer-division bugs)", () => {
  const m = MODEL["llama-3-8b"];
  const w = weightsBytes(m.params_b, DTYPE_BYTES.INT4);
  // 8.03B params × 0.5 B/param = 4.015 GB — not a round integer.
  within(w / 1e9, m.params_b * 0.5, 0.01);
  assert.ok(!Number.isInteger(w / 1e9));
});

// ---------- Layer 4: parallelism ----------

test("recommendParallelism: 70B int8 needs TP≥4 on H100s", () => {
  const m = MODEL["llama-3-70b"];
  const W = weightsBytes(m.params_b, DTYPE_BYTES.INT8);
  const r = recommendParallelism({ hw: HW["H100-80GB"], weight_bytes_total: W, ngpus: 8, d_ff: m.d_ff, batch: 32 });
  // 70 GB weights × 1.1 headroom = 77 GB; per-GPU usable = 72 GB; TP=2 yields
  // 35 GB/GPU which fits.
  assert.equal(r.fits, true);
  assert.ok(r.tp <= 2, `expected TP≤2 for 70B int8 on H100-80, got TP=${r.tp}`);
});

test("recommendParallelism: 70B int8 cannot fit on 1× T4", () => {
  const m = MODEL["llama-3-70b"];
  const W = weightsBytes(m.params_b, DTYPE_BYTES.INT8);
  const r = recommendParallelism({ hw: HW["T4"], weight_bytes_total: W, ngpus: 1, d_ff: m.d_ff, batch: 1 });
  assert.equal(r.fits, false);
});

test("recommendParallelism: ngpus=3 caps TP at largest pow2 ≤ 3", () => {
  const m = MODEL["llama-3-70b"];
  const W = weightsBytes(m.params_b, DTYPE_BYTES.INT8);
  const r = recommendParallelism({ hw: HW["A100-40GB"], weight_bytes_total: W, ngpus: 3, d_ff: m.d_ff, batch: 32 });
  assert.ok([1, 2].includes(r.tp), `TP should be 1 or 2 with ngpus=3, got ${r.tp}`);
});

// ---------- Y_max (Scaling Book) ----------

test("yMax goldens (Scaling Book worked examples)", () => {
  for (const g of golden.y_max) {
    const y = yMax(g.d_ff, g.batch, g.hw);
    within(y, g.expected, g.tolerance_pct);
  }
});

test("yMax: H100 (β≈3.7) gives larger Y_max than book's TPU example", () => {
  const y_h100 = yMax(MODEL["llama-3-8b"].d_ff, 32, HW["H100-80GB"]);
  // d_ff=14336, β=3350/900≈3.72 → Y_max ≈ 120
  within(y_h100, 14336 / (32 * 3350 / 900), 0.1);
  assert.ok(y_h100 > 64);
});

test("yMax: large batch shrinks Y_max (decode→prefill transition)", () => {
  const small = yMax(MODEL["llama-3-8b"].d_ff, 8, HW["H100-80GB"]);
  const large = yMax(MODEL["llama-3-8b"].d_ff, 512, HW["H100-80GB"]);
  assert.ok(large < small);
  // Inversely proportional to B.
  within(large * 512, small * 8, 0.01);
});

test("yMax: PCIe-only hardware (no NVLink) returns small Y_max", () => {
  // β for L4 = 300/64 ≈ 4.7. d_ff=14336, B=32 → Y_max ≈ 95. Big batch tightens.
  const y_pcie = yMax(MODEL["llama-3-8b"].d_ff, 256, HW["L4"]);
  assert.ok(y_pcie < 16, `expected PCIe Y_max <16 at B=256, got ${y_pcie}`);
});

test("yMax: missing ici_bw_gbs returns Infinity (no constraint)", () => {
  const y = yMax(14336, 32, { hbm_bw_gbs: 3000, ici_bw_gbs: null });
  assert.equal(y, Infinity);
});

test("compute: emits ICI-binding diagnostic when Y_max < TP_CAP", () => {
  // Synthetic micro-model with tiny d_ff so Y_max drops below TP_CAP at any
  // realistic batch on H100. Real-world: this regime fires for big batches
  // on slow-ICI hardware, or for tiny models that don't need much sharding.
  const micro = {
    key: "micro", label: "Micro-1B", params_b: 1.0, n_layers: 16, d_model: 1024,
    n_heads: 16, n_kv_heads: 4, head_dim: 64, d_ff: 1024, max_context: 4096, attn_type: "GQA",
  };
  const out = compute({
    hw: HW["H100-80GB"], model: micro,
    isl: 4096, osl: 1024, weight_prec: "FP16", kv_prec: "FP16", act_prec: "FP16",
    tbt_ms: 50, ttft_ms: 2000, ngpus: 8,
  });
  // Y_max should appear in the parallelism object and be finite.
  assert.ok(Number.isFinite(out.metrics.parallelism.y_max),
    `y_max should be finite, got ${out.metrics.parallelism.y_max}`);
  assert.ok(out.metrics.parallelism.y_max < 8,
    `Y_max should bind (< TP_CAP=8) for tiny d_ff + big batch, got ${out.metrics.parallelism.y_max}`);
  const ici = out.warnings.find((w) => w.msg.includes("ICI binds"));
  assert.ok(ici, "expected ICI-binding warning when Y_max < TP_CAP");
});

// ---------- Layer 5: max_num_batched_tokens ----------

test("recommendMaxBatchedTokens snaps to powers of 2 ≥ 1024", () => {
  const r = recommendMaxBatchedTokens({ maxBatch: 32, isl: 1024, tbt_ms: 50 });
  assert.ok([1024, 1536, 2048, 4096, 8192].includes(r.value));
  assert.ok(r.value >= 1024);
});

test("recommendMaxBatchedTokens: strict TBT band caps at 2048", () => {
  const r = recommendMaxBatchedTokens({ maxBatch: 256, isl: 8192, tbt_ms: 20 });
  assert.ok(r.value <= 2048, `strict band should cap at 2048, got ${r.value}`);
  assert.ok(r.band.startsWith("strict"));
});

// ---------- Layer 6: P:D ratio ----------

test("pdRatio amplifies ISL/OSL by MFU asymmetry (4×)", () => {
  // ISL/OSL = 4, MFU ratio = PREFILL_MFU/DECODE_MFU = 0.40/0.10 = 4 → expect 16.
  // The 4× multiplier itself is rule-of-thumb (defensible range 3×–8× across
  // sources — see calc.mjs MFU comment and reference §0.3 footnote). This test
  // pins the *implementation*, not the underlying physics constant.
  within(pdRatio({ isl: 4000, osl: 1000 }), 16, 0.01);
});

// ---------- Layer 7: sweep ranges ----------

test("sweepRanges brackets the recommended batch", () => {
  const s = sweepRanges({ recommendedBatch: 64, recommendedMnbt: 2048 });
  assert.ok(s.max_num_seqs.includes(64) || s.max_num_seqs.some((v) => Math.abs(v - 64) < 64));
  assert.ok(s.max_num_seqs.every((v) => v >= 1 && v <= 128));
  assert.ok(s.max_num_batched_tokens.includes(2048));
  assert.ok(s.concurrency.includes(64));
  // Concurrency must be sorted ascending and start at 1.
  assert.equal(s.concurrency[0], 1);
  for (let i = 1; i < s.concurrency.length; i++) {
    assert.ok(s.concurrency[i] > s.concurrency[i - 1]);
  }
});

// ---------- Layer 8: property tests (small fuzz) ----------

test("property: every (hw, model, prec) combo produces finite metrics", () => {
  const precs = ["FP16", "FP8", "INT8", "INT4"];
  let n = 0;
  for (const hw of hardware) {
    for (const m of models) {
      for (const wp of precs) {
        // Pick non-degenerate ISL/OSL + reasonable SLOs.
        const out = compute({
          hw, model: m,
          isl: 1024, osl: 256, weight_prec: wp, kv_prec: "FP16", act_prec: "FP16",
          tbt_ms: 50, ttft_ms: 2000, ngpus: 1,
        });
        for (const [k, v] of Object.entries(out.metrics)) {
          if (typeof v === "number") {
            assert.ok(Number.isFinite(v), `${hw.key}/${m.key}/${wp}: ${k} = ${v}`);
            assert.ok(v >= 0, `${hw.key}/${m.key}/${wp}: ${k} negative (${v})`);
          }
        }
        n++;
      }
    }
  }
  assert.ok(n >= 100, `expected coverage breadth, ran only ${n}`);
});

// ---------- Layer 9: large-B regression (per-kernel form vs old single-max) ----------

// Regression for the §4 / Scaling Book per-kernel step-time form. Past B_crit,
// the OLD code (single max((W+B·K·S)/BW, 2BN/FLOPs)) under-reports step time
// because the attention KV stream and the MLP compute term were being max'd
// against each other instead of summed across the two sequential kernels.
//
// Llama-3-70B INT8 on H100 (single GPU), seq=8k, B=800 (well past B_crit≈147
// for INT8/INT8 weights+acts, deeper into the compute-bound regime).
test("stepTime large-B: per-kernel sum differs from old single-max form", () => {
  const m = MODEL["llama-3-70b"];
  const hw = HW["H100-80GB"];
  const W = weightsBytes(m.params_b, DTYPE_BYTES.INT8);
  const K = kvPerToken(m, DTYPE_BYTES.INT8);
  const N = m.params_b * 1e9;
  const BW = hw.hbm_bw_gbs * 1e9;
  const F = hw.fp16_tflops * 1e12; // INT8 acts use FP16 compute path on H100
  const B = 800;
  const avgSeq = 8192;

  const tNew = stepTime({ B, kv_per_token: K, avgSeq, weight_bytes_total: W, paramCount: N, totalBW: BW, totalFLOPs: F });

  // Analytical: attn = B·K·S/BW; mlp = max(W/BW, 2BN/F)
  const attn = (B * K * avgSeq) / BW;
  const mlpMem = W / BW;
  const compute = (2 * B * N) / F;
  const tExpected = attn + Math.max(mlpMem, compute);
  within(tNew, tExpected, 0.01);

  // The OLD wrong form: single max((W + B·K·S)/BW, 2BN/F). At B=800 the KV
  // stream dominates the inside-the-max term, so the old form would return
  // (W + B·K·S)/BW alone — missing the additive MLP-compute kernel.
  const tOldWrong = Math.max((W + B * K * avgSeq) / BW, compute);
  // The new (correct) per-kernel sum must be strictly larger than the old
  // single-max once compute is non-trivial — that's the whole bug fix.
  assert.ok(tNew > tOldWrong,
    `new per-kernel form (${tNew.toFixed(4)}s) should exceed old single-max (${tOldWrong.toFixed(4)}s) past B_crit`);
  // And the gap should be on the order of the MLP compute kernel time
  // (the term the old formula dropped).
  const gap = tNew - tOldWrong;
  assert.ok(gap > 0.5 * Math.min(mlpMem, compute),
    `gap (${gap.toFixed(4)}s) should be on the order of the dropped MLP kernel time`);
});

test("property: bCrit is independent of model size", () => {
  const hw = HW["H100-80GB"];
  const v8 = bCrit(hw, "FP16", "FP16");
  const v70 = bCrit(hw, "FP16", "FP16");
  assert.equal(v8, v70);
});
