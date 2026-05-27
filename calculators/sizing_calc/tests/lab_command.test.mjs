// Unit tests for the pure lab_command helpers. These pin the
// calc↔lab rate convention — if the formula changes here, the lab's
// `exp plan` (cli/exp.py:_build_run_grid) must change in lockstep.

import { test } from "node:test";
import assert from "node:assert/strict";

import { recommendedRate } from "../src/lab_command.mjs";

// Little's Law with per-request latency:
//   per_req_s = ttft_ms/1000 + osl * tbt_ms/1000
//   λ_rps    ≈ batch / per_req_s
const perReqS = (tbtMs, oslTokens, ttftMs) =>
  ttftMs / 1000 + oslTokens * (tbtMs / 1000);

test("recommendedRate: golden — batch=128, tbt=50, osl=200, ttft=500 → 12.19 rps", () => {
  // per_req = 0.5 + 200*0.050 = 10.5 s → 128/10.5 = 12.19047… → 12.19 (2dp).
  assert.equal(recommendedRate(128, 50, 200, 500), 12.19);
});

test("recommendedRate: golden — batch=64, tbt=25, osl=100, ttft=200 → 24.62 rps", () => {
  // per_req = 0.2 + 100*0.025 = 2.7 s → 64/2.7 = 23.7037… → 23.7 (2dp).
  // Sanity: 64/2.7 = 23.703703…; rounded to 2dp = 23.7.
  assert.equal(recommendedRate(64, 25, 100, 200), 23.7);
});

test("recommendedRate: floor — batch=0 emits at least 1 rps", () => {
  const r = recommendedRate(0, 50, 200, 500);
  assert.ok(r >= 1, `expected ≥1, got ${r}`);
  assert.ok(Number.isFinite(r));
});

test("recommendedRate: floor — NaN batch emits at least 1 rps", () => {
  const r = recommendedRate(NaN, 50, 200, 500);
  assert.ok(r >= 1);
  assert.ok(Number.isFinite(r));
});

test("recommendedRate: floor — all-zero latency inputs don't divide-by-zero", () => {
  const r = recommendedRate(8, 0, 0, 0);
  assert.ok(Number.isFinite(r));
  assert.ok(r > 0);
});

test("recommendedRate: ttft=0 only — TBT × OSL still drives the rate", () => {
  // per_req = 0 + 200*0.050 = 10.0 s → 128/10 = 12.8.
  assert.equal(recommendedRate(128, 50, 200, 0), 12.8);
});

test("recommendedRate: osl=0 falls back to osl=1 to keep the formula non-degenerate", () => {
  // Treats osl=0 as a single-token reply: per_req = 0.5 + 1*0.050 = 0.55 s.
  // 128/0.55 = 232.72727… → 232.73.
  assert.equal(recommendedRate(128, 50, 0, 500), 232.73);
});

test("recommendedRate: property — matches the explicit formula to 2dp across 50 random cases", () => {
  // Seeded LCG so the test is deterministic.
  let seed = 1337;
  const rng = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  for (let i = 0; i < 50; i++) {
    const batch = 1 + Math.floor(rng() * 1024); // [1, 1024]
    const tbtMs = 5 + Math.floor(rng() * 195); // [5, 199]
    const osl = 1 + Math.floor(rng() * 2048); // [1, 2048]
    const ttftMs = Math.floor(rng() * 5000); // [0, 4999]
    const expected = Math.max(
      1,
      Math.round((batch / perReqS(tbtMs, osl, ttftMs)) * 100) / 100,
    );
    const actual = recommendedRate(batch, tbtMs, osl, ttftMs);
    assert.ok(
      Math.abs(actual - expected) < 1e-6,
      `batch=${batch} tbt=${tbtMs} osl=${osl} ttft=${ttftMs}: got ${actual}, expected ${expected}`,
    );
  }
});

test("recommendedRate: agrees with `exp plan` Little's-Law formula", () => {
  // The lab's _build_run_grid does: rate_rps = round(b / per_req_s, 3).
  // Our calc-side helper rounds to 2dp. Both round towards each other —
  // we assert the values are equal to 2dp for every (batch, tbt, osl, ttft)
  // tuple in a small dense grid. This is the contract that keeps the
  // calc-emitted command and the lab-suggested grid agreeing.
  for (const b of [1, 4, 16, 64, 128, 256]) {
    for (const tbtMs of [10, 25, 50, 100, 200]) {
      for (const osl of [50, 200, 1000]) {
        for (const ttftMs of [100, 500, 2000]) {
          const labRate =
            Math.round((b / perReqS(tbtMs, osl, ttftMs)) * 1000) / 1000;
          const calcRate = recommendedRate(b, tbtMs, osl, ttftMs);
          // Compare to 2dp — the calc's coarser rounding is the limit.
          assert.ok(
            Math.abs(calcRate - Math.max(1, labRate)) <= 0.01,
            `disagree at b=${b} tbt=${tbtMs} osl=${osl} ttft=${ttftMs}: calc=${calcRate}, lab=${labRate}`,
          );
        }
      }
    }
  }
});
