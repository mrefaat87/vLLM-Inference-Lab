// Unit tests for the pure lab_command helpers. These pin the
// calc↔lab rate convention — if the formula changes here, the lab's
// `exp plan` (cli/exp.py:_build_run_grid) must change in lockstep.

import { test } from "node:test";
import assert from "node:assert/strict";

import { recommendedRate } from "../src/lab_command.mjs";

test("recommendedRate: golden — batch=64, tbt=50ms → 1280 rps", () => {
  // Little's Law: 64 / 0.050 = 1280.
  assert.equal(recommendedRate(64, 50), 1280);
});

test("recommendedRate: golden — batch=16, tbt=25ms → 640 rps", () => {
  assert.equal(recommendedRate(16, 25), 640);
});

test("recommendedRate: floor — batch=0 emits at least 1 rps", () => {
  // A degraded prediction must not poison the pasted command with
  // --rate 0 (which would silently produce a zero-arrival run).
  const r = recommendedRate(0, 50);
  assert.ok(r >= 1, `expected ≥1, got ${r}`);
  assert.ok(Number.isFinite(r));
});

test("recommendedRate: floor — NaN batch emits at least 1 rps", () => {
  const r = recommendedRate(NaN, 50);
  assert.ok(r >= 1);
  assert.ok(Number.isFinite(r));
});

test("recommendedRate: floor — tbt=0 falls back to 1ms (doesn't divide-by-zero)", () => {
  const r = recommendedRate(8, 0);
  assert.ok(Number.isFinite(r));
  assert.ok(r > 0);
});

test("recommendedRate: property — matches batch / (tbt/1000) to 2dp across 50 random cases", () => {
  // Seeded LCG so the test is deterministic.
  let seed = 1337;
  const rng = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  for (let i = 0; i < 50; i++) {
    const batch = 1 + Math.floor(rng() * 1024); // [1, 1024]
    const tbtMs = 5 + Math.floor(rng() * 195);  // [5, 199]
    const expected = Math.max(1, Math.round((batch / (tbtMs / 1000)) * 100) / 100);
    const actual = recommendedRate(batch, tbtMs);
    assert.ok(
      Math.abs(actual - expected) < 1e-6,
      `batch=${batch} tbt=${tbtMs}: got ${actual}, expected ${expected}`,
    );
  }
});

test("recommendedRate: agrees with `exp plan` Little's-Law formula", () => {
  // The lab's _build_run_grid does: rate_rps = round(b / tbt_s, 3).
  // Our calc-side helper rounds to 2dp. Both round towards each other —
  // we assert the values are equal to 2dp for every (batch, tbt) pair
  // in a small dense grid. This is the contract that keeps the
  // calc-emitted command and the lab-suggested grid agreeing.
  for (const b of [1, 4, 16, 64, 128, 256]) {
    for (const tbtMs of [10, 25, 50, 100, 200]) {
      const tbtSec = tbtMs / 1000;
      const labRate = Math.round((b / tbtSec) * 1000) / 1000; // lab's 3dp
      const calcRate = recommendedRate(b, tbtMs);
      // Compare to 2dp — the calc's coarser rounding is the limit.
      assert.ok(
        Math.abs(calcRate - labRate) <= 0.01,
        `disagree at batch=${b} tbt=${tbtMs}: calc=${calcRate}, lab=${labRate}`,
      );
    }
  }
});
