// validation.mjs unit tests (node:test).
//
// Owns:
//   C3 — schema-version envelope (unknown fields ignored).
//   C4 — projection determinism.
//   C6 — empty / 404 / malformed paths surface as unavailable strings.
//   C12 (partial) — grid row → Prediction shape.

import { test } from "node:test";
import { strict as assert } from "node:assert";

import {
  loadLabRuns,
  filterRuns,
  projectRun,
  emptyStateMessage,
  fetchRun,
} from "../src/validation.mjs";

function makeFetch(routes) {
  return async (url) => {
    if (!(url in routes)) {
      return { ok: false, status: 404, json: async () => ({}) };
    }
    const v = routes[url];
    if (v instanceof Error) throw v;
    if (v && v.__throwOnJson) {
      return { ok: true, status: 200, json: async () => { throw new Error("bad json"); } };
    }
    return { ok: true, status: 200, json: async () => v };
  };
}

test("loadLabRuns: ok index", async () => {
  const idxBlob = { runs: [{ run_id: "r1", model_ref: "llama-3-70b", hw_ref: "A10G" }] };
  const fetchImpl = makeFetch({ "./lab_runs/index.json": idxBlob });
  const result = await loadLabRuns({ fetchImpl });
  assert.equal(result.unavailable, false);
  assert.equal(result.index.length, 1);
  assert.equal(result.index[0].model_ref, "llama-3-70b");
});

test("loadLabRuns: 404 → unavailable, no throw", async () => {
  const fetchImpl = makeFetch({});
  const result = await loadLabRuns({ fetchImpl });
  assert.ok(result.unavailable);
  assert.equal(result.index.length, 0);
});

test("loadLabRuns: fetch throws → unavailable, no throw", async () => {
  const fetchImpl = async () => { throw new Error("ENETUNREACH"); };
  const result = await loadLabRuns({ fetchImpl });
  assert.ok(result.unavailable);
  assert.match(result.unavailable, /ENETUNREACH/);
});

test("loadLabRuns: malformed JSON → unavailable, no throw", async () => {
  const fetchImpl = makeFetch({ "./lab_runs/index.json": { __throwOnJson: true } });
  const result = await loadLabRuns({ fetchImpl });
  assert.ok(result.unavailable);
});

test("filterRuns: exact match on (model_ref, hw_ref)", () => {
  const idx = [
    { run_id: "a", model_ref: "llama-3-70b", hw_ref: "A10G" },
    { run_id: "b", model_ref: "llama-3-70b", hw_ref: "H100-80GB" },
    { run_id: "c", model_ref: "deepseek-v3", hw_ref: "H100-80GB" },
  ];
  const matched = filterRuns(idx, "llama-3-70b", "A10G");
  assert.deepEqual(matched.map((r) => r.run_id), ["a"]);
});

test("filterRuns: no match returns empty", () => {
  const idx = [{ run_id: "a", model_ref: "llama-3-70b", hw_ref: "A10G" }];
  assert.deepEqual(filterRuns(idx, "deepseek-v3", "H100-80GB"), []);
});

test("filterRuns: case-sensitive (catches drift between calc and lab)", () => {
  const idx = [{ run_id: "a", model_ref: "Llama-3-70b", hw_ref: "A10G" }];
  // Calc uses 'llama-3-70b' (lowercase); a typo in the lab default would
  // fail to match — that's the contract.
  assert.deepEqual(filterRuns(idx, "llama-3-70b", "A10G"), []);
});

const sampleRun = {
  analysis: {
    ttft_s: { p50: 0.05, p95: 0.1, p99: 0.2, mean: 0.07, n: 30 },
    throughput: { tok_per_sec_avg: 480, requests_per_sec_avg: 6.0 },
  },
  prediction: {
    curve: [
      { batch: 1, step_ms: 15, tps: 60 },
      { batch: 8, step_ms: 16, tps: 480 },
      { batch: 32, step_ms: 18, tps: 1500 },
    ],
  },
};

test("projectRun: batchProxy + tps + ttftMs", () => {
  const p = projectRun(sampleRun);
  // rps=6 × ttft=0.05 + 1 = 1.3
  assert.ok(Math.abs(p.batchProxy - 1.3) < 1e-9);
  assert.equal(p.ttftMs, 50);
  assert.equal(p.tps, 480);
});

test("projectRun: divergenceRatio against nearest curve point", () => {
  // measured tps=480, nearest curve point at batch=1 has tps=60 (batchProxy≈1.3)
  const p = projectRun(sampleRun);
  // Nearest to 1.3 is batch=1 (tps=60), so 480/60 = 8.0
  assert.ok(Math.abs(p.divergenceRatio - 8.0) < 1e-9);
});

test("projectRun: deterministic — same input twice → equal output (C4)", () => {
  const a = projectRun(sampleRun);
  const b = projectRun(sampleRun);
  assert.deepStrictEqual(a, b);
});

test("projectRun: missing analysis fields → null projections", () => {
  const p = projectRun({ analysis: {} });
  assert.equal(p.batchProxy, null);
  assert.equal(p.tps, null);
  assert.equal(p.divergenceRatio, null);
});

test("schema-envelope tolerance (C3): unknown top-level fields don't break projection", () => {
  const run = {
    ...sampleRun,
    schema_version: "1.99.0",
    future_extension: { whatever: 42 },
    "experimental.thing": true,
  };
  const p = projectRun(run);
  assert.equal(p.tps, 480);  // still extractable
});

test("fetchRun: caches per href", async () => {
  const cache = new Map();
  let fetchCount = 0;
  const fetchImpl = async () => {
    fetchCount++;
    return { ok: true, status: 200, json: async () => ({ id: "r1" }) };
  };
  await fetchRun({ href: "runs/r1.json", cache, fetchImpl });
  await fetchRun({ href: "runs/r1.json", cache, fetchImpl });
  assert.equal(fetchCount, 1, "cache should prevent re-fetch");
});

test("emptyStateMessage: unavailable bridge", () => {
  const msg = emptyStateMessage({ unavailable: "404" });
  assert.match(msg, /No lab runs available/);
  assert.match(msg, /exp build-portal/);
});

test("emptyStateMessage: no matches but combos available", () => {
  const msg = emptyStateMessage({
    unavailable: false,
    availableCombos: [
      { model_ref: "llama-3-70b", hw_ref: "A10G" },
      { model_ref: "deepseek-v3", hw_ref: "H100-80GB" },
    ],
  });
  assert.match(msg, /No runs match/);
  assert.match(msg, /llama-3-70b/);
});
