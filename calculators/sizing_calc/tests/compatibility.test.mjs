// Unit tests for the compatibility validator.
//
// One test per rule in compatibility.json, plus a few combinatorial cases.
// Every rule firing is a stable contract — if a rule changes ID or severity,
// these tests should fail loudly so we notice.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { validate } from "../src/compatibility.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const hardware = JSON.parse(readFileSync(join(here, "../src/data/hardware.json"))).hardware;
const models   = JSON.parse(readFileSync(join(here, "../src/data/models.json"))).models;
const RULES    = JSON.parse(readFileSync(join(here, "../src/data/compatibility.json"))).rules;
const HW       = Object.fromEntries(hardware.map((h) => [h.key, h]));
const MODEL    = Object.fromEntries(models.map((m) => [m.key, m]));

/** Build a complete-shaped input from a partial override. Sensible defaults so
 *  tests can declare only the field under examination. */
function buildInput(overrides = {}) {
  return {
    hw: HW["H100-80GB"],
    model: MODEL["llama-3-8b"],
    weight_prec: "FP16",
    kv_prec: "FP16",
    act_prec: "FP16",
    isl: 1024,
    osl: 256,
    tbt_ms: 50,
    ttft_ms: 2000,
    ngpus: 1,
    ...overrides,
  };
}

/** Returns true iff `id` appears in the listed issues. */
const has = (issues, id) => issues.some((i) => i.id === id);

// ──────────────────────────────────────────────────────────────────────────
// Per-rule tests
// ──────────────────────────────────────────────────────────────────────────

// int8-non-hopper and awq-marlin-needs-ampere are SOFT warnings — the combo
// runs, just slowly, and the lab auto-calibrates rate. Validates that they
// surface in `warns`, never in `errors` (which would gate the copy button).
test("int8-non-hopper: INT8 + T4 fires warn (not error — runs but slow)", () => {
  const { errors, warns } = validate(buildInput({ hw: HW["T4"], weight_prec: "INT8" }), RULES);
  assert.ok(has(warns, "int8-non-hopper"), "expected int8-non-hopper to fire as warn");
  assert.ok(!has(errors, "int8-non-hopper"), "must not gate the copy button");
});

test("int8-non-hopper: INT8 + A100-80GB fires warn", () => {
  const { errors, warns } = validate(buildInput({ hw: HW["A100-80GB"], weight_prec: "INT8" }), RULES);
  assert.ok(has(warns, "int8-non-hopper"));
  assert.ok(!has(errors, "int8-non-hopper"));
});

test("int8-hopper-suboptimal: INT8 + H100 fires warn, non-hopper rule does not fire", () => {
  const { errors, warns } = validate(buildInput({ hw: HW["H100-80GB"], weight_prec: "INT8" }), RULES);
  assert.ok(has(warns, "int8-hopper-suboptimal"));
  assert.ok(!has(warns, "int8-non-hopper"), "non-hopper rule should not fire on H100");
  assert.ok(!has(errors, "int8-non-hopper"));
});

test("fp8-no-hardware: FP8 + A10G fires error", () => {
  const { errors } = validate(buildInput({ hw: HW["A10G"], weight_prec: "FP8", act_prec: "FP8" }), RULES);
  assert.ok(has(errors, "fp8-no-hardware"));
});

test("fp8-no-hardware: FP8 + H100 does NOT fire", () => {
  const { errors } = validate(buildInput({ hw: HW["H100-80GB"], weight_prec: "FP8", act_prec: "FP8" }), RULES);
  assert.ok(!has(errors, "fp8-no-hardware"));
});

test("awq-no-repo: INT4 + deepseek-v3 fires (no community AWQ for DeepSeek-V3)", () => {
  // deepseek-v3 is also MoE, so we set ngpus=8 to isolate this rule from the
  // moe-needs-many-gpus error.
  const { errors } = validate(
    buildInput({ hw: HW["H100-80GB"], model: MODEL["deepseek-v3"], weight_prec: "INT4", ngpus: 8 }),
    RULES,
  );
  assert.ok(has(errors, "awq-no-repo"));
});

test("awq-no-repo: INT4 + qwen2.5-7b does NOT fire (Qwen has AWQ)", () => {
  const { errors } = validate(
    buildInput({ hw: HW["A100-80GB"], model: MODEL["qwen2.5-7b"], weight_prec: "INT4" }),
    RULES,
  );
  assert.ok(!has(errors, "awq-no-repo"));
});

test("awq-marlin-needs-ampere: INT4 + T4 fires warn (not error — runs but slow)", () => {
  // qwen2.5-7b has an AWQ variant, so awq-no-repo does not fire — isolates the Marlin rule.
  const { errors, warns } = validate(
    buildInput({ hw: HW["T4"], model: MODEL["qwen2.5-7b"], weight_prec: "INT4" }),
    RULES,
  );
  assert.ok(has(warns, "awq-marlin-needs-ampere"), "expected to fire as warn");
  assert.ok(!has(errors, "awq-marlin-needs-ampere"), "must not gate the copy button");
});

test("awq-marlin-needs-ampere: INT4 + A10G does NOT fire", () => {
  const { errors, warns } = validate(
    buildInput({ hw: HW["A10G"], model: MODEL["qwen2.5-7b"], weight_prec: "INT4" }),
    RULES,
  );
  assert.ok(!has(errors, "awq-marlin-needs-ampere"));
  assert.ok(!has(warns, "awq-marlin-needs-ampere"));
});

test("fp8-kv-no-hardware: FP8 KV + T4 fires error", () => {
  const { errors } = validate(buildInput({ hw: HW["T4"], kv_prec: "FP8" }), RULES);
  assert.ok(has(errors, "fp8-kv-no-hardware"));
});

test("fp8-kv-no-hardware: FP8 KV + A100 does NOT fire (Ampere has e5m2)", () => {
  const { errors } = validate(buildInput({ hw: HW["A100-80GB"], kv_prec: "FP8" }), RULES);
  assert.ok(!has(errors, "fp8-kv-no-hardware"));
});

test("kv-prec-int-not-supported: INT8 KV fires regardless of GPU", () => {
  for (const hwKey of ["T4", "A100-80GB", "H100-80GB"]) {
    const { errors } = validate(buildInput({ hw: HW[hwKey], kv_prec: "INT8" }), RULES);
    assert.ok(has(errors, "kv-prec-int-not-supported"), `expected fire on ${hwKey}`);
  }
});

test("moe-needs-many-gpus: deepseek-v3 + ngpus=4 fires error", () => {
  // Use FP8 weights so we don't trip awq-no-repo; we want this rule isolated.
  const { errors } = validate(
    buildInput({ hw: HW["H100-80GB"], model: MODEL["deepseek-v3"], weight_prec: "FP8", act_prec: "FP8", ngpus: 4 }),
    RULES,
  );
  assert.ok(has(errors, "moe-needs-many-gpus"));
});

test("moe-needs-many-gpus: deepseek-v3 + ngpus=8 does NOT fire", () => {
  const { errors } = validate(
    buildInput({ hw: HW["H100-80GB"], model: MODEL["deepseek-v3"], weight_prec: "FP8", act_prec: "FP8", ngpus: 8 }),
    RULES,
  );
  assert.ok(!has(errors, "moe-needs-many-gpus"));
});

test("moe-needs-many-gpus: dense model + ngpus=1 does NOT fire", () => {
  const { errors } = validate(buildInput({ ngpus: 1 }), RULES);
  assert.ok(!has(errors, "moe-needs-many-gpus"));
});

test("context-exceeds-model-max: ISL+OSL beyond max fires error", () => {
  // llama-3-8b max_context = 8192. 5000+5000 > 8192.
  const { errors } = validate(buildInput({ isl: 5000, osl: 5000 }), RULES);
  assert.ok(has(errors, "context-exceeds-model-max"));
});

test("context-exceeds-model-max: at-limit does NOT fire (strict >)", () => {
  const { errors } = validate(buildInput({ isl: 8000, osl: 192 }), RULES);
  assert.ok(!has(errors, "context-exceeds-model-max"));
});

test("weight-act-mismatch-no-fp8-path: FP8 weights + BF16 acts on H100 fires warn", () => {
  const { warns, errors } = validate(
    buildInput({ hw: HW["H100-80GB"], weight_prec: "FP8", act_prec: "BF16" }),
    RULES,
  );
  assert.ok(has(warns, "weight-act-mismatch-no-fp8-path"));
  assert.ok(!has(errors, "weight-act-mismatch-no-fp8-path"), "this rule is warn, not error");
});

test("weight-act-mismatch-no-fp8-path: both FP8 does NOT fire", () => {
  const { warns } = validate(
    buildInput({ hw: HW["H100-80GB"], weight_prec: "FP8", act_prec: "FP8" }),
    RULES,
  );
  assert.ok(!has(warns, "weight-act-mismatch-no-fp8-path"));
});

test("clean combo: FP16 + A10G + llama-3-8b + ngpus=1 produces no errors", () => {
  const { errors } = validate(buildInput({ hw: HW["A10G"], model: MODEL["llama-3-8b"] }), RULES);
  assert.equal(errors.length, 0, `unexpected errors: ${JSON.stringify(errors.map((e) => e.id))}`);
});

// ──────────────────────────────────────────────────────────────────────────
// Combinatorial / contract
// ──────────────────────────────────────────────────────────────────────────

test("combinatorial: INT8 + T4 + deepseek-v3 + ngpus=1 fires the hard blocker; INT8 surfaces as warn", () => {
  const { errors, warns } = validate(
    buildInput({ hw: HW["T4"], model: MODEL["deepseek-v3"], weight_prec: "INT8", kv_prec: "FP16", act_prec: "FP16", ngpus: 1 }),
    RULES,
  );
  // moe-needs-many-gpus is a hard error (capacity issue, lab can't fix it).
  assert.ok(has(errors, "moe-needs-many-gpus"));
  // int8-non-hopper is now a soft warn (lab calibrates the slow rate).
  assert.ok(has(warns, "int8-non-hopper"));
  assert.ok(!has(errors, "int8-non-hopper"));
  // awq-no-repo should NOT fire because weight_prec is INT8, not INT4.
  assert.ok(!has(errors, "awq-no-repo"));
});

test("issue shape: every issue carries id, level, msg, reason", () => {
  const { errors, warns } = validate(
    buildInput({ hw: HW["T4"], weight_prec: "INT8" }),
    RULES,
  );
  for (const issue of [...errors, ...warns]) {
    assert.ok(typeof issue.id === "string" && issue.id.length > 0, `bad id: ${issue.id}`);
    assert.ok(issue.level === "error" || issue.level === "warn");
    assert.ok(typeof issue.msg === "string" && issue.msg.length > 0);
    assert.ok(typeof issue.reason === "string");
  }
});

test("issue.msg includes 'Try:' suffix when a suggest block exists", () => {
  const { warns } = validate(buildInput({ hw: HW["T4"], weight_prec: "INT8" }), RULES);
  const issue = warns.find((e) => e.id === "int8-non-hopper");
  assert.ok(issue, "expected the rule to fire");
  assert.ok(/Try:/.test(issue.msg), `expected 'Try:' in msg: ${issue.msg}`);
});

test("empty rules array returns empty errors/warns", () => {
  const { errors, warns } = validate(buildInput(), []);
  assert.equal(errors.length, 0);
  assert.equal(warns.length, 0);
});

test("unknown predicate key in rule does not fire (safe-by-default)", () => {
  const bogus = [{
    id: "bogus",
    severity: "error",
    when: { not_a_real_predicate: true },
    reason: "should never fire",
  }];
  const { errors } = validate(buildInput(), bogus);
  assert.equal(errors.length, 0);
});

// ──────────────────────────────────────────────────────────────────────────
// Data integrity — every rule referenced above must exist in compatibility.json
// ──────────────────────────────────────────────────────────────────────────

test("compatibility.json: every rule has id, severity, reason, when", () => {
  for (const rule of RULES) {
    assert.ok(typeof rule.id === "string" && rule.id.length > 0, `bad rule id: ${JSON.stringify(rule)}`);
    assert.ok(rule.severity === "error" || rule.severity === "warn",
      `${rule.id} bad severity: ${rule.severity}`);
    assert.ok(typeof rule.reason === "string" && rule.reason.length > 0,
      `${rule.id} missing reason`);
    assert.ok(rule.when && typeof rule.when === "object",
      `${rule.id} missing when block`);
  }
});

test("compatibility.json: rule ids are unique", () => {
  const ids = RULES.map((r) => r.id);
  assert.equal(new Set(ids).size, ids.length, "duplicate rule ids");
});
