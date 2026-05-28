// Compatibility validator: declarative rules from data/compatibility.json
// classify (model × hardware × precision × ngpus) combos as runnable or not.
//
// Why a separate layer (not inside compute()): compute() is pure
// physics/math — roofline, KV bytes, parallelism. Kernel availability and
// HF-repo existence are *facts about the deployment ecosystem*, not facts
// about the math. Mixing them would couple the analytical model to
// fast-moving engine/vendor state.
//
// The output drives both diagnostic rendering and copy-button gating in
// ui.mjs. Each rule is data-driven: `when` describes what must match,
// `severity` is "error" | "warn", `reason` is human-readable, `suggest`
// is a remediation hint.

/**
 * Evaluate every rule against the calculator input.
 *
 * @param {object} input  — same shape compute() consumes
 *                          (hw, model, weight_prec, kv_prec, act_prec, ngpus, isl, osl, …)
 * @param {Array<object>} rules — parsed compatibility.json `rules` array
 * @returns {{ errors: Issue[], warns: Issue[] }}
 *   Issue: { id, level, msg, reason, suggest, rule }
 *   `msg` is `reason` + a one-line "Try: …" derived from `suggest`, suitable
 *   for direct display.
 */
export function validate(input, rules) {
  const errors = [];
  const warns = [];
  if (!input || !Array.isArray(rules)) return { errors, warns };
  for (const rule of rules) {
    if (!rule || !rule.when) continue;
    if (!ruleMatches(rule.when, input)) continue;
    const issue = {
      id: rule.id,
      level: rule.severity === "error" ? "error" : "warn",
      reason: rule.reason || "",
      suggest: rule.suggest || null,
      rule,
      msg: formatMessage(rule),
    };
    if (issue.level === "error") errors.push(issue);
    else warns.push(issue);
  }
  return { errors, warns };
}

/** Render `reason` + a one-line "Try: …" so the validator owns the wording. */
function formatMessage(rule) {
  const parts = [rule.reason || ""];
  const tip = formatSuggestion(rule.suggest);
  if (tip) parts.push(`Try: ${tip}`);
  return parts.join(" ");
}

function formatSuggestion(suggest) {
  if (!suggest || typeof suggest !== "object") return "";
  const bits = [];
  // Known input-field hints first so they read as "change this knob".
  const keyOrder = ["weight_prec", "kv_prec", "act_prec", "ngpus"];
  for (const k of keyOrder) {
    if (suggest[k] != null) bits.push(`${k.replace(/_/g, " ")} = ${suggest[k]}`);
  }
  if (suggest.note) bits.push(suggest.note);
  return bits.join(" · ");
}

/** All `when` keys must match. Unknown keys fail closed (treated as no-match
 *  so a typo in compatibility.json doesn't accidentally fire every rule). */
function ruleMatches(when, input) {
  for (const [key, expected] of Object.entries(when)) {
    const pred = PREDICATES[key];
    if (!pred) return false;            // unknown predicate → don't fire
    if (!pred(expected, input)) return false;
  }
  return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Predicate table — one entry per `when` key documented in compatibility.json.
// Each predicate is (expected, input) → boolean.
// ──────────────────────────────────────────────────────────────────────────
const PREDICATES = {
  weight_prec: (vals, i) => Array.isArray(vals) && vals.includes(i.weight_prec),
  kv_prec:     (vals, i) => Array.isArray(vals) && vals.includes(i.kv_prec),
  act_prec:    (vals, i) => Array.isArray(vals) && vals.includes(i.act_prec),

  // Hardware arch must come from a preset row; custom hw has no arch and
  // therefore can't fire any kernel-availability rule. That's intentional —
  // users in custom mode have opted out of the GPU-knowledge layer.
  gpu_arch: (vals, i) => Array.isArray(vals) && i.hw && vals.includes(i.hw.arch),

  // True when GPU has no hardware FP8 path. Encoded as a boolean predicate so
  // the rule reads naturally; flip in the future if we add "has FP8" rules.
  gpu_fp8_tflops_null: (expect, i) => Boolean(expect) === (i.hw?.fp8_tflops == null),

  model_moe:    (expect, i) => Boolean(expect) === Boolean(i.model?.moe),
  model_family: (vals, i)   => Array.isArray(vals) && vals.includes(i.model?.family),

  // Fires when the model's quant_variants does NOT contain the named token.
  // (Inverted on purpose — "missing" is the failure case we want to surface.)
  // Models without a quant_variants array are treated as missing-all so the
  // rule still fires (safer default — assume not available unless declared).
  model_quant_variants_missing: (token, i) => {
    const arr = Array.isArray(i.model?.quant_variants) ? i.model.quant_variants : [];
    return !arr.includes(token);
  },

  ngpus_lt: (n, i) => Number.isFinite(i.ngpus) && i.ngpus < n,
  ngpus_gt: (n, i) => Number.isFinite(i.ngpus) && i.ngpus > n,

  isl_plus_osl_gt_model_max_context: (expect, i) => {
    if (!expect) return true;           // rule disabled
    const isl = +i.isl || 0;
    const osl = +i.osl || 0;
    const max = +i.model?.max_context || 0;
    return max > 0 && (isl + osl) > max;
  },

  // weight_prec ≠ act_prec AND not both FP8. (When both are FP8 vLLM uses
  // the native FP8 tensor-core path — that's the "good" case.) The FP8+FP8
  // case is filtered out so the warn only surfaces actual mismatch.
  weight_act_prec_mismatch_non_fp8: (expect, i) => {
    if (!expect) return true;
    if (!i.weight_prec || !i.act_prec) return false;
    if (i.weight_prec === i.act_prec) return false;
    if (i.weight_prec === "FP8" && i.act_prec === "FP8") return false;
    return true;
  },
};
