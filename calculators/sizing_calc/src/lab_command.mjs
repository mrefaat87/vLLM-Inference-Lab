// Pure helpers that build the lab-command snippet bits. Extracted from
// ui.mjs so they can be unit-tested without a DOM.
//
// Single source of truth for the calc↔lab rate convention. The lab's
// `exp plan` (experiments/cli/exp.py:_build_run_grid) uses the same
// formula — keep them in sync.

/**
 * Recommend an arrival RPS via Little's Law: λ ≈ batch / per-request-time.
 *
 * Per-request latency is TTFT (one-shot prefill cost) plus OSL × TBT
 * (streaming the output tokens). Using only TBT here is wrong by a
 * factor of OSL — for OSL=200 that pushes the emitted --rate ~200×
 * past what the engine can actually drain, drowning the run.
 *
 * Returned value:
 *   - floored at 1 rps (a degraded prediction with batch=0 must not
 *     produce `--rate 0` or `--rate NaN`),
 *   - rounded to 2 decimals for paste readability.
 *
 * @param {number} batch       recommended in-flight batch size
 * @param {number} tbtMs       target inter-token latency in ms
 * @param {number} oslTokens   expected output sequence length in tokens
 * @param {number} ttftMs      target time-to-first-token in ms
 * @returns {number}
 */
export function recommendedRate(batch, tbtMs, oslTokens, ttftMs) {
  const tbtSec = Number.isFinite(tbtMs) && tbtMs > 0 ? tbtMs / 1000 : 0;
  const ttftSec = Number.isFinite(ttftMs) && ttftMs > 0 ? ttftMs / 1000 : 0;
  const osl = Number.isFinite(oslTokens) && oslTokens > 0 ? oslTokens : 1;
  // Floor at 1ms so a fully-degraded input (all zeros) still divides cleanly.
  const perReqSec = Math.max(ttftSec + osl * tbtSec, 1e-3);
  const b = Number.isFinite(batch) && batch > 0 ? batch : 1;
  return Math.max(1, +((b / perReqSec).toFixed(2)));
}
