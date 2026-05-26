// Pure helpers that build the lab-command snippet bits. Extracted from
// ui.mjs so they can be unit-tested without a DOM.
//
// Single source of truth for the calc↔lab rate convention. The lab's
// `exp plan` (experiments/cli/exp.py:_build_run_grid) uses the same
// formula — keep them in sync.

/**
 * Recommend an arrival RPS via Little's Law: λ ≈ batch / per-request-time.
 * We proxy per-request-time with the per-token SLO (TBT in ms). The
 * approximation is rough — a long-output request takes many tokens —
 * but it lines up with how the lab's `exp plan` advises rates, so the
 * calc-emitted command and the lab-suggested grid agree on the formula.
 *
 * Returned value:
 *   - floored at 1 rps (a degraded prediction with batch=0 must not
 *     produce `--rate 0` or `--rate NaN`),
 *   - rounded to 2 decimals for paste readability.
 *
 * @param {number} batch  recommended in-flight batch size
 * @param {number} tbtMs  target inter-token latency in ms
 * @returns {number}
 */
export function recommendedRate(batch, tbtMs) {
  const tbtSec = Math.max(tbtMs / 1000, 1e-3);
  const b = Number.isFinite(batch) && batch > 0 ? batch : 1;
  return Math.max(1, +((b / tbtSec).toFixed(2)));
}
