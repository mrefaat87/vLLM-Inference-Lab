// Validation overlay: loads measured runs from the lab's bridge dir and
// projects them onto the calculator's scope panels.
//
// Why a separate module: calc.mjs and the existing UI must stay
// independent of any lab-side artifacts (the calculator must work on a
// clone-of-just-the-calc). validation.mjs is opt-in — bootstrap() in
// ui.mjs calls into it only after the main form is wired, and it
// degrades gracefully (empty state) when no bridge dir is present.

// The bundled calculator HTML lives at `calculators/sizing_calculator.html`;
// the bridge dir lives at `calculators/sizing_calc/lab_runs/`. Relative to
// the HTML, that's `./sizing_calc/lab_runs/`. The DEV_BRIDGE_URL is used
// only by tests that run validation.mjs from within `calculators/sizing_calc/`
// directly (where `./lab_runs/` is correct).
export const DEFAULT_BRIDGE_URL = "./sizing_calc/lab_runs/index.json";
export const DEV_BRIDGE_URL = "./lab_runs/index.json";

/**
 * Fetch the lab bridge index and per-run blobs.
 *
 * Returns:
 *   {
 *     index:   [ { run_id, model_ref, hw_ref, result_href, ... } ],
 *     runs:    Map<run_id, full result JSON>,
 *     unavailable: false | "reason string"
 *   }
 *
 * Never throws. Network failures and bad JSON resolve to
 * `unavailable: "<reason>"` so the caller can render an empty-state
 * card without try/catch boilerplate.
 */
export async function loadLabRuns({
  bridgeUrl = DEFAULT_BRIDGE_URL,
  fetchImpl = globalThis.fetch,
} = {}) {
  let index = [];
  try {
    const resp = await fetchImpl(bridgeUrl);
    if (!resp.ok) {
      return { index: [], runs: new Map(), unavailable: `${bridgeUrl} returned ${resp.status}` };
    }
    const blob = await resp.json();
    index = Array.isArray(blob.runs) ? blob.runs : [];
  } catch (e) {
    return { index: [], runs: new Map(), unavailable: `bridge fetch failed: ${e.message}` };
  }
  // Lazy: don't pre-fetch every per-run blob. Caller asks via fetchRun().
  return { index, runs: new Map(), unavailable: false };
}

/**
 * Filter the bridge index to runs whose join keys match the currently
 * selected (model.key, hw.key).
 *
 * Defensive: missing fields don't match anything. The filter is *strict
 * string equality* so any case/spacing drift between the calc's
 * models.json and the lab's --model-ref defaults surfaces here, not as
 * a silent empty overlay.
 */
export function filterRuns(index, modelKey, hwKey) {
  if (!Array.isArray(index)) return [];
  return index.filter(
    (r) => r && r.model_ref === modelKey && r.hw_ref === hwKey,
  );
}

/**
 * Project a fully-loaded run JSON into points consumable by the three
 * scope panels.
 *
 * Returns:
 *   {
 *     batchProxy: number,    // Little's Law: rps × ttft + 1
 *     ttftMs: number | null,
 *     tps: number | null,
 *     costPerMtok: number | null,
 *     divergenceRatio: number | null,  // measured tps / predicted tps at nearest curve point
 *   }
 *
 * The batch axis is a *proxy* — engines don't expose batch directly.
 * Documented prominently in the panel header so users don't read
 * Little's Law as ground truth.
 */
export function projectRun(run) {
  const a = run?.analysis || {};
  const ttft = a.ttft_s?.p50 ?? null;
  const rps = a.throughput?.requests_per_sec_avg ?? null;
  const tps = a.throughput?.tok_per_sec_avg ?? null;
  const batchProxy = (rps != null && ttft != null) ? Math.max(1, rps * ttft + 1) : null;

  const ttftMs = ttft != null ? ttft * 1000 : null;
  // Cost is recoverable from run's hardware row + tps; we lean on the
  // sister calc's cost formula at view time rather than baking $/Mtok
  // into the bridge summary.
  const costPerMtok = null;

  let divergenceRatio = null;
  const curve = run?.prediction?.curve;
  if (batchProxy != null && tps != null && Array.isArray(curve) && curve.length > 0) {
    const nearest = curve.reduce(
      (best, p) => (Math.abs(p.batch - batchProxy) < Math.abs(best.batch - batchProxy) ? p : best),
      curve[0],
    );
    if (nearest.tps > 0) divergenceRatio = tps / nearest.tps;
  }
  return { batchProxy, ttftMs, tps, costPerMtok, divergenceRatio };
}

/**
 * Lazily fetch a per-run blob by href (relative to the bridge dir).
 * Caches in the `runs` Map so the same run isn't re-fetched.
 */
export async function fetchRun({
  bridgeBase = "./lab_runs",
  href,
  cache,
  fetchImpl = globalThis.fetch,
} = {}) {
  if (!href) return null;
  if (cache && cache.has(href)) return cache.get(href);
  let blob = null;
  try {
    const resp = await fetchImpl(`${bridgeBase}/${href}`);
    if (!resp.ok) return null;
    blob = await resp.json();
  } catch {
    return null;
  }
  if (cache) cache.set(href, blob);
  return blob;
}

/**
 * Render an empty-state card into a panel element. Called when no
 * matching runs are found OR when the bridge dir is unreachable.
 *
 * `availableCombos`: optional [{model_ref, hw_ref}, …] so the user knows
 * what's actually been measured. Helps when the missing-match is because
 * the user picked a model that hasn't been benchmarked yet.
 */
export function emptyStateMessage({ unavailable, availableCombos = [] } = {}) {
  if (unavailable) {
    return (
      `No lab runs available (${unavailable}). ` +
      `Build the lab portal with: ` +
      `<code>exp build-portal --calc-bridge calculators/sizing_calc/lab_runs/</code>`
    );
  }
  if (availableCombos.length === 0) {
    return "Bridge is present but contains no runs.";
  }
  const lines = availableCombos
    .slice(0, 8)
    .map((c) => `<code>${c.model_ref} × ${c.hw_ref}</code>`)
    .join(", ");
  return (
    `No runs match the current selection. ` +
    `Available combinations: ${lines}` +
    (availableCombos.length > 8 ? ` (+${availableCombos.length - 8} more)` : "")
  );
}
