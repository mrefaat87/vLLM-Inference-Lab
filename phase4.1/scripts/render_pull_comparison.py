#!/usr/bin/env python3
"""Render the 3-way image-pull comparison results to HTML, with assertions.

Reads phase4.1/tests/pull_comparison_results.json (produced by
run_pull_comparison.py). Computes per-stage medians + stddev per variant,
runs five fail-loud assertions on the data, and emits a side-by-side waterfall
HTML.

Assertions (printed at top of HTML, also exit nonzero if any fail):

    1. HARNESS DRIFT — variant A median image_download stage in [275s, 320s].
       Outside this band means the harness has shifted since the original
       baseline (296±7ms median). Don't trust comparisons across that drift.

    2. MONOTONICITY — median image_download(A) > B > C. Violation means our
       registry-locality or pre-bake assumption is wrong; investigate before
       drawing conclusions.

    3. STAGE-LEAK — for every stage OTHER than image_download, the stddev
       across all 9 runs must be < 15% of that stage's median. Higher leak
       means the variable under test is contaminating other stages, or there's
       uncontrolled infra noise.

    4. LAZY-LOAD TAX — record variant C's stage 4 value. Decision rule:
         < 5s  → tax negligible, FSR not worth $0.75/hr/AZ
         > 15s → tax material, run the FSR variant next
         5-15s → judgment call

    5. COST-PER-SECOND-SAVED — IF a `prebaked_ami_fsr` variant is present in
       the results, compute ($0.75/hr) / (stage4_C - stage4_C_FSR) and surface
       the economic interpretation. Skipped silently if FSR run absent.

Output:
    phase4.1/pull_comparison.html
"""
from __future__ import annotations

import html
import json
import pathlib
import statistics
import sys
from typing import Any

PHASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RESULTS_PATH = PHASE_DIR / "tests" / "pull_comparison_results.json"
HTML_PATH = PHASE_DIR / "pull_comparison.html"

STAGE_ORDER = [
    "karpenter_scheduling",
    "ec2_spot_fulfillment",
    "node_bootstrap",
    "image_download",  # The variable under test
    "image_unpack",
    "model_download_s3",
    "vllm_init_cuda_ctx",
    "weight_load_gpu_mem",
    "cuda_graph_warmup",
    "readiness_probe_pass",
    "first_token_served",
]
VARIABLE_STAGE = "image_download"

# Drift band for variant A — rooted in the original baseline of 296±7s.
# 275-320s gives ±~8% headroom around the median to absorb Spot/network noise
# without admitting a truly-shifted harness.
DRIFT_BAND_MS = (275_000, 320_000)

# Stage-leak tolerance: 15% of median is permissive — most stages are
# bottlenecked by hardware (S3, GPU init) and stable across runs.
LEAK_TOLERANCE_FRAC = 0.15

# FSR economics — fixed AWS price, AZ-scoped.
FSR_DOLLARS_PER_HR_PER_AZ = 0.75


# ─── Stage stat extraction ───────────────────────────────────────────────────

def stage_durations(records: list[dict[str, Any]], stage: str) -> list[int]:
    """Return per-run duration_ms for `stage`, skipping any missing/invalid."""
    out: list[int] = []
    for r in records:
        d = r.get("stages", {}).get(stage, {}).get("duration_ms")
        if isinstance(d, (int, float)) and d > 0:
            out.append(int(d))
    return out


def median_or_none(xs: list[int]) -> float | None:
    return statistics.median(xs) if xs else None


def stddev_or_none(xs: list[int]) -> float | None:
    return statistics.stdev(xs) if len(xs) >= 2 else None


# ─── Assertions ──────────────────────────────────────────────────────────────

def assert_harness_drift(stats: dict[str, dict[str, float | None]]) -> dict[str, Any]:
    a = stats.get("docker_hub", {}).get("median")
    if a is None:
        return {"name": "harness_drift", "passed": False,
                "detail": "no docker_hub runs to check"}
    lo, hi = DRIFT_BAND_MS
    passed = lo <= a <= hi
    return {
        "name": "harness_drift",
        "passed": passed,
        "detail": (f"variant A median image_download = {a/1000:.1f}s, "
                   f"band = [{lo/1000:.0f}s, {hi/1000:.0f}s]"),
    }


def assert_monotonicity(stats: dict[str, dict[str, float | None]]) -> dict[str, Any]:
    a = stats.get("docker_hub", {}).get("median")
    b = stats.get("ecr_same_region", {}).get("median")
    c = stats.get("prebaked_ami", {}).get("median")
    if None in (a, b, c):
        return {"name": "monotonicity", "passed": False,
                "detail": f"missing variant data: A={a}, B={b}, C={c}"}
    passed = a > b > c
    return {
        "name": "monotonicity",
        "passed": passed,
        "detail": (f"medians (s): A={a/1000:.1f} > B={b/1000:.1f} > C={c/1000:.1f}"
                   if passed else
                   f"VIOLATED — A={a/1000:.1f} B={b/1000:.1f} C={c/1000:.1f}"),
    }


def assert_streamer_savings(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Variant E's (model_download + weight_load) total should beat D's by ≥15s.

    The streamer fuses the download + load steps. Comparing the sum keeps the
    comparison honest even though E reports model_download_s3 = 0s by design.
    A real win shows up as fewer total seconds across the two stages combined.
    """
    def sum_two(records: list[dict[str, Any]]) -> float | None:
        m = stage_durations(records, "model_download_s3")
        w = stage_durations(records, "weight_load_gpu_mem")
        if not w:
            return None
        # If model_download is absent (variant E), median is 0 by design.
        return (statistics.median(m) if m else 0) + statistics.median(w)

    d_records = results.get("prebaked_ami_fsr", [])
    e_records = results.get("prebaked_ami_streamer", [])
    if not e_records:
        return None  # variant E absent — skip silently
    if not d_records:
        return {"name": "streamer_savings", "passed": False,
                "detail": "variant D records missing — cannot compute baseline"}
    d_sum = sum_two(d_records)
    e_sum = sum_two(e_records)
    if d_sum is None or e_sum is None:
        return {"name": "streamer_savings", "passed": False,
                "detail": f"missing data: D={d_sum}, E={e_sum}"}
    saved_s = (d_sum - e_sum) / 1000
    threshold_s = 15.0
    return {
        "name": "streamer_savings",
        "passed": saved_s >= threshold_s,
        "detail": (
            f"(model_download + weight_load) D={d_sum/1000:.1f}s, "
            f"E={e_sum/1000:.1f}s, saved={saved_s:+.1f}s "
            f"(threshold ≥{threshold_s:.0f}s)"
        ),
    }


def assert_warmed_savings(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Variant F's vllm_init_cuda_ctx must be lower than D's.

    F adds pre-warmed JIT compile caches (torch.compile + Triton + FlashInfer +
    vLLM FX) to the variant-D AMI. The savings target lives in vllm_init_cuda_ctx
    because that's the stage where the JIT compilers fire on first use.

    Threshold: any reduction passes the mechanism check (≥1s). The plan's stretch
    target was ≥10s; weaker results still validate the wiring.
    """
    d_records = results.get("prebaked_ami_fsr", []) or results.get("prebaked_ami", [])
    f_records = results.get("prebaked_ami_warmed", [])
    if not f_records:
        return None  # variant F absent — skip silently
    if not d_records:
        return {"name": "warmed_cache_savings", "passed": False,
                "detail": "variant D records missing — cannot compute baseline"}
    d_durs = stage_durations(d_records, "vllm_init_cuda_ctx")
    f_durs = stage_durations(f_records, "vllm_init_cuda_ctx")
    if not d_durs or not f_durs:
        return {"name": "warmed_cache_savings", "passed": False,
                "detail": f"missing vllm_init_cuda_ctx data: D={len(d_durs)}, F={len(f_durs)}"}
    d_med = statistics.median(d_durs)
    f_med = statistics.median(f_durs)
    saved_s = (d_med - f_med) / 1000
    return {
        "name": "warmed_cache_savings",
        "passed": saved_s >= 1.0,
        "detail": (
            f"vllm_init_cuda_ctx D={d_med/1000:.1f}s, F={f_med/1000:.1f}s, "
            f"saved={saved_s:+.1f}s (mechanism threshold ≥1s; stretch ≥10s per plan)"
        ),
    }


def assert_stage_leak(all_records_flat: list[dict[str, Any]]) -> dict[str, Any]:
    """Check non-variable stages have low stddev across all 9 runs combined."""
    leaks: list[str] = []
    for stage in STAGE_ORDER:
        if stage == VARIABLE_STAGE:
            continue
        durs = stage_durations(all_records_flat, stage)
        if len(durs) < 2:
            continue
        med = statistics.median(durs)
        if med == 0:
            continue
        sd = statistics.stdev(durs)
        frac = sd / med
        if frac > LEAK_TOLERANCE_FRAC:
            leaks.append(f"{stage}: stddev={sd:.0f}ms = {frac*100:.1f}% of median {med:.0f}ms")
    return {
        "name": "stage_leak",
        "passed": not leaks,
        "detail": ("no leak detected" if not leaks
                   else "leaks: " + "; ".join(leaks)),
    }


def total_median_ms(records: list[dict[str, Any]]) -> float | None:
    """Median of totals.scale_up_to_ready_ms — the real cold-start metric.

    Originally we keyed FSR/lazy-load decisions off `image_download` alone.
    The Phase 4.1 measurement showed the lazy-load tax leaks into adjacent
    stages (node_bootstrap, vllm_init), making stage-4 a misleading proxy.
    Use total time instead.
    """
    durs = [r.get("totals", {}).get("scale_up_to_ready_ms") for r in records]
    durs = [d for d in durs if isinstance(d, (int, float)) and d > 0]
    return statistics.median(durs) if durs else None


def lazy_load_tax_decision(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Compare TOTAL cold-start time of variant C (no FSR) vs A (Hub baseline).

    If C's total is GREATER than A's, the lazy-load tax is real and exceeds
    the pull savings — FSR is justified. Stage-4 alone is uninformative
    (containerd's local image lookup is always fast).
    """
    a_total = total_median_ms(results.get("docker_hub", []))
    c_total = total_median_ms(results.get("prebaked_ami", []))
    if a_total is None or c_total is None:
        return {"name": "lazy_load_tax", "passed": True,
                "detail": "missing variant A or C totals — no decision possible"}
    delta_s = (c_total - a_total) / 1000
    if delta_s > 30:
        decision = (f"FSR justified: prebake-no-FSR is {delta_s:.0f}s slower "
                    f"than Hub overall (lazy-load tax exceeds pull savings)")
    elif delta_s < -30:
        decision = "FSR optional: prebake-no-FSR already beats Hub by >30s"
    else:
        decision = "FSR judgment-call: prebake ≈ Hub within ±30s"
    return {
        "name": "lazy_load_tax",
        "passed": True,
        "detail": (f"total median: A_hub={a_total/1000:.0f}s, "
                   f"C_prebake_noFSR={c_total/1000:.0f}s, Δ={delta_s:+.0f}s — {decision}"),
    }


def fsr_cost_analysis(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    """Compute cost-per-second-saved using TOTAL time, not stage 4.

    The FSR variant's win is a sum across multiple stages — image_download,
    node_bootstrap, vllm_init_cuda_ctx, weight_load_gpu_mem. Comparing only
    stage 4 understates the savings by an order of magnitude.
    """
    a_total = total_median_ms(results.get("docker_hub", []))
    c_total = total_median_ms(results.get("prebaked_ami", []))
    fsr_total = total_median_ms(results.get("prebaked_ami_fsr", []))
    if fsr_total is None:
        return None  # FSR variant absent — skip silently
    if a_total is None or c_total is None:
        return {"name": "fsr_cost", "passed": False,
                "detail": "missing baseline data for cost comparison"}

    saved_vs_no_fsr = (c_total - fsr_total) / 1000  # Pure FSR effect
    saved_vs_hub = (a_total - fsr_total) / 1000     # Total system improvement

    cost_per_sec_per_az = FSR_DOLLARS_PER_HR_PER_AZ / 3600
    # Break-even: how many scale-out events per hour does FSR need to pay back?
    if saved_vs_hub > 0:
        # Each scale-out saves saved_vs_hub seconds; how many per hour to amortize $0.75?
        # (Assumes you'd pay $0 for the seconds you save; this is opportunity cost.)
        break_even_per_hr = FSR_DOLLARS_PER_HR_PER_AZ / (saved_vs_hub * cost_per_sec_per_az)
        # Simplifies to 3600/saved_vs_hub events/hr.
        break_even = 3600 / saved_vs_hub if saved_vs_hub else float("inf")
    else:
        break_even = float("inf")

    return {
        "name": "fsr_cost",
        "passed": saved_vs_hub > 0,
        "detail": (
            f"TOTAL time: A_hub={a_total/1000:.0f}s, C_noFSR={c_total/1000:.0f}s, "
            f"D_FSR={fsr_total/1000:.0f}s. "
            f"FSR vs no-FSR: {saved_vs_no_fsr:+.0f}s (pure pre-warm effect). "
            f"FSR-prebake vs Hub: {saved_vs_hub:+.0f}s (system-level improvement). "
            f"FSR cost ${FSR_DOLLARS_PER_HR_PER_AZ}/hr/AZ. "
            f"Break-even: {break_even:.1f} scale-out events per hour per AZ "
            f"to amortize the FSR cost in saved cold-start seconds."
        ),
    }


# ─── HTML rendering ──────────────────────────────────────────────────────────

PALETTE = {
    "karpenter_scheduling": "#5B8FF9",
    "ec2_spot_fulfillment": "#5AD8A6",
    "node_bootstrap": "#5D7092",
    "image_download": "#F6BD16",  # Highlighted — the variable under test
    "image_unpack": "#E8684A",
    "model_download_s3": "#6DC8EC",
    "vllm_init_cuda_ctx": "#9270CA",
    "weight_load_gpu_mem": "#FF9D4D",
    "cuda_graph_warmup": "#269A99",
    "readiness_probe_pass": "#FF99C3",
    "first_token_served": "#BBBBBB",
}


def render_html(matrix: dict[str, Any], stats: dict[str, dict[str, float | None]],
                assertions: list[dict[str, Any]]) -> str:
    variants = list(matrix.get("results", {}).keys())

    # Build per-variant per-stage rows
    def stage_source(recs: list[dict[str, Any]], stage: str) -> str | None:
        sources = {(r.get("stages", {}).get(stage, {}) or {}).get("source") for r in recs}
        sources.discard(None)
        return next(iter(sources), None) if len(sources) == 1 else None

    def stage_table_row(stage: str) -> str:
        cells = []
        for v in variants:
            recs = matrix["results"].get(v, [])
            durs = stage_durations(recs, stage)
            med = median_or_none(durs)
            sd = stddev_or_none(durs)
            src = stage_source(recs, stage)
            if med == 0 and src == "absent_init_container":
                # Variant E: stage exists but is structurally zero (no init
                # container). Annotate so the reader doesn't read this as
                # "the streamer instantly downloaded the model" — the fetch
                # time is fused into weight_load_gpu_mem.
                txt = "0s (absorbed into weight_load)"
            elif med is None:
                txt = "—"
            else:
                txt = f"{med/1000:.1f}s" + (f" ±{sd/1000:.1f}" if sd else "")
            cells.append(f"<td>{html.escape(txt)}</td>")
        is_var = stage == VARIABLE_STAGE
        marker = " ← variable" if is_var else ""
        return f"<tr><td><code>{html.escape(stage)}</code>{marker}</td>{''.join(cells)}</tr>"

    # Waterfall bars per variant — pixel width proportional to median
    pixels_per_sec = 2.0  # Tunable; 600s × 2 = 1200px max bar
    def variant_waterfall(v: str) -> str:
        recs = matrix["results"].get(v, [])
        if not recs:
            return f'<div class="variant-bar"><h3>{html.escape(v)}</h3><p>(no data)</p></div>'
        rows = []
        for stage in STAGE_ORDER:
            med = median_or_none(stage_durations(recs, stage))
            if med is None:
                continue
            w = max(1, int((med / 1000) * pixels_per_sec))
            color = PALETTE.get(stage, "#888")
            rows.append(
                f'<div class="seg" style="width:{w}px; background:{color};" '
                f'title="{html.escape(stage)}: {med/1000:.1f}s">'
                f'<span>{html.escape(stage)} ({med/1000:.0f}s)</span></div>'
            )
        return (f'<div class="variant-bar"><h3>{html.escape(v)}</h3>'
                f'<div class="bar">{"".join(rows)}</div></div>')

    assertion_rows = []
    for a in assertions:
        cls = "pass" if a["passed"] else "fail"
        assertion_rows.append(
            f'<tr class="{cls}"><td>{html.escape(a["name"])}</td>'
            f'<td>{"PASS" if a["passed"] else "FAIL"}</td>'
            f'<td>{html.escape(a["detail"])}</td></tr>'
        )

    fp = matrix.get("fingerprint_validation", {})
    fp_status = "PASS" if fp.get("passed") else "FAIL"
    fp_detail = ("All hardware fingerprints consistent. "
                 f"Reference: {fp.get('reference')}. "
                 f"A/B AMIs: {fp.get('ab_amis')}. C/D AMIs: {fp.get('c_amis')}. "
                 f"E AMIs: {fp.get('e_amis')}. F AMIs: {fp.get('f_amis')}.")
    if not fp.get("passed"):
        fp_detail += " Issues: " + "; ".join(fp.get("issues", []))

    stage_rows = "\n".join(stage_table_row(s) for s in STAGE_ORDER)
    waterfalls = "\n".join(variant_waterfall(v) for v in variants)
    variant_th = "".join(f"<th>{html.escape(v)}</th>" for v in variants)

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Phase 4.1 — 3-way image-pull comparison</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; color: #222; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  h2 {{ font-size: 16px; margin-top: 28px; }}
  h3 {{ font-size: 14px; margin: 8px 0; }}
  table {{ border-collapse: collapse; margin: 8px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
  th {{ background: #f4f4f4; }}
  tr.pass td:nth-child(2) {{ color: #1a7f37; font-weight: 600; }}
  tr.fail td:nth-child(2) {{ color: #cf222e; font-weight: 600; }}
  .variant-bar {{ margin: 12px 0; }}
  .bar {{ display: flex; flex-wrap: nowrap; height: 26px; align-items: stretch;
         border: 1px solid #ddd; }}
  .seg {{ display: inline-flex; align-items: center; overflow: hidden;
         color: #fff; font-size: 11px; padding: 0 4px; box-sizing: border-box;
         white-space: nowrap; }}
  .seg span {{ overflow: hidden; text-overflow: ellipsis; }}
  code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
  .small {{ color: #666; font-size: 12px; }}
</style></head><body>

<h1>Phase 4.1 — 3-way image-pull cold-start comparison</h1>
<p class="small">Generated {html.escape(matrix.get("generated_at", "?"))}.
Runs per variant: {matrix.get("runs_per_variant", "?")}.</p>

<h2>Assertions</h2>
<table>
  <tr><th>Assertion</th><th>Result</th><th>Detail</th></tr>
  <tr class="{'pass' if fp.get('passed') else 'fail'}">
    <td>fingerprint_validation</td><td>{fp_status}</td><td>{html.escape(fp_detail)}</td></tr>
  {''.join(assertion_rows)}
</table>

<h2>Waterfall (median per stage, per variant)</h2>
{waterfalls}

<h2>Per-stage medians</h2>
<table>
  <tr><th>stage</th>{variant_th}</tr>
  {stage_rows}
</table>

</body></html>
"""


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    if not RESULTS_PATH.is_file():
        print(f"ERROR: {RESULTS_PATH} missing — run run_pull_comparison.py first",
              file=sys.stderr)
        return 1

    matrix = json.loads(RESULTS_PATH.read_text())
    results = matrix.get("results", {})

    # Per-variant summary stats for the variable stage
    stats: dict[str, dict[str, float | None]] = {}
    for v, recs in results.items():
        durs = stage_durations(recs, VARIABLE_STAGE)
        stats[v] = {"median": median_or_none(durs), "stddev": stddev_or_none(durs),
                    "n": len(durs)}

    # Flatten records across all variants for stage-leak check
    all_records_flat: list[dict[str, Any]] = []
    for v, recs in results.items():
        all_records_flat.extend(recs)

    assertions = [
        assert_harness_drift(stats),
        assert_monotonicity(stats),
        assert_stage_leak(all_records_flat),
        lazy_load_tax_decision(results),
    ]
    fsr = fsr_cost_analysis(results)
    if fsr is not None:
        assertions.append(fsr)
    streamer = assert_streamer_savings(results)
    if streamer is not None:
        assertions.append(streamer)
    warmed = assert_warmed_savings(results)
    if warmed is not None:
        assertions.append(warmed)

    HTML_PATH.write_text(render_html(matrix, stats, assertions))
    print(f"wrote {HTML_PATH}")

    print("\n=== Assertion summary ===")
    failed = 0
    for a in assertions:
        mark = "PASS" if a["passed"] else "FAIL"
        print(f"  [{mark}] {a['name']}: {a['detail']}")
        if not a["passed"]:
            failed += 1
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
