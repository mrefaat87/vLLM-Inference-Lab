"""Auto-rate calibration.

Before the measurement window starts, probe the running engine at a
geometric sweep of rates [1, 2, 4, 8, 16, 32] rps to find the highest
rate that does NOT saturate, then bisect once between the last
unsaturated and the first saturated rate. The measurement window then
runs at ``0.8 × ceiling`` for noise headroom.

The whole point of doing this in the lab (rather than reading the
calc's recommended_rate) is that the calc's analytical model assumes
optimal kernels exist for every (model × quant × hw) combo. For combos
like INT8 on T4 the engine falls back to slow kernels (bitsandbytes
runtime dequant) and the analytical recommendation runs 5–10× over the
real ceiling. Only the running engine knows its real ceiling.

Saturation rule (any TWO of three signals trip, OR hard-fail short-circuit):
  1. Completion lag: completed/dispatched < 0.90.
  2. TTFT p95 > 2 × ttft_slo_ms.
  3. Achieved RPS < 0.85 × target rate.
  Hard fail: > 10% of requests errored.

Two-of-three (not any-of-three) is intentional: each signal alone can
trip from transient noise; two together is a much stronger signal.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from experiments.runner.schema import Calibration, CalibrationProbe, RequestRecord

PROBE_S: float = 5.0
# Per-probe wall-clock cap: PROBE_S × this factor. Calibrates the
# patience/throughput tradeoff:
#   - too tight (×2): slow engines like AWQ-on-T4 (12s residence) get false
#     "saturated" reads at probe(1) when one of the 5 requests doesn't drain.
#   - too loose (×10): on truly broken engines a single probe burns 50s and
#     the budget runs out before saturation is found.
# Factor 4 (20s cap) gives ~8s drain headroom on top of typical 12s residence.
# In-flight requests at the cap are cancelled and recorded with error
# "ProbeCutoff", which the saturation rule counts as completion lag.
PROBE_WALL_CLOCK_FACTOR: float = 4.0
# The probe schedule starts at ``start_rate`` (default 1.0) and doubles
# geometrically until either saturation trips or the budget runs out.
# Fixed-length schedules under-report the ceiling when nothing saturates.
DEFAULT_START_RATE: float = 1.0
# Cap on how far we'll double. Keeps a runaway probe (misconfigured engine
# that never saturates) from chewing the whole budget.
MAX_PROBE_RATE: float = 256.0
HEADROOM: float = 0.8
# Number of bisection refinements after the coarse sweep. K=3 narrows the
# ceiling window by 2^3 = 8× — for a [4, 8] coarse window, the final
# uncertainty is ~0.5 rps (≈ 12% precision relative to the ceiling).
K_BISECTIONS: int = 3
# If last_safe and first_sat are already within this gap, additional
# bisections add no useful precision — skip remaining iterations.
MIN_BISECT_GAP: float = 0.25
# Budget covers the worst case: ~9 probes (seed + doubled past 32 up to 256)
# at PROBE_S × PROBE_WALL_CLOCK_FACTOR = 10s each → 90s nominal. Reserve the
# rest as headroom for engine warm-up bias on the first probe.
DEFAULT_BUDGET_S: float = 240.0
DEFAULT_TTFT_SLO_MS: float = 1500.0
TTFT_MULTIPLIER: float = 2.0
SUCCESS_FLOOR: float = 0.90
ACHIEVED_FLOOR: float = 0.85
HARD_FAIL_FRAC: float = 0.10
# If even the lowest probe saturates, we still want to log *something*
# rather than abort — half the smallest probe rate is conservative and
# usually low enough to drain.
FALLBACK_RATE_RPS: float = 0.5

# A workload/run callable: given a target arrival rate and a seed,
# dispatch a PROBE_S-second burst and return per-request records. The
# helper accepts this as an injection point so unit tests can drive
# saturation curves analytically without spinning up an engine.
ProbeFn = Callable[[float, int], Awaitable[list[RequestRecord]]]


def _percentile(xs: list[float], q: float) -> float:
    """Nearest-rank percentile. Returns 0.0 on empty input.

    We deliberately do NOT raise on empty — an empty probe means every
    request failed, which is handled by the hard-fail short-circuit
    upstream (success_rate=0 trips the lag signal too).
    """
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def evaluate_probe(
    *,
    records: list[RequestRecord],
    probe_s: float,
    target_rate: float,
    ttft_slo_ms: float,
) -> CalibrationProbe:
    """Score one burst of records against the saturation rule."""
    dispatched = len(records)
    if dispatched == 0:
        # Engine refused all connections or workload yielded nothing.
        # Conservative: treat as saturated so we don't push higher.
        return CalibrationProbe(
            rate=target_rate, success_rate=0.0,
            ttft_p95_ms=0.0, achieved_rps=0.0, saturated=True,
        )
    completed = [r for r in records if r.error is None and r.ttft_s is not None]
    # ProbeCutoff is a SOFT signal — the engine accepted the request but the
    # probe's wall-clock cap fired before completion. It indicates in-flight
    # overhang (a real saturation symptom) but is NOT an engine error like
    # 5xx / network failure. The hard-fail short-circuit, intended for true
    # engine misconfig, only counts hard errors; cutoffs flow into the
    # two-of-three rule via signals 1 and 3 instead.
    hard_errors = [
        r for r in records
        if r.error is not None and not r.error.startswith("ProbeCutoff")
    ]
    success_rate = len(completed) / dispatched
    achieved_rps = len(completed) / probe_s
    ttft_ms_list = [r.ttft_s * 1000.0 for r in completed if r.ttft_s is not None]
    ttft_p95_ms = _percentile(ttft_ms_list, 0.95)

    hard_fail = (len(hard_errors) / dispatched) > HARD_FAIL_FRAC
    sig_lag = success_rate < SUCCESS_FLOOR
    sig_ttft = ttft_p95_ms > TTFT_MULTIPLIER * ttft_slo_ms
    sig_rps = achieved_rps < ACHIEVED_FLOOR * target_rate
    tripped = int(sig_lag) + int(sig_ttft) + int(sig_rps)
    saturated = hard_fail or tripped >= 2

    return CalibrationProbe(
        rate=target_rate,
        success_rate=success_rate,
        ttft_p95_ms=ttft_p95_ms,
        achieved_rps=achieved_rps,
        saturated=saturated,
    )


def _probe_schedule(
    start_rate: float,
    *,
    max_rate: float,
) -> list[float]:
    """Geometric probe rates starting at ``start_rate`` and doubling up.

    Calibration consumes this lazily and stops at the first saturated probe
    or when the budget is exhausted, so the long tail isn't wasted work — it
    only matters when the schedule didn't saturate, in which case we want
    to keep pushing instead of bailing with an under-reported ceiling.

    When ``start_rate > 1.0`` callers can skip the cheap low probes
    (~17 s each) if they already know the engine handles them — e.g. a
    re-run on a calibrated config.
    """
    if start_rate <= 0:
        start_rate = DEFAULT_START_RATE
    rates = [start_rate]
    while rates[-1] * 2 <= max_rate:
        rates.append(rates[-1] * 2)
    return rates


async def calibrate(
    *,
    probe_fn: ProbeFn,
    ttft_slo_ms: float = DEFAULT_TTFT_SLO_MS,
    probe_s: float = PROBE_S,
    budget_s: float = DEFAULT_BUDGET_S,
    start_rate: float = DEFAULT_START_RATE,
    max_probe_rate: float = MAX_PROBE_RATE,
    seed_offset: int = 7919,  # large prime, won't collide with measurement seed
    k_bisections: int = K_BISECTIONS,
) -> Calibration:
    """Run the geometric sweep + K bisection probes; return a Calibration."""
    deadline = time.monotonic() + budget_s
    probes: list[CalibrationProbe] = []
    last_safe: float | None = None
    first_sat: float | None = None
    seed_step = 0

    for r in _probe_schedule(start_rate, max_rate=max_probe_rate):
        if time.monotonic() > deadline:
            break
        records = await probe_fn(r, seed_offset + seed_step)
        seed_step += 1
        probe = evaluate_probe(
            records=records, probe_s=probe_s,
            target_rate=r, ttft_slo_ms=ttft_slo_ms,
        )
        probes.append(probe)
        if probe.saturated:
            first_sat = r
            break
        last_safe = r

    # K-bisection refinement: each bisection halves the [last_safe, first_sat]
    # window, so K=3 narrows by 8× — e.g. a coarse [4, 8] window settles to
    # ~0.5 rps precision. The loop respects both the budget and a minimum-
    # gap floor: once the window is below MIN_BISECT_GAP, additional probes
    # don't move the ceiling meaningfully.
    for _ in range(k_bisections):
        if last_safe is None or first_sat is None:
            break
        if first_sat - last_safe < MIN_BISECT_GAP:
            break
        if time.monotonic() > deadline:
            break
        mid = (last_safe + first_sat) / 2.0
        records = await probe_fn(mid, seed_offset + seed_step)
        seed_step += 1
        probe = evaluate_probe(
            records=records, probe_s=probe_s,
            target_rate=mid, ttft_slo_ms=ttft_slo_ms,
        )
        probes.append(probe)
        if probe.saturated:
            first_sat = mid
        else:
            last_safe = mid

    if last_safe is None:
        # Even the lowest probe saturated. Selected_rate falls back to a
        # safe floor; capacity_ceiling reports the smallest probed rate
        # (the only thing we know is "ceiling is at or below this"). The
        # explicit FALLBACK_RATE_RPS keeps the measurement run viable
        # rather than aborting on a hard floor.
        floor_rate = probes[0].rate if probes else start_rate
        return Calibration(
            method="auto", probes=probes,
            selected_rate=FALLBACK_RATE_RPS,
            capacity_ceiling=floor_rate,
        )

    # If we exhausted the schedule (or the budget) without ever saturating,
    # last_safe is the highest probed rate but the real ceiling is unknown
    # (it's at least last_safe, possibly much higher). selected_rate still
    # uses the headroom factor so the measurement window is bounded — but
    # any downstream reader can tell from `probes[-1].saturated == False`
    # that the ceiling is a lower bound, not an exact value.
    return Calibration(
        method="auto", probes=probes,
        selected_rate=HEADROOM * last_safe,
        capacity_ceiling=last_safe,
    )


def explicit_calibration(rate: float) -> Calibration:
    """Uniform-shape record for an explicit --rate run (no probes)."""
    return Calibration(
        method="explicit", probes=[],
        selected_rate=rate, capacity_ceiling=rate,
    )
