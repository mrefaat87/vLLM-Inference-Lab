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
# The probe schedule is a *seed* — calibrate() keeps doubling past the last
# entry until either saturation trips or the budget runs out. Fixed-length
# schedules under-report the ceiling when no rate in the seed saturates.
PROBE_SCHEDULE_SEED: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
# Cap on how far we'll double past the seed. Keeps a runaway probe (mis-
# configured engine that never saturates) from chewing the whole budget.
MAX_PROBE_RATE: float = 256.0
HEADROOM: float = 0.8
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
    seed: tuple[float, ...],
    *,
    max_rate: float,
) -> list[float]:
    """Geometric probe rates: the seed, then keep doubling past it.

    Calibration consumes this lazily and stops at the first saturated probe
    or when the budget is exhausted, so the long tail isn't wasted work — it
    only matters when the seed didn't saturate, in which case we want to keep
    pushing instead of bailing with an under-reported ceiling.
    """
    rates = list(seed)
    last = rates[-1] if rates else 1.0
    while last * 2 <= max_rate:
        last *= 2
        rates.append(last)
    return rates


async def calibrate(
    *,
    probe_fn: ProbeFn,
    ttft_slo_ms: float = DEFAULT_TTFT_SLO_MS,
    probe_s: float = PROBE_S,
    budget_s: float = DEFAULT_BUDGET_S,
    schedule: tuple[float, ...] = PROBE_SCHEDULE_SEED,
    max_probe_rate: float = MAX_PROBE_RATE,
    seed_offset: int = 7919,  # large prime, won't collide with measurement seed
) -> Calibration:
    """Run the geometric sweep + one bisection probe; return a Calibration."""
    deadline = time.monotonic() + budget_s
    probes: list[CalibrationProbe] = []
    last_safe: float | None = None
    first_sat: float | None = None
    seed_step = 0

    for r in _probe_schedule(schedule, max_rate=max_probe_rate):
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

    # One bisection refinement between last_safe and first_sat — cheap
    # and meaningfully tightens the ceiling (e.g. coarse says 1≤cap<2,
    # bisect at 1.5 says 1.5≤cap<2 → selected jumps from 0.8 to 1.2 rps).
    if (
        last_safe is not None
        and first_sat is not None
        and time.monotonic() <= deadline
    ):
        mid = (last_safe + first_sat) / 2.0
        if mid > last_safe + 1e-6:
            records = await probe_fn(mid, seed_offset + seed_step)
            probe = evaluate_probe(
                records=records, probe_s=probe_s,
                target_rate=mid, ttft_slo_ms=ttft_slo_ms,
            )
            probes.append(probe)
            if not probe.saturated:
                last_safe = mid

    if last_safe is None:
        # Even the lowest probe saturated. Selected_rate falls back to a
        # safe floor; capacity_ceiling reports the smallest probed rate
        # (the only thing we know is "ceiling is at or below this"). The
        # explicit FALLBACK_RATE_RPS keeps the measurement run viable
        # rather than aborting on a hard floor.
        floor_rate = schedule[0] if schedule else 1.0
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
