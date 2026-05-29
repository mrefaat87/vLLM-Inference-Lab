"""Unit + property tests for the auto-rate calibration helper.

The helper takes a `probe_fn(rate, seed) -> list[RequestRecord]` so we
can drive it with synthetic saturation curves and assert behavior end
to end without spinning up an engine.
"""

from __future__ import annotations

import asyncio

import pytest

from experiments.runner.calibration import (
    HEADROOM,
    PROBE_S,
    PROBE_SCHEDULE,
    calibrate,
    evaluate_probe,
    explicit_calibration,
)
from experiments.runner.schema import RequestRecord

pytestmark = pytest.mark.unit


# ──────────────────────────────────────────────────────────────────────────
# Synthetic curve: an engine with a hard capacity ceiling at `capacity` rps.
# Below the ceiling: every request completes with a clean TTFT just below SLO.
# At/above the ceiling: completions cap at capacity × probe_s, the rest fail,
# and completed-request TTFT spikes 5× — a deliberately stark transition so the
# saturation rule's behavior is easy to reason about in tests.
# ──────────────────────────────────────────────────────────────────────────
def _synth_records(
    *, rate: float, probe_s: float, capacity: float, ttft_slo_ms: float
) -> list[RequestRecord]:
    dispatched = max(1, int(round(rate * probe_s)))
    completed_count = min(dispatched, int(round(capacity * probe_s)))
    util = min(rate / capacity, 1.0) if capacity > 0 else 1.0
    base_ttft_s = (ttft_slo_ms / 1000.0) * (0.2 + 0.3 * util)
    spike = 5.0 if rate > capacity else 1.0
    ttft_s = base_ttft_s * spike

    records: list[RequestRecord] = []
    for i in range(completed_count):
        records.append(
            RequestRecord(
                request_id=f"ok-{i}",
                label="t",
                submit_offset_s=i / max(rate, 1.0),
                prompt_tokens=1,
                max_new_tokens=1,
                completion_tokens=1,
                ttft_s=ttft_s,
                end_to_end_s=ttft_s + 0.05,
            )
        )
    for i in range(dispatched - completed_count):
        records.append(
            RequestRecord(
                request_id=f"err-{i}",
                label="t",
                submit_offset_s=0.0,
                prompt_tokens=1,
                max_new_tokens=1,
                error="TimeoutError: saturated",
            )
        )
    return records


def _make_probe_fn(capacity: float, ttft_slo_ms: float):
    async def _probe(rate: float, seed: int) -> list[RequestRecord]:  # noqa: ARG001
        return _synth_records(
            rate=rate, probe_s=PROBE_S,
            capacity=capacity, ttft_slo_ms=ttft_slo_ms,
        )
    return _probe


# ──────────────────────────────────────────────────────────────────────────
# evaluate_probe — saturation rule
# ──────────────────────────────────────────────────────────────────────────
class TestEvaluateProbe:
    def test_clean_burst_below_capacity_not_saturated(self) -> None:
        records = _synth_records(rate=1.0, probe_s=PROBE_S, capacity=10.0, ttft_slo_ms=500.0)
        p = evaluate_probe(records=records, probe_s=PROBE_S, target_rate=1.0, ttft_slo_ms=500.0)
        assert p.saturated is False
        assert p.success_rate == pytest.approx(1.0)
        # achieved RPS roughly matches target
        assert p.achieved_rps == pytest.approx(1.0, abs=0.2)

    def test_far_above_capacity_saturates(self) -> None:
        # 50 rps target, capacity 2 → completions cap at ~16, lots of errors,
        # TTFT spikes — all three signals trip.
        records = _synth_records(rate=50.0, probe_s=PROBE_S, capacity=2.0, ttft_slo_ms=500.0)
        p = evaluate_probe(records=records, probe_s=PROBE_S, target_rate=50.0, ttft_slo_ms=500.0)
        assert p.saturated is True
        assert p.success_rate < 0.5

    def test_hard_fail_short_circuit(self) -> None:
        """> 10% errors trips saturation even if other signals look OK."""
        records: list[RequestRecord] = []
        for i in range(80):
            records.append(RequestRecord(
                request_id=f"ok-{i}", label="t", submit_offset_s=0.0,
                prompt_tokens=1, max_new_tokens=1, completion_tokens=1,
                ttft_s=0.1, end_to_end_s=0.15,
            ))
        for i in range(20):  # 20% error rate
            records.append(RequestRecord(
                request_id=f"err-{i}", label="t", submit_offset_s=0.0,
                prompt_tokens=1, max_new_tokens=1, error="HTTP 503",
            ))
        p = evaluate_probe(records=records, probe_s=PROBE_S, target_rate=12.5, ttft_slo_ms=1500.0)
        assert p.saturated is True

    def test_empty_records_treated_saturated(self) -> None:
        p = evaluate_probe(records=[], probe_s=PROBE_S, target_rate=8.0, ttft_slo_ms=500.0)
        assert p.saturated is True
        assert p.success_rate == 0.0

    def test_ttft_slo_drives_signal2_trip_point(self) -> None:
        """Tightening ttft_slo_ms turns a previously-safe probe into a saturated
        one even with identical records. Guards the CLI wiring: the calc's
        --ttft-target-ms must actually reach evaluate_probe to be useful."""
        # Build records where TTFT is 1200ms, success+achieved are clean and
        # over the floors. Two signals MUST trip for saturation under
        # two-of-three rule, so this also doubles as a "load the other two
        # so signal 2 makes or breaks it" setup.
        records: list[RequestRecord] = []
        for i in range(40):  # achieved 5 rps over 8s; target 12 → trips sig 3
            records.append(RequestRecord(
                request_id=f"ok-{i}", label="t", submit_offset_s=i * 0.2,
                prompt_tokens=1, max_new_tokens=1, completion_tokens=1,
                ttft_s=1.2, end_to_end_s=1.25,
            ))
        # Loose SLO (1500ms): signal 2 trips at >3000ms. TTFT=1200 < 3000 → safe.
        # Signal 3 trips (5 < 0.85×12=10.2) but that's only one signal → not sat.
        p_loose = evaluate_probe(
            records=records, probe_s=PROBE_S,
            target_rate=12.0, ttft_slo_ms=1500.0,
        )
        assert p_loose.saturated is False
        # Tight SLO (500ms): signal 2 trips at >1000ms. TTFT=1200 > 1000 → trip.
        # Now two signals (sig 2 + sig 3) → saturated.
        p_tight = evaluate_probe(
            records=records, probe_s=PROBE_S,
            target_rate=12.0, ttft_slo_ms=500.0,
        )
        assert p_tight.saturated is True

    def test_single_signal_not_enough_with_clean_traffic(self) -> None:
        """One signal alone shouldn't trip — only TTFT slightly over SLO,
        success rate and achieved RPS clean. Demonstrates the two-of-three rule."""
        records: list[RequestRecord] = []
        # 80 requests at target 10rps over 8s — achieved 10, success 1.0.
        # TTFT just over 2× SLO (signal 2 trips), but signals 1 and 3 don't.
        for i in range(80):
            records.append(RequestRecord(
                request_id=f"ok-{i}", label="t", submit_offset_s=i * 0.1,
                prompt_tokens=1, max_new_tokens=1, completion_tokens=1,
                ttft_s=1.1,  # 1100ms > 2 × 500ms SLO
                end_to_end_s=1.15,
            ))
        p = evaluate_probe(records=records, probe_s=PROBE_S, target_rate=10.0, ttft_slo_ms=500.0)
        # Only TTFT signal trips → not saturated under two-of-three rule.
        assert p.saturated is False


# ──────────────────────────────────────────────────────────────────────────
# calibrate — full sweep + bisection
# ──────────────────────────────────────────────────────────────────────────
class TestCalibrate:
    def test_low_capacity_calibrates_below_first_saturated(self) -> None:
        # Capacity = 1.3 rps. Probes: 1=ok, 2=sat. Bisect at 1.5: 1.5>1.3 → sat.
        # last_safe stays at 1 → selected = 0.8 × 1.0 = 0.8.
        cal = asyncio.run(calibrate(probe_fn=_make_probe_fn(capacity=1.3, ttft_slo_ms=500.0)))
        assert cal.method == "auto"
        assert cal.selected_rate == pytest.approx(HEADROOM * 1.0)
        assert cal.capacity_ceiling == pytest.approx(1.0)
        # Probes contain the coarse sweep up to first saturation + one bisection.
        rates = [p.rate for p in cal.probes]
        assert rates[:2] == [1.0, 2.0]
        assert 1.5 in rates  # bisection probe ran

    def test_high_capacity_bisection_refines_ceiling(self) -> None:
        # Capacity = 20 rps. Probes: 1, 2, 4, 8, 16 all OK; 32 saturated.
        # Bisect at 24: still > 20 → saturated. last_safe stays at 16,
        # selected = 0.8 × 16 = 12.8.
        cal = asyncio.run(calibrate(probe_fn=_make_probe_fn(capacity=20.0, ttft_slo_ms=500.0)))
        assert cal.method == "auto"
        assert cal.capacity_ceiling == pytest.approx(16.0)
        assert cal.selected_rate == pytest.approx(HEADROOM * 16.0)

    def test_bisection_can_advance_ceiling(self) -> None:
        # Capacity = 7 rps. 1, 2, 4 OK; 8 saturated. Bisect at 6 → still OK.
        # last_safe advances to 6, selected = 0.8 × 6 = 4.8.
        cal = asyncio.run(calibrate(probe_fn=_make_probe_fn(capacity=7.0, ttft_slo_ms=500.0)))
        assert cal.capacity_ceiling == pytest.approx(6.0)
        assert cal.selected_rate == pytest.approx(HEADROOM * 6.0)
        rates = [p.rate for p in cal.probes]
        assert 6.0 in rates

    def test_floor_saturation_falls_back_to_safe_rate(self) -> None:
        # Capacity = 0.3 rps. Even R=1 saturates → no last_safe, fallback path.
        cal = asyncio.run(calibrate(probe_fn=_make_probe_fn(capacity=0.3, ttft_slo_ms=500.0)))
        assert cal.method == "auto"
        # Fallback rate is 0.5 rps; ceiling reports the lowest probed rate.
        assert cal.selected_rate == pytest.approx(0.5)
        assert cal.capacity_ceiling == pytest.approx(PROBE_SCHEDULE[0])

    def test_explicit_calibration_shape(self) -> None:
        cal = explicit_calibration(5.0)
        assert cal.method == "explicit"
        assert cal.probes == []
        assert cal.selected_rate == 5.0
        assert cal.capacity_ceiling == 5.0


# ──────────────────────────────────────────────────────────────────────────
# Property: monotonicity. For a noise-free synthetic engine, if probe at
# rate R is unsaturated, all R' < R in the schedule must also be unsaturated.
# This is the core invariant the early-stopping rule relies on — without it,
# we could stop sweeping at a transient false-saturation and report a
# wildly-too-low ceiling.
# ──────────────────────────────────────────────────────────────────────────
class TestMonotonicity:
    @pytest.mark.parametrize("capacity", [0.5, 1.3, 3.0, 7.0, 13.0, 25.0, 100.0])
    def test_saturation_monotone_in_rate(self, capacity: float) -> None:
        """As probe rate increases, saturation can only flip false→true once."""
        ttft_slo_ms = 500.0
        seen_saturated = False
        # Sweep across PROBE_SCHEDULE + a few midpoints for finer resolution.
        rates = sorted(set(list(PROBE_SCHEDULE) + [1.5, 3.0, 6.0, 12.0, 24.0]))
        for r in rates:
            records = _synth_records(
                rate=r, probe_s=PROBE_S,
                capacity=capacity, ttft_slo_ms=ttft_slo_ms,
            )
            p = evaluate_probe(
                records=records, probe_s=PROBE_S,
                target_rate=r, ttft_slo_ms=ttft_slo_ms,
            )
            if seen_saturated:
                assert p.saturated, (
                    f"capacity={capacity}: probe at r={r} unsaturated after "
                    f"a lower rate already saturated — monotonicity broken"
                )
            if p.saturated:
                seen_saturated = True
