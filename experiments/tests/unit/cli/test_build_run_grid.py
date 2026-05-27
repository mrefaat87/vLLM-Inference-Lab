"""Unit tests for ``exp plan``'s ``_build_run_grid`` rate computation.

These pin the lab-side half of the calc↔lab rate contract that the JS
``recommendedRate`` (calculators/sizing_calc/src/lab_command.mjs) also
asserts. If this test moves, the JS contract test must move with it.
"""

from __future__ import annotations

import pytest

from experiments.cli.exp import _build_run_grid
from experiments.runner.schema import Prediction, PredictionCurvePoint

pytestmark = pytest.mark.unit


def _pred(batches: list[int], recommended: float | None = None) -> Prediction:
    """Build a minimal Prediction with a flat curve over ``batches``."""
    # model_construct skips validation — we only exercise grid math here,
    # so envelope fields (calc_version, data_hash, inputs) are irrelevant.
    return Prediction.model_construct(
        calc_version="test",
        data_hash="test",
        inputs=None,
        b_crit=float(batches[len(batches) // 2]),
        recommended_batch=recommended,
        curve=[
            PredictionCurvePoint(batch=b, step_ms=50.0, tps=100.0) for b in batches
        ],
    )


def _expected_rate(batch: float, *, tbt_ms: float, osl: int, ttft_ms: float) -> float:
    """Mirror of the formula under test — used as the truth oracle."""
    per_req_s = max(ttft_ms / 1000.0 + osl * tbt_ms / 1000.0, 1e-3)
    return round(batch / per_req_s, 3)


def test_smoke_golden_chatbot_shape() -> None:
    # OSL=200, TBT=50ms, TTFT=500ms → per_req=10.5s → 128/10.5=12.190 rps.
    pred = _pred([32, 64, 128, 256, 512], recommended=128.0)
    grid = _build_run_grid(
        pred, rows=1, tbt_ms=50.0, osl_tokens=200, ttft_ms=500.0
    )
    assert len(grid) == 1
    # Row centers on recommended_batch=128 with rows=1 → batch_target=lo=32.
    # We only assert the rate computation matches the formula — the batch
    # spacing is out-of-scope here.
    row = grid[0]
    assert row["rate_rps"] == _expected_rate(
        row["batch_target"], tbt_ms=50.0, osl=200, ttft_ms=500.0
    )


def test_chatbot_emits_double_digit_not_thousand_rps() -> None:
    """Regression: the old formula emitted ~2780 rps for this shape."""
    pred = _pred([128], recommended=128.0)
    grid = _build_run_grid(
        pred, rows=1, tbt_ms=50.0, osl_tokens=200, ttft_ms=500.0
    )
    # If someone reintroduces `b / tbt_s` (no OSL), this row jumps ~200×.
    # Hard cap at 100 rps catches that regression.
    assert grid[0]["rate_rps"] < 100, (
        f"rate {grid[0]['rate_rps']} suggests OSL dropped out of the denominator"
    )


def test_zero_latency_inputs_do_not_divide_by_zero() -> None:
    pred = _pred([64], recommended=64.0)
    grid = _build_run_grid(pred, rows=1, tbt_ms=0.0, osl_tokens=0, ttft_ms=0.0)
    assert len(grid) == 1
    # rate = batch_target / 1e-3 — finite, large, but not NaN/inf.
    assert grid[0]["rate_rps"] == pytest.approx(
        grid[0]["batch_target"] / 1e-3, rel=1e-3
    )


def test_empty_curve_returns_empty_grid() -> None:
    pred = Prediction.model_construct(
        calc_version="t", data_hash="t", inputs=None, curve=[]
    )
    assert _build_run_grid(
        pred, rows=3, tbt_ms=50.0, osl_tokens=200, ttft_ms=500.0
    ) == []


@pytest.mark.parametrize("tbt_ms", [10.0, 25.0, 50.0, 100.0, 200.0])
@pytest.mark.parametrize("osl", [50, 200, 1000])
@pytest.mark.parametrize("ttft_ms", [100.0, 500.0, 2000.0])
def test_parity_with_explicit_formula(
    tbt_ms: float, osl: int, ttft_ms: float
) -> None:
    pred = _pred([8, 32, 128, 512, 2048], recommended=128.0)
    grid = _build_run_grid(
        pred, rows=5, tbt_ms=tbt_ms, osl_tokens=osl, ttft_ms=ttft_ms
    )
    for row in grid:
        expected = _expected_rate(
            row["batch_target"], tbt_ms=tbt_ms, osl=osl, ttft_ms=ttft_ms
        )
        assert abs(row["rate_rps"] - expected) <= 0.001, (
            f"rate {row['rate_rps']} disagrees with formula {expected} at "
            f"batch={row['batch_target']} tbt={tbt_ms} osl={osl} ttft={ttft_ms}"
        )


def test_calc_js_contract_pair() -> None:
    """The JS golden ``recommendedRate(128, 50, 200, 500) ≈ 12.19`` must
    agree with the Python implementation to 2dp (the calc's rounding limit).
    """
    # rows=3 with center=128 yields batch_targets [32, 128, 512]; the
    # middle row is the apples-to-apples comparison with the JS golden.
    pred = _pred([32, 128, 512], recommended=128.0)
    grid = _build_run_grid(
        pred, rows=3, tbt_ms=50.0, osl_tokens=200, ttft_ms=500.0
    )
    middle = next(r for r in grid if r["batch_target"] == 128.0)
    js_rate = 12.19  # from lab_command.test.mjs golden
    assert abs(middle["rate_rps"] - js_rate) <= 0.01
