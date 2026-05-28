"""Unit tests for cost-per-Mtok enrichment in the calc bridge.

The Results Explorer renders a "cost vs batch" curve straight from
``result.prediction.curve[*].cost_per_mtok`` (contract C15: no calc.mjs
at view time). The lab's responsibility is to put a non-null value in
that field whenever the hardware row has a known ``price_per_hour_usd``,
and to leave it null otherwise so the panel can degrade gracefully.

Numerical golden anchor:
  Cost ≡ price_per_hour_usd × ngpus × 1e6 / (3600 × tps).
  For batch=64, price=$1.30/hr, ngpus=1, tps=70 →
    1.30 × 1 × 1_000_000 / (3600 × 70) ≈ 5.1587 $/Mtok.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runner.calc_bridge import CalcBridge, CalcInputs
from experiments.runner.schema import (
    Prediction,
    PredictionCurvePoint,
)

pytestmark = pytest.mark.unit


def test_curve_point_accepts_cost_per_mtok() -> None:
    """A PredictionCurvePoint round-trips cost_per_mtok cleanly; the
    field is optional so v1.0.0-shaped result files still parse.
    """
    p = PredictionCurvePoint(batch=64, step_ms=15.0, tps=70.0, cost_per_mtok=5.16)
    assert p.cost_per_mtok == pytest.approx(5.16)
    p_old = PredictionCurvePoint(batch=8, step_ms=20.0, tps=400.0)
    assert p_old.cost_per_mtok is None


def test_grid_lookup_enriches_curve_with_cost(tmp_path: Path) -> None:
    """When a grid row carries cost_per_mtok on its curve points, the
    bridge surfaces it on the returned Prediction without re-deriving.
    Golden = 1.30 × 1e6 / (3600 × 70) ≈ 5.1587.
    """
    calc_root = tmp_path / "calc"
    (calc_root / "predictions").mkdir(parents=True)
    (calc_root / "scripts").mkdir()
    expected_cost = 1.30 * 1 * 1e6 / (3600 * 70)
    grid = {
        "calc_version": "test-1.0",
        "data_hash": "f" * 64,
        "rows": [
            {
                "inputs": {
                    "model_key": "llama-3-8b", "hw_key": "A10G",
                    "weight_prec": "BF16", "kv_prec": "FP16", "act_prec": "BF16",
                    "isl": 200, "osl": 150, "ngpus": 1, "tbt_ms": 50,
                    "price_per_hour_usd": 1.30,
                },
                "curve": [
                    {"batch": 64, "step_ms": 15.0, "tps": 70.0,
                     "cost_per_mtok": expected_cost},
                ],
                "warnings": [],
            }
        ],
    }
    (calc_root / "predictions" / "grid.json").write_text(json.dumps(grid))
    bridge = CalcBridge(calc_root=calc_root)
    pred = bridge.predict(CalcInputs(
        model_key="llama-3-8b", hw_key="A10G",
        weight_prec="BF16", kv_prec="FP16", act_prec="BF16",
        isl=200, osl=150, ngpus=1, tbt_ms=50, price_per_hour_usd=1.30,
    ))
    assert pred is not None
    assert pred.curve[0].cost_per_mtok == pytest.approx(5.158730, rel=1e-5)


def test_grid_curve_without_cost_field_yields_none(tmp_path: Path) -> None:
    """An older grid that omits cost_per_mtok must still parse; the
    field defaults to None so the cost panel degrades gracefully.
    """
    calc_root = tmp_path / "calc"
    (calc_root / "predictions").mkdir(parents=True)
    (calc_root / "scripts").mkdir()
    grid = {
        "calc_version": "test-old",
        "data_hash": "0" * 64,
        "rows": [
            {
                "inputs": {
                    "model_key": "llama-3-8b", "hw_key": "T4",
                    "weight_prec": "BF16", "kv_prec": "FP16", "act_prec": "BF16",
                    "isl": 200, "osl": 150, "ngpus": 1, "tbt_ms": 50,
                },
                "curve": [{"batch": 1, "step_ms": 800.0, "tps": 1.0}],
                "warnings": [],
            }
        ],
    }
    (calc_root / "predictions" / "grid.json").write_text(json.dumps(grid))
    bridge = CalcBridge(calc_root=calc_root)
    pred = bridge.predict(CalcInputs(
        model_key="llama-3-8b", hw_key="T4",
        weight_prec="BF16", kv_prec="FP16", act_prec="BF16",
        isl=200, osl=150, ngpus=1, tbt_ms=50,
    ))
    assert pred is not None
    assert pred.curve[0].cost_per_mtok is None


def test_lookup_hw_price_reads_calc_hardware_json(tmp_path: Path) -> None:
    """`CalcBridge.lookup_hw_price` resolves the calc's per-hour price
    for a known hw key, returns None for unknown keys, and degrades to
    None when the data file is missing (clone-only-the-lab scenario).
    """
    calc_root = tmp_path / "calc"
    (calc_root / "src" / "data").mkdir(parents=True)
    (calc_root / "src" / "data" / "hardware.json").write_text(json.dumps({
        "hardware": [
            {"key": "T4", "price_per_hour_usd": 0.526, "label": "NVIDIA T4"},
            {"key": "A10G", "price_per_hour_usd": 1.006, "label": "NVIDIA A10G"},
            {"key": "Custom", "label": "Custom GPU"},
        ],
    }))
    bridge = CalcBridge(calc_root=calc_root)
    assert bridge.lookup_hw_price("T4") == pytest.approx(0.526)
    assert bridge.lookup_hw_price("A10G") == pytest.approx(1.006)
    assert bridge.lookup_hw_price("Custom") is None
    assert bridge.lookup_hw_price("unknown-key") is None
    bridge_empty = CalcBridge(calc_root=tmp_path / "no-such-calc")
    assert bridge_empty.lookup_hw_price("T4") is None


def test_back_compat_old_prediction_blob_parses() -> None:
    """v1.0.x-shaped Prediction blobs (no cost_per_mtok) still validate."""
    blob = {
        "calc_version": "old-0.9",
        "data_hash": "1" * 64,
        "inputs": {
            "model_key": "llama-3-8b", "hw_key": "A10G",
            "weight_prec": "BF16", "kv_prec": "FP16", "act_prec": "BF16",
            "isl": 200, "osl": 150, "ngpus": 1, "tbt_ms": 50.0,
        },
        "curve": [
            {"batch": 1, "step_ms": 100.0, "tps": 10.0},
            {"batch": 8, "step_ms": 120.0, "tps": 65.0},
        ],
        "warnings": [],
    }
    pred = Prediction.model_validate(blob)
    assert all(p.cost_per_mtok is None for p in pred.curve)
    assert pred.inputs.price_per_hour_usd is None
