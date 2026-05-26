"""Unit tests for calc_bridge.py — grid-hit, off-grid live, missing-Node.

Owns contracts:
  C14 — graceful degrade when Node is absent.
  C12 partial — grid row resolves to the same Prediction shape as a live call.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from experiments.runner.calc_bridge import CalcBridge, CalcInputs

pytestmark = pytest.mark.unit


def _fixture_grid(tmp_path: Path) -> Path:
    """Write a tiny but well-formed grid.json next to a fake calc tree."""
    calc_root = tmp_path / "calc"
    (calc_root / "predictions").mkdir(parents=True)
    (calc_root / "scripts").mkdir()
    grid = {
        "calc_version": "test-1.0",
        "data_hash": "a" * 64,
        "generated_at": "2026-05-24T00:00:00.000Z",
        "rows": [
            {
                "inputs": {
                    "model_key": "llama-3-70b",
                    "hw_key": "A10G",
                    "weight_prec": "INT4",
                    "kv_prec": "FP16",
                    "act_prec": "BF16",
                    "isl": 128,
                    "osl": 128,
                    "ngpus": 4,
                    "tbt_ms": 50,
                    "price_per_hour_usd": 1.624,
                },
                "b_crit": 30,
                "b_slo": 90,
                "b_kv": 400,
                "recommended_batch": 90,
                "y_max": 8,
                "curve": [
                    {"batch": 1, "step_ms": 15.0, "tps": 60.0, "cost_per_mtok": 7.5},
                    {"batch": 8, "step_ms": 16.0, "tps": 480.0, "cost_per_mtok": 1.0},
                    {"batch": 32, "step_ms": 18.0, "tps": 1500.0, "cost_per_mtok": 0.4},
                ],
                "warnings": [],
                "unavailable_reason": None,
            }
        ],
    }
    (calc_root / "predictions" / "grid.json").write_text(json.dumps(grid))
    return calc_root


def test_grid_hit_returns_prediction_without_subprocess(tmp_path: Path, monkeypatch) -> None:
    calc_root = _fixture_grid(tmp_path)
    bridge = CalcBridge(calc_root=calc_root)
    # subprocess.run would only fire on live invocation; failing it loudly
    # if called proves the test is exercising the grid path.
    import subprocess
    calls: list = []

    def boom(*args, **kwargs):  # noqa: ARG001
        calls.append(args)
        raise RuntimeError("should not have shelled out")

    monkeypatch.setattr(subprocess, "run", boom)
    pred = bridge.predict(
        CalcInputs(
            model_key="llama-3-70b", hw_key="A10G",
            weight_prec="INT4", kv_prec="FP16", act_prec="BF16",
            isl=128, osl=128, ngpus=4, tbt_ms=50,
        )
    )
    assert pred is not None
    assert calls == []
    assert pred.calc_version == "test-1.0"
    assert pred.data_hash == "a" * 64
    assert pred.b_crit == 30
    assert len(pred.curve) == 3
    assert pred.curve[1].batch == 8 and pred.curve[1].tps == 480.0


def test_off_grid_falls_back_to_live_when_node_present(tmp_path: Path) -> None:
    """Off-grid lookup tries node — but the fixture has no compute_cli.mjs,
    so the bridge should return None gracefully rather than crash."""
    calc_root = _fixture_grid(tmp_path)
    bridge = CalcBridge(calc_root=calc_root)
    pred = bridge.predict(
        CalcInputs(
            model_key="llama-3-70b", hw_key="A10G",
            weight_prec="INT4", kv_prec="FP16", act_prec="BF16",
            isl=999, osl=999, ngpus=4, tbt_ms=50,  # no row matches
        )
    )
    assert pred is None  # bridge degraded; the run would proceed with prediction: null


def test_missing_node_returns_none(tmp_path: Path, monkeypatch) -> None:
    """C14: with no `node` on PATH, the bridge returns None without raising."""
    calc_root = _fixture_grid(tmp_path)
    bridge = CalcBridge(calc_root=calc_root)
    monkeypatch.setattr("shutil.which", lambda _: None)
    pred = bridge.predict(
        CalcInputs(
            model_key="llama-3-70b", hw_key="A10G",
            weight_prec="INT4", kv_prec="FP16", act_prec="BF16",
            isl=777, osl=777, ngpus=4, tbt_ms=50,
        )
    )
    assert pred is None


def test_calc_root_env_override(tmp_path: Path, monkeypatch) -> None:
    calc_root = _fixture_grid(tmp_path)
    monkeypatch.setenv("SIZING_CALC_ROOT", str(calc_root))
    bridge = CalcBridge()  # no explicit calc_root
    assert bridge.grid_path == calc_root / "predictions" / "grid.json"


def test_missing_grid_returns_none_without_calling_subprocess(tmp_path: Path, monkeypatch) -> None:
    """No grid and no compute_cli.mjs → None, no crash."""
    bridge = CalcBridge(calc_root=tmp_path / "nonexistent")
    monkeypatch.setattr("shutil.which", lambda _: None)
    pred = bridge.predict(
        CalcInputs(
            model_key="llama-3-70b", hw_key="A10G",
            weight_prec="INT4", kv_prec="FP16", act_prec="BF16",
            isl=128, osl=128, ngpus=4, tbt_ms=50,
        )
    )
    assert pred is None


def test_malformed_grid_returns_none(tmp_path: Path) -> None:
    calc_root = tmp_path / "calc"
    (calc_root / "predictions").mkdir(parents=True)
    (calc_root / "predictions" / "grid.json").write_text("not json")
    bridge = CalcBridge(calc_root=calc_root)
    pred = bridge.predict(
        CalcInputs(
            model_key="llama-3-70b", hw_key="A10G",
            weight_prec="INT4", kv_prec="FP16", act_prec="BF16",
            isl=128, osl=128, ngpus=4, tbt_ms=50,
        )
    )
    assert pred is None
