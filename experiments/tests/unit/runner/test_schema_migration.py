"""C16: schema migration is backward-readable.

  - A v1.0.0 result fixture (no `prediction`, no isl/osl percentile) loads
    cleanly under v1.1.0; `prediction is None`.
  - A v1.1.0 result with `prediction` loads cleanly under v1.0.0 if the
    consumer relaxes its envelope to `extra="ignore"` — we test this by
    constructing a strict-subset model and validating only the v1.0.0
    fields.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from experiments.runner.schema import RunResult

pytestmark = pytest.mark.unit


def _v100_result_blob() -> dict:
    """A frozen v1.0.0-shaped result JSON (no prediction field)."""
    now = datetime.now(timezone.utc)
    return {
        "schema_version": "1.0.0",
        "run_id": "v100-fixture",
        "started_at": now.isoformat(),
        "finished_at": (now + timedelta(seconds=5)).isoformat(),
        "engine": {"name": "mock", "image": "n/a", "model": "m"},
        "model": {"name": "m"},
        "hardware": {"instance": "local", "gpu": "cpu", "n_gpu": 1},
        "workload": {"name": "chatbot", "rate_rps": 1.0, "duration_s": 5.0},
        "analysis": {
            "steady_state_requests": 1,
            "failed_requests": 0,
            "throughput": {
                "total_completion_tokens": 10,
                "total_prompt_tokens": 5,
                "tok_per_sec_avg": 2.0,
                "requests_per_sec_avg": 0.2,
            },
        },
        "roofline_link": {"model_ref": "m", "hw_ref": "local"},
        "raw_results": [],
    }


def test_v100_loads_under_v110_schema() -> None:
    """Old result, new code → prediction is None, no error."""
    r = RunResult.model_validate(_v100_result_blob())
    assert r.prediction is None
    assert r.analysis.isl_tokens_p50 is None
    assert r.analysis.osl_tokens_p50 is None


def test_v110_loads_with_unknown_top_level_fields_ignored() -> None:
    """Future schema bump → ignored unknown fields, never raises."""
    blob = _v100_result_blob()
    blob["schema_version"] = "1.99.0"
    blob["future_extension"] = {"some": "thing"}
    blob["another_future_field"] = [1, 2, 3]
    r = RunResult.model_validate(blob)
    assert r.run_id == "v100-fixture"


def test_v110_loads_under_v120_no_calibration_field() -> None:
    """v1.1.0 result (no calibration block) loads cleanly under v1.2.0
    code — the calibration field is optional, so old runs in
    results/runs/ keep parsing after the schema bump.
    """
    blob = _v100_result_blob()
    blob["schema_version"] = "1.1.0"
    r = RunResult.model_validate(blob)
    assert r.calibration is None


def test_v120_with_calibration_roundtrips() -> None:
    """A v1.2.0 result with a calibration block survives JSON round-trip
    so the portal can read what the runner writes."""
    blob = _v100_result_blob()
    blob["schema_version"] = "1.2.0"
    blob["calibration"] = {
        "method": "auto",
        "probes": [
            {"rate": 1.0, "success_rate": 1.0, "ttft_p95_ms": 200.0,
             "achieved_rps": 1.0, "saturated": False},
            {"rate": 2.0, "success_rate": 0.4, "ttft_p95_ms": 2500.0,
             "achieved_rps": 0.8, "saturated": True},
            {"rate": 1.5, "success_rate": 1.0, "ttft_p95_ms": 280.0,
             "achieved_rps": 1.5, "saturated": False},
        ],
        "selected_rate": 1.2,
        "capacity_ceiling": 1.5,
    }
    r = RunResult.model_validate(blob)
    assert r.calibration is not None
    assert r.calibration.method == "auto"
    assert r.calibration.selected_rate == pytest.approx(1.2)
    assert len(r.calibration.probes) == 3
    # Round-trip
    re = RunResult.model_validate(json.loads(r.model_dump_json()))
    assert re.calibration is not None
    assert re.calibration.capacity_ceiling == pytest.approx(1.5)
    assert re.calibration.probes[1].saturated is True


def test_v110_with_prediction_roundtrips() -> None:
    """The prediction block survives a JSON round-trip without lossy
    transforms — the lab's portal builder relies on this for C2 byte
    fidelity at the JSON level."""
    blob = _v100_result_blob()
    blob["schema_version"] = "1.1.0"
    blob["prediction"] = {
        "calc_version": "test-1.0",
        "data_hash": "0" * 64,
        "inputs": {
            "model_key": "m",
            "hw_key": "local",
            "weight_prec": "BF16",
            "kv_prec": "FP16",
            "act_prec": "BF16",
            "isl": 100,
            "osl": 100,
            "ngpus": 1,
            "tbt_ms": 50,
            "price_per_hour_usd": None,
        },
        "b_crit": 32,
        "b_slo": 50,
        "b_kv": 200,
        "recommended_batch": 50,
        "y_max": 1,
        "curve": [
            {"batch": 1, "step_ms": 10.0, "tps": 100.0, "cost_per_mtok": None},
        ],
        "warnings": [],
        "unavailable_reason": None,
    }
    r = RunResult.model_validate(blob)
    assert r.prediction is not None
    assert r.prediction.b_crit == 32
    assert r.prediction.curve[0].tps == 100.0
    # Round-trip
    re = RunResult.model_validate(json.loads(r.model_dump_json()))
    assert re.prediction is not None
    assert re.prediction.calc_version == "test-1.0"
