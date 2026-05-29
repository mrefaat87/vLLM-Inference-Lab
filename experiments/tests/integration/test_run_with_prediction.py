"""End-to-end: mock run captures a Prediction block.

Owns:
  C11 — prediction provenance (calc_version + data_hash present).
  Portion of C15 verified at the HTML-string level (overlay code present).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments.drivers.mock import MockEngineDriver
from experiments.portal.build import BuildInputs, build as build_portal
from experiments.runner.calc_bridge import CalcBridge
from experiments.runner.schema import (
    EngineConfig,
    HardwareSpec,
    ModelSpec,
    RooflineLink,
    WorkloadConfig,
)
from experiments.runner.sweep import SweepRunner
from experiments.workloads.chatbot import ChatbotParams, ChatbotWorkload

pytestmark = pytest.mark.integration


@pytest.fixture
def calc_root_available() -> Path | None:
    """Skip if the sibling calc isn't checked out (lab-only clones)."""
    root = Path(__file__).resolve().parents[3] / "calculators" / "sizing_calc"
    if not (root / "scripts" / "compute_cli.mjs").exists():
        pytest.skip("sizing_calc not present in this clone")
    if shutil.which("node") is None:
        pytest.skip("node not on PATH")
    return root


@pytest.mark.asyncio
async def test_mock_run_captures_prediction(tmp_path: Path, calc_root_available: Path) -> None:
    runner = SweepRunner(results_dir=tmp_path, calc_bridge=CalcBridge(calc_root=calc_root_available))
    path = await runner.run_one(
        engine=MockEngineDriver(),
        engine_cfg=EngineConfig(name="mock", image="n/a", model="m", quantization="awq"),
        workload=ChatbotWorkload(seed=1, params=ChatbotParams(rate_rps=15.0)),
        workload_cfg=WorkloadConfig(name="chatbot", rate_rps=15.0, duration_s=1.5, warmup_s=0.0),
        model=ModelSpec(name="Llama-3-70B-AWQ", quant="awq", tp=4),
        hardware=HardwareSpec(instance="g5.12xlarge", gpu="A10G", n_gpu=4),
        roofline_link=RooflineLink(model_ref="llama-3-70b", hw_ref="A10G"),
        tbt_target_ms=50.0,
    )
    blob = json.loads(Path(path).read_text())
    # Loose-check the schema minor of the current contract — bump alongside
    # SCHEMA_VERSION when the runner is supposed to write a new minor.
    assert blob["schema_version"].startswith("1.2.")

    pred = blob.get("prediction")
    assert pred is not None, "calc bridge should have populated a prediction"
    # C11: provenance present
    assert pred["calc_version"]
    assert len(pred["data_hash"]) == 64
    # Curve usable for overlay rendering.
    assert pred["curve"], "predicted curve should be non-empty"
    first = pred["curve"][0]
    assert first["batch"] >= 1 and first["tps"] > 0
    # Inputs include the measured ISL/OSL p50, not the workload defaults.
    assert pred["inputs"]["isl"] == blob["analysis"]["isl_tokens_p50"]
    assert pred["inputs"]["osl"] == blob["analysis"]["osl_tokens_p50"]


@pytest.mark.asyncio
async def test_prediction_renders_in_portal(tmp_path: Path, calc_root_available: Path) -> None:
    """Built results_explorer.html embeds the overlay code that reads
    result.prediction.curve. Doesn't *open a browser* — that's the calc
    side's Playwright job. Owns the static half of C15.
    """
    runner = SweepRunner(
        results_dir=tmp_path / "results",
        calc_bridge=CalcBridge(calc_root=calc_root_available),
    )
    await runner.run_one(
        engine=MockEngineDriver(),
        engine_cfg=EngineConfig(name="mock", image="n/a", model="m", quantization="awq"),
        workload=ChatbotWorkload(seed=1, params=ChatbotParams(rate_rps=15.0)),
        workload_cfg=WorkloadConfig(name="chatbot", rate_rps=15.0, duration_s=1.2, warmup_s=0.0),
        model=ModelSpec(name="m", quant="awq", tp=4),
        hardware=HardwareSpec(instance="g5.12xlarge", gpu="A10G", n_gpu=4),
        roofline_link=RooflineLink(model_ref="llama-3-70b", hw_ref="A10G"),
    )
    build_portal(BuildInputs(results_dir=tmp_path / "results", out_dir=tmp_path / "_site"))
    html = (tmp_path / "_site" / "results_explorer.html").read_text()
    # Code paths present (independent of whether the data is there).
    assert "renderRooflineOverlay" in html
    assert "roofline-chart" in html
    assert "prediction-banner" in html
    # Null-safe access: the overlay must short-circuit when prediction is null.
    # We assert presence of a guard like `pred &&` or `if (!pred)` in the JS.
    assert ("if (!pred)" in html) or ("!pred" in html) or ("pred && " in html), (
        "results_explorer overlay JS must guard against missing prediction"
    )
