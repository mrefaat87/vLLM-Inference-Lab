"""Auto-rate calibration end-to-end against the mock engine.

Verifies that:
  - The runner does a calibration sweep when given a workload_factory.
  - The resulting JSON has calibration.method == "auto", probes populated,
    and a positive selected_rate.
  - The persisted workload.rate_rps is the calibrated rate (not the
    placeholder the caller passed in).
  - The explicit-rate path writes calibration.method == "explicit" with
    probes empty so downstream readers see a uniform shape either way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.drivers.mock import MockEngineDriver
from experiments.runner.schema import (
    EngineConfig,
    HardwareSpec,
    ModelSpec,
    RooflineLink,
    WorkloadConfig,
)
from experiments.runner.sweep import SweepRunner, load_result
from experiments.workloads.chatbot import ChatbotParams, ChatbotWorkload

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_auto_calibration_writes_probe_trace(tmp_path: Path) -> None:
    runner = SweepRunner(results_dir=tmp_path)
    driver = MockEngineDriver()
    # Placeholder rate that gets overridden by calibration. Schema requires gt=0.
    workload_cfg = WorkloadConfig(
        name="chatbot", rate_rps=1.0,
        duration_s=1.0, warmup_s=0.0, drain_s=0.0, seed=1,
    )

    path = await runner.run_one(
        engine=driver,
        engine_cfg=EngineConfig(name="mock", image="n/a", model="mock-model"),
        workload_factory=lambda r, s: ChatbotWorkload(
            seed=s, params=ChatbotParams(rate_rps=r)
        ),
        workload_cfg=workload_cfg,
        model=ModelSpec(name="mock-model", quant="none", tp=1),
        hardware=HardwareSpec(instance="local", gpu="cpu", n_gpu=1),
        roofline_link=RooflineLink(model_ref="mock-model", hw_ref="local-cpu"),
    )

    result = load_result(path)
    assert result.calibration is not None
    assert result.calibration.method == "auto"
    assert len(result.calibration.probes) >= 1
    assert result.calibration.selected_rate > 0.0
    assert result.calibration.capacity_ceiling > 0.0
    # The persisted workload rate IS the calibrated rate (not the placeholder).
    assert result.workload.rate_rps == pytest.approx(
        result.calibration.selected_rate
    )


@pytest.mark.asyncio
async def test_explicit_rate_writes_uniform_calibration_shape(tmp_path: Path) -> None:
    """Explicit --rate runs still write a calibration block (method=explicit,
    probes=[]) so the portal / readers can render uniformly."""
    runner = SweepRunner(results_dir=tmp_path)
    driver = MockEngineDriver()
    workload_cfg = WorkloadConfig(
        name="chatbot", rate_rps=5.0,
        duration_s=1.0, warmup_s=0.0, drain_s=0.0, seed=1,
    )

    path = await runner.run_one(
        engine=driver,
        engine_cfg=EngineConfig(name="mock", image="n/a", model="mock-model"),
        workload=ChatbotWorkload(seed=1, params=ChatbotParams(rate_rps=5.0)),
        workload_cfg=workload_cfg,
        model=ModelSpec(name="mock-model", quant="none", tp=1),
        hardware=HardwareSpec(instance="local", gpu="cpu", n_gpu=1),
        roofline_link=RooflineLink(model_ref="mock-model", hw_ref="local-cpu"),
    )

    result = load_result(path)
    assert result.calibration is not None
    assert result.calibration.method == "explicit"
    assert result.calibration.probes == []
    assert result.calibration.selected_rate == pytest.approx(5.0)
    assert result.calibration.capacity_ceiling == pytest.approx(5.0)
    # Explicit path leaves workload.rate_rps as the caller specified.
    assert result.workload.rate_rps == pytest.approx(5.0)
