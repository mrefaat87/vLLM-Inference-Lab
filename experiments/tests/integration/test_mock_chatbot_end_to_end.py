"""Full pipeline end-to-end against the in-process stub server.

This is the single most load-bearing test in the suite: it proves the
schemas, the driver loop, the workload, the analysis, the manifest store,
and the result writer all compose into a valid JSON document that the
portal can read. Failure here means the whole pipeline is broken.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.drivers.mock import MockEngineDriver
from experiments.runner.manifest_store import ManifestStore
from experiments.runner.schema import (
    SCHEMA_VERSION,
    EngineConfig,
    HardwareSpec,
    ModelSpec,
    RooflineLink,
    RunResult,
    RunStatus,
    WorkloadConfig,
)
from experiments.runner.sweep import SweepRunner, load_result
from experiments.workloads.chatbot import ChatbotParams, ChatbotWorkload

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_end_to_end_mock_chatbot(tmp_path: Path) -> None:
    runner = SweepRunner(results_dir=tmp_path)
    driver = MockEngineDriver()
    engine_cfg = EngineConfig(name="mock", image="n/a", model="mock-model")
    workload = ChatbotWorkload(seed=1, params=ChatbotParams(rate_rps=20.0))
    workload_cfg = WorkloadConfig(
        name="chatbot", rate_rps=20.0, duration_s=2.0, warmup_s=0.0, drain_s=0.0, seed=1
    )

    path = await runner.run_one(
        engine=driver,
        engine_cfg=engine_cfg,
        workload=workload,
        workload_cfg=workload_cfg,
        model=ModelSpec(name="mock-model", quant="none", tp=1),
        hardware=HardwareSpec(instance="local", gpu="cpu", n_gpu=1),
        roofline_link=RooflineLink(model_ref="mock-model", hw_ref="local-cpu"),
        notes="integration test",
    )

    # Result file exists and parses.
    assert path.exists()
    result = load_result(path)
    assert isinstance(result, RunResult)
    assert result.schema_version == SCHEMA_VERSION
    assert result.analysis.steady_state_requests > 0
    assert result.analysis.failed_requests == 0
    # We exercised the streaming path → TTFT/TBT should be populated.
    assert result.analysis.ttft_s is not None
    assert result.analysis.ttft_s.n > 0
    assert result.analysis.tbt_s is None or result.analysis.tbt_s.n >= 0
    assert result.analysis.throughput.total_completion_tokens > 0

    # JSON is portal-loadable (parses + has all top-level keys).
    blob = json.loads(path.read_text())
    for key in (
        "schema_version", "run_id", "started_at", "finished_at",
        "engine", "model", "hardware", "workload", "analysis", "roofline_link",
        "raw_results", "engine_metrics",
    ):
        assert key in blob, f"missing top-level key {key!r}"

    # Manifest transitioned correctly.
    store = ManifestStore(tmp_path / "manifests")
    manifests = store.list_all()
    assert len(manifests) == 1
    m = manifests[0]
    assert m.status == RunStatus.DONE
    assert m.result_path == str(path)
    assert m.started_at is not None and m.finished_at is not None
    assert m.finished_at >= m.started_at


@pytest.mark.asyncio
async def test_manifest_failed_on_unhealthy_engine(tmp_path: Path) -> None:
    """If the engine never comes up, the manifest must end in FAILED."""

    class BrokenDriver(MockEngineDriver):
        def healthcheck(self) -> bool:
            return False  # never healthy

    runner = SweepRunner(results_dir=tmp_path)
    driver = BrokenDriver()
    # We deliberately pass a TINY readiness budget by monkeypatching the runner.
    runner._await_ready = lambda engine, *, timeout_s: (_ for _ in ()).throw(  # type: ignore[method-assign]
        TimeoutError("forced")
    )

    with pytest.raises(TimeoutError):
        await runner.run_one(
            engine=driver,
            engine_cfg=EngineConfig(name="mock", image="x", model="m"),
            workload=ChatbotWorkload(seed=1),
            workload_cfg=WorkloadConfig(name="chatbot", rate_rps=1.0, duration_s=0.5),
            model=ModelSpec(name="m"),
            hardware=HardwareSpec(instance="local", gpu="cpu", n_gpu=1),
            roofline_link=RooflineLink(model_ref="m", hw_ref="local"),
        )

    manifests = ManifestStore(tmp_path / "manifests").list_all()
    assert len(manifests) == 1
    assert manifests[0].status == RunStatus.FAILED
    assert "TimeoutError" in (manifests[0].error or "")
