"""Unit tests for runner.schema."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from experiments.runner.schema import (
    SCHEMA_VERSION,
    Analysis,
    EngineConfig,
    HardwareSpec,
    ModelSpec,
    Percentiles,
    RequestRecord,
    RooflineLink,
    RunManifest,
    RunResult,
    RunStatus,
    ThroughputStats,
    WorkloadConfig,
)

pytestmark = pytest.mark.unit


def _good_engine() -> EngineConfig:
    return EngineConfig(name="vllm", image="vllm:latest", model="meta-llama/Llama-3-70B-Instruct")


def _good_result() -> RunResult:
    now = datetime.now(timezone.utc)
    return RunResult(
        run_id="r1",
        started_at=now,
        finished_at=now + timedelta(seconds=10),
        engine=_good_engine(),
        model=ModelSpec(name="Llama-3-70B-AWQ", quant="awq-int4", tp=4),
        hardware=HardwareSpec(instance="g5.12xlarge", gpu="A10G", n_gpu=4),
        workload=WorkloadConfig(name="chatbot", rate_rps=8.0, duration_s=60.0),
        analysis=Analysis(
            steady_state_requests=100,
            failed_requests=0,
            throughput=ThroughputStats(
                total_completion_tokens=10_000,
                total_prompt_tokens=2_000,
                tok_per_sec_avg=166.6,
                requests_per_sec_avg=7.9,
            ),
        ),
        roofline_link=RooflineLink(model_ref="llama-3-70b", hw_ref="a10g-x4"),
    )


class TestEngineConfig:
    def test_accepts_known_engine(self) -> None:
        c = EngineConfig(name="vllm", image="x", model="m")
        assert c.name == "vllm"

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            EngineConfig(name="  ", image="x", model="m")

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            EngineConfig(name="vllm", image="x", model="m", bogus=1)  # type: ignore[call-arg]

    def test_tp_bounds(self) -> None:
        with pytest.raises(ValidationError):
            EngineConfig(name="vllm", image="x", model="m", tensor_parallel=0)
        with pytest.raises(ValidationError):
            EngineConfig(name="vllm", image="x", model="m", tensor_parallel=99)


class TestWorkloadConfig:
    def test_rejects_zero_rate(self) -> None:
        with pytest.raises(ValidationError):
            WorkloadConfig(name="chatbot", rate_rps=0.0, duration_s=10.0)

    def test_rejects_negative_duration(self) -> None:
        with pytest.raises(ValidationError):
            WorkloadConfig(name="chatbot", rate_rps=1.0, duration_s=-1.0)


class TestRunResult:
    def test_roundtrip(self) -> None:
        r = _good_result()
        blob = r.model_dump_json()
        r2 = RunResult.model_validate_json(blob)
        assert r2 == r
        assert r2.schema_version == SCHEMA_VERSION

    def test_timeline_validator(self) -> None:
        r = _good_result()
        # model_copy + model_dump don't re-run validators in current pydantic
        # (they're a fast path for already-validated instances). The validator
        # fires on model_validate, which is what callers use when loading from
        # disk — the path that actually matters.
        with pytest.raises(ValidationError):
            RunResult.model_validate(
                {
                    **r.model_dump(mode="json"),
                    "finished_at": (r.started_at - timedelta(seconds=1)).isoformat(),
                }
            )


class TestRunManifest:
    def test_planned_minimal(self) -> None:
        m = RunManifest(
            run_id="r1", status=RunStatus.PLANNED, engine_name="vllm", workload_name="chatbot"
        )
        assert m.status == RunStatus.PLANNED

    def test_running_requires_started(self) -> None:
        with pytest.raises(ValidationError):
            RunManifest(
                run_id="r1",
                status=RunStatus.RUNNING,
                engine_name="vllm",
                workload_name="chatbot",
            )

    def test_done_requires_result_path(self) -> None:
        now = datetime.now(timezone.utc)
        with pytest.raises(ValidationError):
            RunManifest(
                run_id="r1",
                status=RunStatus.DONE,
                engine_name="vllm",
                workload_name="chatbot",
                started_at=now,
                finished_at=now,
            )

    def test_failed_requires_error(self) -> None:
        now = datetime.now(timezone.utc)
        with pytest.raises(ValidationError):
            RunManifest(
                run_id="r1",
                status=RunStatus.FAILED,
                engine_name="vllm",
                workload_name="chatbot",
                started_at=now,
                finished_at=now,
            )


class TestPercentilesAndRecord:
    def test_percentiles_n_nonneg(self) -> None:
        Percentiles(p50=1.0, p95=2.0, p99=3.0, mean=1.5, n=0)
        with pytest.raises(ValidationError):
            Percentiles(p50=1.0, p95=2.0, p99=3.0, mean=1.5, n=-1)

    def test_request_record_optional_ttft(self) -> None:
        r = RequestRecord(
            request_id="x",
            label="chatbot.turn1",
            submit_offset_s=0.0,
            prompt_tokens=10,
            max_new_tokens=50,
        )
        assert r.ttft_s is None
        assert r.error is None
