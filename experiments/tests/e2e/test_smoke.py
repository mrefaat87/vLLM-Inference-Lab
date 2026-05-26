"""End-to-end smoke test against a real cluster.

Skipped unless ``RUN_E2E=1``. The CI ``e2e.yml`` workflow:
  1. Provisions the inference-lab cluster via ./eks/bringup.sh
  2. Sets RUN_E2E=1
  3. Runs `pytest -m e2e`
  4. Tears down via ./eks/teardown.sh

This test then:
  - Runs a short sweep against the engine selected by EXP_ENGINE.
  - Asserts the result JSON parses, TTFT/throughput are non-zero,
    and at least one request succeeded.

Validates the entire pipeline: real engine pod → real driver loop → real
result on disk → portal-loadable.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from experiments.cli.exp import DRIVERS, ENGINE_IMAGES, _workload
from experiments.runner.schema import (
    EngineConfig,
    HardwareSpec,
    ModelSpec,
    RooflineLink,
    WorkloadConfig,
)
from experiments.runner.sweep import SweepRunner, load_result

pytestmark = pytest.mark.e2e

_ENABLED = os.environ.get("RUN_E2E") == "1"


@pytest.mark.skipif(not _ENABLED, reason="RUN_E2E=1 not set")
def test_smoke_against_real_cluster(tmp_path: Path) -> None:
    engine_name = os.environ.get("EXP_ENGINE", "vllm")
    duration_s = float(os.environ.get("EXP_DURATION_S", "60"))
    rate = float(os.environ.get("EXP_RATE", "2"))

    driver_cls = DRIVERS[engine_name]
    driver = driver_cls()
    engine_cfg = EngineConfig(
        name=engine_name,
        image=ENGINE_IMAGES[engine_name],
        model="meta-llama/Llama-3-70B-Instruct-AWQ",
        quantization="awq",
        tensor_parallel=4,
    )
    workload_cfg = WorkloadConfig(
        name="chatbot", rate_rps=rate, duration_s=duration_s, warmup_s=10.0
    )
    workload = _workload("chatbot", seed=1, rate=rate)

    runner = SweepRunner(results_dir=tmp_path)

    started = datetime.now(timezone.utc)
    path = asyncio.run(
        runner.run_one(
            engine=driver,
            engine_cfg=engine_cfg,
            workload=workload,
            workload_cfg=workload_cfg,
            model=ModelSpec(name="Llama-3-70B-AWQ", quant="awq", tp=4),
            hardware=HardwareSpec(instance="g5.12xlarge", gpu="A10G", n_gpu=4),
            roofline_link=RooflineLink(model_ref="llama-3-70b", hw_ref="a10g-x4"),
            notes=f"e2e smoke; started_at={started.isoformat()}",
        )
    )

    result = load_result(path)
    assert result.analysis.steady_state_requests > 0
    assert result.analysis.failed_requests <= 0.2 * result.analysis.steady_state_requests
    assert result.analysis.throughput.tok_per_sec_avg > 0
    if result.analysis.ttft_s is not None:
        assert result.analysis.ttft_s.p50 > 0
        # TTFT > 30s on a smoke run means the engine is broken.
        assert result.analysis.ttft_s.p99 < 30.0
