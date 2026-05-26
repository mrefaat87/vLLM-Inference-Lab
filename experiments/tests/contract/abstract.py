"""Abstract contract test classes.

Every EngineDriver implementation and every WorkloadGenerator implementation
subclasses one of these test classes to inherit the full contract suite.
This way a new engine driver gets ~15 tests for free, and the contract
itself can only evolve in one place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import pytest

from experiments.drivers.base import EndpointInfo, EngineDriver
from experiments.runner.schema import EngineConfig
from experiments.workloads.base import Request, WorkloadGenerator


# ---------------------------------------------------------------------------
# Engine driver contract
# ---------------------------------------------------------------------------
class AbstractEngineDriverContract(ABC):
    """Every concrete EngineDriver subclasses this in its test file.

    Subclasses must implement ``make_driver`` and ``make_config``.
    Subclasses requiring real cloud infra should mark themselves
    @pytest.mark.e2e so the contract suite is skipped by default.
    """

    @abstractmethod
    def make_driver(self) -> EngineDriver:
        ...

    @abstractmethod
    def make_config(self) -> EngineConfig:
        ...

    @pytest.mark.contract
    def test_name_matches_config(self) -> None:
        driver = self.make_driver()
        cfg = self.make_config()
        assert driver.name, "driver.name must be set"
        assert driver.name == cfg.name, "driver.name must match the config it accepts"

    @pytest.mark.contract
    def test_start_returns_endpoint(self) -> None:
        driver = self.make_driver()
        cfg = self.make_config()
        ep = driver.start(cfg)
        try:
            assert isinstance(ep, EndpointInfo)
            assert ep.base_url.startswith(("http://", "https://"))
        finally:
            driver.stop()

    @pytest.mark.contract
    def test_healthcheck_true_after_start(self) -> None:
        driver = self.make_driver()
        cfg = self.make_config()
        driver.start(cfg)
        try:
            assert driver.healthcheck() is True
        finally:
            driver.stop()

    @pytest.mark.contract
    def test_metrics_returns_dict(self) -> None:
        driver = self.make_driver()
        cfg = self.make_config()
        driver.start(cfg)
        try:
            m = driver.metrics()
            assert isinstance(m, dict)
        finally:
            driver.stop()

    @pytest.mark.contract
    def test_stop_is_idempotent(self) -> None:
        driver = self.make_driver()
        cfg = self.make_config()
        driver.start(cfg)
        driver.stop()
        # Second stop must not raise.
        driver.stop()
        # Healthcheck must be False after stop.
        assert driver.healthcheck() is False

    @pytest.mark.contract
    def test_context_manager(self) -> None:
        driver = self.make_driver()
        cfg = self.make_config()
        with driver as d:
            d.start(cfg)
            assert d.healthcheck()
        assert driver.healthcheck() is False


# ---------------------------------------------------------------------------
# Workload contract
# ---------------------------------------------------------------------------
class AbstractWorkloadContract(ABC):
    """Every WorkloadGenerator subclass tests against this."""

    @abstractmethod
    def make_workload(self, seed: int) -> WorkloadGenerator:
        ...

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_name_set(self) -> None:
        w = self.make_workload(seed=1)
        assert w.name, "workload.name must be set"

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_arrivals_monotonic_and_bounded(self) -> None:
        w = self.make_workload(seed=1)
        duration = 5.0
        previous = -1.0
        count = 0
        async for req in w.requests(duration_s=duration):
            assert isinstance(req, Request)
            assert req.arrival_offset_s >= 0.0
            assert req.arrival_offset_s < duration
            assert req.arrival_offset_s >= previous, "arrivals must be non-decreasing"
            previous = req.arrival_offset_s
            count += 1
            if count > 10_000:
                pytest.fail("workload produced unreasonably many requests in 5s")

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_determinism_same_seed(self) -> None:
        a = await _collect(self.make_workload(seed=42).requests(duration_s=3.0))
        b = await _collect(self.make_workload(seed=42).requests(duration_s=3.0))
        assert len(a) == len(b)
        for ra, rb in zip(a, b, strict=True):
            assert ra.prompt == rb.prompt
            assert ra.prompt_tokens == rb.prompt_tokens
            assert ra.max_new_tokens == rb.max_new_tokens
            assert ra.arrival_offset_s == pytest.approx(rb.arrival_offset_s, abs=1e-9)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_different_seeds_differ(self) -> None:
        a = await _collect(self.make_workload(seed=1).requests(duration_s=3.0))
        b = await _collect(self.make_workload(seed=2).requests(duration_s=3.0))
        # At least one prompt or arrival should differ. If a workload is
        # deterministic in length but random in content, this still passes.
        differs = (
            len(a) != len(b)
            or any(ra.prompt != rb.prompt for ra, rb in zip(a, b, strict=False))
            or any(
                ra.arrival_offset_s != rb.arrival_offset_s
                for ra, rb in zip(a, b, strict=False)
            )
        )
        assert differs, "different seeds must produce a different request stream"

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_positive_lengths(self) -> None:
        w = self.make_workload(seed=1)
        async for req in w.requests(duration_s=2.0):
            assert req.prompt_tokens > 0
            assert req.max_new_tokens > 0


async def _collect(it: AsyncIterator[Request]) -> list[Request]:
    out: list[Request] = []
    async for r in it:
        out.append(r)
    return out
