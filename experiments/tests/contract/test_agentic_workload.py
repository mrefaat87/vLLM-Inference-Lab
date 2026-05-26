"""AgenticCodingWorkload vs AbstractWorkloadContract."""

from __future__ import annotations

from experiments.tests.contract.abstract import AbstractWorkloadContract
from experiments.workloads.agentic_coding import AgenticCodingWorkload, AgenticParams


class TestAgenticContract(AbstractWorkloadContract):
    def make_workload(self, seed: int) -> AgenticCodingWorkload:
        return AgenticCodingWorkload(seed=seed, params=AgenticParams(rate_rps=4.0))
