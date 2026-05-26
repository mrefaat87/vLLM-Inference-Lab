"""RagWorkload vs AbstractWorkloadContract."""

from __future__ import annotations

from experiments.tests.contract.abstract import AbstractWorkloadContract
from experiments.workloads.rag import RagParams, RagWorkload


class TestRagContract(AbstractWorkloadContract):
    def make_workload(self, seed: int) -> RagWorkload:
        return RagWorkload(seed=seed, params=RagParams(rate_rps=8.0))
