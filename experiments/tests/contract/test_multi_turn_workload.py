"""MultiTurnWorkload vs AbstractWorkloadContract."""

from __future__ import annotations

from experiments.tests.contract.abstract import AbstractWorkloadContract
from experiments.workloads.multi_turn import MultiTurnParams, MultiTurnWorkload


class TestMultiTurnContract(AbstractWorkloadContract):
    def make_workload(self, seed: int) -> MultiTurnWorkload:
        return MultiTurnWorkload(
            seed=seed,
            params=MultiTurnParams(rate_rps=10.0, concurrent_sessions=8),
        )
