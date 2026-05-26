"""MixWorkload vs AbstractWorkloadContract."""

from __future__ import annotations

from experiments.tests.contract.abstract import AbstractWorkloadContract
from experiments.workloads.agentic_coding import AgenticCodingWorkload, AgenticParams
from experiments.workloads.chatbot import ChatbotParams, ChatbotWorkload
from experiments.workloads.mix import MixWorkload, WeightedChild


class TestMixContract(AbstractWorkloadContract):
    def make_workload(self, seed: int) -> MixWorkload:
        return MixWorkload(
            [
                WeightedChild(
                    ChatbotWorkload(seed=seed, params=ChatbotParams(rate_rps=10.0)),
                    weight=0.7,
                ),
                WeightedChild(
                    AgenticCodingWorkload(seed=seed + 1, params=AgenticParams(rate_rps=2.0)),
                    weight=0.3,
                ),
            ]
        )
