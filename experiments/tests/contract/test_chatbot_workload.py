"""ChatbotWorkload vs AbstractWorkloadContract."""

from __future__ import annotations

from experiments.tests.contract.abstract import AbstractWorkloadContract
from experiments.workloads.chatbot import ChatbotParams, ChatbotWorkload


class TestChatbotContract(AbstractWorkloadContract):
    def make_workload(self, seed: int) -> ChatbotWorkload:
        return ChatbotWorkload(seed=seed, params=ChatbotParams(rate_rps=10.0))
