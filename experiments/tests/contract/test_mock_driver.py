"""MockEngineDriver vs AbstractEngineDriverContract."""

from __future__ import annotations

from experiments.drivers.mock import MockEngineDriver
from experiments.runner.schema import EngineConfig
from experiments.tests.contract.abstract import AbstractEngineDriverContract


class TestMockDriverContract(AbstractEngineDriverContract):
    def make_driver(self) -> MockEngineDriver:
        return MockEngineDriver()

    def make_config(self) -> EngineConfig:
        return EngineConfig(name="mock", image="n/a", model="mock-model")
