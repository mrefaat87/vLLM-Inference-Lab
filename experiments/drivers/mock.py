"""Mock driver.

Backed by a local stub OpenAI-compatible server (see ``experiments.runner.stub_server``).
Used everywhere CI runs to exercise the full pipeline without GPU.
"""

from __future__ import annotations

from typing import Any

from experiments.drivers.base import EndpointInfo, EngineDriver
from experiments.runner.schema import EngineConfig
from experiments.runner.stub_server import StubServer


class MockEngineDriver(EngineDriver):
    name = "mock"

    def __init__(self, *, port: int = 0) -> None:
        # port=0 → OS picks a free one.
        self._server: StubServer | None = None
        self._port = port

    def start(self, cfg: EngineConfig) -> EndpointInfo:
        if cfg.name != self.name:
            raise ValueError(f"MockEngineDriver expects cfg.name=mock, got {cfg.name!r}")
        if self._server is not None:
            return EndpointInfo(base_url=self._server.url, openai_compat=True)
        self._server = StubServer(port=self._port)
        self._server.start()
        return EndpointInfo(base_url=self._server.url, openai_compat=True)

    def healthcheck(self) -> bool:
        return self._server is not None and self._server.is_running

    def metrics(self) -> dict[str, Any]:
        if self._server is None:
            return {}
        return self._server.metrics()

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None
