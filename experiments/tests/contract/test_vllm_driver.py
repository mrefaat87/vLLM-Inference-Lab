"""VLLMDriver contract test.

Uses a fake kubectl + a local stub server so the same AbstractEngineDriverContract
runs against the real driver code without a real cluster.
"""

from __future__ import annotations

import socket
import tempfile
from pathlib import Path
from typing import Any

import pytest

from experiments.drivers.k8s_common import KubectlConfig
from experiments.drivers.vllm_driver import VLLMDriver
from experiments.runner.schema import EngineConfig
from experiments.runner.stub_server import StubServer
from experiments.tests.contract._fake_kubectl import install_fake_kubectl
from experiments.tests.contract.abstract import AbstractEngineDriverContract


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _FakeKubectlVLLMDriver(VLLMDriver):
    """VLLMDriver wired to a fake kubectl and a local stub server.

    Overrides _await_ready (no real Deployment to wait on) and endpoint_url
    (point at the stub port instead of 8000).
    """

    def __init__(self, stub: StubServer, kubectl_path: Path) -> None:
        super().__init__(
            kubectl=KubectlConfig(kubectl_path=str(kubectl_path)),
            readiness_timeout_s=5.0,
        )
        self._stub = stub

    def _await_ready(self) -> None:  # type: ignore[override]
        # In real life we'd wait on Deployment Available + port-forward.
        # Here, the stub server is already running.
        return None

    def endpoint_url(self) -> str:  # type: ignore[override]
        return self._stub.url


class TestVLLMDriverContract(AbstractEngineDriverContract):
    _tmpdir: Path
    _kubectl: Path
    _stub: StubServer

    @classmethod
    def setup_class(cls) -> None:
        cls._tmpdir = Path(tempfile.mkdtemp(prefix="vllm-driver-test-"))
        cls._kubectl = install_fake_kubectl(cls._tmpdir)
        cls._stub = StubServer(port=_free_port())
        cls._stub.start()

    @classmethod
    def teardown_class(cls) -> None:
        cls._stub.stop()

    def make_driver(self) -> _FakeKubectlVLLMDriver:
        return _FakeKubectlVLLMDriver(self._stub, self._kubectl)

    def make_config(self) -> EngineConfig:
        return EngineConfig(
            name="vllm",
            image="vllm/vllm-openai:latest",
            model="meta-llama/Llama-3-70B-Instruct-AWQ",
            quantization="awq",
            tensor_parallel=4,
            max_model_len=8192,
        )
