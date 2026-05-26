"""SGLangDriver contract test (fake kubectl + stub server)."""

from __future__ import annotations

import socket
import tempfile
from pathlib import Path

from experiments.drivers.k8s_common import KubectlConfig
from experiments.drivers.sglang_driver import SGLangDriver
from experiments.runner.schema import EngineConfig
from experiments.runner.stub_server import StubServer
from experiments.tests.contract._fake_kubectl import install_fake_kubectl
from experiments.tests.contract.abstract import AbstractEngineDriverContract


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _FakeKubectlSGLangDriver(SGLangDriver):
    def __init__(self, stub: StubServer, kubectl_path: Path) -> None:
        super().__init__(
            kubectl=KubectlConfig(kubectl_path=str(kubectl_path)),
            readiness_timeout_s=5.0,
        )
        self._stub = stub

    def _await_ready(self) -> None:  # type: ignore[override]
        return None

    def endpoint_url(self) -> str:  # type: ignore[override]
        return self._stub.url


class TestSGLangDriverContract(AbstractEngineDriverContract):
    _tmpdir: Path
    _kubectl: Path
    _stub: StubServer

    @classmethod
    def setup_class(cls) -> None:
        cls._tmpdir = Path(tempfile.mkdtemp(prefix="sglang-driver-test-"))
        cls._kubectl = install_fake_kubectl(cls._tmpdir)
        cls._stub = StubServer(port=_free_port())
        cls._stub.start()

    @classmethod
    def teardown_class(cls) -> None:
        cls._stub.stop()

    def make_driver(self) -> _FakeKubectlSGLangDriver:
        return _FakeKubectlSGLangDriver(self._stub, self._kubectl)

    def make_config(self) -> EngineConfig:
        return EngineConfig(
            name="sglang",
            image="lmsysorg/sglang:latest",
            model="meta-llama/Llama-3-70B-Instruct-AWQ",
            quantization="awq",
            tensor_parallel=4,
            max_model_len=8192,
        )
