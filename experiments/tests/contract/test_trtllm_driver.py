"""TRTLLMDriver contract test (fake kubectl + stub server).

The driver invokes an engine-build Job before deploying the serve
Deployment. The fake kubectl shim no-ops both, so the contract still
exercises the driver's actual code path including the build step.
"""

from __future__ import annotations

import socket
import tempfile
from pathlib import Path

from experiments.drivers.k8s_common import KubectlConfig
from experiments.drivers.trtllm_driver import TRTLLMDriver
from experiments.runner.schema import EngineConfig
from experiments.runner.stub_server import StubServer
from experiments.tests.contract._fake_kubectl import install_fake_kubectl
from experiments.tests.contract.abstract import AbstractEngineDriverContract


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _FakeKubectlTRTLLMDriver(TRTLLMDriver):
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


class TestTRTLLMDriverContract(AbstractEngineDriverContract):
    _tmpdir: Path
    _kubectl: Path
    _stub: StubServer

    @classmethod
    def setup_class(cls) -> None:
        cls._tmpdir = Path(tempfile.mkdtemp(prefix="trtllm-driver-test-"))
        cls._kubectl = install_fake_kubectl(cls._tmpdir)
        cls._stub = StubServer(port=_free_port())
        cls._stub.start()

    @classmethod
    def teardown_class(cls) -> None:
        cls._stub.stop()

    def make_driver(self) -> _FakeKubectlTRTLLMDriver:
        return _FakeKubectlTRTLLMDriver(self._stub, self._kubectl)

    def make_config(self) -> EngineConfig:
        return EngineConfig(
            name="trtllm",
            image="nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3",
            model="meta-llama/Llama-3-70B-Instruct-AWQ",
            quantization="awq",
            tensor_parallel=4,
            max_model_len=8192,
        )
