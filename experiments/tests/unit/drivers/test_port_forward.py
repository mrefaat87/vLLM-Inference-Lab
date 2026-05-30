"""Tests for the K8s driver's port-forward lifecycle.

The port-forward path is mocked end-to-end: subprocess.Popen is stubbed
so the test asserts the *contract* (we spawn a kubectl port-forward, we
terminate it on stop) without needing a live cluster.

The local-side /health verification is patched out — it's covered in
test_port_forward_verification.py with a real local HTTP server.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from experiments.drivers.k8s_common import K8sEngineDriver, KubectlConfig
from experiments.runner.schema import EngineConfig

pytestmark = pytest.mark.unit


class _FakeDriver(K8sEngineDriver):
    """Concrete K8sEngineDriver with the kubectl shellouts stubbed.

    The real ``_kubectl_apply`` and ``_await_ready`` would hit a live
    cluster; we replace them with no-ops so we can isolate the
    port-forward lifecycle.
    """

    name = "vllm"
    template_path = None  # render() is overridden, never reads disk
    service_port = 8000

    def render(self, cfg: EngineConfig) -> str:
        return "manifest: stub"

    def _kubectl_apply(self, manifest: str) -> None:  # noqa: ARG002
        pass

    def _await_ready(self) -> None:
        pass

    def _verify_realized_instance(self, cfg: EngineConfig) -> None:  # noqa: ARG002
        # The realized-instance check shells out to kubectl too —
        # short-circuit it in tests so we don't need to mock 4 calls.
        pass

    def _kubectl_delete(self) -> None:
        pass


def _cfg() -> EngineConfig:
    return EngineConfig(
        name="vllm",
        image="vllm:test",
        model="meta-llama/Llama-3-8B",
        tensor_parallel=1,
    )


@patch.object(K8sEngineDriver, "_wait_for_port_forward", return_value=True)
@patch("experiments.drivers.k8s_common.time.sleep", lambda _: None)
@patch("experiments.drivers.k8s_common.subprocess.Popen")
def test_start_spawns_port_forward_by_default(
    popen_mock: MagicMock, _wait_mock: MagicMock,
) -> None:
    proc = MagicMock()
    proc.wait.return_value = 0
    popen_mock.return_value = proc

    driver = _FakeDriver(kubectl=KubectlConfig(kubectl_path="/fake/kubectl"))
    driver.start(_cfg())

    popen_mock.assert_called_once()
    cmd = popen_mock.call_args.args[0]
    # The spawned command must include "port-forward" and the service ref.
    assert "port-forward" in cmd
    assert "svc/vllm-engine" in cmd
    assert "8000:8000" in cmd


@patch.object(K8sEngineDriver, "_wait_for_port_forward", return_value=True)
@patch("experiments.drivers.k8s_common.time.sleep", lambda _: None)
@patch("experiments.drivers.k8s_common.subprocess.Popen")
def test_stop_terminates_port_forward(
    popen_mock: MagicMock, _wait_mock: MagicMock,
) -> None:
    proc = MagicMock()
    proc.wait.return_value = 0
    popen_mock.return_value = proc

    driver = _FakeDriver(kubectl=KubectlConfig(kubectl_path="/fake/kubectl"))
    driver.start(_cfg())
    driver.stop()

    proc.terminate.assert_called_once()


@patch("experiments.drivers.k8s_common.time.sleep", lambda _: None)
@patch("experiments.drivers.k8s_common.subprocess.Popen")
def test_no_port_forward_flag_skips_spawn(popen_mock: MagicMock) -> None:
    driver = _FakeDriver(
        kubectl=KubectlConfig(kubectl_path="/fake/kubectl"),
        port_forward=False,
    )
    driver.start(_cfg())

    popen_mock.assert_not_called()


@patch.object(K8sEngineDriver, "_wait_for_port_forward", return_value=True)
@patch("experiments.drivers.k8s_common.time.sleep", lambda _: None)
@patch("experiments.drivers.k8s_common.subprocess.Popen")
def test_stop_force_kills_unresponsive_port_forward(
    popen_mock: MagicMock, _wait_mock: MagicMock,
) -> None:
    proc = MagicMock()
    proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="kubectl", timeout=5.0), 0]
    popen_mock.return_value = proc

    driver = _FakeDriver(kubectl=KubectlConfig(kubectl_path="/fake/kubectl"))
    driver.start(_cfg())
    driver.stop()

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()


@patch.object(K8sEngineDriver, "_wait_for_port_forward", return_value=True)
@patch("experiments.drivers.k8s_common.time.sleep", lambda _: None)
@patch("experiments.drivers.k8s_common.subprocess.Popen")
def test_endpoint_url_points_at_localhost_under_port_forward(
    popen_mock: MagicMock, _wait_mock: MagicMock,
) -> None:
    proc = MagicMock()
    popen_mock.return_value = proc

    driver = _FakeDriver(kubectl=KubectlConfig(kubectl_path="/fake/kubectl"))
    endpoint = driver.start(_cfg())

    assert endpoint.base_url == "http://127.0.0.1:8000"


# New: when verification fails, start() must raise a clear RuntimeError
# rather than returning a dead endpoint that quietly drops every request.
@patch.object(K8sEngineDriver, "_wait_for_port_forward", return_value=False)
@patch("experiments.drivers.k8s_common.time.sleep", lambda _: None)
@patch("experiments.drivers.k8s_common.subprocess.Popen")
def test_start_raises_when_port_forward_never_responds(
    popen_mock: MagicMock, _wait_mock: MagicMock,
) -> None:
    proc = MagicMock()
    proc.wait.return_value = 0
    # Empty stderr — covers the "<none>" branch of the error message.
    proc.stderr.read.return_value = b""
    popen_mock.return_value = proc

    driver = _FakeDriver(kubectl=KubectlConfig(kubectl_path="/fake/kubectl"))
    with pytest.raises(RuntimeError, match="kubectl port-forward never responded"):
        driver.start(_cfg())
    # One restart was attempted: Popen called twice (initial + retry).
    assert popen_mock.call_count == 2
