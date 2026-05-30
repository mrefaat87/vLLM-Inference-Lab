"""Port-forward verification.

The K8s driver used to fire `kubectl port-forward` as a `subprocess.Popen`
and immediately return — if the forward wasn't actually carrying traffic
(stale local port, transient API hiccup, Service not Ready yet), every
load-driver POST came back as ServerDisconnectedError and we'd burn a full
measurement window before noticing. The driver now polls /health through
the local port and either confirms the forward is live or restarts it
once.

Tests stand up a real local HTTP server on a free port (matching the
driver's polling pattern) and verify the wait helper returns the right
boolean. We avoid mocking the subprocess + urllib internals — the value
of the helper is exactly the loop around them.
"""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from experiments.drivers.k8s_common import K8sEngineDriver, KubectlConfig

pytestmark = pytest.mark.unit


class _Probe(K8sEngineDriver):
    """Bare subclass — we only exercise the port-forward helpers."""
    name = "probe"
    template_path = Path("/dev/null")


def _free_port() -> int:
    """Snag a kernel-assigned free port. The probe binds it next."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _HealthServer(BaseHTTPRequestHandler):
    """Minimal /health responder; everything else 404."""
    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args, **kwargs) -> None:  # noqa: D401
        """Silence the default request log so pytest output stays clean."""


def _start_server(port: int) -> HTTPServer:
    srv = HTTPServer(("127.0.0.1", port), _HealthServer)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


class _FakeProc:
    """Subprocess stand-in: `poll()` returns None while ``alive`` is True."""
    def __init__(self, alive: bool = True) -> None:
        self._alive = alive
        self.stderr = None

    def poll(self) -> int | None:
        return None if self._alive else 1

    def kill(self) -> None:
        self._alive = False


def test_wait_returns_true_when_local_health_responds() -> None:
    port = _free_port()
    srv = _start_server(port)
    try:
        driver = _Probe(kubectl=KubectlConfig(kubectl_path="/bin/true"))
        driver.service_port = port
        driver._port_forward_proc = _FakeProc(alive=True)  # subprocess "alive"
        assert driver._wait_for_port_forward(timeout_s=3.0) is True
    finally:
        srv.shutdown()
        srv.server_close()


def test_wait_returns_false_when_subprocess_died() -> None:
    """If the port-forward process exited, no point polling — short-circuit."""
    driver = _Probe(kubectl=KubectlConfig(kubectl_path="/bin/true"))
    driver.service_port = _free_port()  # nothing bound here
    driver._port_forward_proc = _FakeProc(alive=False)
    assert driver._wait_for_port_forward(timeout_s=3.0) is False


def test_wait_returns_false_when_nothing_responds_within_budget() -> None:
    """Subprocess "alive" but nothing actually listening on the local port —
    the wait loop must respect its deadline rather than hanging forever."""
    driver = _Probe(kubectl=KubectlConfig(kubectl_path="/bin/true"))
    driver.service_port = _free_port()  # free port = nothing accepting
    driver._port_forward_proc = _FakeProc(alive=True)
    # 1.5s budget — long enough to make several poll attempts, short enough
    # to keep the test snappy.
    assert driver._wait_for_port_forward(timeout_s=1.5) is False
