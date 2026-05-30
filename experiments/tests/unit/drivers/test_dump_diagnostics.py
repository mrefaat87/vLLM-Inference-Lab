"""K8s driver pod-log + events capture.

Stages a fake `kubectl` shell script that echoes a recognisable token to
stdout. The test confirms ``dump_diagnostics`` writes the token into the
log file at the requested path and returns the path in its result dict.

We test against a fake binary rather than mocking subprocess.run so the
arg-list construction (context, namespace, deployment name, --tail, etc.)
is also exercised — those are exactly the strings that would silently
break on a real cluster.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from experiments.drivers.k8s_common import K8sEngineDriver, KubectlConfig
from experiments.runner.schema import EngineConfig

pytestmark = pytest.mark.unit


class _Probe(K8sEngineDriver):
    """Minimal K8s driver subclass — no real engine, just exercises the
    base class plumbing."""
    name = "probe"
    template_path = Path("/dev/null")  # never read; we don't call start()


def _write_fake_kubectl(target: Path) -> Path:
    """Stage a kubectl shim that echoes its argv tail and a marker."""
    script = target / "kubectl"
    script.write_text(
        "#!/bin/sh\n"
        "echo \"[fake-kubectl] args: $*\"\n"
        "echo \"PROBE_MARKER_OK\"\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


def test_dump_diagnostics_writes_logs_and_events(tmp_path: Path) -> None:
    fake_kubectl = _write_fake_kubectl(tmp_path)
    driver = _Probe(
        kubectl=KubectlConfig(
            context="ctx", namespace="ns", kubectl_path=str(fake_kubectl),
        ),
    )
    # Drive enough of the lifecycle that dump_diagnostics will fire — we
    # don't actually `start()` (template is /dev/null) but we set _cfg
    # and _release_name directly since those are what dump checks.
    driver._cfg = EngineConfig(name="probe", image="x", model="m")
    driver._release_name = "probe-engine"

    out_dir = tmp_path / "runs"
    paths = driver.dump_diagnostics(out_dir, basename="run-abc")

    log = Path(paths["log_path"])
    events = Path(paths["events_path"])
    assert log.exists() and events.exists()
    # Marker proves we actually invoked the fake kubectl, not just wrote
    # an empty file. Both captures hit the same shim and see the marker.
    log_text = log.read_text()
    events_text = events.read_text()
    assert "PROBE_MARKER_OK" in log_text
    assert "PROBE_MARKER_OK" in events_text
    # Arg-list construction is the load-bearing detail — confirm the right
    # context and namespace and the right kubectl subcommands made it through.
    assert "--context ctx" in log_text
    assert "-n ns" in log_text
    assert "logs deployment/probe-engine" in log_text
    assert "get events" in events_text
    assert "involvedObject.name=probe-engine" in events_text


def test_dump_diagnostics_returns_empty_when_engine_not_started(
    tmp_path: Path,
) -> None:
    driver = _Probe(kubectl=KubectlConfig(kubectl_path="/bin/true"))
    # No _cfg, no _release_name → diagnostics has nothing to capture.
    assert driver.dump_diagnostics(tmp_path, basename="run") == {}


def test_dump_diagnostics_swallows_kubectl_failure(tmp_path: Path) -> None:
    """A failing kubectl must produce a stub file with the error inside —
    never raise. Diagnostics are best-effort; raising would mask the real
    failure the runner is trying to capture context for."""
    # /bin/false exits 1 immediately with no output.
    driver = _Probe(kubectl=KubectlConfig(kubectl_path="/bin/false"))
    driver._cfg = EngineConfig(name="probe", image="x", model="m")
    driver._release_name = "probe-engine"
    paths = driver.dump_diagnostics(tmp_path, basename="run")
    # Files still written, just with the "[no logs captured]" stub from
    # an empty stdout — the capture itself did not raise.
    assert Path(paths["log_path"]).exists()
    assert Path(paths["events_path"]).exists()
