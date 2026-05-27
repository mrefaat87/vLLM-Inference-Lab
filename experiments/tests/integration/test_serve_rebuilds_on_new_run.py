"""`exp serve` rebuilds the portal when a new result JSON lands.

Skipped if `watchdog` isn't installed — it's an optional dev dep.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("watchdog")

from experiments.cli.exp import _start_portal_server, _try_start_watcher  # noqa: E402
from experiments.portal.build import BuildInputs, build as build_portal  # noqa: E402

pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Minimal RunResult-shaped blob the build script can parse for the index.
# We bypass SweepRunner here to keep the watch loop isolated from the
# (much slower) full driver run — this test is about the watcher firing,
# not about result correctness.
def _minimal_result(run_id: str) -> dict:
    return {
        "schema_version": "1.1.0",
        "run_id": run_id,
        "engine": {"name": "mock", "image": "n/a", "model": "m", "quantization": "awq",
                   "tensor_parallel": 1},
        "model": {"name": "m", "quant": "awq", "tp": 1},
        "hardware": {"instance": "g5.xlarge", "gpu": "A10G", "n_gpu": 1},
        "workload": {"name": "chatbot", "rate_rps": 1.0, "duration_s": 1.0,
                     "warmup_s": 0.0, "seed": 1},
        "roofline_link": {"model_ref": "llama-3-70b", "hw_ref": "A10G"},
        "started_at": "2026-05-26T00:00:00+00:00",
        "finished_at": "2026-05-26T00:00:01+00:00",
        "analysis": {
            "ttft_s": None,
            "tbt_s": None,
            "e2e_s": None,
            "throughput": {"tok_per_sec_avg": 0.0, "req_per_sec_avg": 0.0,
                           "completed": 0, "failed": 0},
            "steady_state_requests": 0,
            "isl_tokens_p50": 0,
            "osl_tokens_p50": 0,
        },
        "records": [],
        "prediction": None,
    }


def test_watcher_rebuilds_on_new_result(tmp_path: Path) -> None:
    results = tmp_path / "results"
    portal = tmp_path / "_site"
    bridge = tmp_path / "bridge"
    runs_dir = results / "runs"
    runs_dir.mkdir(parents=True)

    # Initial build so portal exists.
    build_portal(BuildInputs(results_dir=results, out_dir=portal, calc_bridge=bridge))
    initial_index = json.loads((portal / "assets" / "runs.json").read_text())
    assert initial_index["runs"] == []

    observer = _try_start_watcher(results_dir=results, portal_dir=portal, bridge=bridge)
    assert observer is not None, "watchdog should be installed for this test"

    try:
        # Drop a new result file; watcher should pick it up within ~1s.
        run_id = "test-run-xyz"
        (runs_dir / f"{run_id}.json").write_text(json.dumps(_minimal_result(run_id)))

        index_path = portal / "assets" / "runs.json"
        initial_mtime = index_path.stat().st_mtime_ns

        deadline = time.time() + 3.0
        rebuilt = False
        while time.time() < deadline:
            time.sleep(0.1)
            if index_path.stat().st_mtime_ns != initial_mtime:
                rebuilt = True
                break
        assert rebuilt, "watcher did not rebuild portal within 3s of new file event"
    finally:
        observer.stop()
        observer.join(timeout=2.0)


def test_portal_server_serves_built_index(tmp_path: Path) -> None:
    """Smoke: the static server actually returns the built index page."""
    import urllib.request

    portal = tmp_path / "_site"
    portal.mkdir()
    (portal / "results_explorer.html").write_text("<html>hello</html>")

    port = _free_port()
    server = _start_portal_server(portal, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/results_explorer.html", timeout=2.0) as r:
            body = r.read().decode()
        assert "hello" in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
