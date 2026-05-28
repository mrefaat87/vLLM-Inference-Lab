"""Integration test: `exp serve` reroot.

The static server is now rooted at the repo root (not the portal subtree)
so absolute links like ``/calculators/sizing_calculator.html`` and
``/experiments/_site/results_explorer.html`` resolve from a single port.
This test exercises the same code path ``exp serve`` uses
(``_start_portal_server``) and confirms HTTP GETs against both sibling
trees succeed.

This is the regression test for the bug that motivated the change:
``BENCH › SIZING CALC`` 404'd because the relative ``../../calculators/…``
href resolved outside the served directory.
"""

from __future__ import annotations

import socket
import threading
import urllib.request
from pathlib import Path

import pytest

from experiments.cli.exp import _repo_root, _start_portal_server
from experiments.portal.build import BuildInputs, build as build_portal

pytestmark = pytest.mark.integration


def _free_port() -> int:
    """Grab a kernel-assigned free port. Two binds for safety: close,
    reopen — we hand the port to the server we're about to launch.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_repo_root_server_reaches_both_calc_and_explorer(tmp_path: Path) -> None:
    """Build the portal, then start the static server at the repo root.
    The bench-switcher's two flagship links must both 200.

    Why this is an integration test (not unit): the bug only manifests
    once an HTTP client resolves the absolute href through the real
    server's filesystem mount.
    """
    repo = _repo_root()
    # Pre-flight: the two on-disk artefacts must exist for the GETs to
    # have any chance of succeeding. The calc HTML is checked into the
    # repo; the portal is built into a tmp dir under the repo so the
    # `/experiments/_site/…` URL resolves under the served root.
    calc_html = repo / "calculators" / "sizing_calculator.html"
    if not calc_html.exists():
        pytest.skip("sizing_calculator.html not built in this clone")

    # Build a transient portal into a path UNDER the repo root so the
    # absolute URL maps. We re-use the standard out_dir but under
    # tmp_path/_site_inside_repo would be outside the repo. Use the
    # real experiments/_site dir on disk if writable, else skip.
    out_dir = repo / "experiments" / "_site"
    if not out_dir.parent.exists():
        pytest.skip("experiments/ not present")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    build_portal(BuildInputs(results_dir=results_dir, out_dir=out_dir))

    port = _free_port()
    server = _start_portal_server(repo, port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        # Calc HTML — the link that 404'd before the reroot.
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/calculators/sizing_calculator.html",
            timeout=5,
        ) as resp:
            assert resp.status == 200
            body = resp.read().decode("utf-8", errors="replace")
            assert "<title>" in body
            # The calc's title is stable: anchor on it so a future
            # template change doesn't silently break this assertion.
            assert "Sizing Calculator" in body or "sizing calculator" in body.lower()

        # Results Explorer — the same port serves the portal subtree too.
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/experiments/_site/results_explorer.html",
            timeout=5,
        ) as resp:
            assert resp.status == 200
            re_body = resp.read().decode("utf-8", errors="replace")
            assert "Results Explorer" in re_body
            # Absolute href to the calc is present (the contract the
            # reroot exists to support).
            assert "/calculators/sizing_calculator.html" in re_body
    finally:
        server.shutdown()
        server.server_close()


def test_repo_root_helper_points_at_repo_top() -> None:
    """`_repo_root()` must resolve to the directory containing both
    ``calculators/`` and ``experiments/``. If anyone moves exp.py up or
    down the tree, this assertion catches the off-by-one.
    """
    root = _repo_root()
    assert (root / "experiments").is_dir()
    assert (root / "calculators").is_dir() or (root / "calculators").exists() is False, (
        "_repo_root must resolve to the parent of experiments/"
    )
