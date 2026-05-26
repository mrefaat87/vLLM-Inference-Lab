"""End-to-end smoke for `exp launch`.

Drives the composite verb with the mock engine + `--no-serve --no-open`,
then asserts the artifacts the calc's Validation panel relies on are in
place: a result JSON with a baked-in prediction, a built portal, a
bridge dir with an index, and the stdout URL with the right run-id fragment.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from experiments.cli.exp import cli

pytestmark = pytest.mark.integration


@pytest.fixture
def calc_root() -> Path:
    """Skip if the sibling calc isn't present — `exp launch` itself
    works without it, but we want to assert the bridge dir got the run.
    """
    root = Path(__file__).resolve().parents[3] / "calculators" / "sizing_calc"
    if not (root / "scripts" / "compute_cli.mjs").exists():
        pytest.skip("sizing_calc not present in this clone")
    if shutil.which("node") is None:
        pytest.skip("node not on PATH")
    return root


def test_launch_writes_result_portal_and_bridge(tmp_path: Path, calc_root: Path) -> None:
    results = tmp_path / "results"
    portal = tmp_path / "_site"
    bridge = tmp_path / "bridge"

    result = CliRunner().invoke(
        cli,
        [
            "launch",
            "--engine", "mock",
            "--workload", "chatbot",
            "--rate", "8",
            "--duration", "1.0",
            "--warmup", "0.0",
            "--preflight", "off",
            "--results-dir", str(results),
            "--portal-dir", str(portal),
            "--calc-bridge", str(bridge),
            "--no-serve",
            "--no-open",
        ],
    )
    assert result.exit_code == 0, result.output

    # Exactly one result JSON should have been written.
    runs = list((results / "runs").glob("*.json"))
    assert len(runs) == 1, f"expected one run, got {runs}"
    run_path = runs[0]
    run_id = run_path.stem
    blob = json.loads(run_path.read_text())
    # Prediction baked in (C11).
    assert blob.get("prediction") is not None
    assert blob["prediction"]["calc_version"]

    # Portal built.
    assert (portal / "results_explorer.html").exists()
    assert (portal / "assets" / "runs.json").exists()
    assert (portal / "runs" / f"{run_id}.json").exists()

    # Calc bridge populated.
    assert (bridge / "index.json").exists()
    assert (bridge / "runs" / f"{run_id}.json").exists()
    idx = json.loads((bridge / "index.json").read_text())
    assert any(r["run_id"] == run_id for r in idx["runs"])

    # Stdout points the user at the right URL with run-id fragment.
    assert "open: http://127.0.0.1:" in result.output
    assert f"#{run_id}" in result.output


def test_launch_help_lists_orchestration_flags() -> None:
    """`exp launch --help` must expose all the orchestration flags the
    pasted-from-calc command relies on. Guards against accidental
    decorator drift that would silently drop a flag.
    """
    result = CliRunner().invoke(cli, ["launch", "--help"])
    assert result.exit_code == 0, result.output
    for flag in [
        "--engine", "--workload", "--rate",
        "--serve", "--no-serve", "--open", "--no-open",
        "--port", "--calc-bridge", "--portal-dir",
    ]:
        assert flag in result.output, f"missing {flag} in launch --help"


def test_serve_help_lists_watch_flag() -> None:
    result = CliRunner().invoke(cli, ["serve", "--help"])
    assert result.exit_code == 0, result.output
    for flag in ["--watch", "--no-watch", "--port", "--calc-bridge", "--portal-dir"]:
        assert flag in result.output, f"missing {flag} in serve --help"


def test_launch_reuses_running_server(tmp_path: Path, calc_root: Path) -> None:
    """If another server is already on the chosen port, `exp launch`
    must not crash with "address in use" — it announces the URL and
    relies on the existing server to handle traffic.
    """
    import socket

    # Bind a stub listener on a free port to simulate a running `exp serve`.
    stub = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stub.bind(("127.0.0.1", 0))
    stub.listen(1)
    busy_port = stub.getsockname()[1]

    try:
        result = CliRunner().invoke(
            cli,
            [
                "launch",
                "--engine", "mock",
                "--workload", "chatbot",
                "--rate", "8",
                "--duration", "1.0",
                "--warmup", "0.0",
                "--preflight", "off",
                "--results-dir", str(tmp_path / "results"),
                "--portal-dir", str(tmp_path / "_site"),
                "--calc-bridge", str(tmp_path / "bridge"),
                "--serve",
                "--no-open",
                "--port", str(busy_port),
            ],
        )
    finally:
        stub.close()

    assert result.exit_code == 0, result.output
    assert "already in use — reusing" in result.output
    assert f"http://127.0.0.1:{busy_port}/" in result.output
