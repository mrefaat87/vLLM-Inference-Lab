"""CLI --help works for every subcommand. Regression check for accidental
import errors / broken click definitions.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from experiments.cli.exp import cli

pytestmark = pytest.mark.build


@pytest.mark.parametrize(
    "cmd",
    [[], ["run"], ["list"], ["build-portal"], ["plan"], ["launch"], ["serve"]],
)
def test_help_does_not_explode(cmd: list[str]) -> None:
    result = CliRunner().invoke(cli, [*cmd, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_launch_help_lists_orchestration_flags() -> None:
    """`exp launch --help` must expose the new flags so the calc-emitted
    command + manual overrides both work without surprises."""
    result = CliRunner().invoke(cli, ["launch", "--help"])
    assert result.exit_code == 0, result.output
    for flag in ["--engine", "--workload", "--rate", "--serve", "--no-serve",
                 "--open", "--no-open", "--port", "--calc-bridge", "--portal-dir"]:
        assert flag in result.output, f"missing {flag} in launch --help"


def test_run_help_documents_auto_rate() -> None:
    """`exp run --help` must surface the auto-vs-explicit --rate contract,
    so a user reading help text understands the default behavior is
    engine-side calibration (not the prior 8.0 rps default)."""
    result = CliRunner().invoke(cli, ["run", "--help"])
    assert result.exit_code == 0, result.output
    assert "--rate" in result.output
    assert "auto" in result.output
    # The help must mention explicit-float behavior, not just "auto".
    assert "float" in result.output


def test_serve_help_lists_watch_flag() -> None:
    result = CliRunner().invoke(cli, ["serve", "--help"])
    assert result.exit_code == 0, result.output
    for flag in ["--watch", "--no-watch", "--port", "--calc-bridge", "--portal-dir"]:
        assert flag in result.output, f"missing {flag} in serve --help"
