"""C1: every default --model-ref / --hw-ref in the CLI maps to a real
`key` in the calc's data files. Catches accidental drift when either
side renames a model or hardware row.

Skipped if the calc tree isn't present (lab-only clone).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from experiments.cli.exp import cli

pytestmark = pytest.mark.build


def _calc_data_root() -> Path | None:
    root = Path(__file__).resolve().parents[3] / "calculators" / "sizing_calc" / "src" / "data"
    if not (root / "models.json").exists():
        return None
    return root


def test_cli_defaults_resolve_to_calc_keys() -> None:
    data = _calc_data_root()
    if data is None:
        pytest.skip("sizing_calc data files not in this clone")
    models = {m["key"] for m in json.loads((data / "models.json").read_text())["models"]}
    hw = {h["key"] for h in json.loads((data / "hardware.json").read_text())["hardware"]}

    out = CliRunner().invoke(cli, ["run", "--help"])
    assert out.exit_code == 0
    # Pull defaults out of --help output. Click prints "default: <value>" for
    # options. Match the two we care about.
    text = out.output
    # Extract default --model-ref (parsing via regex would be fragile; instead
    # rely on the well-known defaults defined in cli/exp.py).
    from experiments.cli.exp import run_cmd  # noqa: WPS433
    defaults: dict[str, str] = {}
    for param in run_cmd.params:
        if param.name in {"model_ref", "hw_ref"}:
            defaults[param.name] = param.default
    assert defaults["model_ref"] in models, (
        f"--model-ref default '{defaults['model_ref']}' not a calc model key. "
        f"Available: {sorted(models)}"
    )
    assert defaults["hw_ref"] in hw, (
        f"--hw-ref default '{defaults['hw_ref']}' not a calc hardware key. "
        f"Available: {sorted(hw)}"
    )
    _ = text  # quieted to avoid unused-var linter
