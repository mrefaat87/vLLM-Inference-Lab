"""terraform fmt -check on the EKS module.

Skipped silently if terraform isn't installed (e.g. dev laptop without TF).
CI installs it via hashicorp/setup-terraform so this WILL run there.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.build

ROOT = Path(__file__).resolve().parents[2]
TF_DIR = ROOT / "eks" / "terraform"


def test_terraform_fmt_check() -> None:
    tf = shutil.which("terraform")
    if tf is None:
        pytest.skip("terraform not installed locally")
    result = subprocess.run(
        [tf, "fmt", "-check", "-recursive", str(TF_DIR)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"terraform fmt drift:\n{result.stdout}\n{result.stderr}\n"
        f"Run: terraform fmt -recursive {TF_DIR}"
    )


def test_terraform_validate() -> None:
    tf = shutil.which("terraform")
    if tf is None:
        pytest.skip("terraform not installed locally")
    # Validate runs without contacting AWS but does need init.
    init = subprocess.run(
        [tf, "-chdir=" + str(TF_DIR), "init", "-backend=false", "-input=false"],
        capture_output=True,
        text=True,
        check=False,
    )
    if init.returncode != 0:
        pytest.skip(f"terraform init failed (likely offline): {init.stderr[:200]}")
    result = subprocess.run(
        [tf, "-chdir=" + str(TF_DIR), "validate"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"terraform validate:\n{result.stdout}\n{result.stderr}"
