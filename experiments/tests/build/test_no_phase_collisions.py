"""Guard test: nothing under experiments/ may reference sibling phase* stacks.

This catches accidental copy-paste from phase4.5 manifests, which would
otherwise quietly point the inference-lab cluster at the wrong IAM role,
VPC tag, or NodePool.

The allowlist below covers human prose (README, CONTRIBUTING, docs) that
intentionally mentions the sibling stacks for context.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.build

ROOT = Path(__file__).resolve().parents[2]  # experiments/

# File suffixes that we scan. We do NOT scan Markdown — prose can mention
# the sibling stacks freely. We do NOT scan tests — they may compare strings.
SCAN_SUFFIXES = {".tf", ".tfvars", ".yaml", ".yml", ".sh", ".py", ".json"}

# Files / paths we deliberately allow (relative to ROOT).
ALLOWLIST = {
    Path("tests/build/test_no_phase_collisions.py"),  # this file
    # Defensive name guard — explicitly refuses cluster names containing
    # "inference-phase" to keep us from clobbering sibling stacks. The
    # string is the thing we're protecting against, not a leak of it.
    Path("eks/terraform/variables.tf"),
}

PATTERN = re.compile(r"inference-phase")


def test_no_phase_string_in_artifacts() -> None:
    offenders: list[tuple[Path, int, str]] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in SCAN_SUFFIXES:
            continue
        # Skip vendored dirs.
        rel = path.relative_to(ROOT)
        if any(part in {".venv", "node_modules", "_site", ".terraform", "build", "dist"} for part in rel.parts):
            continue
        if rel in ALLOWLIST:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if PATTERN.search(line):
                offenders.append((rel, i, line.strip()))
    assert not offenders, (
        "Found references to sibling 'inference-phase*' stacks. Either rename "
        "or add the file to ALLOWLIST with justification.\n"
        + "\n".join(f"  {p}:{ln}  {txt}" for p, ln, txt in offenders)
    )
