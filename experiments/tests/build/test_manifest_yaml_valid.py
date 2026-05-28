"""Every YAML manifest under eks/manifests/ must parse cleanly and target
the inference-lab cluster, not anything else."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.build

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "eks" / "manifests"


def _docs() -> list[tuple[Path, dict]]:
    out: list[tuple[Path, dict]] = []
    if not MANIFESTS.exists():
        return out
    for p in MANIFESTS.rglob("*.y*ml"):
        for doc in yaml.safe_load_all(p.read_text()):
            if doc is None:
                continue
            out.append((p, doc))
    return out


def test_yaml_documents_parse() -> None:
    docs = _docs()
    assert docs, "no manifests found — eks/manifests/ should not be empty"


def test_names_use_inference_lab_prefix() -> None:
    for p, doc in _docs():
        name = (doc.get("metadata") or {}).get("name", "")
        if not name:
            continue
        # Allow upstream things like 'nvidia-device-plugin', cert-manager, etc.
        # We're only enforcing that our OWN named resources use the project prefix.
        if doc.get("apiVersion", "").startswith("karpenter.") and not name.startswith(
            "inference-lab"
        ):
            pytest.fail(f"{p}: karpenter resource '{name}' must use inference-lab prefix")
