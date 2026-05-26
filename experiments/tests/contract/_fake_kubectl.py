"""A fake kubectl shim used by driver contract tests.

The K8s drivers shell out to ``kubectl``. We don't want CI to need a
real cluster, so we ship a tiny script that pretends to be kubectl: it
exits 0 on ``apply`` / ``delete`` / ``wait`` calls. Behind the scenes a
StubServer is what actually serves the OpenAI-compatible health/inference
endpoints; the driver thinks it's talking to a cluster Service via port
forwarding, but the URL really points at the stub.
"""

from __future__ import annotations

import os
import stat
import tempfile
import textwrap
from pathlib import Path


def install_fake_kubectl(tmpdir: Path) -> Path:
    """Write a fake kubectl script into tmpdir and return its path."""
    script = tmpdir / "kubectl"
    script.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            # Fake kubectl for contract tests.
            # Read manifest from stdin on apply, ignore it. Exit 0 on the verbs
            # the driver uses.
            for arg in "$@"; do
              case "$arg" in
                apply|delete|wait)
                  # Drain stdin so the caller's `kubectl apply -f -` doesn't SIGPIPE.
                  cat >/dev/null 2>&1 || true
                  exit 0
                  ;;
              esac
            done
            exit 0
            """
        )
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script
