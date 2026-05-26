"""Filesystem-backed run manifest store.

Single source of truth that the static portal's command-center page reads.
The CLI / sweep runner writes a manifest at every state transition
(planned → running → done | failed) so a refresh of the HTML page shows
current state. No daemon, no DB.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from experiments.runner.schema import RunManifest, RunStatus


class ManifestStore:
    def __init__(self, base_dir: Path | str) -> None:
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def path(self, run_id: str) -> Path:
        return self._dir / f"{run_id}.json"

    def write(self, manifest: RunManifest) -> Path:
        """Atomically write a manifest. tmp+rename so a torn write never appears."""
        target = self.path(manifest.run_id)
        blob = manifest.model_dump_json(indent=2)
        # NamedTemporaryFile in the same dir → rename is atomic on POSIX.
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{manifest.run_id}.", suffix=".json.tmp", dir=str(self._dir)
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(blob)
            os.replace(tmp_path, target)
        except Exception:
            with _suppress(FileNotFoundError):
                os.unlink(tmp_path)
            raise
        return target

    def read(self, run_id: str) -> RunManifest:
        return RunManifest.model_validate_json(self.path(run_id).read_text())

    def list_all(self) -> list[RunManifest]:
        out: list[RunManifest] = []
        for p in sorted(self._dir.glob("*.json")):
            try:
                out.append(RunManifest.model_validate_json(p.read_text()))
            except Exception:  # noqa: BLE001 — skip corrupt manifests, don't kill the listing
                continue
        return out

    # ----- convenience state-transition helpers -----
    def mark_running(self, run_id: str, log_path: str | None = None) -> RunManifest:
        m = self.read(run_id)
        updated = m.model_copy(
            update={
                "status": RunStatus.RUNNING,
                "started_at": datetime.now(timezone.utc),
                "log_path": log_path,
            }
        )
        self.write(updated)
        return updated

    def mark_done(self, run_id: str, result_path: str) -> RunManifest:
        m = self.read(run_id)
        updated = m.model_copy(
            update={
                "status": RunStatus.DONE,
                "finished_at": datetime.now(timezone.utc),
                "result_path": result_path,
            }
        )
        self.write(updated)
        return updated

    def mark_failed(self, run_id: str, error: str) -> RunManifest:
        m = self.read(run_id)
        # If never marked running, set started_at = finished_at = now.
        now = datetime.now(timezone.utc)
        updated = m.model_copy(
            update={
                "status": RunStatus.FAILED,
                "started_at": m.started_at or now,
                "finished_at": now,
                "error": error,
            }
        )
        self.write(updated)
        return updated


class _suppress:
    def __init__(self, *excs: type[BaseException]) -> None:
        self._excs = excs

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: type[BaseException] | None, *_: object) -> bool:
        return exc_type is not None and issubclass(exc_type, self._excs)
