"""Unit tests for ManifestStore."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from experiments.runner.manifest_store import ManifestStore
from experiments.runner.schema import RunManifest, RunStatus

pytestmark = pytest.mark.unit


def _planned(run_id: str = "r1") -> RunManifest:
    return RunManifest(
        run_id=run_id, status=RunStatus.PLANNED, engine_name="vllm", workload_name="chatbot"
    )


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)
    m = _planned()
    p = store.write(m)
    assert p.exists()
    assert store.read("r1") == m


def test_list_all_sorted_and_skips_corrupt(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)
    store.write(_planned("a"))
    store.write(_planned("b"))
    (tmp_path / "garbage.json").write_text("not json at all")
    all_ = store.list_all()
    assert [m.run_id for m in all_] == ["a", "b"]


def test_state_transitions(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)
    store.write(_planned("r1"))

    store.mark_running("r1", log_path="/tmp/log")
    m = store.read("r1")
    assert m.status == RunStatus.RUNNING
    assert m.started_at is not None
    assert m.log_path == "/tmp/log"

    store.mark_done("r1", result_path="/tmp/out.json")
    m = store.read("r1")
    assert m.status == RunStatus.DONE
    assert m.result_path == "/tmp/out.json"
    assert m.finished_at is not None


def test_mark_failed_without_started(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)
    store.write(_planned("r1"))
    store.mark_failed("r1", error="kaboom")
    m = store.read("r1")
    assert m.status == RunStatus.FAILED
    assert m.started_at is not None
    assert m.finished_at is not None
    assert m.error == "kaboom"


def test_atomic_write_no_partial_file(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)
    store.write(_planned("r1"))
    # No .tmp files should remain.
    assert list(tmp_path.glob("*.tmp")) == []


def test_running_requires_started_already_at_schema_layer() -> None:
    # Sanity check that schema-layer validation rejects bad transitions
    # even if a caller bypasses ManifestStore.
    with pytest.raises(Exception):  # noqa: B017, PT011 — pydantic.ValidationError exact type doesn't matter
        RunManifest(
            run_id="r1",
            status=RunStatus.RUNNING,
            engine_name="vllm",
            workload_name="chatbot",
        )


def test_started_before_finished_at_done(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)
    now = datetime.now(timezone.utc)
    store.write(_planned("r1"))
    store.mark_running("r1")
    store.mark_done("r1", result_path=str(tmp_path / "x.json"))
    m = store.read("r1")
    assert m.started_at is not None and m.finished_at is not None
    assert m.started_at <= m.finished_at
    assert m.started_at >= now - (now - now)  # silence unused; just keeps the import meaningful
