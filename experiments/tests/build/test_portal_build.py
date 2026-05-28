"""Build-time validation of the static portal.

These run in CI on every PR without GPUs. They prove the build script
produces well-formed HTML, valid JSON, and that internal links resolve.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from experiments.portal.build import BuildInputs, build
from experiments.runner.manifest_store import ManifestStore
from experiments.runner.schema import (
    Analysis,
    EngineConfig,
    HardwareSpec,
    ModelSpec,
    RequestRecord,
    RooflineLink,
    RunManifest,
    RunResult,
    RunStatus,
    ThroughputStats,
    WorkloadConfig,
)

pytestmark = pytest.mark.build


def _seed_results(results_dir: Path) -> None:
    runs = results_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    manifests_dir = results_dir / "manifests"
    store = ManifestStore(manifests_dir)
    now = datetime.now(timezone.utc)
    rid = "20260522T000000Z-mock-chatbot-aaaaaa"
    result = RunResult(
        run_id=rid,
        started_at=now,
        finished_at=now + timedelta(seconds=10),
        engine=EngineConfig(name="mock", image="n/a", model="mock-model"),
        model=ModelSpec(name="mock-model"),
        hardware=HardwareSpec(instance="local", gpu="cpu", n_gpu=1),
        workload=WorkloadConfig(name="chatbot", rate_rps=5.0, duration_s=10.0),
        analysis=Analysis(
            steady_state_requests=50,
            failed_requests=0,
            throughput=ThroughputStats(
                total_completion_tokens=500,
                total_prompt_tokens=200,
                tok_per_sec_avg=50.0,
                requests_per_sec_avg=5.0,
            ),
        ),
        roofline_link=RooflineLink(model_ref="mock", hw_ref="cpu"),
        raw_results=[
            RequestRecord(
                request_id=f"r{i}",
                label="chatbot.turn1",
                submit_offset_s=float(i) * 0.2,
                prompt_tokens=10,
                max_new_tokens=10,
                completion_tokens=10,
                ttft_s=0.05 + i * 0.001,
                end_to_end_s=0.1,
            )
            for i in range(20)
        ],
    )
    (runs / f"{rid}.json").write_text(result.model_dump_json(indent=2))
    store.write(
        RunManifest(
            run_id=rid,
            status=RunStatus.DONE,
            engine_name="mock",
            workload_name="chatbot",
            started_at=now,
            finished_at=now + timedelta(seconds=10),
            result_path=str(runs / f"{rid}.json"),
        )
    )


def test_build_produces_html_and_json(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    out_dir = tmp_path / "_site"
    _seed_results(results_dir)

    build(BuildInputs(results_dir=results_dir, out_dir=out_dir))

    cc = out_dir / "command_center.html"
    re_ = out_dir / "results_explorer.html"
    runs = out_dir / "assets" / "runs.json"
    css = out_dir / "assets" / "style.css"
    assert cc.exists() and re_.exists() and runs.exists() and css.exists()

    # HTML parses.
    soup_cc = BeautifulSoup(cc.read_text(), "html.parser")
    soup_re = BeautifulSoup(re_.read_text(), "html.parser")
    assert soup_cc.find("title")
    assert soup_re.find("title")

    # Internal nav targets must point into a known top-level under the
    # repo root. The portal now emits absolute paths (`/calculators/…`,
    # `/experiments/_site/…`, `/index.html`) because `exp serve` is rooted
    # at the repo, not the portal subtree. We don't have the repo on disk
    # at test time, so we just sanity-check the prefix shape.
    ALLOWED_ABS_PREFIXES = (
        "/calculators/",
        "/experiments/_site/",
        "/index.html",
    )
    for href in {a["href"] for a in soup_cc.find_all("a") if a.get("href")}:
        if href.startswith(("http", "#")):
            continue
        if href.startswith("/"):
            assert href.startswith(ALLOWED_ABS_PREFIXES), (
                f"unexpected absolute link in command_center.html: {href}"
            )
            continue
        # Bare relative paths (no leading "/") still resolve under out_dir
        # — e.g. JS-built `runs/<id>.json` fetches.
        path = (out_dir / href.split("#", 1)[0]).resolve()
        assert path.exists(), f"broken link in command_center.html: {href}"

    # Runs index has our seeded run.
    blob = json.loads(runs.read_text())
    assert len(blob["runs"]) == 1
    r = blob["runs"][0]
    assert r["status"] == "done"
    assert r["result_href"]
    # Result blob was copied to output.
    assert (out_dir / r["result_href"]).exists()


def test_build_handles_empty_results(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    out_dir = tmp_path / "_site"
    build(BuildInputs(results_dir=results_dir, out_dir=out_dir))
    runs = json.loads((out_dir / "assets" / "runs.json").read_text())
    assert runs == {"runs": []}
    assert (out_dir / "command_center.html").exists()
    assert (out_dir / "results_explorer.html").exists()
