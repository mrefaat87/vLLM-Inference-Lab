"""Build the static portal pages.

Usage:
    python -m experiments.portal.build --out _site
    python -m experiments.portal.build --out _site --results-dir results

The output directory contains:
    command_center.html       — manifest list with status badges
    results_explorer.html     — engine/workload picker, charts
    assets/style.css          — shared styling
    assets/runs.json          — bundled manifest + result index for the JS
    runs/*.json               — per-run result blobs (copied from results_dir)

The page is fully static: it loads ``assets/runs.json`` and the per-run
blobs via ``fetch``. No backend required.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.runner.manifest_store import ManifestStore
from experiments.runner.schema import RunManifest, RunResult

HERE = Path(__file__).parent
TEMPLATES = HERE / "templates"
ASSETS = HERE / "assets"


@dataclass
class BuildInputs:
    results_dir: Path
    out_dir: Path
    # Optional bridge dir for the sizing calculator's Validation tab.
    # Writes a sibling index.json + per-run blobs (byte-identical copies
    # of the source result JSONs) so the calc can fetch them via a
    # relative URL without reaching into experiments/_site.
    calc_bridge: Path | None = None


def build(inputs: BuildInputs) -> None:
    inputs.out_dir.mkdir(parents=True, exist_ok=True)
    (inputs.out_dir / "assets").mkdir(parents=True, exist_ok=True)
    (inputs.out_dir / "runs").mkdir(parents=True, exist_ok=True)

    # 1. Gather manifests + result summaries.
    manifests_dir = inputs.results_dir / "manifests"
    runs_dir = inputs.results_dir / "runs"
    manifests = (
        ManifestStore(manifests_dir).list_all() if manifests_dir.exists() else []
    )
    summaries: list[dict[str, Any]] = []
    for m in manifests:
        summary: dict[str, Any] = {
            "run_id": m.run_id,
            "status": m.status.value,
            "engine": m.engine_name,
            "workload": m.workload_name,
            "created_at": m.created_at.isoformat(),
            "started_at": m.started_at.isoformat() if m.started_at else None,
            "finished_at": m.finished_at.isoformat() if m.finished_at else None,
            "result_href": None,
            "error": m.error,
        }
        if m.result_path:
            # Copy the result JSON into the build output so the portal can be
            # served from any static host without exposing the source tree.
            # `result_path` may be:
            #   - absolute (test fixtures use tmp_path),
            #   - CWD-relative (real CLI invocations),
            #   - results_dir-relative (legacy callers).
            # Try each in order; the convention isn't strict enough to assume.
            candidates = [Path(m.result_path)]
            if not candidates[0].is_absolute():
                candidates.append(inputs.results_dir / m.result_path)
            candidates.append(inputs.results_dir / "runs" / f"{m.run_id}.json")
            src = next((c for c in candidates if c.exists()), candidates[0])
            if src.exists():
                dst = inputs.out_dir / "runs" / f"{m.run_id}.json"
                shutil.copyfile(src, dst)
                summary["result_href"] = f"runs/{m.run_id}.json"
                # Add lightweight summary fields for the list view.
                try:
                    result = RunResult.model_validate_json(src.read_text())
                    summary.update(
                        {
                            "model": result.model.name,
                            "hardware": result.hardware.instance,
                            "rate_rps": result.workload.rate_rps,
                            "ttft_p50_s": _opt_pct(result, "ttft_s", "p50"),
                            "ttft_p95_s": _opt_pct(result, "ttft_s", "p95"),
                            "throughput_tok_per_s": result.analysis.throughput.tok_per_sec_avg,
                            "roofline_link": {
                                "model_ref": result.roofline_link.model_ref,
                                "hw_ref": result.roofline_link.hw_ref,
                            },
                        }
                    )
                except Exception as exc:  # noqa: BLE001 — keep going past one bad file
                    summary["parse_error"] = str(exc)
        summaries.append(summary)

    # Also include any orphan result JSONs that lack a manifest (defensive).
    if runs_dir.exists():
        seen = {s["run_id"] for s in summaries}
        for p in sorted(runs_dir.glob("*.json")):
            rid = p.stem
            if rid in seen:
                continue
            try:
                result = RunResult.model_validate_json(p.read_text())
            except Exception:
                continue
            dst = inputs.out_dir / "runs" / p.name
            shutil.copyfile(p, dst)
            summaries.append(
                {
                    "run_id": result.run_id,
                    "status": "done",
                    "engine": result.engine.name,
                    "workload": result.workload.name,
                    "model": result.model.name,
                    "hardware": result.hardware.instance,
                    "result_href": f"runs/{p.name}",
                    "throughput_tok_per_s": result.analysis.throughput.tok_per_sec_avg,
                    "roofline_link": {
                        "model_ref": result.roofline_link.model_ref,
                        "hw_ref": result.roofline_link.hw_ref,
                    },
                }
            )

    (inputs.out_dir / "assets" / "runs.json").write_text(
        json.dumps({"runs": summaries}, indent=2)
    )

    # 2. Copy static assets.
    if ASSETS.exists():
        for asset in ASSETS.iterdir():
            if asset.is_file():
                shutil.copyfile(asset, inputs.out_dir / "assets" / asset.name)

    # 3. Render pages from templates (simple {{var}} substitution; no Jinja
    # to keep the build dep-free).
    _render(TEMPLATES / "command_center.html.tpl", inputs.out_dir / "command_center.html", {})
    _render(TEMPLATES / "results_explorer.html.tpl", inputs.out_dir / "results_explorer.html", {})

    # 4. Optionally emit the calc-side bridge dir. Byte-identical copies
    # of the source result JSONs go under `bridge/runs/`; a parallel
    # `bridge/index.json` mirrors the in-portal summary. This satisfies
    # C2 (bridge byte fidelity) by using shutil.copyfile rather than
    # re-serializing through pydantic.
    if inputs.calc_bridge is not None:
        _emit_bridge(inputs, summaries)


def _emit_bridge(inputs: BuildInputs, summaries: list[dict[str, Any]]) -> None:
    bridge = inputs.calc_bridge
    assert bridge is not None
    bridge_runs = bridge / "runs"
    bridge.mkdir(parents=True, exist_ok=True)
    bridge_runs.mkdir(parents=True, exist_ok=True)
    bridge_summaries: list[dict[str, Any]] = []
    runs_root = inputs.results_dir
    for summary in summaries:
        if summary.get("status") != "done":
            continue
        href = summary.get("result_href")
        if not href:
            continue
        # Build the source path from the _site copy we already wrote — that's
        # known to be a byte-identical copy of the original. Falling back to
        # the original on results_dir keeps tests that bypass the _site copy
        # working too.
        src_site = inputs.out_dir / href
        src_orig = runs_root / "runs" / f"{summary['run_id']}.json"
        src = src_site if src_site.exists() else src_orig
        if not src.exists():
            continue
        dst = bridge_runs / f"{summary['run_id']}.json"
        shutil.copyfile(src, dst)
        bridge_summaries.append(
            {
                "run_id": summary["run_id"],
                "engine": summary.get("engine"),
                "workload": summary.get("workload"),
                "model": summary.get("model"),
                "hardware": summary.get("hardware"),
                "model_ref": (summary.get("roofline_link") or {}).get("model_ref"),
                "hw_ref": (summary.get("roofline_link") or {}).get("hw_ref"),
                "throughput_tok_per_s": summary.get("throughput_tok_per_s"),
                "ttft_p50_s": summary.get("ttft_p50_s"),
                "ttft_p95_s": summary.get("ttft_p95_s"),
                "result_href": f"runs/{summary['run_id']}.json",
            }
        )
    (bridge / "index.json").write_text(
        json.dumps({"runs": bridge_summaries}, indent=2)
    )


def _opt_pct(result: RunResult, field: str, pct: str) -> float | None:
    pcts = getattr(result.analysis, field)
    if pcts is None:
        return None
    return float(getattr(pcts, pct))


def _render(template_path: Path, out_path: Path, vars_: dict[str, str]) -> None:
    text = template_path.read_text()
    for k, v in vars_.items():
        text = text.replace("{{" + k + "}}", v)
    out_path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.portal.build")
    parser.add_argument("--out", default="_site", type=Path)
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    parser.add_argument("--calc-bridge", default=None, type=Path)
    args = parser.parse_args(argv)
    build(
        BuildInputs(
            results_dir=args.results_dir,
            out_dir=args.out,
            calc_bridge=args.calc_bridge,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
