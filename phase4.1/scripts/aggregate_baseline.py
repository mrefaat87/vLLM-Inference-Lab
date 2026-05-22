#!/usr/bin/env python3
"""Aggregate fresh-*.json runs into baseline_summary.json.

For each stage: median, min, max, stddev across the matrix runs. Also include
the per-run dot values so the HTML can show variance directly.

Refuses to merge runs whose probe_spec_hash differs, hardware fingerprint
differs, or that have validation.all_stages_present=false. Better to surface
the problem than to silently produce an apples-to-oranges aggregate.
"""
import json
import pathlib
import statistics
import sys

PHASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RUNS_DIR = PHASE_DIR / "tests" / "baseline_runs"
SUMMARY_PATH = RUNS_DIR.parent / "baseline_summary.json"

STAGE_ORDER = [
    "karpenter_scheduling",
    "ec2_spot_fulfillment",
    "node_bootstrap",
    "image_download",
    "image_unpack",
    "model_download_s3",
    "vllm_init_cuda_ctx",
    "weight_load_gpu_mem",
    "cuda_graph_warmup",
    "readiness_probe_pass",
    "first_token_served",
]


def main() -> int:
    runs = sorted(RUNS_DIR.glob("fresh-*.json"))
    if not runs:
        print("no fresh-*.json runs found", file=sys.stderr)
        return 1

    records = [json.loads(p.read_text()) for p in runs]

    # Validation gates.
    bad = [r["run_id"] for r in records
           if not r.get("validation", {}).get("all_stages_present")]
    if bad:
        print(f"refusing to aggregate — invalid runs: {bad}", file=sys.stderr)
        return 2

    hashes = {r["validation"]["probe_spec_hash"] for r in records}
    if len(hashes) > 1:
        print(f"refusing to aggregate — probe_spec_hash differs: {hashes}",
              file=sys.stderr)
        return 3

    fps = {(r["node"]["az"], r["node"]["instance_type"], r["node"]["cpu_model"])
           for r in records}
    if len(fps) > 1:
        print(f"refusing to aggregate — hardware fingerprints differ: {fps}",
              file=sys.stderr)
        return 4

    # Per-stage stats.
    summary_stages: dict[str, dict] = {}
    for stage in STAGE_ORDER:
        durations = [r["stages"][stage]["duration_ms"] for r in records]
        summary_stages[stage] = {
            "n": len(durations),
            "median_ms": statistics.median(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "stddev_ms": (statistics.pstdev(durations) if len(durations) > 1 else 0),
            "per_run": durations,
        }

    totals_ready = [r["totals"]["scale_up_to_ready_ms"] for r in records]
    totals_first = [r["totals"]["scale_up_to_first_token_ms"] for r in records]

    summary = {
        "fresh_node": {
            "n": len(records),
            "run_ids": [r["run_id"] for r in records],
            "hardware": {
                "az": records[0]["node"]["az"],
                "instance_type": records[0]["node"]["instance_type"],
                "cpu_model": records[0]["node"]["cpu_model"],
                "kernel": records[0]["node"]["kernel"],
                "ami_id": records[0]["node"]["ami_id"],
            },
            "probe_spec_hash": records[0]["validation"]["probe_spec_hash"],
            "stages": summary_stages,
            "totals": {
                "scale_up_to_ready_ms": {
                    "median": statistics.median(totals_ready),
                    "min": min(totals_ready),
                    "max": max(totals_ready),
                    "per_run": totals_ready,
                },
                "scale_up_to_first_token_ms": {
                    "median": statistics.median(totals_first),
                    "min": min(totals_first),
                    "max": max(totals_first),
                    "per_run": totals_first,
                },
            },
        }
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
