#!/usr/bin/env python3
"""Post-hoc fix for variant-E runs whose tracer didn't emit weight_load_done.

Background:
  RunAI Model Streamer doesn't log vLLM's stock "Loading weights took" message,
  so cold_start_tracer.py never matched the end-of-weight-load pattern. The
  affected run JSONs have:
    stages.weight_load_gpu_mem.start_ms  = <real timestamp>
    stages.weight_load_gpu_mem.end_ms    = None
    stages.weight_load_gpu_mem.duration_ms = None

What this does:
  Set end_ms = stages.cuda_graph_warmup.start_ms (first CUDA graph capture
  message timestamp). Graph capture cannot start before weights finish
  loading, so this is a tight upper bound — never an under-estimate.
  Mark source="tracer_jsonl+graph_capture_inferred" so the audit trail is
  preserved.

Updated tracer (cold_start_tracer.py) does this implicitly for new runs;
this script handles already-collected runs.

Usage:
  python3 phase4.1/scripts/patch_streamer_weight_load.py phase4.1/tests/baseline_runs/streamer-*.json
"""
from __future__ import annotations

import json
import pathlib
import sys


def patch_one(path: pathlib.Path) -> bool:
    data = json.loads(path.read_text())
    stages = data.get("stages", {})
    wlg = stages.get("weight_load_gpu_mem", {})
    cgw = stages.get("cuda_graph_warmup", {})
    if wlg.get("end_ms") is not None:
        return False
    if wlg.get("start_ms") is None or cgw.get("start_ms") is None:
        return False
    new_end = cgw["start_ms"]
    wlg["end_ms"] = new_end
    wlg["duration_ms"] = new_end - wlg["start_ms"]
    wlg["source"] = "tracer_jsonl+graph_capture_inferred"

    # Refresh validation notes.
    val = data.setdefault("validation", {})
    notes = [n for n in val.get("notes", [])
             if "weight_load_gpu_mem" not in n]
    val["notes"] = notes
    # Recompute all_stages_present.
    missing = [k for k, v in stages.items() if v.get("duration_ms") is None]
    val["all_stages_present"] = not missing
    if missing:
        val["notes"].append(f"missing_stages: {missing}")

    path.write_text(json.dumps(data, indent=2))
    return True


def main() -> int:
    paths = [pathlib.Path(a) for a in sys.argv[1:]]
    if not paths:
        print("usage: patch_streamer_weight_load.py <run.json> ...", file=sys.stderr)
        return 2
    n = 0
    for p in paths:
        if not p.is_file():
            print(f"  [skip] missing: {p}")
            continue
        if patch_one(p):
            print(f"  [patched] {p}")
            n += 1
        else:
            print(f"  [no-op]  {p} (already has weight_load end_ms or missing inputs)")
    print(f"Patched {n} run(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
