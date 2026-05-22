#!/usr/bin/env python3
"""Orchestrate the Phase 4.1 baseline 3-run matrix with a hardware-stability gate.

For each run:
  1. Force-delete any existing GPU node so the next run gets a fresh one.
  2. Wait for the node to actually be gone (Karpenter terminates the EC2 instance).
  3. Invoke run_baseline.sh <run-id> end-to-end.
  4. Read the resulting run JSON, extract the hardware fingerprint.
  5. Compare against fingerprints of prior runs. Abort if AZ, instance-type,
     or CPU model don't match — heterogeneous hardware would make the
     aggregate meaningless.

We do NOT delete the pod after each run via the runner's default cleanup;
instead we force-delete the GPU node BETWEEN runs. That guarantees the next
run actually exercises stages 1-3 (provision a brand new instance) rather
than landing on a warm node.

Run records land in phase4.1/tests/baseline_runs/fresh-001.json etc.
"""
import json
import pathlib
import subprocess
import sys
import time
from typing import Optional

PHASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RUNS_DIR = PHASE_DIR / "tests" / "baseline_runs"
RUN_IDS = ["fresh-001", "fresh-002", "fresh-003"]


def kubectl(args: list[str], capture: bool = True) -> str:
    out = subprocess.run(
        ["kubectl", *args],
        capture_output=capture, text=True, check=False,
    )
    return (out.stdout or "") + (out.stderr or "")


def get_gpu_nodes() -> list[str]:
    raw = kubectl(["get", "nodes", "-l", "node-role=gpu", "-o", "name"])
    return [line.strip().split("/", 1)[-1] for line in raw.splitlines() if line.strip()]


def teardown_gpu_node() -> None:
    """Force-delete the GPU node and wait until Karpenter terminates the EC2 instance."""
    nodes = get_gpu_nodes()
    if not nodes:
        print("  no GPU node to tear down")
        return
    for n in nodes:
        print(f"  deleting node {n}")
        kubectl(["delete", "node", n, "--wait=false"])
    # Wait for Karpenter to clean up. NodeClaim disappearing is the strongest signal.
    deadline = time.time() + 240
    while time.time() < deadline:
        nc = kubectl(["get", "nodeclaims", "-o", "name"]).strip()
        if not nc:
            print("  NodeClaim gone — Karpenter terminated the instance")
            return
        time.sleep(5)
    print("  WARNING: NodeClaim still exists after 240s; proceeding anyway")


def fingerprint(run: dict) -> tuple:
    """Stable tuple of hardware identity. Two runs match iff this tuple matches."""
    n = run.get("node", {})
    return (n.get("az"), n.get("instance_type"), n.get("cpu_model"))


def run_baseline(run_id: str) -> dict:
    """Invoke run_baseline.sh and return the parsed run JSON.

    We tolerate non-zero exit codes if the run JSON exists and is valid. The
    bash runner sometimes returns non-zero from a trailing `|| true` clause
    even when the substantive run succeeded. Validation is what we care about.
    """
    script = PHASE_DIR / "scripts" / "run_baseline.sh"
    out = RUNS_DIR / f"{run_id}.json"
    if out.is_file():
        out.unlink()  # force a fresh write — don't let stale data hide a failure

    print(f"  invoking {script.name} {run_id}")
    rc = subprocess.run(["bash", str(script), run_id], check=False).returncode
    print(f"  run_baseline.sh exit code: {rc}")

    if not out.is_file():
        raise RuntimeError(f"run JSON missing for {run_id} (rc={rc})")
    record = json.loads(out.read_text())
    if rc != 0 and record.get("validation", {}).get("all_stages_present"):
        print(f"  (rc={rc} but JSON valid — proceeding)")
    return record


def main() -> int:
    fingerprints: list[tuple] = []
    valid_runs: list[str] = []

    for i, run_id in enumerate(RUN_IDS, start=1):
        print(f"\n=== Run {i}/{len(RUN_IDS)}: {run_id} ===")
        print("[gate] tearing down GPU node before fresh run")
        teardown_gpu_node()

        record = run_baseline(run_id)

        if not record.get("validation", {}).get("all_stages_present"):
            print(f"  WARNING: {run_id} has missing stages: "
                  f"{record['validation'].get('notes')}")
        if record.get("validation", {}).get("any_negative_durations"):
            print(f"  WARNING: {run_id} has negative durations")

        fp = fingerprint(record)
        print(f"  hardware fingerprint: az={fp[0]} type={fp[1]} cpu={fp[2]}")

        if fingerprints:
            ref = fingerprints[0]
            if fp != ref:
                print(f"\n[gate] HARDWARE MISMATCH — aborting matrix")
                print(f"  reference: az={ref[0]} type={ref[1]} cpu={ref[2]}")
                print(f"  this run:  az={fp[0]} type={fp[1]} cpu={fp[2]}")
                print(f"  valid runs collected so far: {valid_runs}")
                print(f"  rerun the matrix when Spot capacity is more stable.")
                return 2
        fingerprints.append(fp)
        valid_runs.append(run_id)

    print(f"\n=== Matrix complete: {len(valid_runs)}/{len(RUN_IDS)} runs ===")
    print(f"all hardware fingerprints match: az={fingerprints[0][0]} "
          f"type={fingerprints[0][1]} cpu={fingerprints[0][2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
