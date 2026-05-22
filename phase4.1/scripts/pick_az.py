#!/usr/bin/env python3
"""Pick the AZ with the highest Spot capacity for g4dn.xlarge in us-east-1.

Why: Phase 4.1 baseline runs need stable hardware across the 3-run matrix.
EC2 Spot capacity varies per AZ; if Karpenter shifts AZ between runs, we'd
mix instance generations and contaminate stages 2-8 of the waterfall.
This script asks AWS for current Spot placement scores, picks the top AZ,
and writes the ranking + winner to phase4.1/tests/baseline_runs/_az_selection.json.

It also patches the NodePool's topology.kubernetes.io/zone constraint to the
winning AZ so Karpenter only considers that AZ for the next 3 runs.

Spot Placement Scores API:
  - Returns a 1-10 score per AZ for a hypothetical capacity request.
  - Higher = lower likelihood of capacity issues / interruptions.
  - Refreshed continuously by AWS based on current pool depth.
"""
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

REGION = "us-east-1"
INSTANCE_TYPE = "g4dn.xlarge"
TARGET_CAPACITY = 1
PHASE_DIR = pathlib.Path(__file__).resolve().parent.parent
OUT_PATH = PHASE_DIR / "tests" / "baseline_runs" / "_az_selection.json"
NODEPOOL_PATH = PHASE_DIR / "k8s" / "karpenter" / "gpu-nodepool.yaml"


def get_az_id_to_name() -> dict:
    """AZ ids (use1-az1 etc.) are stable across accounts; AZ names (us-east-1a)
    are account-specific. We need the mapping to filter and report cleanly."""
    out = subprocess.run(
        ["aws", "ec2", "describe-availability-zones",
         "--region", REGION, "--output", "json"],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    return {z["ZoneId"]: z["ZoneName"] for z in data["AvailabilityZones"]}


def get_cluster_az_names() -> set:
    """Read the 4 AZ names the cluster's NodePool currently allows.
    Avoids picking an AZ Karpenter has no subnet for."""
    out = subprocess.run(
        ["kubectl", "get", "nodepool", "gpu-pool", "-o", "json"],
        capture_output=True, text=True, check=True,
    )
    np = json.loads(out.stdout)
    for req in np["spec"]["template"]["spec"]["requirements"]:
        if req["key"] == "topology.kubernetes.io/zone":
            return set(req["values"])
    raise RuntimeError("NodePool has no zone requirement")


def get_spot_placement_scores(az_id_to_name: dict, allowed_az_names: set) -> list[dict]:
    """Query Spot scores, attach AZ names, filter to cluster's AZs, sort.
    --single-availability-zone forces per-AZ breakdown."""
    out = subprocess.run(
        [
            "aws", "ec2", "get-spot-placement-scores",
            "--instance-types", INSTANCE_TYPE,
            "--target-capacity", str(TARGET_CAPACITY),
            "--single-availability-zone",
            "--region-names", REGION,
            "--region", REGION,
            "--output", "json",
        ],
        capture_output=True, text=True, check=True,
    )
    raw = json.loads(out.stdout).get("SpotPlacementScores", [])
    scores = []
    for s in raw:
        az_name = az_id_to_name.get(s["AvailabilityZoneId"])
        if az_name is None or az_name not in allowed_az_names:
            continue
        scores.append({
            "AvailabilityZone": az_name,
            "AvailabilityZoneId": s["AvailabilityZoneId"],
            "Score": s["Score"],
        })
    # Highest score wins. Tie-break alphabetically by AZ name.
    scores.sort(key=lambda s: (-s["Score"], s["AvailabilityZone"]))
    return scores


def patch_nodepool(winning_az: str):
    """Rewrite the NodePool zone constraint to the single winning AZ.

    Uses kubectl patch (server-side) instead of editing the yaml file in place,
    so the manifest stays as the canonical 4-AZ template and the patch is
    visible in cluster state for audit.
    """
    patch = json.dumps({
        "spec": {
            "template": {
                "spec": {
                    "requirements": [
                        # Note: this REPLACES the full requirements list, so we
                        # have to repeat the other constraints. JSON merge patch
                        # at the array level isn't granular enough.
                        {
                            "key": "node.kubernetes.io/instance-type",
                            "operator": "In",
                            "values": [INSTANCE_TYPE],
                        },
                        {
                            "key": "karpenter.sh/capacity-type",
                            "operator": "In",
                            "values": ["spot"],
                        },
                        {
                            "key": "topology.kubernetes.io/zone",
                            "operator": "In",
                            "values": [winning_az],
                        },
                        {
                            "key": "kubernetes.io/arch",
                            "operator": "In",
                            "values": ["amd64"],
                        },
                    ]
                }
            }
        }
    })
    subprocess.run(
        [
            "kubectl",
            "patch",
            "nodepool",
            "gpu-pool",
            "--type=merge",
            "-p",
            patch,
        ],
        check=True,
    )


def main():
    az_map = get_az_id_to_name()
    cluster_azs = get_cluster_az_names()
    scores = get_spot_placement_scores(az_map, cluster_azs)
    if not scores:
        print("ERROR: no Spot placement scores returned for cluster AZs.", file=sys.stderr)
        sys.exit(1)

    winner = scores[0]
    winner_az = winner["AvailabilityZone"]

    record = {
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "instance_type": INSTANCE_TYPE,
        "region": REGION,
        "target_capacity": TARGET_CAPACITY,
        "winner": {
            "availability_zone": winner_az,
            "az_id": winner["AvailabilityZoneId"],
            "score": winner["Score"],
        },
        "ranking": [
            {
                "availability_zone": s["AvailabilityZone"],
                "az_id": s["AvailabilityZoneId"],
                "score": s["Score"],
            }
            for s in scores
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(record, indent=2))
    print(f"AZ selection written to {OUT_PATH}")
    print(f"Winner: {winner_az} (score {winner['Score']})")

    print(f"Patching NodePool zone constraint -> {winner_az}")
    patch_nodepool(winner_az)
    print("Done. Karpenter will only place GPU nodes in this AZ now.")


if __name__ == "__main__":
    main()
