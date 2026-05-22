#!/usr/bin/env bash
# run_variantH.sh — Run Variant H (FSx Lustre weights) measurements.
#
# Loops 3 fresh-node runs in series, with Karpenter consolidation gates
# between runs. The matrix runner is purpose-built for the A/B/C/D matrix
# and hardcodes run IDs; we want h-001/h-002/h-003 instead.
#
# Each iteration:
#   1. Wait for the H NodePool's node count to drop to 0 (consolidation
#      gate — guarantees a fresh-node measurement).
#   2. Run run_baseline.sh with MANIFEST_PATH pointed at the H pod manifest.
#   3. Eyeball the result before continuing.
#
# Exits early if a run's total exceeds 300s (Stop condition from the plan).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export KUBECONFIG="${PHASE_DIR}/.kubeconfig"

H_MANIFEST="${PHASE_DIR}/k8s/baseline/baseline-pod-fsx-weights.yaml"
H_NODEPOOL="gpu-pool-prebaked-fsx"
NS="baseline"

if [ ! -f "$H_MANIFEST" ]; then
  echo "ERROR: $H_MANIFEST not found — Stage 6 manifests missing" >&2
  exit 2
fi

wait_for_consolidation() {
  # Karpenter's `consolidateAfter: 300s` means an empty node lingers for 5 min
  # before being killed. Poll until the H NodePool reports zero nodes.
  echo "[wait] polling for $H_NODEPOOL to drain..."
  local timeout_seconds=600
  local elapsed=0
  while [ $elapsed -lt $timeout_seconds ]; do
    local count
    count=$(kubectl get nodes -l "image-source=prebaked-fsx" -o name 2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -eq 0 ]; then
      echo "[wait] $H_NODEPOOL drained ($(date '+%H:%M:%S'))"
      return 0
    fi
    echo "[wait] still $count node(s) on H pool, sleeping 30s..."
    sleep 30
    elapsed=$((elapsed + 30))
  done
  echo "ERROR: H pool didn't drain in ${timeout_seconds}s" >&2
  return 1
}

for i in 001 002 003; do
  RUN_ID="h-$i"
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  Variant H run: $RUN_ID  ($(date))"
  echo "════════════════════════════════════════════════════════════"

  if [ "$i" != "001" ]; then
    # First run doesn't need to wait — there's no node yet. Subsequent runs
    # MUST wait for the previous run's node to consolidate or we measure a
    # warm node.
    if ! wait_for_consolidation; then
      echo "ABORT: consolidation gate timed out before $RUN_ID" >&2
      exit 1
    fi
  fi

  # MANIFEST_PATH override — read by run_baseline.sh.
  if ! MANIFEST_PATH="$H_MANIFEST" bash "${SCRIPT_DIR}/run_baseline.sh" "$RUN_ID"; then
    echo "ABORT: run_baseline.sh failed for $RUN_ID" >&2
    exit 1
  fi

  # Eyeball the result. Stop if total > 300s (plan stop condition).
  RESULT_JSON="${PHASE_DIR}/tests/baseline_runs/${RUN_ID}.json"
  if [ -f "$RESULT_JSON" ]; then
    TOTAL_MS=$(python3 -c "
import json
d = json.load(open('$RESULT_JSON'))
total = d.get('totals', {}).get('scale_up_to_ready_ms')
print(total or 0)
")
    TOTAL_S=$((TOTAL_MS / 1000))
    echo "[$RUN_ID] scale_up_to_ready=${TOTAL_S}s"

    if [ "$TOTAL_S" -gt 300 ]; then
      echo "STOP: $RUN_ID total ${TOTAL_S}s exceeds 300s threshold." >&2
      echo "Investigate FSx throughput / SG / mount before continuing runs 2-3." >&2
      exit 1
    fi

    # Validate the new fsx_csi_mount stage was captured.
    HAS_FSX_STAGE=$(python3 -c "
import json
d = json.load(open('$RESULT_JSON'))
opt = d.get('optional_stages', {})
print('1' if 'fsx_csi_mount_s' in opt else '0')
")
    if [ "$HAS_FSX_STAGE" = "0" ]; then
      echo "WARNING: $RUN_ID has no fsx_csi_mount_s in optional_stages — tracer regression?" >&2
    fi
  fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All 3 Variant H runs complete"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Aggregate via:"
echo "  python3 ${SCRIPT_DIR}/aggregate_baseline.py --runs h-001 h-002 h-003"
