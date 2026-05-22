#!/bin/bash
# run_baseline_matrix.sh — bash version of the 3-run matrix orchestrator.
#
# Earlier python orchestrator (run_baseline_matrix.py) had subprocess-mode
# rc-propagation issues that didn't repro in foreground bash. This is the
# pragmatic alternative: same shell, same env, sequential runs with a
# hardware-stability gate between them.
#
# Usage: bash scripts/run_baseline_matrix.sh
# Output: phase4.1/tests/baseline_runs/fresh-{001,002,003}.json
set -uo pipefail   # NOT set -e — we manage rc explicitly so harvest cleanup
                   # warnings never abort the whole matrix.

PHASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUNS=( "fresh-001" "fresh-002" "fresh-003" )
RUN_OK=()
FP_AZ="" ; FP_TYPE="" ; FP_CPU=""

teardown_node() {
    # macOS bash 3.2 doesn't have mapfile; use plain `for` loop.
    local nodes_str
    nodes_str="$(kubectl get nodes -l node-role=gpu -o name 2>/dev/null)"
    if [[ -z "$nodes_str" ]]; then
        echo "  no GPU node to tear down"
        return
    fi
    for n in $nodes_str; do
        echo "  deleting node $n"
        kubectl delete "$n" --wait=false >/dev/null 2>&1 || true
    done
    # Wait up to 4 min for NodeClaim cleanup.
    for i in {1..48}; do
        if [[ -z "$(kubectl get nodeclaims -o name 2>/dev/null)" ]]; then
            echo "  NodeClaim gone — node terminated"
            return
        fi
        sleep 5
    done
    echo "  WARNING: NodeClaim still present after 240s — proceeding"
}

for i in "${!RUNS[@]}"; do
    run_id="${RUNS[$i]}"
    echo
    echo "=== Run $((i+1))/${#RUNS[@]}: $run_id ==="
    echo "[gate] tearing down GPU node before fresh run"
    teardown_node

    out_file="$PHASE_DIR/tests/baseline_runs/$run_id.json"
    rm -f "$out_file"

    echo "  invoking run_baseline.sh $run_id"
    bash "$PHASE_DIR/scripts/run_baseline.sh" "$run_id"
    rc=$?
    echo "  run_baseline.sh exit code: $rc"

    if [[ ! -f "$out_file" ]]; then
        echo "  FATAL: $out_file missing — aborting matrix"
        exit 2
    fi

    valid=$(python3 -c "import json; d=json.load(open('$out_file')); print(d['validation']['all_stages_present'] and not d['validation']['any_negative_durations'])")
    if [[ "$valid" != "True" ]]; then
        echo "  $run_id validation failed:"
        python3 -c "import json; print(json.load(open('$out_file'))['validation'])"
        echo "  aborting matrix — fix the pipeline before continuing"
        exit 3
    fi

    az=$(python3 -c "import json; print(json.load(open('$out_file'))['node']['az'])")
    typ=$(python3 -c "import json; print(json.load(open('$out_file'))['node']['instance_type'])")
    cpu=$(python3 -c "import json; print(json.load(open('$out_file'))['node']['cpu_model'])")
    echo "  hardware: az=$az type=$typ cpu=$cpu"

    if [[ -z "$FP_AZ" ]]; then
        FP_AZ="$az" ; FP_TYPE="$typ" ; FP_CPU="$cpu"
    else
        if [[ "$az" != "$FP_AZ" || "$typ" != "$FP_TYPE" || "$cpu" != "$FP_CPU" ]]; then
            echo
            echo "[gate] HARDWARE MISMATCH — aborting matrix"
            echo "  reference: az=$FP_AZ type=$FP_TYPE cpu=$FP_CPU"
            echo "  this run:  az=$az type=$typ cpu=$cpu"
            echo "  valid runs collected so far: ${RUN_OK[*]}"
            exit 4
        fi
    fi
    RUN_OK+=( "$run_id" )
done

echo
echo "=== Matrix complete: ${#RUN_OK[@]}/${#RUNS[@]} runs ==="
echo "all hardware fingerprints match: az=$FP_AZ type=$FP_TYPE cpu=$FP_CPU"
