#!/bin/bash
# run_baseline.sh — execute one Phase 4.1 baseline cold-start run end-to-end.
#
# What it does:
#   1. Renders baseline-pod.yaml with RUN_ID + IRSA role ARN + tracer script
#   2. Apply manifest, capture wall-clock t0 (Karpenter sees Pending pod)
#   3. Wait for pod Ready (timeout 20 min — well past worst-case)
#   4. Issue smoke request, capture first-token wall-clock
#   5. Read EC2 instance metadata for the node (id, az, type, AMI)
#   6. Read CPU model + kernel from the node via `kubectl debug`
#   7. Pull containerd journal from the node
#   8. Pull tracer JSONL from the node hostPath
#   9. Run collect_cold_start_run.py to assemble the run record
#  10. Delete the pod (Karpenter consolidates the node within 60s)
#
# Usage:
#   ./run_baseline.sh <run-id> [--skip-cleanup]
#
# Outputs:
#   phase4.1/tests/baseline_runs/<run-id>.json
#   phase4.1/tests/baseline_runs/raw/<run-id>.{events.json,containerd.log,tracer.jsonl,smoke.log,pod.json}
set -euo pipefail

RUN_ID="${1:?usage: $0 <run-id> [--skip-cleanup]}"
SKIP_CLEANUP="${2:-}"

PHASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$PHASE_DIR/tests/baseline_runs/raw"
mkdir -p "$RAW_DIR"

NS=baseline
POD="baseline-${RUN_ID}"

# Manifest path is overridable so the 3-way pull comparison can swap variants
# (baseline-pod.yaml / baseline-pod-ecr.yaml / baseline-pod-prebaked.yaml).
# Default preserves the original behaviour for existing single-variant callers.
MANIFEST_PATH="${MANIFEST_PATH:-$PHASE_DIR/k8s/baseline/baseline-pod.yaml}"

# --- Render manifest ---------------------------------------------------------
echo "[$RUN_ID] rendering manifest"
ROLE_ARN="$(cd "$PHASE_DIR/terraform" && terraform output -raw vllm_worker_role_arn)"
TRACER_PATH="$PHASE_DIR/scripts/cold_start_tracer.py"
# Indent the tracer script by 4 spaces to fit inside ConfigMap data block.
INDENTED_TRACER="$(sed 's/^/    /' "$TRACER_PATH")"

RENDERED="$(mktemp)"
trap 'rm -f "$RENDERED"' EXIT

# Substitute placeholders. Use python for the multi-line tracer substitution
# to avoid shell escaping pain.
python3 - <<PY > "$RENDERED"
import os, pathlib
tpl = pathlib.Path("$MANIFEST_PATH").read_text()
tracer = pathlib.Path("$TRACER_PATH").read_text()
indented = "\n".join("    " + line for line in tracer.splitlines())
out = (tpl
    .replace("RUN_ID", "$RUN_ID")
    .replace("ROLE_ARN_PLACEHOLDER", "$ROLE_ARN")
    .replace("    PLACEHOLDER_TRACER_SCRIPT", indented))
print(out)
PY

# --- Apply -------------------------------------------------------------------
echo "[$RUN_ID] applying"
kubectl apply -f "$RENDERED"

# --- Wait for Ready ----------------------------------------------------------
echo "[$RUN_ID] waiting for pod Ready (timeout 20m)"
if ! kubectl wait --for=condition=Ready pod/"$POD" -n "$NS" --timeout=20m; then
    echo "[$RUN_ID] pod did not become Ready in time"
    kubectl describe pod -n "$NS" "$POD" > "$RAW_DIR/$RUN_ID.describe.txt" || true
    kubectl logs -n "$NS" "$POD" --tail=200 > "$RAW_DIR/$RUN_ID.logs.txt" || true
    [[ "$SKIP_CLEANUP" != "--skip-cleanup" ]] && kubectl delete pod -n "$NS" "$POD" --grace-period=0 --force || true
    exit 1
fi

POD_READY_MS="$(python3 -c "import time; print(int(time.time()*1000))")"
echo "[$RUN_ID] pod Ready at $POD_READY_MS ms"

# --- Smoke request -----------------------------------------------------------
echo "[$RUN_ID] running smoke request"
PFWD_LOG="$(mktemp)"
kubectl port-forward -n "$NS" pod/"$POD" 18000:8000 > "$PFWD_LOG" 2>&1 &
PFWD_PID=$!
trap 'kill $PFWD_PID 2>/dev/null || true; rm -f "$RENDERED" "$PFWD_LOG"' EXIT

# Wait for port-forward to be live.
for i in {1..20}; do
    sleep 0.5
    curl -fsS http://127.0.0.1:18000/health > /dev/null 2>&1 && break
done

SMOKE_START_MS="$(python3 -c "import time; print(int(time.time()*1000))")"
SMOKE_LOG="$RAW_DIR/$RUN_ID.smoke.log"

# vLLM exposes the served model name via /v1/models. Variant E uses an
# s3://… model arg (so its served name is the S3 URL); variants A-D use the
# local /models/Qwen… path. Query first instead of hard-coding.
SERVED_MODEL="$(curl -sS http://127.0.0.1:18000/v1/models 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' \
    2>/dev/null || echo "/models/Qwen2.5-7B-Instruct-AWQ")"
echo "[$RUN_ID] smoke request against served model: $SERVED_MODEL"

curl -sS -N -X POST http://127.0.0.1:18000/v1/completions \
    -H "Content-Type: application/json" \
    -d "$(printf '{"model":%s,"prompt":"Hello","max_tokens":1,"stream":true}' \
            "$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$SERVED_MODEL")")" \
    -o "$SMOKE_LOG" || true
SMOKE_FIRST_TOKEN_MS="$(python3 -c "import time; print(int(time.time()*1000))")"
kill $PFWD_PID 2>/dev/null || true

echo "[$RUN_ID] smoke first-token at $SMOKE_FIRST_TOKEN_MS ms"

# --- Harvest pod + node metadata --------------------------------------------
# Disable strict-error-mode for the harvest section. Many of the kubectl/aws/
# python commands here can return non-zero on transient API hiccups (control
# plane DNS flaps, debug-pod timing) without anything actually being wrong.
# We rely on collect_cold_start_run.py's validation to catch real data gaps.
set +e
echo "[$RUN_ID] harvesting metadata"
NODE="$(kubectl get pod -n "$NS" "$POD" -o jsonpath='{.spec.nodeName}')"
INSTANCE_ID="$(kubectl get node "$NODE" -o jsonpath='{.spec.providerID}' | sed 's|.*/||')"

# EC2 metadata
EC2_JSON="$RAW_DIR/$RUN_ID.ec2.json"
aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0]' --output json > "$EC2_JSON"
INSTANCE_TYPE="$(jq -r '.InstanceType' "$EC2_JSON")"
AZ="$(jq -r '.Placement.AvailabilityZone' "$EC2_JSON")"
AMI_ID="$(jq -r '.ImageId' "$EC2_JSON")"
EC2_LAUNCH_TIME="$(jq -r '.LaunchTime' "$EC2_JSON")"
EC2_RUNNING_MS="$(python3 -c "from datetime import datetime; print(int(datetime.fromisoformat('$EC2_LAUNCH_TIME'.replace('Z','+00:00')).timestamp()*1000))")"

# Karpenter NodeClaim creationTimestamp = when Karpenter decided to launch the
# instance (just before the EC2 CreateFleet API call). We use this instead of
# CloudTrail RunInstances because:
#  - CloudTrail can lag 5-15 min, often empty when we look
#  - Karpenter v0.37 uses CreateFleet for spot, not RunInstances
#  - NodeClaim is in our cluster — instant access, no IAM gymnastics
NC_NAME="$(kubectl get nodeclaims -o json | jq -r ".items[] | select(.status.providerID==\"$(kubectl get node $NODE -o jsonpath='{.spec.providerID}')\") | .metadata.name")"
NC_CREATED="$(kubectl get nodeclaim "$NC_NAME" -o jsonpath='{.metadata.creationTimestamp}' 2>/dev/null || echo "")"
EC2_RUN_REQUEST_MS=""
if [[ -n "$NC_CREATED" ]]; then
    EC2_RUN_REQUEST_MS="$(python3 -c "from datetime import datetime; print(int(datetime.fromisoformat('$NC_CREATED'.replace('Z','+00:00')).timestamp()*1000))")"
fi
echo "[$RUN_ID] NodeClaim=$NC_NAME created at $NC_CREATED ($EC2_RUN_REQUEST_MS ms)"

# --- Pull pod object + events to raw -----------------------------------------
kubectl get pod -n "$NS" "$POD" -o json > "$RAW_DIR/$RUN_ID.pod.json"
kubectl get events -n "$NS" --field-selector "involvedObject.name=$POD" -o json > "$RAW_DIR/$RUN_ID.events.json"

# --- Pull node-level data via kubectl debug ----------------------------------
DEBUGGER="kubectl debug node/$NODE -it --image=public.ecr.aws/amazonlinux/amazonlinux:2 -- chroot /host"
# Note: kubectl debug is interactive; for scripting we use --quiet + redirect.
# This step may fail on locked-down nodes; we proceed without it if so.
JOURNAL_PATH="$RAW_DIR/$RUN_ID.containerd.log"
TRACER_JSONL="$RAW_DIR/$RUN_ID.tracer.jsonl"

CPU_MODEL=""
KERNEL=""
echo "[$RUN_ID] pulling tracer events from kubectl logs"
# Extract tracer JSONL events from the vllm container's stdout. This is more
# reliable than hostPath because:
#  - kubectl logs survives pod deletion (--previous flag) for ~10 min
#  - no kubectl debug pod hassle
#  - works on locked-down nodes where SSM/exec might be blocked
kubectl logs -n "$NS" "$POD" -c vllm 2>/dev/null \
    | grep -E '^\{"event":' > "$TRACER_JSONL" || echo "" > "$TRACER_JSONL"

# containerd journal + cpu/kernel via `kubectl debug node`.
# Critical detail: `kubectl debug node/X` with default --attach=true does NOT
# include the command's stdout in kubectl's stdout — only meta-messages. Use
# --attach=false so kubectl just creates the debug pod and returns immediately,
# then `kubectl logs` reads the actual command output once the pod completes.
echo "[$RUN_ID] pulling node-level data (containerd + cpu/kernel) via debug pod"

DBG_META="$(mktemp)"
kubectl debug node/"$NODE" --image=public.ecr.aws/amazonlinux/amazonlinux:2 --attach=false \
    -- chroot /host sh -c \
    "echo '===CPU==='; grep -m1 'model name' /proc/cpuinfo; echo '===KERNEL==='; uname -r; echo '===JOURNAL==='; journalctl -u containerd --since '1 hour ago' -o short-iso" \
    > "$DBG_META" 2>&1 || true

DBG_POD="$(grep -oE 'node-debugger-[a-z0-9.\-]+' "$DBG_META" 2>/dev/null | head -1 || true)"
rm -f "$DBG_META"

if [[ -n "$DBG_POD" ]]; then
    # Wait for the debug pod to actually run and complete. We can't rely on
    # `kubectl wait --for=condition=Ready=false` alone because on a fresh node
    # the AL2 image pull can take 20-30s, and the pod stays in
    # ContainerCreating that whole time. Poll until pod phase is Succeeded
    # or Failed, with a generous total budget.
    DBG_OUT="$(mktemp)"
    for i in $(seq 1 60); do  # up to 5 min total (60 * 5s)
        phase="$(kubectl get pod -n default "$DBG_POD" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")"
        if [[ "$phase" == "Succeeded" || "$phase" == "Failed" ]]; then
            break
        fi
        sleep 5
    done
    sleep 2  # allow log backend to flush
    kubectl logs -n default "$DBG_POD" > "$DBG_OUT" 2>&1
    if grep -q "===JOURNAL===" "$DBG_OUT"; then
        awk '/===CPU===/{f="cpu";next} /===KERNEL===/{f="kernel";next} /===JOURNAL===/{f="journal";next} f{print > ("/tmp/dbg-" f)}' "$DBG_OUT"
        CPU_MODEL="$(sed 's/^.*: //' /tmp/dbg-cpu 2>/dev/null | head -1 || echo unknown)"
        KERNEL="$(head -1 /tmp/dbg-kernel 2>/dev/null || echo unknown)"
        cp /tmp/dbg-journal "$JOURNAL_PATH" 2>/dev/null || echo "" > "$JOURNAL_PATH"
    else
        echo "[$RUN_ID] debug pod $DBG_POD output missing sentinels; saving raw to $JOURNAL_PATH for inspection"
        cp "$DBG_OUT" "$JOURNAL_PATH"
        CPU_MODEL="unknown"
        KERNEL="unknown"
    fi
    kubectl delete pod -n default "$DBG_POD" --grace-period=0 --force >/dev/null 2>&1 || true
    rm -f "$DBG_OUT" /tmp/dbg-cpu /tmp/dbg-kernel /tmp/dbg-journal
else
    echo "[$RUN_ID] could not create debug pod"
    echo "" > "$JOURNAL_PATH"
    CPU_MODEL="unknown"
    KERNEL="unknown"
fi

# --- Build run record --------------------------------------------------------
# Re-enable strict-error-mode for the final phases — collector failures or
# missing JSON are real bugs we want to surface.
set -e
echo "[$RUN_ID] building run record"
python3 "$PHASE_DIR/scripts/collect_cold_start_run.py" \
    --run-id "$RUN_ID" \
    --arm fresh_node \
    --namespace "$NS" \
    --pod "$POD" \
    --node "$NODE" \
    --instance-id "$INSTANCE_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --az "$AZ" \
    --ami-id "$AMI_ID" \
    --cpu-model "$CPU_MODEL" \
    --kernel "$KERNEL" \
    ${EC2_RUN_REQUEST_MS:+--ec2-run-request-ms "$EC2_RUN_REQUEST_MS"} \
    --ec2-running-ms "$EC2_RUNNING_MS" \
    --pod-ready-ms "$POD_READY_MS" \
    --smoke-first-token-ms "$SMOKE_FIRST_TOKEN_MS" \
    --tracer-jsonl "$TRACER_JSONL" \
    --containerd-journal "$JOURNAL_PATH" \
    --out-dir "$PHASE_DIR/tests/baseline_runs"

# --- Cleanup -----------------------------------------------------------------
if [[ "$SKIP_CLEANUP" != "--skip-cleanup" ]]; then
    echo "[$RUN_ID] deleting pod (Karpenter will consolidate the node ~60s after empty)"
    kubectl delete pod -n "$NS" "$POD" --wait=false || true
fi

echo "[$RUN_ID] done"
