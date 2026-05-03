#!/usr/bin/env bash
# run-experiment.sh — Apply a DGD, wait for ready, run the configured load,
# capture results, and tear down the DGD.
#
# Usage:
#   run-experiment.sh A W1                # apply dgd-A-baseline.yaml, run W1
#   run-experiment.sh B W2 --rate 1.5     # apply dgd-B-kvrouter.yaml, run W2
#
# Per design: each (experiment, workload) pair writes its own JSON to
# tests/exp-X-Wy-results.json. The load tester refuses to overwrite, so
# re-runs require deleting the existing file first.
set -euo pipefail

EXP="${1:?missing experiment letter (A/B/C/D/E)}"
WL="${2:?missing workload (W1/W2/W3/W4)}"
shift 2

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TESTS="$ROOT/tests"
mkdir -p "$TESTS"

# Map shortcuts to filenames.
declare -A DGD_MAP=(
  [A]="dgd-A-baseline.yaml"
  [B]="dgd-B-kvrouter.yaml"
  [C]="dgd-C-disagg.yaml"
  [D]="dgd-D-modelexpress.yaml"
  [E]="dgd-E-specdecode.yaml"
)
declare -A WL_MAP=(
  [W1]="w1_random"
  [W2]="w2_shared_prefix"
  [W3]="w3_bursty"
  [W4]="w4_cold_start"
)

DGD="${DGD_MAP[$EXP]:?unknown experiment $EXP}"
WORKLOAD="${WL_MAP[$WL]:?unknown workload $WL}"
DGD_NAME=$(yq '.metadata.name' "$ROOT/k8s/$DGD")
NS=dynamo-system

# Cluster guard.
CTX=$(kubectl config current-context)
[[ "$CTX" =~ inference-phase4-5 ]] || { echo "wrong cluster context: $CTX" >&2; exit 1; }

OUT="$TESTS/exp-${EXP}-${WL}-results.json"
[[ -e "$OUT" ]] && { echo "result file exists, aborting: $OUT" >&2; exit 2; }

# Apply DGD.
kubectl apply -f "$ROOT/k8s/$DGD"

# Wait for frontend, router, workers to be ready. The exact wait condition
# depends on the operator's status fields — adjust after first apply.
echo "Waiting for DGD '$DGD_NAME' to reach Ready..."
kubectl -n $NS wait --for=condition=Ready dynamographdeployment/$DGD_NAME --timeout=15m

FRONTEND_SVC="$DGD_NAME-frontend"
ENDPOINT="http://$FRONTEND_SVC.$NS.svc:8000"
METRICS="http://$FRONTEND_SVC.$NS.svc:9090/metrics"

# Defaults; allow caller overrides via remaining args.
RATE=1.0
DURATION=90
WARMUP=20
DRAIN=10
case "$WL" in
  W1) RATE=2.0;;
  W2) RATE=1.5;;
  W3) RATE=1.0; DURATION=120;;        # longer, bursts need observation window
  W4) RATE=1.0; DURATION=10; WARMUP=0; DRAIN=0;;  # one-shot probe
esac

PHASE45_EXPERIMENT_LABEL="exp-${EXP}-${WL}" \
python3 "$HERE/load-test.py" \
  --endpoint "$ENDPOINT" \
  --metrics-endpoint "$METRICS" \
  --workload "$WORKLOAD" \
  --rate "$RATE" \
  --duration "$DURATION" \
  --warmup "$WARMUP" \
  --drain "$DRAIN" \
  --output "$OUT" \
  "$@"

echo "Done. Result: $OUT"

# Leave DGD running — the next experiment will replace it. Caller decides
# when to teardown via `kubectl delete dgd $DGD_NAME -n $NS`.
