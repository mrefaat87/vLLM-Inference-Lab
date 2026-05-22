#!/bin/bash
# deploy.sh — Patch the vllm-worker deployment to use the Phase 3 worker image.
#
# WHAT THIS DOES:
#   1. Installs Grafana Image Renderer (for /render API screenshot export)
#   2. Updates the queue-worker container image to the Phase 3 tag
#   3. Adds admission config env vars (strategy, threshold, metric)
#   4. Waits for the rollout to complete
#
# WHY patch (not full manifest apply): we don't want to touch the vLLM server
# container, volumes, tolerations, or resource limits. Only the queue-worker
# sidecar changes. This minimizes blast radius.
#
# Usage:
#   bash phase3/scripts/deploy.sh                     # deploy with default (reactive, decode_throughput)
#   bash phase3/scripts/deploy.sh static              # static strategy
#   bash phase3/scripts/deploy.sh predictive 10 0.80 compound  # predictive + compound metric

set -euo pipefail

STRATEGY="${1:-reactive}"
NAMESPACE="inference-system"
DEPLOYMENT="vllm-worker"
CONTAINER="queue-worker"
PREFETCH="${2:-10}"
THRESHOLD="${3:-0.80}"
METRIC="${4:-decode_throughput}"

# Validate strategy
if [[ "$STRATEGY" != "static" && "$STRATEGY" != "reactive" && "$STRATEGY" != "predictive" ]]; then
    echo "ERROR: Invalid strategy '$STRATEGY'. Must be: static, reactive, or predictive"
    exit 1
fi

# Get AWS account for ECR URL
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region || echo "us-east-1")
ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
PHASE3_IMAGE="${ECR_URL}/inference-lab/worker:phase3"

echo "=== Phase 3 — Deploy ==="
echo "Strategy:  ${STRATEGY}"
echo "Image:     ${PHASE3_IMAGE}"
echo "Namespace: ${NAMESPACE}"
echo ""

# Check current state
echo "Current deployment state:"
kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" -o wide 2>/dev/null || {
    echo "ERROR: Deployment $DEPLOYMENT not found in namespace $NAMESPACE."
    echo "Is the cluster running? (check: kubectl get nodes)"
    exit 1
}
echo ""

# Step 0: Install Grafana Image Renderer for screenshot export
# WHY: the backpressure comparison test captures Grafana panel PNGs after each
# run via the /render API. This API requires the Image Renderer plugin, which
# kube-prometheus-stack doesn't install by default. The renderer runs as a
# sidecar container alongside Grafana (~200Mi memory).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
echo "Installing Grafana Image Renderer..."
helm upgrade kube-prometheus-stack prometheus-community/kube-prometheus-stack \
    -n monitoring \
    -f "${PROJECT_DIR}/../phase2/k8s/monitoring/prometheus-values.yaml" \
    -f "${PROJECT_DIR}/k8s/grafana-image-renderer-values.yaml" \
    --wait --timeout 3m 2>/dev/null && \
    echo "  Image Renderer installed" || \
    echo "  WARNING: Could not install Image Renderer — screenshots will fall back to URLs"
echo ""

# Step 1: Update the container image to Phase 3
# WHY kubectl set image: atomic operation that triggers a rolling update.
# Equivalent to updating the launch template version in an ASG.
echo "Updating container image to phase3..."
kubectl set image \
    "deployment/${DEPLOYMENT}" \
    -n "$NAMESPACE" \
    "${CONTAINER}=${PHASE3_IMAGE}"

# Step 2: Set the admission strategy env var
# WHY separate command: kubectl set env and set image can't be combined in one call.
# The rolling update from Step 1 hasn't completed yet, but K8s batches the changes.
echo "Setting admission config: strategy=${STRATEGY}, metric=${METRIC}, threshold=${THRESHOLD}..."
kubectl set env \
    "deployment/${DEPLOYMENT}" \
    -n "$NAMESPACE" \
    -c "$CONTAINER" \
    "ADMISSION_STRATEGY=${STRATEGY}" \
    "PREFETCH_COUNT=${PREFETCH}" \
    "ADMISSION_THRESHOLD=${THRESHOLD}" \
    "BOTTLENECK_METRIC=${METRIC}"

# Step 3: Wait for rollout
echo ""
echo "Waiting for rollout to complete (up to 5 min)..."
echo "  (vLLM model reload takes ~2-3 min on T4)"
kubectl rollout status \
    "deployment/${DEPLOYMENT}" \
    -n "$NAMESPACE" \
    --timeout=300s

echo ""
echo "=== Deployment complete ==="
echo "Strategy: ${STRATEGY}"
echo ""

# Verify the pod is running with the correct image and strategy
echo "Verifying pod state:"
kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=vllm-worker -o wide
echo ""

# Show the worker container's env vars for confirmation
echo "Worker container env vars:"
kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="queue-worker")].env[*]}' | \
    python3 -c "import sys,json; [print(f'  {e[\"name\"]}={e.get(\"value\",\"<secret>\")}') for e in json.loads('['+sys.stdin.read().replace('} {','},{').replace('}{','},{')+']')]" 2>/dev/null || true
echo ""

echo "Next steps:"
echo "  1. Start port-forward:  bash phase3/scripts/port-forward.sh"
echo "  2. Run comparison test: python3 phase3/tests/backpressure_comparison.py --host http://localhost:8080"
echo "  3. To switch strategy:  bash phase3/scripts/deploy.sh <static|reactive|predictive> [prefetch] [threshold] [metric]"
