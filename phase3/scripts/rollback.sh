#!/bin/bash
# rollback.sh — Revert the vllm-worker to the Phase 1 worker image.
#
# WHY: if the Phase 3 worker has issues, this restores the known-good Phase 1
# worker. Like rolling back a deployment to a previous revision.

set -euo pipefail

NAMESPACE="inference-system"
DEPLOYMENT="vllm-worker"
CONTAINER="queue-worker"

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region || echo "us-east-1")
ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
PHASE1_IMAGE="${ECR_URL}/inference-lab/worker:latest"

echo "=== Rollback to Phase 1 Worker ==="
echo "Image: ${PHASE1_IMAGE}"
echo ""

kubectl set image \
    "deployment/${DEPLOYMENT}" \
    -n "$NAMESPACE" \
    "${CONTAINER}=${PHASE1_IMAGE}"

# Remove the ADMISSION_STRATEGY env var (Phase 1 worker doesn't use it)
kubectl set env \
    "deployment/${DEPLOYMENT}" \
    -n "$NAMESPACE" \
    -c "$CONTAINER" \
    "ADMISSION_STRATEGY-"

echo "Waiting for rollout..."
kubectl rollout status \
    "deployment/${DEPLOYMENT}" \
    -n "$NAMESPACE" \
    --timeout=300s

echo ""
echo "Rollback complete. Worker is back to Phase 1 (Strategy 2 hardcoded)."
