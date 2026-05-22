#!/bin/bash
# build_and_push.sh — Build Phase 3 worker image and push to ECR.
#
# WHY a separate image (not reusing Phase 1 worker):
#   Phase 3 worker has pluggable admission strategies controlled by
#   ADMISSION_STRATEGY env var. Phase 1 worker is hardcoded to Strategy 2.
#   We push to a different ECR tag so we can roll back if needed.
#
# The vLLM sidecar image doesn't change — only the queue-worker container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# AWS account and region
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region || echo "us-east-1")
ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "=== Phase 3 — Build & Push ==="
echo "Account: ${AWS_ACCOUNT_ID}"
echo "Region:  ${AWS_REGION}"
echo "ECR URL: ${ECR_URL}"
echo ""

# Ensure ECR repo exists
echo "Ensuring ECR repository exists..."
aws ecr create-repository \
    --repository-name inference-lab/worker \
    --region "$AWS_REGION" \
    --image-scanning-configuration scanOnPush=true \
    2>/dev/null || true

echo "ECR repository ready"

# Docker login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ECR_URL"
echo ""

# Build and push the Phase 3 worker
# WHY tag "phase3": separate tag from "latest" so we can switch back to Phase 1
# worker by redeploying with the "latest" tag. Belt-and-suspenders rollback.
echo "Building phase3 worker..."
docker buildx build \
    --platform linux/amd64 \
    -t "${ECR_URL}/inference-lab/worker:phase3" \
    "${PROJECT_DIR}/app/worker/" \
    --push

echo ""
echo "=== Image pushed ==="
echo "  ${ECR_URL}/inference-lab/worker:phase3"
echo ""
echo "Next: run phase3/scripts/deploy.sh to patch the K8s deployment"
