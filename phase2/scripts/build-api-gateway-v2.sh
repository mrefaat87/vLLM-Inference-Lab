#!/usr/bin/env bash
# build-api-gateway-v2.sh — Build and push the metrics-instrumented API gateway.
#
# WHAT THIS DOES: Builds the Phase 2 API gateway Docker image (with prometheus-client)
# and pushes it to ECR with the :v2-metrics tag. The Phase 2 deployment patch
# references this tag instead of :latest.
#
# WHY a separate image (not updating :latest): keeps Phase 1 deployable independently.
# If Phase 2 has issues, we can roll back by re-applying Phase 1's deployment.yaml
# which still points to :latest.
#
# USAGE: bash phase2/scripts/build-api-gateway-v2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_DIR="$PROJECT_ROOT/phase2/app/api_gateway"

# Get AWS account info
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region || echo "us-east-1")
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/inference-lab/api-gateway"
IMAGE_TAG="v2-metrics"

echo "=== Building API Gateway v2 (metrics-instrumented) ==="
echo "  ECR Repo: ${ECR_REPO}"
echo "  Tag: ${IMAGE_TAG}"
echo "  Source: ${APP_DIR}"
echo ""

# Step 1: Ensure ECR repo exists (idempotent)
echo "[1/4] Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names inference-lab/api-gateway \
    --region "$AWS_REGION" 2>/dev/null || \
aws ecr create-repository \
    --repository-name inference-lab/api-gateway \
    --image-scanning-configuration scanOnPush=true \
    --region "$AWS_REGION"

# Step 2: Login to ECR
echo "[2/4] Logging in to ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Step 3: Build for amd64 (EC2 target, not Apple Silicon)
echo "[3/4] Building Docker image for linux/amd64..."
# WHY buildx: cross-platform build from Apple M4 → amd64 EC2 instances.
docker buildx build \
    --platform linux/amd64 \
    --tag "${ECR_REPO}:${IMAGE_TAG}" \
    --push \
    "$APP_DIR"

# Step 4: Verify
echo "[4/4] Verifying image..."
aws ecr describe-images \
    --repository-name inference-lab/api-gateway \
    --image-ids imageTag="${IMAGE_TAG}" \
    --region "$AWS_REGION" \
    --query 'imageDetails[0].{Tag: imageTags[0], Size: imageSizeInBytes, Pushed: imagePushedAt}' \
    --output table

echo ""
echo "=== API Gateway v2 image pushed successfully ==="
echo "  Image: ${ECR_REPO}:${IMAGE_TAG}"
echo "  Apply the deployment patch: kubectl apply -f phase2/k8s/patches/api-gateway-deployment.yaml"
