#!/bin/bash
# build_and_push.sh — Build container images and push to ECR.
#
# WHY ECR (not Docker Hub): EKS pulls from ECR within the same AWS region over
# the VPC endpoint — no internet egress, no Docker Hub rate limits, and IAM-based
# auth instead of managing registry credentials as K8s secrets.
#
# WHY buildx with --platform linux/amd64: we're building on Apple Silicon (M4)
# but deploying to x86 EC2 instances. Without explicit platform targeting,
# Docker would build arm64 images that crash on EKS nodes.

set -euo pipefail

# ---------------------------------------------------------------------------
# Determine project root (one level up from scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# AWS account and region — used to construct ECR URLs
# ---------------------------------------------------------------------------
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region || echo "us-east-1")

# ECR URL format: {account}.dkr.ecr.{region}.amazonaws.com
ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "=== Build & Push ==="
echo "Account: ${AWS_ACCOUNT_ID}"
echo "Region:  ${AWS_REGION}"
echo "ECR URL: ${ECR_URL}"
echo ""

# ---------------------------------------------------------------------------
# Create ECR repos if they don't exist
# WHY || true: create-repository fails if the repo already exists. We want
# idempotency — run this script multiple times without error.
# ---------------------------------------------------------------------------
echo "Ensuring ECR repositories exist..."
aws ecr create-repository \
    --repository-name inference-lab/api-gateway \
    --region "$AWS_REGION" \
    --image-scanning-configuration scanOnPush=true \
    2>/dev/null || true
# WHY scanOnPush: automatically scans images for CVEs on push.
# Free tier includes 100 scans/month — good hygiene for no extra cost.

aws ecr create-repository \
    --repository-name inference-lab/worker \
    --region "$AWS_REGION" \
    --image-scanning-configuration scanOnPush=true \
    2>/dev/null || true

echo "ECR repositories ready"

# ---------------------------------------------------------------------------
# Docker login to ECR
# WHY get-login-password: generates a 12-hour token. The old get-login command
# is deprecated and printed credentials to stdout (security risk).
# ---------------------------------------------------------------------------
echo "Logging into ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ECR_URL"
echo ""

# ---------------------------------------------------------------------------
# Build and push api-gateway
# WHY --push flag: builds and pushes in a single step. Without it, buildx
# builds to a local cache that's not accessible via `docker images` (buildx
# uses its own storage backend).
# ---------------------------------------------------------------------------
echo "Building api-gateway..."
docker buildx build \
    --platform linux/amd64 \
    -t "${ECR_URL}/inference-lab/api-gateway:latest" \
    "${PROJECT_DIR}/app/api_gateway/" \
    --push

echo "api-gateway pushed successfully"
echo ""

# ---------------------------------------------------------------------------
# Build and push worker
# ---------------------------------------------------------------------------
echo "Building worker..."
docker buildx build \
    --platform linux/amd64 \
    -t "${ECR_URL}/inference-lab/worker:latest" \
    "${PROJECT_DIR}/app/worker/" \
    --push

echo "worker pushed successfully"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=== Images pushed ==="
echo "  ${ECR_URL}/inference-lab/api-gateway:latest"
echo "  ${ECR_URL}/inference-lab/worker:latest"
echo ""
echo "Next: run scripts/setup.sh to deploy to EKS"
