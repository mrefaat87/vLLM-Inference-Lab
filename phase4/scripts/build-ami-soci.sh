#!/bin/bash
# build-ami-soci.sh — Build GPU AMI with SOCI lazy image loading.
#
# WHAT THIS DOES:
#   1. Runs Packer to build an AMI with SOCI snapshotter installed
#   2. Enables EBS Fast Snapshot Restore (FSR) for the AMI
#   3. Updates Karpenter NodePool to use the SOCI AMI tag
#
# PREREQUISITES:
#   - Run setup-soci.sh first (pushes vLLM image + SOCI index to ECR)
#   - Packer installed (brew install packer)
#   - AWS CLI configured with EC2/EBS/ECR permissions
#
# USAGE:
#   bash phase4/scripts/build-ami-soci.sh
#
# COST:
#   - Build instance: ~$0.16-0.50/hr for 10-15 min = ~$0.05
#   - FSR: $0.75/hr per AZ while enabled (disable after testing)
#   - AMI snapshot: ~$0.05/month
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKER_DIR="${SCRIPT_DIR}/../packer"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_VLLM_IMAGE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/inference-lab/vllm-openai:v0.19.0-slim"

echo "================================================================"
echo " Building SOCI-enabled GPU AMI"
echo "================================================================"
echo ""
echo "Region:     ${REGION}"
echo "ECR image:  ${ECR_VLLM_IMAGE}"
echo "Packer dir: ${PACKER_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Verify SOCI index exists in ECR
# ---------------------------------------------------------------------------
echo "── Step 1: Verifying SOCI index exists in ECR ──────────────────"
# WHY check: building the AMI without a SOCI index in ECR means SOCI will
# fall back to normal (full) decompression — wasting the entire effort.
IMAGE_DIGEST=$(aws ecr describe-images \
    --repository-name "inference-lab/vllm-openai" \
    --image-ids imageTag="v0.19.0-slim" \
    --query 'imageDetails[0].imageDigest' \
    --output text \
    --region "${REGION}" 2>/dev/null || echo "NOT_FOUND")

if [ "${IMAGE_DIGEST}" = "NOT_FOUND" ]; then
    echo "ERROR: vLLM image not found in ECR. Run setup-soci.sh first."
    exit 1
fi
echo "vLLM image found in ECR: ${IMAGE_DIGEST}"

# Check for SOCI index (OCI referrer)
# WHY: SOCI index is stored as an OCI artifact referencing the image manifest.
# If it doesn't exist, SOCI snapshotter falls back to normal overlayfs.
REFERRERS=$(aws ecr describe-images \
    --repository-name "inference-lab/vllm-openai" \
    --query 'imageDetails | length(@)' \
    --output text \
    --region "${REGION}" 2>/dev/null || echo "0")

if [ "${REFERRERS}" -le 1 ]; then
    echo "WARNING: Only 1 image found in ECR — SOCI index may not exist."
    echo "Run setup-soci.sh to create and push the SOCI index."
    echo "Continuing anyway (SOCI will fall back to normal pull if no index found)."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 2: Initialize and run Packer
# ---------------------------------------------------------------------------
echo "── Step 2: Building AMI with Packer ────────────────────────────"
cd "${PACKER_DIR}"

# Initialize Packer plugins (idempotent)
packer init gpu-node-soci.pkr.hcl

# Build the AMI
# WHY -on-error=ask: if the build fails, Packer prompts whether to clean up
# the instance. Useful for debugging SSH into the instance.
packer build \
    -var "region=${REGION}" \
    -var "ecr_vllm_image=${ECR_VLLM_IMAGE}" \
    -on-error=ask \
    gpu-node-soci.pkr.hcl

echo ""

# ---------------------------------------------------------------------------
# Step 3: Find the newly built AMI and enable FSR
# ---------------------------------------------------------------------------
echo "── Step 3: Enabling Fast Snapshot Restore (FSR) ────────────────"

# Find the latest SOCI AMI
AMI_ID=$(aws ec2 describe-images \
    --owners self \
    --filters "Name=name,Values=inference-lab-gpu-node-soci-*" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text \
    --region "${REGION}")

echo "Latest SOCI AMI: ${AMI_ID}"

# Find the root EBS snapshot
SNAPSHOT_ID=$(aws ec2 describe-images \
    --image-ids "${AMI_ID}" \
    --query 'Images[0].BlockDeviceMappings[?DeviceName==`/dev/xvda`].Ebs.SnapshotId' \
    --output text \
    --region "${REGION}")

echo "Root snapshot: ${SNAPSHOT_ID}"

# Enable FSR for both AZs
# WHY FSR: even with SOCI lazy loading, the EBS volume backing the root
# filesystem needs to be fully initialized for fast I/O. Without FSR,
# first access to cold EBS blocks goes to S3 (100-200ms per block).
# With FSR, all blocks are pre-warmed (~0.1ms per block).
echo "Enabling FSR in us-east-1a and us-east-1b..."
aws ec2 enable-fast-snapshot-restores \
    --availability-zones us-east-1a us-east-1b \
    --source-snapshot-ids "${SNAPSHOT_ID}" \
    --region "${REGION}" || true

echo ""
echo "================================================================"
echo " SOCI AMI Build Complete"
echo "================================================================"
echo ""
echo "AMI ID:       ${AMI_ID}"
echo "Snapshot:     ${SNAPSHOT_ID}"
echo "FSR:          Enabled (us-east-1a, us-east-1b)"
echo ""
echo "AMI tag 'inference-lab-gpu-node-soci' matches Karpenter amiSelectorTerms."
echo ""
echo "NEXT STEPS:"
echo "  1. Update gpu-nodepool.yaml amiSelectorTerms tag to 'inference-lab-gpu-node-soci'"
echo "  2. Deploy: bash phase4/scripts/deploy.sh policy-d"
echo ""
echo "COST WARNING:"
echo "  FSR costs \$0.75/hr per AZ (\$1.50/hr for 2 AZs)."
echo "  Disable when not testing:"
echo "    aws ec2 disable-fast-snapshot-restores \\"
echo "      --availability-zones us-east-1a us-east-1b \\"
echo "      --source-snapshot-ids ${SNAPSHOT_ID}"
