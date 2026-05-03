#!/bin/bash
# setup-soci.sh — Push vLLM image to ECR and create SOCI index for lazy loading.
#
# WHAT THIS DOES:
#   1. Pulls the vLLM image from Docker Hub (if not cached)
#   2. Creates an ECR repository (if it doesn't exist)
#   3. Pushes the image to ECR (required — SOCI only works with ECR)
#   4. Installs the soci-snapshotter CLI (if not installed)
#   5. Creates a SOCI index for the image (maps filesystem paths to layer offsets)
#   6. Pushes the SOCI index to ECR alongside the image
#
# WHY SOCI (Seekable OCI):
#   The vLLM image is 3.8GB compressed. On gp3 EBS (125 MB/s throughput),
#   decompressing all layers to overlayfs takes ~110 seconds — this is the #1
#   cold start bottleneck. Pre-baking the AMI doesn't help because containerd
#   re-unpacks layers from the local cache (it decompresses, not downloads).
#
#   SOCI solves this by creating a secondary index artifact that maps individual
#   files to byte ranges within compressed layers. The SOCI containerd snapshotter
#   uses this index to lazily fetch only the files the container actually accesses,
#   on demand. The container starts in ~10-20s instead of waiting for full unpack.
#
#   ANALOGY: think of it like Amazon S3 Select vs downloading the whole object.
#   Instead of decompressing a 3.8GB tarball to read a few Python files, SOCI
#   random-accesses specific byte ranges from the compressed layer — like range
#   GETs on S3. The container starts with a "skeleton" filesystem and fills in
#   files as they're accessed.
#
# PREREQUISITES:
#   - AWS CLI configured with ECR push permissions
#   - Docker daemon running (for image pull/push)
#   - Linux x86_64 (SOCI CLI is Linux-only; run on an EC2 instance or in WSL)
#
# COST: ~$0.10/GB/month for the ECR image. The SOCI index is ~10-50MB.
#
# USAGE:
#   bash phase4.5/scripts/setup-soci.sh
#
# AFTER RUNNING THIS:
#   1. Build the SOCI-enabled AMI: bash phase4.5/scripts/build-ami-soci.sh
#   2. Update gpu-nodepool.yaml amiSelectorTerms tag if using a new AMI name
#   3. Deploy: bash phase4.5/scripts/deploy.sh policy-d
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
VLLM_IMAGE="vllm/vllm-openai:v0.19.0"  # Only used for SOCI index reference — slim image pushed separately
ECR_REPO="inference-lab/vllm-openai"
ECR_TAG="v0.19.0-slim"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${ECR_TAG}"
SOCI_VERSION="0.7.0"  # Latest stable as of 2025

echo "================================================================"
echo " SOCI Setup — Lazy Image Loading for vLLM"
echo "================================================================"
echo ""
echo "Source image:  ${VLLM_IMAGE}"
echo "Target ECR:    ${ECR_URI}"
echo "SOCI version:  ${SOCI_VERSION}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Pull vLLM image from Docker Hub
# ---------------------------------------------------------------------------
echo "── Step 1: Pulling vLLM image from Docker Hub ──────────────────"
# WHY pull first: we need the image locally to create the SOCI index.
# If already cached, Docker will skip the download.
if docker inspect "${VLLM_IMAGE}" >/dev/null 2>&1; then
    echo "Image already cached locally, skipping pull"
else
    echo "Pulling ${VLLM_IMAGE} (~8GB, may take a few minutes)..."
    docker pull "${VLLM_IMAGE}"
fi

# ---------------------------------------------------------------------------
# Step 2: Create ECR repository (idempotent)
# ---------------------------------------------------------------------------
echo ""
echo "── Step 2: Creating ECR repository ─────────────────────────────"
# WHY ECR: SOCI uses the OCI Referrers API to store the index alongside the
# image. ECR supports this API natively. Docker Hub and most other registries
# do not support OCI referrers yet.
if aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${REGION}" >/dev/null 2>&1; then
    echo "ECR repository ${ECR_REPO} already exists"
else
    echo "Creating ECR repository ${ECR_REPO}..."
    aws ecr create-repository \
        --repository-name "${ECR_REPO}" \
        --region "${REGION}" \
        --image-tag-mutability IMMUTABLE \
        --image-scanning-configuration scanOnPush=true
    echo "Created ECR repository"
fi

# ---------------------------------------------------------------------------
# Step 3: Push vLLM image to ECR
# ---------------------------------------------------------------------------
echo ""
echo "── Step 3: Pushing vLLM image to ECR ───────────────────────────"
# WHY push to ECR: SOCI lazy loading requires the image in a registry that
# supports OCI referrers (for the index) and HTTP range requests (for lazy
# fetching layer byte ranges). ECR supports both.

# Authenticate Docker with ECR
echo "Authenticating with ECR..."
aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# Tag and push
docker tag "${VLLM_IMAGE}" "${ECR_URI}"
echo "Pushing to ECR (${ECR_URI})..."
echo "This pushes ~8GB — may take 5-10 minutes on first push."
docker push "${ECR_URI}"
echo "Push complete"

# ---------------------------------------------------------------------------
# Step 4: Install SOCI CLI
# ---------------------------------------------------------------------------
echo ""
echo "── Step 4: Installing SOCI CLI ─────────────────────────────────"
# WHY install: the soci CLI creates the index artifact from a local or remote
# image. It's a standalone Go binary — no daemon needed.
if command -v soci >/dev/null 2>&1; then
    INSTALLED_VERSION=$(soci --version 2>&1 | grep -oP '[\d.]+' | head -1 || echo "unknown")
    echo "SOCI CLI already installed (version: ${INSTALLED_VERSION})"
else
    echo "Installing SOCI CLI v${SOCI_VERSION}..."

    # WHY /usr/local/bin: standard location for user-installed binaries.
    SOCI_URL="https://github.com/awslabs/soci-snapshotter/releases/download/v${SOCI_VERSION}/soci-snapshotter-${SOCI_VERSION}-linux-amd64.tar.gz"
    TMPDIR=$(mktemp -d)
    curl -fsSL "${SOCI_URL}" -o "${TMPDIR}/soci.tar.gz"
    tar -xzf "${TMPDIR}/soci.tar.gz" -C "${TMPDIR}"

    # The tarball contains: soci, soci-snapshotter-grpc
    sudo cp "${TMPDIR}/soci" /usr/local/bin/
    sudo cp "${TMPDIR}/soci-snapshotter-grpc" /usr/local/bin/
    sudo chmod +x /usr/local/bin/soci /usr/local/bin/soci-snapshotter-grpc
    rm -rf "${TMPDIR}"

    echo "SOCI CLI installed: $(soci --version 2>&1)"
fi

# ---------------------------------------------------------------------------
# Step 5: Create SOCI index for the vLLM image
# ---------------------------------------------------------------------------
echo ""
echo "── Step 5: Creating SOCI index ─────────────────────────────────"
# WHY SOCI index: the index is a table-of-contents that maps every file in
# every layer to its byte offset in the compressed layer blob. When the SOCI
# snapshotter needs a file, it does an HTTP range request for just those bytes
# instead of downloading and decompressing the entire layer.
#
# The index is stored as an OCI artifact (referrer) in ECR, linked to the
# image manifest via the OCI Referrers API. Containerd discovers it automatically.
#
# Index creation reads the image from ECR (not local Docker) so it matches
# the exact manifest/digest that nodes will pull.
echo "Creating SOCI index for ${ECR_URI}..."
echo "This reads image layers from ECR and builds the file-to-offset table."
echo "May take 2-5 minutes for an 8GB image."

# WHY --min-layer-size 10M: only index layers larger than 10MB. Small layers
# (config, metadata) decompress instantly — not worth the index overhead.
# The vLLM image has a few massive layers (PyTorch, CUDA) and many small ones.
soci create "${ECR_URI}" --min-layer-size 10485760

echo "SOCI index created"

# ---------------------------------------------------------------------------
# Step 6: Push SOCI index to ECR
# ---------------------------------------------------------------------------
echo ""
echo "── Step 6: Pushing SOCI index to ECR ───────────────────────────"
# WHY push: the index must be in ECR alongside the image so containerd can
# discover it via the OCI Referrers API. Without the index in the registry,
# the SOCI snapshotter falls back to normal (full) image pull.
echo "Pushing SOCI index..."
soci push "${ECR_URI}"
echo "SOCI index pushed to ECR"

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " SOCI Setup Complete"
echo "================================================================"
echo ""
echo "Image in ECR:  ${ECR_URI}"
echo "SOCI index:    attached as OCI referrer (discoverable by containerd)"
echo ""
echo "NEXT STEPS:"
echo "  1. Build SOCI-enabled AMI:"
echo "     bash phase4.5/packer/build-ami-soci.sh"
echo "  2. Deploy with the new AMI:"
echo "     bash phase4.5/scripts/deploy.sh policy-d"
echo ""
echo "HOW TO VERIFY SOCI IS WORKING:"
echo "  After deploying, SSH into a GPU node and check containerd logs:"
echo "    journalctl -u soci-snapshotter -f"
echo "  You should see 'remote snapshot prepared' for each layer — this means"
echo "  the layer is being lazy-loaded from ECR, not fully unpacked."
echo ""
echo "EXPECTED IMPROVEMENT:"
echo "  Without SOCI: ~250s image unpack (gp3 125 MB/s × 9.5GB v0.19.0 image)"
echo "  With SOCI:    10-20s (only fetches accessed files on demand)"
echo "  Savings:      ~230-240s per cold start"
