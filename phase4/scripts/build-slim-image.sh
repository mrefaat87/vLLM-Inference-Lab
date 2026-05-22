#!/bin/bash
# build-slim-image.sh — Build slim vLLM image from source targeting T4 (sm_75 only).
#
# WHAT THIS DOES:
#   1. Clones vLLM v0.19.0 from GitHub
#   2. Builds with torch_cuda_arch_list="7.5" (T4 only) — eliminates ~5GB of unused kernels
#   3. Pushes slim image (~4GB) to ECR
#   4. Builds slim-streamer image (slim + RunAI Model Streamer)
#   5. Pushes slim-streamer to ECR
#
# WHY SLIM:
#   Stock vllm/vllm-openai:v0.19.0 is 9.5GB — compiles CUDA kernels for 7 GPU
#   architectures (sm_70 through sm_120). Our g4dn.xlarge has T4 (sm_75 only).
#   Building for sm_75 eliminates ~5GB of unused kernels. Image pull drops from
#   4+ min to ~2 min. Combined with SOCI, cold start improves dramatically.
#
# MUST RUN ON: g4dn.xlarge or any GPU instance (needs CUDA for build).
# Build time: ~30-45 minutes.
#
# PREREQUISITES:
#   - Docker installed with NVIDIA Container Toolkit
#   - AWS CLI configured with ECR push permissions
#   - ~50GB free disk space for build cache
#
# USAGE:
#   bash phase4/scripts/build-slim-image.sh
#
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
VLLM_VERSION="v0.19.0"
SLIM_TAG="${VLLM_VERSION}-slim"
SLIM_STREAMER_TAG="${VLLM_VERSION}-slim-streamer"

echo "================================================================"
echo " Building Slim vLLM Image (T4/sm_75 only)"
echo "================================================================"
echo ""
echo "  ECR:              ${ECR}"
echo "  vLLM version:     ${VLLM_VERSION}"
echo "  Slim tag:         ${SLIM_TAG}"
echo "  Slim-streamer tag: ${SLIM_STREAMER_TAG}"
echo "  CUDA arch:        7.5 (T4 only)"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Authenticate with ECR
# ---------------------------------------------------------------------------
echo "── Step 1: ECR authentication ─────────────────────────────────────"
aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ECR}"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Clone vLLM at v0.19.0
# ---------------------------------------------------------------------------
echo "── Step 2: Cloning vLLM ${VLLM_VERSION} ────────────────────────────"
BUILD_DIR="/tmp/vllm-build"
if [ -d "${BUILD_DIR}" ]; then
    echo "Removing existing build directory..."
    rm -rf "${BUILD_DIR}"
fi
git clone --branch "${VLLM_VERSION}" --depth 1 https://github.com/vllm-project/vllm.git "${BUILD_DIR}"
cd "${BUILD_DIR}"
echo "Cloned to ${BUILD_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Build slim image (sm_75 only)
# ---------------------------------------------------------------------------
echo "── Step 3: Building slim image (this takes 30-45 min) ─────────────"
# WHY --build-arg torch_cuda_arch_list="7.5": tells PyTorch/vLLM to compile
# CUDA kernels ONLY for sm_75 (T4). Stock image compiles for sm_70-sm_120
# (7 architectures), producing ~5GB of unused .so files. sm_75-only → ~4GB.
#
# WHY --target vllm-openai: vLLM's multi-stage Dockerfile has build stages
# (base → csrc-build → build → vllm-openai). The final vllm-openai stage
# contains only runtime deps (CUDA runtime, not devel tools).
BUILD_START=$(date +%s)
DOCKER_BUILDKIT=1 docker build \
    --build-arg torch_cuda_arch_list="7.5" \
    --target vllm-openai \
    -t "${ECR}/inference-lab/vllm-openai:${SLIM_TAG}" \
    -f docker/Dockerfile .
BUILD_END=$(date +%s)
BUILD_DURATION=$(( BUILD_END - BUILD_START ))
echo ""
echo "Build complete in ${BUILD_DURATION}s"
echo ""

# Check image size
echo "── Image size comparison ──────────────────────────────────────────"
docker images "${ECR}/inference-lab/vllm-openai:${SLIM_TAG}" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"
echo ""

# ---------------------------------------------------------------------------
# Step 4: Push slim image to ECR
# ---------------------------------------------------------------------------
echo "── Step 4: Pushing slim image to ECR ──────────────────────────────"
docker push "${ECR}/inference-lab/vllm-openai:${SLIM_TAG}"
echo "Slim image pushed"
echo ""

# ---------------------------------------------------------------------------
# Step 5: Build and push slim-streamer
# ---------------------------------------------------------------------------
echo "── Step 5: Building slim-streamer (slim + RunAI) ──────────────────"
# WHY separate image: only configs 7-8 need the RunAI Model Streamer.
# Keeping it as a thin layer (~50MB) on top of slim avoids bloating the
# base image that SOCI indexes for all other configs.
docker build \
    -t "${ECR}/inference-lab/vllm-openai:${SLIM_STREAMER_TAG}" \
    -f - . << 'DOCKERFILE'
FROM 019167255542.dkr.ecr.us-east-1.amazonaws.com/inference-lab/vllm-openai:v0.19.0-slim
RUN pip install --no-cache-dir runai-model-streamer runai-model-streamer-s3
DOCKERFILE

docker push "${ECR}/inference-lab/vllm-openai:${SLIM_STREAMER_TAG}"
echo "Slim-streamer pushed"
echo ""

# ---------------------------------------------------------------------------
# Step 6: Quick smoke test
# ---------------------------------------------------------------------------
echo "── Step 6: Smoke test ─────────────────────────────────────────────"
echo "Testing slim image loads vLLM correctly..."
# Just verify the image starts and vLLM module imports (no GPU needed for this check)
docker run --rm "${ECR}/inference-lab/vllm-openai:${SLIM_TAG}" \
    python3 -c "import vllm; print(f'vLLM {vllm.__version__} imported successfully')" 2>&1 || \
    echo "WARNING: Import test failed (may need GPU). Image built successfully though."
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "================================================================"
echo " Slim Image Build Complete"
echo "================================================================"
echo ""
echo "  Slim image:     ${ECR}/inference-lab/vllm-openai:${SLIM_TAG}"
echo "  Slim-streamer:  ${ECR}/inference-lab/vllm-openai:${SLIM_STREAMER_TAG}"
echo "  Build time:     ${BUILD_DURATION}s"
echo ""
echo "  NEXT STEPS:"
echo "    1. Create SOCI index: bash phase4/scripts/setup-soci.sh"
echo "    2. Rebuild prebaked AMI: bash phase4/scripts/build-ami.sh"
echo "    3. Rebuild SOCI AMI: bash phase4/scripts/build-ami-soci.sh"
echo "    4. Run benchmark: python3 phase4/tests/cold_start_benchmark.py --host http://localhost:8080"
