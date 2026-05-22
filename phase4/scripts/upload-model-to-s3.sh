#!/usr/bin/env bash
# upload-model-to-s3.sh — One-time prerequisite: download model from HuggingFace
# and upload to S3 for fast cold starts.
#
# WHY THIS SCRIPT:
#   Phase 4's init container downloads model weights from S3 instead of HuggingFace.
#   S3 same-region transfer: ~15-20s. HuggingFace download: ~60s+ (varies by CDN).
#   This script prepopulates the S3 bucket so the init container has something to pull.
#   It's a one-time operation — run once, then every pod cold start benefits.
#
#   Analogy: pre-baking an AMI with your application code vs downloading it from
#   GitHub on every instance boot. The S3 model cache is the "golden AMI" for
#   model weights.
#
# PREREQUISITES:
#   - AWS CLI v2 configured with credentials (aws configure)
#   - huggingface-cli installed (pip install huggingface-hub)
#   - HF_TOKEN env var set if model is gated (Qwen2.5 is public, so optional)
#   - S3 bucket must already exist (this script does NOT create buckets)
#
# USAGE:
#   bash phase4/scripts/upload-model-to-s3.sh                                     # defaults
#   bash phase4/scripts/upload-model-to-s3.sh my-custom-bucket                    # custom bucket
#   MODEL_NAME="meta-llama/Llama-3-8B" bash phase4/scripts/upload-model-to-s3.sh  # custom model
#
# COST: S3 storage for Qwen2.5-7B-AWQ is ~4.5GB. At S3 Standard pricing ($0.023/GB/mo)
# that's ~$0.10/month. INTELLIGENT_TIERING auto-archives if unused for 90 days.

set -euo pipefail

# ── Color Output ─────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Configuration ────────────────────────────────────────────────────────
# WHY env vars with defaults: allows overriding for different models or buckets
# without editing the script. Same pattern as passing parameters to a
# CloudFormation stack via --parameter-overrides.
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
BUCKET="${1:-${BUCKET:-inference-lab-model-cache}}"

# S3 key prefix — mirrors the HuggingFace org/repo structure so the init
# container can find it at a predictable path.
S3_PREFIX="${MODEL_NAME}"

# Local staging directory for the download.
# WHY /tmp: ephemeral, cleaned up at the end. The model is ~4.5GB — make sure
# /tmp has enough space (most EC2 instances and macOS have plenty).
LOCAL_DIR="/tmp/inference-lab-model-cache"

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Upload Model to S3${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
info "Model:  ${MODEL_NAME}"
info "Bucket: s3://${BUCKET}/${S3_PREFIX}/"
info "Local:  ${LOCAL_DIR}"
echo ""

# ── Step 1: Check Prerequisites ─────────────────────────────────────────

# Check AWS CLI — required for S3 upload.
# WHY check explicitly: a missing aws CLI produces a cryptic "command not found"
# error deep in the script. Better to fail with a clear message upfront.
info "Checking prerequisites..."

if ! command -v aws &>/dev/null; then
    error "AWS CLI not found. Install it:"
    echo "  macOS:  brew install awscli"
    echo "  Linux:  curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o awscliv2.zip && unzip awscliv2.zip && sudo ./aws/install"
    exit 1
fi
success "AWS CLI found: $(aws --version 2>&1 | head -1)"

# Verify AWS credentials are configured and valid.
# WHY: aws s3 sync with expired/missing credentials fails with a confusing XML error.
if ! aws sts get-caller-identity &>/dev/null; then
    error "AWS credentials not configured or expired."
    echo "  Run: aws configure"
    echo "  Or:  export AWS_PROFILE=your-profile"
    exit 1
fi
success "AWS credentials valid: $(aws sts get-caller-identity --query Arn --output text)"

# Check huggingface-cli — required for model download.
# WHY huggingface-cli (not git clone): the HF CLI handles large file downloads
# with resume support, progress bars, and proper caching. Git LFS requires
# git-lfs installed and doesn't resume partial downloads.
if ! command -v huggingface-cli &>/dev/null; then
    warn "huggingface-cli not found. Installing huggingface-hub..."
    pip install --quiet huggingface-hub
    if ! command -v huggingface-cli &>/dev/null; then
        error "Failed to install huggingface-hub. Install manually:"
        echo "  pip install huggingface-hub"
        exit 1
    fi
    success "huggingface-hub installed"
else
    success "huggingface-cli found"
fi
echo ""

# ── Step 2: Check If Model Already Exists in S3 ─────────────────────────
# WHY check first: uploading 4.5GB takes ~2 min even on fast networks. If the
# model is already there, skip the upload entirely. Like checking if an AMI
# already exists before building a new one.
info "Checking if model already exists in S3..."

# Count objects in the S3 prefix. If >0, model is already uploaded.
# WHY --summarize: gives us TotalObjects count without downloading anything.
S3_OBJECT_COUNT=$(aws s3 ls "s3://${BUCKET}/${S3_PREFIX}/" --recursive --summarize 2>/dev/null \
    | grep "Total Objects:" | awk '{print $3}' || echo "0")

if [ "${S3_OBJECT_COUNT}" -gt 5 ]; then
    # WHY >5 (not >0): a partial upload might leave 1-2 files. A complete Qwen2.5
    # upload has ~20+ files (config, tokenizer, weights shards, etc.).
    success "Model already exists in S3 (${S3_OBJECT_COUNT} objects found)"
    echo ""
    info "S3 contents:"
    aws s3 ls "s3://${BUCKET}/${S3_PREFIX}/" --summarize --human-readable 2>/dev/null | tail -5
    echo ""
    echo "To force re-upload, delete the S3 prefix first:"
    echo "  aws s3 rm s3://${BUCKET}/${S3_PREFIX}/ --recursive"
    echo "  bash phase4/scripts/upload-model-to-s3.sh"
    exit 0
fi
info "Model not found in S3 (or partial upload detected). Proceeding with upload."
echo ""

# ── Step 3: Download Model from HuggingFace ─────────────────────────────
# WHY download to local first (not stream to S3): huggingface-cli doesn't
# support piping to S3 directly. We download to /tmp, then aws s3 sync uploads.
# The two-step process also lets us verify the download before uploading.
info "Downloading model from HuggingFace: ${MODEL_NAME}..."
info "This may take 5-15 minutes depending on your connection speed."
echo ""

# Clean up any partial previous download to avoid stale files.
# WHY rm -rf first: huggingface-cli's --local-dir may not overwrite partial files
# correctly. Starting fresh ensures a clean download.
rm -rf "${LOCAL_DIR}"
mkdir -p "${LOCAL_DIR}"

DOWNLOAD_START=$(date +%s)

huggingface-cli download "${MODEL_NAME}" \
    --local-dir "${LOCAL_DIR}" \
    --local-dir-use-symlinks False
    # WHY --local-dir-use-symlinks False: we need real files for S3 upload.
    # By default, HF CLI creates symlinks to its cache dir — aws s3 sync
    # would upload the symlinks (0 bytes) instead of the actual weight files.

DOWNLOAD_END=$(date +%s)
DOWNLOAD_DURATION=$(( DOWNLOAD_END - DOWNLOAD_START ))

success "Download complete in ${DOWNLOAD_DURATION}s"
info "Local model size: $(du -sh "${LOCAL_DIR}" | cut -f1)"
echo ""

# ── Step 4: Sync to S3 ──────────────────────────────────────────────────
# WHY s3 sync (not s3 cp): sync is idempotent — if interrupted, re-running
# only uploads the missing/changed files. Like rsync for S3.
# WHY INTELLIGENT_TIERING: if the model isn't accessed for 90 days (between
# experiment sessions), S3 auto-archives to cheaper storage. No cost if actively
# used. Perfect for a learning lab with intermittent usage.
info "Uploading to S3: s3://${BUCKET}/${S3_PREFIX}/..."
info "Using INTELLIGENT_TIERING storage class for cost optimization."
echo ""

UPLOAD_START=$(date +%s)

aws s3 sync \
    "${LOCAL_DIR}" \
    "s3://${BUCKET}/${S3_PREFIX}/" \
    --storage-class INTELLIGENT_TIERING \
    --no-progress
    # WHY --no-progress: the per-file progress bars from aws s3 sync are noisy
    # in CI/terminal logs. We show overall timing instead.

UPLOAD_END=$(date +%s)
UPLOAD_DURATION=$(( UPLOAD_END - UPLOAD_START ))

success "Upload complete in ${UPLOAD_DURATION}s"
echo ""

# ── Step 5: Verify Upload ───────────────────────────────────────────────
# WHY verify: "trust but verify" — confirm the upload actually landed in S3
# and the object count matches what we downloaded. A partial upload would cause
# the init container to fail silently (missing weight shards = vLLM crashes on load).
info "Verifying S3 contents..."
echo ""
aws s3 ls "s3://${BUCKET}/${S3_PREFIX}/" --summarize --human-readable
echo ""

# Compare local file count with S3 object count as a sanity check.
LOCAL_COUNT=$(find "${LOCAL_DIR}" -type f | wc -l | tr -d ' ')
S3_COUNT=$(aws s3 ls "s3://${BUCKET}/${S3_PREFIX}/" --recursive --summarize \
    | grep "Total Objects:" | awk '{print $3}')
if [ "${LOCAL_COUNT}" -eq "${S3_COUNT}" ]; then
    success "File count matches: ${LOCAL_COUNT} local = ${S3_COUNT} in S3"
else
    warn "File count mismatch: ${LOCAL_COUNT} local vs ${S3_COUNT} in S3"
    warn "Some files may have been filtered. Check the S3 prefix manually."
fi
echo ""

# ── Step 6: Clean Up ────────────────────────────────────────────────────
# WHY clean up: the model is ~4.5GB. Leaving it in /tmp wastes disk space,
# especially on a laptop with limited SSD. The canonical copy is now in S3.
info "Cleaning up local temp files (${LOCAL_DIR})..."
rm -rf "${LOCAL_DIR}"
success "Local cache cleaned"
echo ""

# ── Summary ──────────────────────────────────────────────────────────────
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Upload Complete${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo "  Model:    ${MODEL_NAME}"
echo "  S3 Path:  s3://${BUCKET}/${S3_PREFIX}/"
echo "  Download: ${DOWNLOAD_DURATION}s"
echo "  Upload:   ${UPLOAD_DURATION}s"
echo ""
echo "  The Phase 4 init container will pull from this S3 path on cold start."
echo "  Make sure the Deployment's MODEL_CACHE_BUCKET matches: ${BUCKET}"
echo ""
echo "  Verify the init container config:"
echo "    grep MODEL_CACHE_BUCKET phase4/k8s/worker/deployment-patch.yaml"
echo ""
