#!/usr/bin/env bash
# build-ami.sh — Build a pre-baked GPU node AMI using Packer.
#
# WHAT THIS DOES:
#   1. Validates Packer is installed
#   2. Initializes Packer plugins (Amazon EBS builder)
#   3. Builds a custom AMI based on the EKS-optimized AL2 GPU AMI
#   4. Pre-caches the vLLM container image (~4-9.5GB) in containerd
#   5. Outputs the AMI ID for use in Karpenter EC2NodeClass
#   6. (optional, --enable-fsr) enables EBS Fast Snapshot Restore
#
# TEMPLATES SUPPORTED:
#   gpu-node.pkr.hcl       — slim 4GB image, sm_75-only (Phase 6 baseline)
#   gpu-node-stock.pkr.hcl — stock 9.5GB image (Phase 4.1 pull comparison)
#   gpu-node-soci.pkr.hcl  — SOCI lazy snapshotter (Phase 6+)
#
# Each template uses a distinct AMI Name tag (inference-lab-gpu-node,
# inference-lab-gpu-node-stock, inference-lab-gpu-node-soci) so they coexist
# in the AWS account without collision.
#
# USAGE:
#   bash phase4.1/scripts/build-ami.sh                                       # default: slim
#   bash phase4.1/scripts/build-ami.sh --template gpu-node-stock.pkr.hcl     # stock 9.5GB
#   bash phase4.1/scripts/build-ami.sh --region us-west-2                    # different region
#   bash phase4.1/scripts/build-ami.sh --enable-fsr                          # opt into FSR
#   bash phase4.1/scripts/build-ami.sh --force                               # don't prompt on existing AMI
#
# WHY FSR IS OPT-IN NOW (changed from prior version):
#   Phase 4.1's pull-comparison experiment deliberately measures cold-start
#   WITHOUT FSR first, to surface the EBS S3-lazy-load tax. Auto-enabling FSR
#   would erase that data point. See phase4.1/LEARNINGS.md.
#
# COST:
#   Build: ~$0.03-0.05 on Spot g4dn.xlarge for ~5-10 minutes
#   FSR (if enabled): $0.75/hr/AZ while active

set -euo pipefail

# ── Color Output ────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Parse Arguments ────────────────────────────────────────────────────
REGION="us-east-1"
TEMPLATE="gpu-node.pkr.hcl"
ENABLE_FSR="no"
FORCE="no"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --template)
            TEMPLATE="$2"; shift 2 ;;
        --region)
            REGION="$2"; shift 2 ;;
        --enable-fsr)
            ENABLE_FSR="yes"; shift ;;
        --force)
            FORCE="yes"; shift ;;
        --help|-h)
            grep -E '^#( |$)' "$0" | sed 's/^# //; s/^#//'
            exit 0 ;;
        *)
            error "unknown arg: $1"; exit 2 ;;
    esac
done

# Resolve PACKER_DIR relative to this script — works regardless of CWD.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PACKER_DIR="${PHASE_DIR}/packer"

# Map each template to its AMI Name tag and packer working dir.
# Each variant lives in its own subdir so packer init/build doesn't merge
# duplicate variable/data blocks across templates.
case "${TEMPLATE}" in
    gpu-node.pkr.hcl)
        AMI_NAME_TAG="inference-lab-gpu-node"
        TEMPLATE_DIR="${PACKER_DIR}" ;;
    gpu-node-stock.pkr.hcl)
        AMI_NAME_TAG="inference-lab-gpu-node-stock"
        TEMPLATE_DIR="${PACKER_DIR}/stock" ;;
    gpu-node-stock-warmed.pkr.hcl)
        AMI_NAME_TAG="inference-lab-gpu-node-stock-warmed"
        TEMPLATE_DIR="${PACKER_DIR}/stock-warmed" ;;
    gpu-node-streamer.pkr.hcl)
        AMI_NAME_TAG="inference-lab-gpu-node-streamer"
        TEMPLATE_DIR="${PACKER_DIR}/streamer" ;;
    gpu-node-model-baked.pkr.hcl)
        AMI_NAME_TAG="inference-lab-gpu-node-model-baked"
        TEMPLATE_DIR="${PACKER_DIR}/model-baked" ;;
    gpu-node-lustre.pkr.hcl)
        AMI_NAME_TAG="inference-lab-gpu-node-lustre"
        TEMPLATE_DIR="${PACKER_DIR}/lustre-client" ;;
    gpu-node-soci.pkr.hcl)
        AMI_NAME_TAG="inference-lab-gpu-node-soci"
        TEMPLATE_DIR="${PACKER_DIR}" ;;
    *)
        warn "Unknown template ${TEMPLATE} — falling back to inference-lab-gpu-node tag filter."
        AMI_NAME_TAG="inference-lab-gpu-node"
        TEMPLATE_DIR="${PACKER_DIR}" ;;
esac

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Phase 4.1 — GPU Node AMI Build${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
info "Template:    ${TEMPLATE}"
info "Region:      ${REGION}"
info "AMI tag:     ${AMI_NAME_TAG}"
info "FSR:         ${ENABLE_FSR}"
echo ""

# ── Pre-flight Checks ──────────────────────────────────────────────────
info "Checking for Packer..."
if ! command -v packer &>/dev/null; then
    error "Packer not found. Install: brew install packer (macOS) or https://developer.hashicorp.com/packer/install"
    exit 1
fi
success "Packer: $(packer version | head -1)"

info "Checking AWS credentials..."
if ! aws sts get-caller-identity &>/dev/null; then
    error "AWS credentials not configured. Run: aws configure"
    exit 1
fi
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
success "AWS account: ${AWS_ACCOUNT}"

info "Checking template..."
if [ ! -f "${TEMPLATE_DIR}/${TEMPLATE}" ]; then
    error "Template not found: ${TEMPLATE_DIR}/${TEMPLATE}"
    exit 1
fi
success "Template found: ${TEMPLATE_DIR}/${TEMPLATE}"
echo ""

# ── Check for Existing AMI ─────────────────────────────────────────────
info "Checking for existing AMI with tag Name=${AMI_NAME_TAG}..."
EXISTING_AMI=$(aws ec2 describe-images \
    --owners self \
    --filters "Name=tag:Name,Values=${AMI_NAME_TAG}" \
              "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text \
    --region "${REGION}" 2>/dev/null || echo "None")

if [ "${EXISTING_AMI}" != "None" ] && [ "${EXISTING_AMI}" != "null" ] && [ -n "${EXISTING_AMI}" ]; then
    EXISTING_DATE=$(aws ec2 describe-images --image-ids "${EXISTING_AMI}" \
        --query 'Images[0].CreationDate' --output text --region "${REGION}")
    warn "Existing AMI: ${EXISTING_AMI} (created ${EXISTING_DATE})"
    if [[ "${FORCE}" == "yes" ]]; then
        info "--force given; rebuilding anyway"
    else
        echo ""
        read -p "Build a new AMI anyway? (y/N) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            info "Skipping build. Using existing AMI: ${EXISTING_AMI}"
            exit 0
        fi
    fi
fi
echo ""

# ── Initialize Packer ──────────────────────────────────────────────────
info "Initializing Packer plugins (dir: ${TEMPLATE_DIR})..."
cd "${TEMPLATE_DIR}"
packer init .
success "Packer plugins initialized"
echo ""

# ── Build AMI ──────────────────────────────────────────────────────────
info "Starting AMI build..."
BUILD_START=$(date +%s)
packer build -var "region=${REGION}" -color=true "${TEMPLATE}"
BUILD_END=$(date +%s)
BUILD_DURATION=$(( BUILD_END - BUILD_START ))
cd - >/dev/null

echo ""
success "AMI build complete in ${BUILD_DURATION}s"

# ── Find the New AMI ───────────────────────────────────────────────────
info "Locating the newly created AMI (tag Name=${AMI_NAME_TAG})..."
NEW_AMI=$(aws ec2 describe-images \
    --owners self \
    --filters "Name=tag:Name,Values=${AMI_NAME_TAG}" \
              "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text \
    --region "${REGION}")

if [ -z "${NEW_AMI}" ] || [ "${NEW_AMI}" == "None" ]; then
    error "Could not find the new AMI."
    exit 1
fi
success "New AMI: ${NEW_AMI}"

AMI_SNAP=$(aws ec2 describe-images --image-ids "${NEW_AMI}" \
    --query 'Images[0].BlockDeviceMappings[0].Ebs.SnapshotId' \
    --output text --region "${REGION}")
info "Root snapshot: ${AMI_SNAP}"
echo ""

# ── Optional: Enable FSR ───────────────────────────────────────────────
if [[ "${ENABLE_FSR}" == "yes" ]]; then
    info "Enabling EBS Fast Snapshot Restore (FSR)..."
    info "Cost: \$0.75/hr per AZ. Disable with:"
    echo "  aws ec2 disable-fast-snapshot-restores --availability-zones us-east-1a --source-snapshot-ids ${AMI_SNAP}"
    aws ec2 enable-fast-snapshot-restores \
        --availability-zones us-east-1a \
        --source-snapshot-ids "${AMI_SNAP}" \
        --region "${REGION}" 2>&1 | head -5
    success "FSR enabling on us-east-1a (5-15 min to be ready)"
else
    info "FSR not enabled (use --enable-fsr to opt in)."
    info "For Phase 4.1 pull-comparison: keep FSR off for the first variant-C measurement"
    info "to surface the EBS S3-lazy-load tax. See phase4.1/LEARNINGS.md."
fi

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  AMI Build Complete${NC}"
echo -e "${GREEN}================================================================${NC}"
echo "  AMI ID:       ${NEW_AMI}"
echo "  Snapshot:     ${AMI_SNAP}"
echo "  Region:       ${REGION}"
echo "  Build time:   ${BUILD_DURATION}s"
echo "  FSR:          ${ENABLE_FSR}"
echo ""
echo "  Apply the matching NodePool to use this AMI:"
case "${TEMPLATE}" in
    gpu-node-stock.pkr.hcl)
        echo "    kubectl apply -f phase4.1/k8s/karpenter/gpu-nodepool-prebaked.yaml" ;;
    *)
        echo "    kubectl apply -f phase4.1/k8s/karpenter/gpu-nodepool.yaml" ;;
esac
echo ""
