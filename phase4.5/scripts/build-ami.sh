#!/usr/bin/env bash
# build-ami.sh — Build the pre-baked GPU node AMI using Packer.
#
# WHAT THIS DOES:
#   1. Validates Packer is installed
#   2. Initializes Packer plugins (Amazon EBS builder)
#   3. Builds a custom AMI based on the EKS-optimized AL2 GPU AMI
#   4. Pre-caches the vLLM container image (~8GB) in containerd
#   5. Outputs the AMI ID for use in Karpenter EC2NodeClass
#
# WHY PRE-BAKE THE AMI:
#   Phase 2 showed that container image pull takes ~90s of the 5-min cold start.
#   By caching the vLLM image on the AMI, new nodes skip the pull entirely.
#   Combined with the S3 model init container, cold start drops from ~300s to ~165s.
#
# ANALOGY: Creating a Golden AMI for an ASG. Every new instance in the group
#   starts with the application binary already on disk instead of downloading
#   it from S3/ECR on boot.
#
# USAGE:
#   bash phase4.5/scripts/build-ami.sh
#   bash phase4.5/scripts/build-ami.sh --region us-west-2
#
# PREREQUISITES:
#   - Packer installed: brew install packer (macOS) or https://packer.io/downloads
#   - AWS credentials configured (same as terraform/kubectl)
#   - The build uses a Spot g4dn.xlarge (~$0.16/hr for ~10 min = ~$0.03 per build)
#
# REBUILD WHEN:
#   - vLLM version changes (update vllm_image var in gpu-node.pkr.hcl)
#   - EKS version upgrades (update eks_version var)
#   - NOT needed when model changes (model comes from S3, not AMI)

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
REGION="${1:-us-east-1}"
PACKER_DIR="phase4.5/packer"

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Phase 4 — GPU Node AMI Build${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

# ── Pre-flight Checks ──────────────────────────────────────────────────

# Check Packer
info "Checking for Packer..."
if ! command -v packer &>/dev/null; then
    error "Packer not found. Install it:"
    echo "  macOS:  brew install packer"
    echo "  Linux:  https://developer.hashicorp.com/packer/install"
    exit 1
fi
PACKER_VERSION=$(packer version | head -1)
success "Packer found: ${PACKER_VERSION}"

# Check AWS credentials
info "Checking AWS credentials..."
if ! aws sts get-caller-identity &>/dev/null; then
    error "AWS credentials not configured. Run: aws configure"
    exit 1
fi
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
success "AWS account: ${AWS_ACCOUNT}"

# Check Packer template exists
info "Checking Packer template..."
if [ ! -f "${PACKER_DIR}/gpu-node.pkr.hcl" ]; then
    error "Packer template not found at ${PACKER_DIR}/gpu-node.pkr.hcl"
    exit 1
fi
success "Template found"
echo ""

# ── Check for Existing AMI ─────────────────────────────────────────────
# WHY check first: avoid building a duplicate AMI if one already exists.
# The AMI build costs ~$0.03 and takes ~10 min, so skipping saves time.
info "Checking for existing pre-baked AMI..."
EXISTING_AMI=$(aws ec2 describe-images \
    --owners self \
    --filters "Name=tag:Name,Values=inference-lab-gpu-node" \
              "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text \
    --region "${REGION}" 2>/dev/null || echo "None")

if [ "${EXISTING_AMI}" != "None" ] && [ "${EXISTING_AMI}" != "null" ] && [ -n "${EXISTING_AMI}" ]; then
    warn "Existing AMI found: ${EXISTING_AMI}"
    EXISTING_DATE=$(aws ec2 describe-images \
        --image-ids "${EXISTING_AMI}" \
        --query 'Images[0].CreationDate' \
        --output text \
        --region "${REGION}")
    warn "Created: ${EXISTING_DATE}"
    echo ""
    read -p "Build a new AMI anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Skipping build. Using existing AMI: ${EXISTING_AMI}"
        echo ""
        echo "To use this AMI in Karpenter, update phase4.5/k8s/karpenter/gpu-nodepool.yaml:"
        echo "  amiSelectorTerms:"
        echo "    - id: ${EXISTING_AMI}"
        exit 0
    fi
fi
echo ""

# ── Initialize Packer ──────────────────────────────────────────────────
info "Initializing Packer plugins..."
cd "${PACKER_DIR}"
packer init .
success "Packer plugins initialized"
echo ""

# ── Build AMI ──────────────────────────────────────────────────────────
info "Starting AMI build (region: ${REGION})..."
info "This takes ~10 minutes (mostly pulling the 8GB vLLM image)..."
info "Build uses a Spot g4dn.xlarge (~\$0.03 total cost)"
echo ""

# WHY -color=true: Packer's output is dense. Colors help distinguish
# provisioner output (what's happening on the instance) from build metadata.
BUILD_START=$(date +%s)
packer build -var "region=${REGION}" -color=true gpu-node.pkr.hcl
BUILD_END=$(date +%s)
BUILD_DURATION=$(( BUILD_END - BUILD_START ))

cd - >/dev/null

echo ""
success "AMI build complete in ${BUILD_DURATION}s"

# ── Find the New AMI ───────────────────────────────────────────────────
info "Finding the newly created AMI..."
NEW_AMI=$(aws ec2 describe-images \
    --owners self \
    --filters "Name=tag:Name,Values=inference-lab-gpu-node" \
              "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text \
    --region "${REGION}")

if [ -z "${NEW_AMI}" ] || [ "${NEW_AMI}" == "None" ]; then
    error "Could not find the new AMI. Check the Packer output above for errors."
    exit 1
fi

success "New AMI: ${NEW_AMI}"
echo ""

# ── Enable EBS Fast Snapshot Restore ───────────────────────────────────
# WHY FSR: EBS volumes created from snapshots use lazy loading — first access
# to each block fetches from S3 (where the snapshot is stored). For a 100GB root
# volume with 8GB of cached container images, this makes the "local" image pull
# as slow as a network pull (~107s measured). FSR pre-initializes the volume so
# ALL blocks are immediately available on launch — image pull drops to ~0s.
#
# COST: $0.75/hr per AZ while enabled. Disable when not actively testing.
# ANALOGY: like EBS pre-warming (dd if=/dev/xvda) but automated by AWS.
info "Enabling EBS Fast Snapshot Restore (FSR)..."
info "This eliminates lazy loading so container images are instantly available."
info "Cost: \$0.75/hr per AZ while enabled. Disable when not testing."

# Find the AMI's root volume snapshot
AMI_SNAP=$(aws ec2 describe-images --image-ids "${NEW_AMI}" \
    --query 'Images[0].BlockDeviceMappings[0].Ebs.SnapshotId' \
    --output text --region "${REGION}")

if [ -z "${AMI_SNAP}" ] || [ "${AMI_SNAP}" == "None" ]; then
    warn "Could not find AMI root snapshot. Enable FSR manually."
else
    info "AMI root snapshot: ${AMI_SNAP}"
    aws ec2 enable-fast-snapshot-restores \
        --availability-zones us-east-1a us-east-1b \
        --source-snapshot-ids "${AMI_SNAP}" \
        --region "${REGION}" 2>&1 | head -5
    success "FSR enabling on us-east-1a and us-east-1b"
    info "FSR takes 5-15 min to fully enable. Check with:"
    echo "  aws ec2 describe-fast-snapshot-restores --filters Name=snapshot-id,Values=${AMI_SNAP}"
    echo ""
    info "To disable FSR when done testing (stops the \$0.75/hr/AZ charge):"
    echo "  aws ec2 disable-fast-snapshot-restores --availability-zones us-east-1a us-east-1b --source-snapshot-ids ${AMI_SNAP}"
fi

echo ""

# ── Print Next Steps ───────────────────────────────────────────────────
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  AMI Build Complete${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo "  AMI ID:       ${NEW_AMI}"
echo "  Snapshot:     ${AMI_SNAP}"
echo "  FSR:          enabling (wait 5-15 min before testing)"
echo "  Region:       ${REGION}"
echo "  Build time:   ${BUILD_DURATION}s"
echo ""
echo "  The Karpenter EC2NodeClass (gpu-nodepool.yaml) is already configured"
echo "  to discover this AMI via the 'inference-lab-gpu-node' tag."
echo ""
echo "  To apply the updated NodePool:"
echo "    kubectl apply -f phase4.5/k8s/karpenter/gpu-nodepool.yaml"
echo ""
echo "  Expected cold start improvement (with FSR enabled):"
echo "    Phase 2 baseline:          ~300s"
echo "    + S3 init container:       ~255s  (saves ~45s model download)"
echo "    + Pre-baked AMI + FSR:     ~165s  (saves ~90s image pull)"
echo "    Total savings:             ~135s  (45% faster)"
echo ""
