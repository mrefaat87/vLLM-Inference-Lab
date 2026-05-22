#!/usr/bin/env bash
# build-prerequisites.sh — Build ALL cold start benchmark prerequisites.
#
# This orchestrates the 3 prerequisites needed before running the benchmark:
#   1. Pre-baked AMI (Packer — runs FROM your Mac, builds ON a Spot instance)
#   2. SOCI setup + Streamer image (needs Linux x86_64 — runs ON an EC2 instance)
#   3. SOCI AMI (Packer — runs FROM your Mac, builds ON a Spot instance)
#
# WHY A SINGLE SCRIPT:
#   The SOCI setup (push image to ECR, create SOCI index) and streamer Docker
#   build both require Linux x86_64. You can't run them on Apple Silicon. This
#   script launches a temporary EC2 instance that handles both, then continues
#   with the Packer AMI builds from your Mac.
#
# SAFETY:
#   - Only touches inference-phase4 resources (ECR, AMIs, EBS snapshots)
#   - Does NOT touch any EKS clusters (no kubectl)
#   - Does NOT touch Phase 3 resources
#   - The EC2 builder instance uses the Phase 4 Karpenter node role for ECR access
#
# COST:
#   - EC2 builder: t3.xlarge Spot ~$0.04/hr × 20 min = ~$0.02
#   - Pre-baked AMI build: g4dn.xlarge Spot ~$0.16/hr × 10 min = ~$0.03
#   - SOCI AMI build: g4dn.xlarge Spot ~$0.16/hr × 10 min = ~$0.03
#   - FSR: $0.75/hr per AZ (disable when not testing)
#   - Total build cost: ~$0.08 (excluding FSR)
#
# USAGE:
#   bash phase4/scripts/build-prerequisites.sh
#   bash phase4/scripts/build-prerequisites.sh --skip-ami      # skip pre-baked AMI
#   bash phase4/scripts/build-prerequisites.sh --skip-soci-ami # skip SOCI AMI
#   bash phase4/scripts/build-prerequisites.sh --only-ec2      # only run SOCI+streamer on EC2
#
# PREREQUISITES:
#   - AWS CLI configured
#   - Packer installed (brew install packer) — for AMI builds
#   - jq installed (brew install jq) — for JSON parsing
#
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Image versions (v0.19.0 is the Phase 4 version)
VLLM_VERSION="v0.19.0"
VLLM_DOCKERHUB="vllm/vllm-openai:${VLLM_VERSION}"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
ECR_VLLM="${ECR_REGISTRY}/inference-lab/vllm-openai:${VLLM_VERSION}"
ECR_STREAMER="${ECR_REGISTRY}/inference-lab/vllm-openai:${VLLM_VERSION}-streamer"
SOCI_VERSION="0.7.0"

# EC2 builder config
BUILDER_INSTANCE_TYPE="t3.xlarge"  # 4 vCPU, 16GB — enough for Docker build + SOCI
BUILDER_IAM_PROFILE="inference-phase4-karpenter-node"  # has ECR push permissions

# ── Color Output ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Parse Arguments ────────────────────────────────────────────────────────
SKIP_AMI=false
SKIP_SOCI_AMI=false
ONLY_EC2=false
for arg in "$@"; do
    case $arg in
        --skip-ami)      SKIP_AMI=true ;;
        --skip-soci-ami) SKIP_SOCI_AMI=true ;;
        --only-ec2)      ONLY_EC2=true ;;
        --help|-h)
            echo "Usage: bash phase4/scripts/build-prerequisites.sh [OPTIONS]"
            echo "  --skip-ami       Skip pre-baked AMI build (Packer)"
            echo "  --skip-soci-ami  Skip SOCI AMI build (Packer)"
            echo "  --only-ec2       Only run SOCI+streamer EC2 builder"
            exit 0
            ;;
        *) error "Unknown argument: $arg"; exit 1 ;;
    esac
done

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Phase 4 — Build All Benchmark Prerequisites${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo "  Region:          ${REGION}"
echo "  Account:         ${ACCOUNT_ID}"
echo "  vLLM version:    ${VLLM_VERSION}"
echo "  SOCI version:    ${SOCI_VERSION}"
echo ""

# ── Safety Check ──────────────────────────────────────────────────────────
# WHY: ensure we're not accidentally using Phase 3 resources.
# This script doesn't use kubectl, so there's no cluster context to check.
# Instead, verify the IAM instance profile exists in Phase 4.
info "Verifying Phase 4 IAM resources..."
if ! aws iam get-instance-profile --instance-profile-name "${BUILDER_IAM_PROFILE}" &>/dev/null; then
    error "IAM instance profile '${BUILDER_IAM_PROFILE}' not found."
    error "This means Phase 4 Terraform hasn't been applied, or the naming is wrong."
    error "Expected: inference-phase4-karpenter-node (from Phase 4 Terraform)."
    exit 1
fi
success "Phase 4 IAM profile exists: ${BUILDER_IAM_PROFILE}"

# ── Step 1: Ensure ECR Repository Exists ──────────────────────────────────
info "Checking ECR repository..."
if aws ecr describe-repositories --repository-names "inference-lab/vllm-openai" --region "${REGION}" &>/dev/null; then
    success "ECR repository inference-lab/vllm-openai exists"
else
    info "Creating ECR repository..."
    aws ecr create-repository \
        --repository-name "inference-lab/vllm-openai" \
        --region "${REGION}" \
        --image-tag-mutability MUTABLE \
        --image-scanning-configuration scanOnPush=true
    success "ECR repository created"
fi

# Also ensure the streamer image can be pushed (same repo, different tag)
echo ""

# ── Step 2: Launch EC2 Builder for SOCI + Streamer ────────────────────────
# WHY EC2: SOCI CLI is Linux-only. Docker buildx cross-compilation of the
# 9.5GB vLLM image would be painfully slow on Apple Silicon. A t3.xlarge
# Spot instance handles this in ~15 min for ~$0.02.
echo "================================================================"
echo " Step 2: SOCI Index + Streamer Image (EC2 Builder)"
echo "================================================================"
echo ""

# Find the latest Amazon Linux 2 AMI for the builder
BUILDER_AMI=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
              "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text \
    --region "${REGION}")
info "Builder AMI: ${BUILDER_AMI}"

# Find a subnet tagged for Karpenter (reuse Phase 4 VPC)
SUBNET_ID=$(aws ec2 describe-subnets \
    --filters "Name=tag:karpenter.sh/discovery,Values=inference-phase4" \
    --query 'Subnets[0].SubnetId' \
    --output text \
    --region "${REGION}")

if [ -z "${SUBNET_ID}" ] || [ "${SUBNET_ID}" == "None" ]; then
    error "No subnet found with tag karpenter.sh/discovery=inference-phase4"
    error "Phase 4 VPC may not be set up. Run Terraform first."
    exit 1
fi
info "Using subnet: ${SUBNET_ID}"

# Find a security group in the Phase 4 VPC
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=tag:karpenter.sh/discovery,Values=inference-phase4" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --region "${REGION}")
info "Using security group: ${SG_ID}"

# Generate the userdata script that runs on the builder instance
# WHY inline: the builder instance is ephemeral. No SSH, no monitoring — it
# does its job and signals completion via an SSM parameter.
USERDATA=$(cat <<'USERDATAEOF'
#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/benchmark-prereq-build.log) 2>&1

echo "=== Phase 4 Benchmark Prerequisites Builder ==="
echo "Started at: $(date -u)"

REGION="__REGION__"
ACCOUNT_ID="__ACCOUNT_ID__"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
VLLM_VERSION="__VLLM_VERSION__"
SOCI_VERSION="__SOCI_VERSION__"
ECR_VLLM="${ECR_REGISTRY}/inference-lab/vllm-openai:${VLLM_VERSION}"
ECR_STREAMER="${ECR_REGISTRY}/inference-lab/vllm-openai:${VLLM_VERSION}-streamer"

# Install Docker
echo "── Installing Docker ──"
amazon-linux-extras install docker -y
systemctl start docker
systemctl enable docker

# Authenticate with ECR
echo "── Authenticating with ECR ──"
aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# ── Task 1: Push vLLM image to ECR ──
echo ""
echo "══ Task 1: Push vLLM v0.19.0 to ECR ══"
# Check if already pushed
EXISTING=$(aws ecr describe-images \
    --repository-name "inference-lab/vllm-openai" \
    --image-ids imageTag="${VLLM_VERSION}" \
    --query 'imageDetails[0].imageDigest' \
    --output text \
    --region "${REGION}" 2>/dev/null || echo "NOT_FOUND")

if [ "${EXISTING}" != "NOT_FOUND" ] && [ "${EXISTING}" != "None" ]; then
    echo "vLLM ${VLLM_VERSION} already in ECR: ${EXISTING}"
else
    echo "Pulling ${VLLM_VERSION} from Docker Hub (~9.5GB)..."
    docker pull "vllm/vllm-openai:${VLLM_VERSION}"
    docker tag "vllm/vllm-openai:${VLLM_VERSION}" "${ECR_VLLM}"
    echo "Pushing to ECR..."
    docker push "${ECR_VLLM}"
    echo "Push complete"
fi

# ── Task 2: Create and push SOCI index ──
echo ""
echo "══ Task 2: Create SOCI Index ══"
echo "Installing SOCI CLI v${SOCI_VERSION}..."
cd /tmp
curl -fsSL "https://github.com/awslabs/soci-snapshotter/releases/download/v${SOCI_VERSION}/soci-snapshotter-${SOCI_VERSION}-linux-amd64.tar.gz" -o soci.tar.gz
tar -xzf soci.tar.gz -C /usr/local/bin/ soci soci-snapshotter-grpc 2>/dev/null || tar -xzf soci.tar.gz -C /usr/local/bin/
chmod +x /usr/local/bin/soci /usr/local/bin/soci-snapshotter-grpc 2>/dev/null || true
rm soci.tar.gz

echo "Creating SOCI index for ${ECR_VLLM}..."
echo "This reads layers from ECR and builds the file-to-offset table (~5 min)."
# SOCI needs containerd running to access the image content store
# Use the --from-registry flag to work directly against ECR
soci create "${ECR_VLLM}" --min-layer-size 10485760 || {
    echo "WARNING: soci create failed. This may need containerd running."
    echo "Trying alternative: pull image into containerd first..."

    # Start containerd if available
    if command -v containerd &>/dev/null; then
        containerd &
        sleep 3
        ECR_PASS=$(aws ecr get-login-password --region "${REGION}")
        echo "${ECR_PASS}" | ctr -n k8s.io images pull --user AWS "${ECR_VLLM}" || true
        soci create "${ECR_VLLM}" --min-layer-size 10485760
    else
        echo "ERROR: containerd not available. SOCI index creation requires containerd."
        echo "The SOCI AMI build may fail without the index."
    fi
}

echo "Pushing SOCI index to ECR..."
soci push "${ECR_VLLM}" || echo "WARNING: SOCI push failed — index may need manual creation"

# ── Task 3: Build and push vLLM Streamer image ──
echo ""
echo "══ Task 3: Build vLLM Streamer Image ══"
# Check if already pushed
STREAMER_EXISTING=$(aws ecr describe-images \
    --repository-name "inference-lab/vllm-openai" \
    --image-ids imageTag="${VLLM_VERSION}-streamer" \
    --query 'imageDetails[0].imageDigest' \
    --output text \
    --region "${REGION}" 2>/dev/null || echo "NOT_FOUND")

if [ "${STREAMER_EXISTING}" != "NOT_FOUND" ] && [ "${STREAMER_EXISTING}" != "None" ]; then
    echo "Streamer image already in ECR: ${STREAMER_EXISTING}"
else
    echo "Building streamer image..."
    mkdir -p /tmp/vllm-streamer
    cat > /tmp/vllm-streamer/Dockerfile << 'DEOF'
FROM vllm/vllm-openai:v0.19.0
RUN pip install --no-cache-dir runai-model-streamer runai-model-streamer-s3
DEOF
    docker build -t "${ECR_STREAMER}" /tmp/vllm-streamer/
    echo "Pushing streamer image to ECR..."
    docker push "${ECR_STREAMER}"
    echo "Streamer image pushed"
fi

# ── Signal Completion ──
echo ""
echo "══ All Tasks Complete ══"
echo "  vLLM in ECR:     ${ECR_VLLM}"
echo "  SOCI index:      pushed (or attempted)"
echo "  Streamer in ECR: ${ECR_STREAMER}"
echo ""
echo "Completed at: $(date -u)"

# Write completion marker to SSM Parameter Store
aws ssm put-parameter \
    --name "/inference-phase4/prereq-build-status" \
    --value "COMPLETE:$(date -u +%Y%m%dT%H%M%SZ)" \
    --type String \
    --overwrite \
    --region "${REGION}" 2>/dev/null || echo "WARNING: Could not write SSM parameter"

# Self-terminate after 2 min (time for logs to flush)
echo "Builder will self-terminate in 120s..."
sleep 120
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}" 2>/dev/null || true
USERDATAEOF
)

# Replace placeholders in userdata
USERDATA="${USERDATA//__REGION__/${REGION}}"
USERDATA="${USERDATA//__ACCOUNT_ID__/${ACCOUNT_ID}}"
USERDATA="${USERDATA//__VLLM_VERSION__/${VLLM_VERSION}}"
USERDATA="${USERDATA//__SOCI_VERSION__/${SOCI_VERSION}}"

# Base64 encode for EC2 launch
USERDATA_B64=$(echo "${USERDATA}" | base64)

# Clear any previous completion marker
aws ssm put-parameter \
    --name "/inference-phase4/prereq-build-status" \
    --value "RUNNING:$(date -u +%Y%m%dT%H%M%SZ)" \
    --type String \
    --overwrite \
    --region "${REGION}" 2>/dev/null || \
aws ssm put-parameter \
    --name "/inference-phase4/prereq-build-status" \
    --value "RUNNING:$(date -u +%Y%m%dT%H%M%SZ)" \
    --type String \
    --region "${REGION}" 2>/dev/null || true

info "Launching EC2 builder instance..."
info "Instance type: ${BUILDER_INSTANCE_TYPE} (Spot)"
info "This will: push vLLM to ECR, create SOCI index, build streamer image"
info "Estimated time: ~15-20 minutes"
info "Estimated cost: ~\$0.02 (Spot)"
echo ""

# Request Spot instance
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "${BUILDER_AMI}" \
    --instance-type "${BUILDER_INSTANCE_TYPE}" \
    --iam-instance-profile "Name=${BUILDER_IAM_PROFILE}" \
    --subnet-id "${SUBNET_ID}" \
    --security-group-ids "${SG_ID}" \
    --user-data "${USERDATA_B64}" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":120,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=phase4-prereq-builder},{Key=Project,Value=inference-lab},{Key=Purpose,Value=benchmark-prerequisites}]" \
    --query 'Instances[0].InstanceId' \
    --output text \
    --region "${REGION}")

if [ -z "${INSTANCE_ID}" ] || [ "${INSTANCE_ID}" == "None" ]; then
    error "Failed to launch EC2 builder instance"
    exit 1
fi

success "EC2 builder launched: ${INSTANCE_ID}"
info "View logs: aws ssm start-session --target ${INSTANCE_ID} (then: tail -f /var/log/benchmark-prereq-build.log)"
echo ""

# Poll for completion via SSM parameter
info "Waiting for EC2 builder to complete..."
info "This typically takes 15-20 minutes (mostly pulling the 9.5GB vLLM image)."
info "You can safely Ctrl+C — the builder continues autonomously and self-terminates."
echo ""

MAX_WAIT=1800  # 30 min max
ELAPSED=0
while [ ${ELAPSED} -lt ${MAX_WAIT} ]; do
    STATUS=$(aws ssm get-parameter \
        --name "/inference-phase4/prereq-build-status" \
        --query 'Parameter.Value' \
        --output text \
        --region "${REGION}" 2>/dev/null || echo "UNKNOWN")

    if [[ "${STATUS}" == COMPLETE:* ]]; then
        echo ""
        success "EC2 builder completed! ${STATUS}"
        break
    fi

    # Also check if instance is still running
    INSTANCE_STATE=$(aws ec2 describe-instances \
        --instance-ids "${INSTANCE_ID}" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text \
        --region "${REGION}" 2>/dev/null || echo "unknown")

    if [ "${INSTANCE_STATE}" == "terminated" ] || [ "${INSTANCE_STATE}" == "shutting-down" ]; then
        echo ""
        warn "Instance terminated. Checking if build completed..."
        if [[ "${STATUS}" == COMPLETE:* ]]; then
            success "Build completed before termination"
        else
            warn "Instance terminated without completion signal. Check ECR manually:"
            echo "  aws ecr describe-images --repository-name inference-lab/vllm-openai --region ${REGION}"
        fi
        break
    fi

    printf "\r  Waiting... %dm %ds elapsed (instance: %s)" $((ELAPSED/60)) $((ELAPSED%60)) "${INSTANCE_STATE}"
    sleep 30
    ELAPSED=$((ELAPSED + 30))
done

if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
    warn "Timed out waiting for EC2 builder after ${MAX_WAIT}s"
    warn "The builder may still be running. Check:"
    echo "  aws ssm get-parameter --name /inference-phase4/prereq-build-status --region ${REGION}"
    echo "  aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --region ${REGION}"
fi
echo ""

# Verify ECR images
echo "── Verifying ECR images ──"
for TAG in "${VLLM_VERSION}" "${VLLM_VERSION}-streamer"; do
    DIGEST=$(aws ecr describe-images \
        --repository-name "inference-lab/vllm-openai" \
        --image-ids imageTag="${TAG}" \
        --query 'imageDetails[0].imageDigest' \
        --output text \
        --region "${REGION}" 2>/dev/null || echo "NOT_FOUND")
    if [ "${DIGEST}" != "NOT_FOUND" ] && [ "${DIGEST}" != "None" ]; then
        success "ECR: inference-lab/vllm-openai:${TAG} — ${DIGEST}"
    else
        error "ECR: inference-lab/vllm-openai:${TAG} — NOT FOUND"
    fi
done
echo ""

if [ "${ONLY_EC2}" = true ]; then
    echo "  --only-ec2 flag set. Skipping Packer AMI builds."
    exit 0
fi

# ── Step 3: Build Pre-baked AMI (Packer) ──────────────────────────────────
if [ "${SKIP_AMI}" = true ]; then
    info "Skipping pre-baked AMI build (--skip-ami)"
else
    echo "================================================================"
    echo " Step 3: Pre-baked AMI (Packer)"
    echo "================================================================"
    echo ""
    bash "${SCRIPT_DIR}/build-ami.sh" "${REGION}"
fi
echo ""

# ── Step 4: Build SOCI AMI (Packer) ──────────────────────────────────────
if [ "${SKIP_SOCI_AMI}" = true ]; then
    info "Skipping SOCI AMI build (--skip-soci-ami)"
else
    echo "================================================================"
    echo " Step 4: SOCI AMI (Packer)"
    echo "================================================================"
    echo ""
    bash "${SCRIPT_DIR}/build-ami-soci.sh"
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  All Prerequisites Built${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo "  ECR images:"
echo "    inference-lab/vllm-openai:${VLLM_VERSION}           (for SOCI)"
echo "    inference-lab/vllm-openai:${VLLM_VERSION}-streamer  (for configs 7-8)"
echo ""
echo "  AMIs:"
echo "    inference-lab-gpu-node       (pre-baked, for configs 3-5)"
echo "    inference-lab-gpu-node-soci  (SOCI-enabled, for configs 6-8)"
echo ""
echo "  NEXT: Run the benchmark:"
echo "    python3 phase4/tests/cold_start_benchmark.py --host http://localhost:8080"
echo ""
echo "  COST NOTE: FSR is running (\$1.50/hr for 2 AZs). Disable when done:"
echo "    aws ec2 describe-fast-snapshot-restores --region ${REGION} --query 'FastSnapshotRestores[].SnapshotId'"
echo ""
