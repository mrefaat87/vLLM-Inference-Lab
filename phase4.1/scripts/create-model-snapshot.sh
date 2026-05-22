#!/usr/bin/env bash
# create-model-snapshot.sh — Create an EBS snapshot pre-loaded with model weights.
#
# WHAT THIS DOES:
#   1. Launches a temp t3.medium instance with a 20GiB gp3 data volume
#   2. UserData formats the volume, downloads model from S3, then signals completion
#   3. Waits for instance to reach running + UserData to finish (via stop signal)
#   4. Stops the instance, detaches the data volume, snapshots it
#   5. Cleans up the instance and volume
#   6. Outputs the snapshot ID for use in Karpenter blockDeviceMappings
#
# WHY AN EBS SNAPSHOT:
#   Phase 4 cold start breakdown: ~20s S3 download + ~30s GPU load + ~205s infra.
#   With a pre-loaded EBS snapshot attached via Karpenter blockDeviceMappings, new
#   nodes boot with model weights already on a local volume — no S3 download needed.
#   The init container just verifies the files exist instead of downloading them.
#   Cold start drops from ~255s (with S3) to ~235s (snapshot mount = ~0s).
#
# ANALOGY: Like using a pre-populated EBS volume in a launch template instead of
#   downloading application data from S3 on every instance boot. The snapshot is
#   the "golden data volume" — new volumes are created from it instantly.
#
# USAGE:
#   bash phase4/scripts/create-model-snapshot.sh
#   bash phase4/scripts/create-model-snapshot.sh --region us-west-2
#   bash phase4/scripts/create-model-snapshot.sh --fsr   # enable Fast Snapshot Restore
#
# PREREQUISITES:
#   - AWS CLI configured with credentials
#   - Model already uploaded to S3 (run phase4/scripts/upload-model-to-s3.sh first)
#   - Karpenter subnets tagged with karpenter.sh/discovery: inference-lab
#
# COST:
#   - t3.medium Spot: ~$0.01/hr for ~5 min = ~$0.001
#   - EBS snapshot storage: ~$0.05/GB/month = ~$1/month for 20GiB
#   - FSR (optional): $0.75/hr/AZ while enabled

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
ENABLE_FSR=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --region)
            REGION="$2"
            shift 2
            ;;
        --fsr)
            ENABLE_FSR=true
            shift
            ;;
        *)
            error "Unknown argument: $1"
            echo "Usage: $0 [--region REGION] [--fsr]"
            exit 1
            ;;
    esac
done

# ── Variables ──────────────────────────────────────────────────────────
BUCKET="inference-lab-model-cache"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct-AWQ"
# WHY this path layout: matches HuggingFace hub cache directory structure.
# vLLM's model loader expects models--{org}--{name}/snapshots/{hash}/.
MODEL_CACHE_DIR="models--Qwen--Qwen2.5-7B-Instruct-AWQ"
VOLUME_SIZE=20       # GiB — model is ~4GB, leaving headroom for filesystem overhead
INSTANCE_TYPE="t3.medium"

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Phase 4 — EBS Model Snapshot Builder${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

# ── Pre-flight: AWS CLI ───────────────────────────────────────────────
info "Checking AWS CLI..."
if ! command -v aws &>/dev/null; then
    error "AWS CLI not found. Install it: https://aws.amazon.com/cli/"
    exit 1
fi
if ! aws sts get-caller-identity &>/dev/null; then
    error "AWS credentials not configured. Run: aws configure"
    exit 1
fi
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
success "AWS account: ${AWS_ACCOUNT}, region: ${REGION}"

# ── Pre-flight: S3 model exists ──────────────────────────────────────
info "Checking for model in S3 bucket: s3://${BUCKET}/${MODEL_PATH}/..."
if ! aws s3 ls "s3://${BUCKET}/${MODEL_PATH}/" --region "${REGION}" &>/dev/null; then
    error "Model not found in S3. Run upload-model-to-s3.sh first:"
    echo "  bash phase4/scripts/upload-model-to-s3.sh"
    exit 1
fi
success "Model found in S3"

# ── Find a Karpenter subnet ──────────────────────────────────────────
# WHY use Karpenter discovery tag: launches the temp instance in the same subnet
# that Karpenter uses for GPU nodes. This ensures the instance has the right
# VPC routing/NAT to reach S3 (possibly via VPC endpoint).
info "Finding subnet with Karpenter discovery tag..."
SUBNET_ID=$(aws ec2 describe-subnets \
    --filters "Name=tag:karpenter.sh/discovery,Values=inference-lab" \
    --query 'Subnets[0].SubnetId' \
    --output text \
    --region "${REGION}")

if [ -z "${SUBNET_ID}" ] || [ "${SUBNET_ID}" == "None" ]; then
    error "No subnet found with tag karpenter.sh/discovery=inference-lab"
    error "Ensure your VPC is set up with Karpenter discovery tags."
    exit 1
fi
success "Using subnet: ${SUBNET_ID}"

# ── Find the latest Amazon Linux 2 AMI ──────────────────────────────
# WHY AL2 (not EKS AMI): we just need a basic Linux instance to format a disk
# and run aws s3 sync. No GPU, no Kubernetes — smallest footprint possible.
info "Finding latest Amazon Linux 2 AMI..."
BASE_AMI=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
              "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text \
    --region "${REGION}")

if [ -z "${BASE_AMI}" ] || [ "${BASE_AMI}" == "None" ]; then
    error "Could not find Amazon Linux 2 AMI."
    exit 1
fi
success "Base AMI: ${BASE_AMI}"

# ── Find IAM instance profile ────────────────────────────────────────
# WHY instance profile: the temp instance needs S3 read access to download
# model weights. Reuse the Karpenter node role which already has S3 permissions.
info "Looking for Karpenter node instance profile..."
INSTANCE_PROFILE=$(aws iam list-instance-profiles \
    --query "InstanceProfiles[?contains(InstanceProfileName, 'karpenter') || contains(InstanceProfileName, 'inference-lab')].InstanceProfileName | [0]" \
    --output text 2>/dev/null || echo "None")

IAM_ARGS=""
if [ "${INSTANCE_PROFILE}" != "None" ] && [ -n "${INSTANCE_PROFILE}" ]; then
    IAM_ARGS="--iam-instance-profile Name=${INSTANCE_PROFILE}"
    success "Using instance profile: ${INSTANCE_PROFILE}"
else
    warn "No instance profile found. The instance will need the AWS CLI"
    warn "configured via UserData environment variables or metadata credentials."
fi

# ── Build UserData ───────────────────────────────────────────────────
# WHY UserData approach: simpler than SSM Run Command. The instance boots,
# runs the script automatically, then shuts itself down. We just wait for
# the instance to reach 'stopped' state as the completion signal.
# Analogy: like a Lambda function — fire and forget, check status later.
info "Building UserData script..."

USERDATA=$(cat <<'USERDATA_SCRIPT'
#!/bin/bash
set -euo pipefail
exec > /var/log/model-snapshot-setup.log 2>&1
echo "=== Model snapshot setup starting: $(date -u) ==="

# Format the data volume as ext4
# WHY ext4: simple, stable, well-supported. No need for XFS performance here.
echo "Formatting /dev/xvdf as ext4..."
mkfs.ext4 -q /dev/xvdf

# Mount the data volume
mkdir -p /mnt/model
mount /dev/xvdf /mnt/model
echo "Mounted /dev/xvdf at /mnt/model"

# Install AWS CLI v2 if not present (AL2 ships with v1, we need v2 for s3 sync speed)
if ! command -v aws &>/dev/null || ! aws --version 2>&1 | grep -q "aws-cli/2"; then
    echo "Installing AWS CLI v2..."
    curl -sL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
    cd /tmp && unzip -qo awscliv2.zip && ./aws/install --update
    cd /
fi
echo "AWS CLI version: $(aws --version)"

# Download model weights from S3
# WHY this directory layout: matches HuggingFace hub cache structure.
# vLLM's AutoModelForCausalLM.from_pretrained() looks for this exact path
# when TRANSFORMERS_CACHE or HF_HOME is set.
MODEL_DIR="/mnt/model/models--Qwen--Qwen2.5-7B-Instruct-AWQ"
mkdir -p "${MODEL_DIR}"
echo "Downloading model from S3..."
echo "Start: $(date -u)"
aws s3 sync \
    "s3://inference-lab-model-cache/Qwen/Qwen2.5-7B-Instruct-AWQ/" \
    "${MODEL_DIR}/" \
    --no-progress --only-show-errors
echo "Download complete: $(date -u)"
echo "Model size: $(du -sh ${MODEL_DIR}/ | cut -f1)"
ls -la "${MODEL_DIR}/"

# Unmount cleanly before shutdown
# WHY sync + umount: ensures all writes are flushed to EBS before we snapshot.
sync
umount /mnt/model
echo "Volume unmounted cleanly"

echo "=== Model snapshot setup complete: $(date -u) ==="

# Signal completion by stopping the instance
# WHY shutdown: the parent script polls for 'stopped' state as the completion
# signal. This is simpler than writing a status file to S3 or using CloudWatch.
shutdown -h now
USERDATA_SCRIPT
)

# Base64 encode for the API call
USERDATA_B64=$(echo "${USERDATA}" | base64)

# ── Launch Instance ──────────────────────────────────────────────────
info "Launching ${INSTANCE_TYPE} with 20GiB data volume..."
info "The instance will download the model, then shut itself down."
echo ""

# WHY --block-device-mappings for /dev/xvdf: attaches a second EBS volume
# at launch time. This volume will hold model weights and become our snapshot.
# The root volume (/dev/xvda) is throwaway — we only care about the data volume.
RUN_OUTPUT=$(aws ec2 run-instances \
    --image-id "${BASE_AMI}" \
    --instance-type "${INSTANCE_TYPE}" \
    --subnet-id "${SUBNET_ID}" \
    ${IAM_ARGS} \
    --block-device-mappings "[
        {
            \"DeviceName\": \"/dev/xvda\",
            \"Ebs\": {
                \"VolumeSize\": 10,
                \"VolumeType\": \"gp3\",
                \"DeleteOnTermination\": true
            }
        },
        {
            \"DeviceName\": \"/dev/xvdf\",
            \"Ebs\": {
                \"VolumeSize\": ${VOLUME_SIZE},
                \"VolumeType\": \"gp3\",
                \"DeleteOnTermination\": false,
                \"Encrypted\": true
            }
        }
    ]" \
    --user-data "${USERDATA_B64}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=model-snapshot-builder},{Key=Project,Value=inference-lab}]" \
    --region "${REGION}" \
    --output json)

INSTANCE_ID=$(echo "${RUN_OUTPUT}" | python3 -c "import sys,json; print(json.load(sys.stdin)['Instances'][0]['InstanceId'])")
success "Instance launched: ${INSTANCE_ID}"

# ── Wait for Instance Running ────────────────────────────────────────
info "Waiting for instance to reach 'running' state..."
aws ec2 wait instance-running \
    --instance-ids "${INSTANCE_ID}" \
    --region "${REGION}"
success "Instance is running"

# ── Wait for UserData Completion (instance stops itself) ─────────────
# WHY poll for 'stopped': the UserData script calls 'shutdown -h now' after
# downloading the model and unmounting the volume. When the instance transitions
# to 'stopped', we know the data volume is ready to snapshot.
# Timeout after 15 minutes — S3 download should take 1-3 min for a ~4GB model.
info "Waiting for model download to complete (instance will stop itself)..."
info "This typically takes 3-5 minutes..."
TIMEOUT=900   # 15 minutes
ELAPSED=0
POLL_INTERVAL=15

while [ ${ELAPSED} -lt ${TIMEOUT} ]; do
    STATE=$(aws ec2 describe-instances \
        --instance-ids "${INSTANCE_ID}" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text \
        --region "${REGION}")

    if [ "${STATE}" == "stopped" ]; then
        success "Instance stopped — model download complete"
        break
    elif [ "${STATE}" == "terminated" ]; then
        error "Instance terminated unexpectedly. Check CloudWatch logs."
        exit 1
    fi

    # Show progress every poll
    info "  Instance state: ${STATE} (${ELAPSED}s elapsed, timeout ${TIMEOUT}s)"
    sleep ${POLL_INTERVAL}
    ELAPSED=$(( ELAPSED + POLL_INTERVAL ))
done

if [ ${ELAPSED} -ge ${TIMEOUT} ]; then
    error "Timeout waiting for instance to stop. Cleaning up..."
    aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}" &>/dev/null
    exit 1
fi

# ── Find the Data Volume ─────────────────────────────────────────────
# WHY query by device name: the instance has two volumes. We need the /dev/xvdf
# data volume (model weights), not the /dev/xvda root volume (OS).
info "Finding the data volume (/dev/xvdf)..."
VOLUME_ID=$(aws ec2 describe-instances \
    --instance-ids "${INSTANCE_ID}" \
    --query 'Reservations[0].Instances[0].BlockDeviceMappings[?DeviceName==`/dev/xvdf`].Ebs.VolumeId' \
    --output text \
    --region "${REGION}")

if [ -z "${VOLUME_ID}" ] || [ "${VOLUME_ID}" == "None" ]; then
    error "Could not find data volume. Cleaning up..."
    aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}" &>/dev/null
    exit 1
fi
success "Data volume: ${VOLUME_ID}"

# ── Detach the Data Volume ───────────────────────────────────────────
# WHY detach before snapshot: the volume must be detached (or the instance
# stopped) for a consistent snapshot. Since the instance is already stopped,
# detaching is optional but explicit — it also lets us terminate the instance
# immediately without waiting for the snapshot to complete.
info "Detaching data volume from instance..."
aws ec2 detach-volume \
    --volume-id "${VOLUME_ID}" \
    --instance-id "${INSTANCE_ID}" \
    --region "${REGION}" &>/dev/null

# Wait for detach to complete
aws ec2 wait volume-available \
    --volume-ids "${VOLUME_ID}" \
    --region "${REGION}"
success "Volume detached"

# ── Terminate Instance (no longer needed) ────────────────────────────
info "Terminating temp instance..."
aws ec2 terminate-instances \
    --instance-ids "${INSTANCE_ID}" \
    --region "${REGION}" &>/dev/null
success "Instance ${INSTANCE_ID} terminated"

# ── Create Snapshot ──────────────────────────────────────────────────
info "Creating snapshot from data volume..."
SNAPSHOT_ID=$(aws ec2 create-snapshot \
    --volume-id "${VOLUME_ID}" \
    --description "inference-lab model cache: ${MODEL_PATH}" \
    --tag-specifications "ResourceType=snapshot,Tags=[
        {Key=Name,Value=inference-lab-model-cache},
        {Key=Model,Value=${MODEL_PATH}},
        {Key=Project,Value=inference-lab}
    ]" \
    --query 'SnapshotId' \
    --output text \
    --region "${REGION}")

success "Snapshot creation started: ${SNAPSHOT_ID}"

# ── Wait for Snapshot to Complete ────────────────────────────────────
# WHY wait: the snapshot must be 'completed' before Karpenter can use it in
# blockDeviceMappings. Pending snapshots cause instance launch failures.
info "Waiting for snapshot to complete (this may take a few minutes)..."
aws ec2 wait snapshot-completed \
    --snapshot-ids "${SNAPSHOT_ID}" \
    --region "${REGION}"
success "Snapshot complete: ${SNAPSHOT_ID}"

# ── Delete the Orphaned Volume ───────────────────────────────────────
# WHY delete: the volume was created with DeleteOnTermination=false so it
# survived instance termination. Now that the snapshot is done, the volume
# is just costing $0.08/GB/month for no reason.
info "Deleting orphaned data volume..."
aws ec2 delete-volume \
    --volume-id "${VOLUME_ID}" \
    --region "${REGION}"
success "Volume ${VOLUME_ID} deleted"

# ── Optional: Enable Fast Snapshot Restore ───────────────────────────
if [ "${ENABLE_FSR}" = true ]; then
    # WHY FSR: EBS volumes created from snapshots use lazy loading — the first
    # read of each block fetches from S3. For a 20GiB model volume, this adds
    # latency on the first model load. FSR pre-initializes the volume so all
    # blocks are instantly available.
    # COST: $0.75/hr per AZ while enabled. Disable when not testing.
    info "Enabling Fast Snapshot Restore (FSR) on ${SNAPSHOT_ID}..."
    info "Cost: \$0.75/hr per AZ while enabled."
    aws ec2 enable-fast-snapshot-restores \
        --availability-zones us-east-1a us-east-1b \
        --source-snapshot-ids "${SNAPSHOT_ID}" \
        --region "${REGION}" 2>&1 | head -5
    success "FSR enabling on us-east-1a and us-east-1b (takes 5-15 min to fully enable)"
    echo ""
    info "To disable FSR when done testing (stops the \$0.75/hr/AZ charge):"
    echo "  aws ec2 disable-fast-snapshot-restores --availability-zones us-east-1a us-east-1b --source-snapshot-ids ${SNAPSHOT_ID} --region ${REGION}"
else
    info "FSR not enabled. To enable later (faster first reads, \$0.75/hr/AZ):"
    echo "  bash phase4/scripts/create-model-snapshot.sh --fsr"
    echo "  # or manually:"
    echo "  aws ec2 enable-fast-snapshot-restores --availability-zones us-east-1a us-east-1b --source-snapshot-ids ${SNAPSHOT_ID} --region ${REGION}"
fi

# ── Print Results ────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Model Snapshot Created Successfully${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo "  Snapshot ID:   ${SNAPSHOT_ID}"
echo "  Model:         ${MODEL_PATH}"
echo "  Volume size:   ${VOLUME_SIZE}GiB"
echo "  Region:        ${REGION}"
echo "  FSR:           $([ "${ENABLE_FSR}" = true ] && echo 'enabling' || echo 'disabled')"
echo ""
echo "  Next steps:"
echo "    1. Update gpu-nodepool.yaml with the snapshot ID:"
echo "       blockDeviceMappings:"
echo "         - deviceName: /dev/xvdf"
echo "           ebs:"
echo "             snapshotID: \"${SNAPSHOT_ID}\""
echo "             volumeSize: ${VOLUME_SIZE}Gi"
echo "             volumeType: gp3"
echo "             encrypted: true"
echo "             deleteOnTermination: true"
echo ""
echo "    2. Apply the updated NodePool:"
echo "       kubectl apply -f phase4/k8s/karpenter/gpu-nodepool.yaml"
echo ""
echo "    3. The deployment-patch.yaml init container will detect the"
echo "       pre-loaded volume and skip the S3 download."
echo ""
