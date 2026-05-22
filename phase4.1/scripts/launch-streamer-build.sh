#!/bin/bash
# launch-streamer-build.sh — Build vllm-openai:v0.19.0-streamer on a spot EC2,
# push to ECR, terminate.
#
# WHAT THIS DOES
#   Stage 1 of the variant-E plan. Produces a derived image:
#     FROM <ecr>/inference-lab/vllm-openai:v0.19.0
#     RUN  pip install --no-cache-dir runai-model-streamer runai-model-streamer-s3
#   tagged as <ecr>/inference-lab/vllm-openai:v0.19.0-streamer.
#
# WHY EC2 (NOT MAC LOCAL)
#   The base image is a 9.5GB linux/amd64 CUDA image. Building on macOS via
#   buildx works but is slow and unreliable on M-series; the install is
#   noncompiling (just a pip layer) so a small CPU spot instance is plenty.
#
# WHY NOT JUST KUBECTL EXEC INTO A POD
#   Build needs Docker daemon access; the cluster system nodes run containerd
#   only and we'd be wrapping in kaniko etc. Faster + cleaner to do it on a
#   purpose-built throwaway instance.
#
# COST: ~$0.05 — c5.large spot (~$0.04/hr) for ~10-15 minutes.
#
# PREREQUISITES
#   - Karpenter node role has temp ECR push policy (TempECRPushVariantE)
#   - Account/region default to us-east-1 / current caller
#
# USAGE
#   bash phase4.1/scripts/launch-streamer-build.sh

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
BASE_IMAGE="${ECR}/inference-lab/vllm-openai:v0.19.0"
NEW_IMAGE="${ECR}/inference-lab/vllm-openai:v0.19.0-streamer"

# Use the existing phase4-1 VPC; us-east-1a public-routed subnet so the
# instance has direct internet (ECR pull from base image needs egress).
SUBNET_ID="subnet-0b597d988438e7737"        # public-us-east-1a
SECURITY_GROUP="sg-09fead621b005aa30"       # cluster SG (allows egress)
INSTANCE_PROFILE="inference-phase4-1-karpenter-node"

# Amazon Linux 2023, x86_64 — Docker package available via dnf.
# (Avoid ssm:GetParameter — the spot-lab IAM user lacks that permission.
# DescribeImages with a name filter is equivalent and only needs ec2:*.)
AMI_ID=$(aws ec2 describe-images --owners amazon \
    --filters "Name=name,Values=al2023-ami-2023.*-kernel-6.1-x86_64" \
              "Name=state,Values=available" \
    --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' \
    --output text --region "${REGION}")

echo "ECR:       ${ECR}"
echo "Base:      ${BASE_IMAGE}"
echo "Output:    ${NEW_IMAGE}"
echo "Builder:   c5.large spot in subnet ${SUBNET_ID} (us-east-1a)"
echo "AMI:       ${AMI_ID}"
echo

# Userdata: install Docker, login to ECR, build derived image, push, signal.
# A failure tags the instance with BuildStatus=failed so the controller can
# surface the issue without needing to ssh in.
USERDATA=$(cat <<'BUILDEOF'
#!/bin/bash
set -uxo pipefail
exec > >(tee /var/log/streamer-build.log) 2>&1

# Helper to tag the instance with a build status; surfaces success/failure
# without requiring SSH or SSM into the box.
tag_status() {
    local val="$1"
    local iid
    iid=$(curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT 'http://169.254.169.254/latest/api/token' -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/instance-id)
    local az
    az=$(curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT 'http://169.254.169.254/latest/api/token' -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/placement/availability-zone)
    local region="${az%?}"
    aws ec2 create-tags --region "$region" --resources "$iid" --tags "Key=BuildStatus,Value=${val}"
}
trap 'tag_status failed' ERR

dnf install -y docker
systemctl start docker
systemctl enable docker

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
TOKEN=$(curl -s -X PUT 'http://169.254.169.254/latest/api/token' -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')
AZ=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/availability-zone)
REGION="${AZ%?}"
ECR="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
BASE="${ECR}/inference-lab/vllm-openai:v0.19.0"
NEW="${ECR}/inference-lab/vllm-openai:v0.19.0-streamer"

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR"

# Pull base. Time it so we can sanity-check ECR-VPC throughput.
T0=$(date +%s)
docker pull "$BASE"
echo "base_pull_seconds=$(( $(date +%s) - T0 ))"

# Build derived image: just a pip install layer. Pin to the latest stable
# runai-model-streamer at experiment time (no version pin — the package is
# young and we want the most current S3 reader). Capture resolved version in
# the build log for reproducibility.
mkdir -p /tmp/build && cd /tmp/build
cat > Dockerfile <<DOCKEREOF
FROM ${BASE}
# Verify the streamer package imports inside the build (no CUDA touch).
# We deliberately skip "import vllm" here — vllm initializes CUDA at import,
# which hangs on CPU-only build hosts. We verify vllm + GPU integration in
# Stage 2 (AMI bake test) where a real GPU instance is available.
RUN pip install --no-cache-dir runai-model-streamer runai-model-streamer-s3 \
    && python3 -c "import runai_model_streamer; print('runai_model_streamer ok')" \
    && python3 -c "from runai_model_streamer import SafetensorsStreamer; print('s3 extra ok')" \
    && pip show runai-model-streamer runai-model-streamer-s3
DOCKEREOF

DOCKER_BUILDKIT=1 docker build --progress=plain -t "$NEW" .

# Push (only the new layer should upload — base layers already in ECR).
docker push "$NEW"

# Mark the image so the controller can fetch its digest later.
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep streamer || true

tag_status complete
BUILDEOF
)

echo "Launching c5.large spot builder..."
LAUNCH_SPEC=$(cat <<EOF
{
    "ImageId": "${AMI_ID}",
    "InstanceType": "c5.large",
    "IamInstanceProfile": {"Name": "${INSTANCE_PROFILE}"},
    "SubnetId": "${SUBNET_ID}",
    "SecurityGroupIds": ["${SECURITY_GROUP}"],
    "UserData": "$(echo "${USERDATA}" | base64)",
    "BlockDeviceMappings": [{
        "DeviceName": "/dev/xvda",
        "Ebs": {"VolumeSize": 50, "VolumeType": "gp3", "DeleteOnTermination": true}
    }],
    "TagSpecifications": [{
        "ResourceType": "instance",
        "Tags": [
            {"Key": "Name", "Value": "vllm-streamer-builder"},
            {"Key": "Project", "Value": "inference-lab"},
            {"Key": "Stage", "Value": "phase4.1-variantE"},
            {"Key": "BuildStatus", "Value": "building"}
        ]
    }]
}
EOF
)

# Public IP for outbound to ECR/GitHub via the public-routed subnet.
LAUNCH_SPEC=$(echo "${LAUNCH_SPEC}" | python3 -c "
import sys, json
spec = json.load(sys.stdin)
spec['NetworkInterfaces'] = [{
    'DeviceIndex': 0,
    'AssociatePublicIpAddress': True,
    'SubnetId': spec.pop('SubnetId'),
    'Groups': spec.pop('SecurityGroupIds')
}]
json.dump(spec, sys.stdout)
")

INSTANCE_ID=$(aws ec2 run-instances \
    --cli-input-json "${LAUNCH_SPEC}" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
    --query 'Instances[0].InstanceId' \
    --output text \
    --region "${REGION}")

echo "Instance: ${INSTANCE_ID}"
echo "Polling BuildStatus tag every 30s (timeout 25min)..."

DEADLINE=$(( $(date +%s) + 1500 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    sleep 30
    STATUS=$(aws ec2 describe-tags \
        --filters Name=resource-id,Values="${INSTANCE_ID}" Name=key,Values=BuildStatus \
        --query 'Tags[0].Value' --output text --region "${REGION}" 2>/dev/null || echo "unknown")
    echo "$(date +%H:%M:%S) BuildStatus=${STATUS}"
    case "$STATUS" in
        complete)
            echo "BUILD COMPLETE."
            echo "Terminating ${INSTANCE_ID}..."
            aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}" >/dev/null
            echo "Verify image:"
            echo "  aws ecr describe-images --repository-name inference-lab/vllm-openai --image-ids imageTag=v0.19.0-streamer --region ${REGION}"
            exit 0
            ;;
        failed)
            echo "BUILD FAILED. Instance left running for log retrieval:"
            echo "  aws ssm start-session --target ${INSTANCE_ID}"
            echo "  cat /var/log/streamer-build.log"
            echo "Or fetch the log via SSM run-command:"
            echo "  aws ssm send-command --instance-ids ${INSTANCE_ID} \\"
            echo "    --document-name AWS-RunShellScript \\"
            echo "    --parameters 'commands=[\"tail -200 /var/log/streamer-build.log\"]'"
            exit 2
            ;;
    esac
done

echo "TIMEOUT after 25min. Instance ${INSTANCE_ID} still running; investigate:"
echo "  aws ssm start-session --target ${INSTANCE_ID}"
exit 3
