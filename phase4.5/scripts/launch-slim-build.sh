#!/bin/bash
# launch-slim-build.sh — Launch a spot instance, build slim vLLM image, push to ECR, terminate.
#
# WHY NOT BUILD LOCALLY: vLLM's Dockerfile uses nvidia/cuda base images (Linux amd64).
# Can't build on macOS. This script launches a c5.2xlarge spot (~$0.09/hr, 8 vCPU)
# for faster nvcc compilation than g4dn.xlarge (4 vCPU). No GPU needed for build.
#
# TOTAL COST: ~$0.10 (c5.2xlarge spot for ~60 min)
#
# USAGE:
#   bash phase4.5/scripts/launch-slim-build.sh
#
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
KEY_NAME="${SLIM_BUILD_KEY:-inference-lab-key}"  # SSH key pair name
SECURITY_GROUP="${SLIM_BUILD_SG:-}"  # Optional; will use default VPC SG if empty
SUBNET_ID="${SLIM_BUILD_SUBNET:-}"  # Optional

echo "================================================================"
echo " Launching Slim vLLM Build Instance"
echo "================================================================"
echo ""

# Find latest Amazon Linux 2023 AMI (lightweight, Docker-friendly)
AMI_ID=$(aws ssm get-parameter \
    --name "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-x86_64" \
    --query 'Parameter.Value' --output text --region "${REGION}")
echo "Base AMI: ${AMI_ID}"

# Create userdata script that does everything automatically
USERDATA=$(cat << 'BUILDEOF'
#!/bin/bash
set -euxo pipefail

# Log everything to a file for debugging
exec > >(tee /var/log/slim-build.log) 2>&1

echo "=== Installing Docker ==="
dnf install -y docker git
systemctl start docker
systemctl enable docker

echo "=== Configuring AWS CLI ==="
# Instance role provides ECR permissions

echo "=== ECR Login ==="
REGION=$(ec2-metadata --availability-zone | cut -d' ' -f2 | sed 's/.$//')
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${ECR}"

echo "=== Cloning vLLM v0.19.0 ==="
git clone --branch v0.19.0 --depth 1 https://github.com/vllm-project/vllm.git /tmp/vllm-build
cd /tmp/vllm-build

echo "=== Building slim image (sm_75 only) ==="
BUILD_START=$(date +%s)
DOCKER_BUILDKIT=1 docker build \
    --build-arg torch_cuda_arch_list="7.5" \
    --build-arg max_jobs=6 \
    --target vllm-openai \
    -t "${ECR}/inference-lab/vllm-openai:v0.19.0-slim" \
    -f docker/Dockerfile .
BUILD_END=$(date +%s)
echo "Build took $(( BUILD_END - BUILD_START ))s"

echo "=== Image size ==="
docker images "${ECR}/inference-lab/vllm-openai:v0.19.0-slim"

echo "=== Pushing slim image ==="
docker push "${ECR}/inference-lab/vllm-openai:v0.19.0-slim"

echo "=== Building slim-streamer ==="
cat << 'DOCKERFILE' | docker build -t "${ECR}/inference-lab/vllm-openai:v0.19.0-slim-streamer" -f - .
FROM 019167255542.dkr.ecr.us-east-1.amazonaws.com/inference-lab/vllm-openai:v0.19.0-slim
RUN pip install --no-cache-dir runai-model-streamer runai-model-streamer-s3
DOCKERFILE
docker push "${ECR}/inference-lab/vllm-openai:v0.19.0-slim-streamer"

echo "=== DONE ==="
echo "BUILD_COMPLETE" > /tmp/build-status
# Signal completion via tag
INSTANCE_ID=$(ec2-metadata --instance-id | cut -d' ' -f2)
aws ec2 create-tags --resources "${INSTANCE_ID}" --tags Key=BuildStatus,Value=complete --region "${REGION}"

BUILDEOF
)

# Launch spot instance
echo "Launching c5.2xlarge spot instance..."
LAUNCH_SPEC=$(cat << EOF
{
    "ImageId": "${AMI_ID}",
    "InstanceType": "c5.2xlarge",
    "IamInstanceProfile": {"Name": "inference-phase4-5-karpenter-node"},
    "UserData": "$(echo "${USERDATA}" | base64)",
    "BlockDeviceMappings": [{
        "DeviceName": "/dev/xvda",
        "Ebs": {"VolumeSize": 100, "VolumeType": "gp3", "DeleteOnTermination": true}
    }],
    "TagSpecifications": [{
        "ResourceType": "instance",
        "Tags": [
            {"Key": "Name", "Value": "slim-vllm-builder"},
            {"Key": "Project", "Value": "inference-lab"},
            {"Key": "BuildStatus", "Value": "building"}
        ]
    }]
}
EOF
)

# Add key pair if provided
if [ -n "${KEY_NAME}" ]; then
    LAUNCH_SPEC=$(echo "${LAUNCH_SPEC}" | python3 -c "
import sys, json
spec = json.load(sys.stdin)
spec['KeyName'] = '${KEY_NAME}'
json.dump(spec, sys.stdout)
")
fi

INSTANCE_ID=$(aws ec2 run-instances \
    --cli-input-json "${LAUNCH_SPEC}" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
    --query 'Instances[0].InstanceId' \
    --output text \
    --region "${REGION}")

echo "Instance launched: ${INSTANCE_ID}"
echo ""
echo "================================================================"
echo " Build Running on ${INSTANCE_ID}"
echo "================================================================"
echo ""
echo "  Monitor progress:"
echo "    # Check build status tag (becomes 'complete' when done)"
echo "    aws ec2 describe-tags --filters Name=resource-id,Values=${INSTANCE_ID} Name=key,Values=BuildStatus --query 'Tags[0].Value' --output text"
echo ""
echo "    # SSH in to watch live logs"
echo "    INSTANCE_IP=\$(aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
echo "    ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@\${INSTANCE_IP} 'tail -f /var/log/slim-build.log'"
echo ""
echo "  After build completes (~30-60 min):"
echo "    # Verify images in ECR"
echo "    aws ecr describe-images --repository-name inference-lab/vllm-openai --query 'imageDetails[*].imageTags' --output table"
echo ""
echo "    # Terminate the build instance"
echo "    aws ec2 terminate-instances --instance-ids ${INSTANCE_ID}"
echo ""
echo "    # Then rebuild AMIs and run benchmark:"
echo "    bash phase4.5/scripts/setup-soci.sh   # Create SOCI index for slim image"
echo "    bash phase4.5/scripts/build-ami.sh     # Rebuild prebaked AMI"
echo "    bash phase4.5/scripts/build-ami-soci.sh # Rebuild SOCI AMI"
echo "    python3 phase4.5/tests/cold_start_benchmark.py --host http://localhost:8080"
