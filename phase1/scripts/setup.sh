#!/bin/bash
# setup.sh — End-to-end deployment from terraform to running pods.
#
# WHAT THIS DOES (in order):
#   1. Provisions AWS infra (VPC, EKS, IAM) via Terraform
#   2. Configures kubectl to talk to the new cluster
#   3. Builds and pushes container images to ECR
#   4. Deploys K8s manifests (namespace → deps → app)
#   5. Waits for everything to be healthy
#
# WHY a single script: reproducibility. Running 15 commands manually invites
# ordering mistakes (e.g., deploying the worker before RabbitMQ is ready).
# This script enforces the correct dependency order — like a CloudFormation
# stack with DependsOn constraints.
#
# IDEMPOTENCY: every command either succeeds or is safe to re-run.
# Terraform is inherently idempotent. kubectl apply is idempotent.
# ECR create-repository uses || true. Re-running this script is safe.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TERRAFORM_DIR="${PROJECT_DIR}/terraform"
K8S_DIR="${PROJECT_DIR}/k8s"
NAMESPACE="inference-system"

echo "============================================"
echo "  Inference Lab — Phase 1 Setup"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Terraform — provision AWS infrastructure
# WHY auto-approve: this is a learning lab, not production. In production
# you'd run plan + manual review + apply.
# ---------------------------------------------------------------------------
echo "=== Step 1: Terraform ==="
cd "$TERRAFORM_DIR"

echo "Running terraform init..."
terraform init

echo "Running terraform apply..."
terraform apply -auto-approve

# WHY extract outputs: we need the cluster name and S3 bucket for later steps.
# Terraform outputs are the contract between infra and app layers.
CLUSTER_NAME=$(terraform output -raw cluster_name 2>/dev/null || echo "inference-lab")
AWS_REGION=$(terraform output -raw region 2>/dev/null || aws configure get region || echo "us-east-1")
# WHY fallback to variable default: if terraform outputs aren't defined yet
# (first run, outputs not wired), use the known defaults from variables.tf

echo "Cluster: ${CLUSTER_NAME}"
echo "Region:  ${AWS_REGION}"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Configure kubectl
# WHY update-kubeconfig: creates a context in ~/.kube/config that uses IAM
# authentication to the EKS API server. Without this, kubectl can't talk
# to the cluster.
# ---------------------------------------------------------------------------
echo "=== Step 2: Configure kubectl ==="
aws eks update-kubeconfig \
    --name "$CLUSTER_NAME" \
    --region "$AWS_REGION"

# Verify connectivity
echo "Verifying cluster access..."
kubectl cluster-info
echo ""

# ---------------------------------------------------------------------------
# Step 3: Build and push images
# WHY separate script: build_and_push.sh is useful standalone (rebuild after
# code changes without re-running terraform).
# ---------------------------------------------------------------------------
echo "=== Step 3: Build and push images ==="
bash "${SCRIPT_DIR}/build_and_push.sh"
echo ""

# ---------------------------------------------------------------------------
# Step 4: Deploy K8s resources (order matters!)
# WHY this specific order:
#   1. Namespace first — all other resources live in it
#   2. GPU NodePool — Karpenter needs this to provision GPU nodes
#   3. NVIDIA device plugin — makes GPUs visible to K8s scheduler
#   4. RabbitMQ + Redis — must be ready before app pods start
#   5. App pods — depend on all of the above
# This mirrors CloudFormation DependsOn chains.
# ---------------------------------------------------------------------------
echo "=== Step 4: Deploy K8s manifests ==="

# --- 4a: Namespace ---
echo "Creating namespace..."
kubectl apply -f "${K8S_DIR}/namespace.yaml"

# --- 4b: GPU NodePool (Karpenter CRDs) ---
echo "Applying GPU NodePool..."
kubectl apply -f "${K8S_DIR}/gpu-nodepool.yaml"

# --- 4c: NVIDIA device plugin ---
# WHY DaemonSet from NVIDIA: this DaemonSet runs on every GPU node and
# registers GPUs as a schedulable resource (nvidia.com/gpu). Without it,
# K8s doesn't know the node has GPUs. Like installing CloudWatch agent
# to expose custom metrics to ASG scaling policies.
echo "Installing NVIDIA device plugin..."
kubectl apply -f "${K8S_DIR}/nvidia-device-plugin.yaml"

# --- 4d: RabbitMQ and Redis ---
echo "Deploying RabbitMQ..."
kubectl apply -f "${K8S_DIR}/rabbitmq/"

echo "Deploying Redis..."
kubectl apply -f "${K8S_DIR}/redis/"

# WHY wait: pods need time to pull images and start. The API gateway and
# worker depend on these being ready. 120s timeout covers slow image pulls
# on cold nodes.
echo "Waiting for RabbitMQ to be ready..."
kubectl -n "$NAMESPACE" wait --for=condition=ready pod -l app.kubernetes.io/name=rabbitmq --timeout=120s || {
    echo "WARNING: RabbitMQ not ready after 120s — check pod status"
    kubectl -n "$NAMESPACE" get pods -l app.kubernetes.io/name=rabbitmq
}

echo "Waiting for Redis to be ready..."
kubectl -n "$NAMESPACE" wait --for=condition=ready pod -l app.kubernetes.io/name=redis --timeout=120s || {
    echo "WARNING: Redis not ready after 120s — check pod status"
    kubectl -n "$NAMESPACE" get pods -l app.kubernetes.io/name=redis
}

# --- 4e: Update K8s manifests with actual ECR URLs ---
# WHY sed: K8s manifests use placeholder image refs. We replace them with
# the actual ECR URLs from this AWS account. This avoids hardcoding account
# IDs in version-controlled YAML.
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "Updating manifests with ECR URL: ${ECR_URL}"

# Replace ECR image placeholders in deployment manifests
find "${K8S_DIR}" -name "*.yaml" -exec \
    sed -i.bak "s|ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com|${ECR_URL}|g" {} \;
find "${K8S_DIR}" -name "*.yaml.bak" -delete 2>/dev/null || true

# Replace remaining ACCOUNT_ID references (e.g., IRSA role ARN in service account)
# WHY separate pass: the ECR URL replacement above handles the full ECR pattern.
# This catches standalone ACCOUNT_ID in IAM ARNs like "arn:aws:iam::ACCOUNT_ID:role/..."
find "${K8S_DIR}" -name "*.yaml" -exec \
    sed -i.bak "s|ACCOUNT_ID|${AWS_ACCOUNT_ID}|g" {} \;
find "${K8S_DIR}" -name "*.yaml.bak" -delete 2>/dev/null || true

# Replace S3 bucket name if terraform output is available
S3_BUCKET=$(terraform -chdir="$TERRAFORM_DIR" output -raw model_cache_bucket_name 2>/dev/null || echo "")
if [ -n "$S3_BUCKET" ]; then
    echo "Updating manifests with S3 bucket: ${S3_BUCKET}"
    find "${K8S_DIR}" -name "*.yaml" -exec \
        sed -i.bak "s|BUCKET_NAME_PLACEHOLDER|${S3_BUCKET}|g" {} \;
    find "${K8S_DIR}" -name "*.yaml.bak" -delete 2>/dev/null || true
fi

# --- 4f: Deploy application pods ---
echo "Deploying API Gateway..."
kubectl apply -f "${K8S_DIR}/api-gateway/"

echo "Deploying vLLM Worker..."
# WHY apply the whole directory: picks up service account, PV, and deployment
# in the correct order (K8s handles dependency resolution within a single apply).
if [ -d "${K8S_DIR}/vllm-worker" ]; then
    kubectl apply -f "${K8S_DIR}/vllm-worker/"
fi

# ---------------------------------------------------------------------------
# Step 5: Wait for application pods
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 5: Waiting for pods ==="

echo "Waiting for API Gateway..."
kubectl -n "$NAMESPACE" wait --for=condition=ready pod -l app.kubernetes.io/name=api-gateway --timeout=120s || {
    echo "WARNING: API Gateway not ready after 120s"
    kubectl -n "$NAMESPACE" get pods -l app.kubernetes.io/name=api-gateway
    kubectl -n "$NAMESPACE" describe pods -l app.kubernetes.io/name=api-gateway | tail -20
}

# WHY not waiting for vLLM worker: it needs a GPU node, which Karpenter
# provisions on-demand. This can take 2-5 minutes for Spot instances.
# We don't block the script — the user can watch progress with kubectl.
echo ""
echo "NOTE: vLLM worker pods require GPU nodes (provisioned by Karpenter on-demand)."
echo "GPU node provisioning takes 2-5 minutes. Watch progress with:"
echo "  kubectl -n $NAMESPACE get pods -w"
echo ""

# ---------------------------------------------------------------------------
# Final status
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Cluster:   ${CLUSTER_NAME} (${AWS_REGION})"
echo "Namespace: ${NAMESPACE}"
echo ""
echo "Pod status:"
kubectl -n "$NAMESPACE" get pods -o wide
echo ""
echo "Services:"
kubectl -n "$NAMESPACE" get svc
echo ""
echo "=== Next Steps ==="
echo "1. Wait for GPU node + vLLM pod: kubectl -n $NAMESPACE get pods -w"
echo "2. Port forward:                 bash scripts/port_forward.sh"
echo "3. Test:                         curl http://localhost:8080/health"
echo "4. Submit inference:             curl -X POST http://localhost:8080/generate \\"
echo "                                   -H 'Content-Type: application/json' \\"
echo "                                   -d '{\"prompt\": \"Hello!\", \"max_tokens\": 50}'"
