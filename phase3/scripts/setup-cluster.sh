#!/bin/bash
# setup-cluster.sh — Provision a separate EKS cluster for Phase 3 experiments.
#
# WHAT THIS DOES:
#   1. Terraform: VPC, EKS, Karpenter, IAM (reuses ECR + S3 from Phase 1)
#   2. kubectl: configure access to the new cluster
#   3. K8s: deploy namespace, GPU nodepool, NVIDIA plugin, RabbitMQ, Redis,
#           API gateway, vLLM worker with Phase 3 image
#   4. Helm: Prometheus + Grafana + KEDA + Image Renderer
#
# WHY separate cluster: Phase 4 modified the main cluster (worker image,
# KEDA policies, Karpenter NodePool). This cluster is fully isolated so
# Phase 3 experiments can't affect production infra.
#
# COST: ~$0.55/hr when running ($0.10 EKS + $0.08 CPU + $0.045 NAT + $0.22 GPU Spot)
#
# Usage: bash phase3/scripts/setup-cluster.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE3_DIR="$(dirname "$SCRIPT_DIR")"
PHASE1_DIR="$(dirname "$PHASE3_DIR")/phase1"
PHASE2_DIR="$(dirname "$PHASE3_DIR")/phase2"
TF_DIR="${PHASE3_DIR}/terraform"
K8S_DIR="${PHASE1_DIR}/k8s"
NAMESPACE="inference-system"

echo "============================================"
echo "  Phase 3 — Backpressure Experiment Cluster"
echo "============================================"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Terraform
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Step 1: Terraform (VPC + EKS + Karpenter) ==="
cd "$TF_DIR"

terraform init
terraform apply -auto-approve

CLUSTER_NAME=$(terraform output -raw cluster_name)
AWS_REGION=$(terraform output -raw region)
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "Cluster: ${CLUSTER_NAME}"
echo "Region:  ${AWS_REGION}"
echo "ECR:     ${ECR_URL}"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Configure kubectl for the NEW cluster
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Step 2: Configure kubectl ==="
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$AWS_REGION"
kubectl cluster-info
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Deploy K8s manifests (reuse Phase 1 manifests)
# WHY reuse: same architecture (namespace, GPU nodepool, NVIDIA plugin,
# RabbitMQ, Redis, API gateway, vLLM worker sidecar). Only the worker
# image tag differs.
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Step 3: Deploy K8s manifests ==="

# Create a temporary working copy of K8s manifests to sed without modifying originals
WORK_DIR=$(mktemp -d)
cp -r "$K8S_DIR"/* "$WORK_DIR/"
trap "rm -rf $WORK_DIR" EXIT

# Replace placeholders with actual values
echo "Patching manifests with account ID and ECR URL..."
find "$WORK_DIR" -name "*.yaml" -exec \
    sed -i.bak "s|ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com|${ECR_URL}|g" {} \;
find "$WORK_DIR" -name "*.yaml" -exec \
    sed -i.bak "s|ACCOUNT_ID|${AWS_ACCOUNT_ID}|g" {} \;

# Replace ALL "inference-lab" references with the Phase 3 cluster name.
# WHY global replace: the GPU nodepool yaml has "inference-lab" in the IAM role
# name, subnet/SG discovery tags, and project tags — not just the Karpenter
# discovery tag. Missing any of these causes GPU nodes to launch in Phase 1's
# VPC or with the wrong IAM role, failing to register with Phase 3's cluster.
find "$WORK_DIR" -name "*.yaml" -exec \
    sed -i.bak "s|inference-lab|${CLUSTER_NAME}|g" {} \;
find "$WORK_DIR" -name "*.yaml.bak" -delete 2>/dev/null || true

# Also update the vLLM worker IRSA role ARN to the Phase 3 role
VLLM_ROLE_ARN=$(terraform -chdir="$TF_DIR" output -raw vllm_worker_role_arn 2>/dev/null || echo "")
if [ -n "$VLLM_ROLE_ARN" ]; then
    echo "Patching vLLM worker ServiceAccount with role: ${VLLM_ROLE_ARN}"
    find "$WORK_DIR" -name "*.yaml" -exec \
        sed -i.bak "s|eks.amazonaws.com/role-arn:.*|eks.amazonaws.com/role-arn: ${VLLM_ROLE_ARN}|g" {} \;
    find "$WORK_DIR" -name "*.yaml.bak" -delete 2>/dev/null || true
fi

# Deploy in dependency order
echo "Creating namespace..."
kubectl apply -f "$WORK_DIR/namespace.yaml"

echo "Applying GPU NodePool..."
kubectl apply -f "$WORK_DIR/gpu-nodepool.yaml"

echo "Installing NVIDIA device plugin..."
kubectl apply -f "$WORK_DIR/nvidia-device-plugin.yaml"

echo "Deploying RabbitMQ..."
kubectl apply -f "$WORK_DIR/rabbitmq/"

echo "Deploying Redis..."
kubectl apply -f "$WORK_DIR/redis/"

echo "Waiting for RabbitMQ..."
kubectl -n "$NAMESPACE" wait --for=condition=ready pod -l app.kubernetes.io/name=rabbitmq --timeout=180s || echo "WARNING: RabbitMQ not ready"

echo "Waiting for Redis..."
kubectl -n "$NAMESPACE" wait --for=condition=ready pod -l app.kubernetes.io/name=redis --timeout=120s || echo "WARNING: Redis not ready"

echo "Deploying API Gateway..."
kubectl apply -f "$WORK_DIR/api-gateway/"

echo "Deploying vLLM Worker (with phase3 image)..."
# Deploy vLLM worker from Phase 1 manifests, then patch to Phase 3 image
kubectl apply -f "$WORK_DIR/vllm-worker/"

# Patch the worker to use Phase 3 image and config
echo "Patching worker to Phase 3 image + PREFETCH_COUNT=10..."
kubectl set image deployment/vllm-worker \
    -n "$NAMESPACE" \
    queue-worker="${ECR_URL}/inference-lab/worker:phase3"

kubectl set env deployment/vllm-worker \
    -n "$NAMESPACE" \
    -c queue-worker \
    ADMISSION_STRATEGY=static \
    PREFETCH_COUNT=10 \
    MAX_BATCH_SIZE=8

# Set maxUnavailable=1 so strategy switches don't need a second GPU node
kubectl patch deployment vllm-worker -n "$NAMESPACE" \
    -p '{"spec":{"strategy":{"rollingUpdate":{"maxSurge":0,"maxUnavailable":1}}}}'

echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Install monitoring stack (Prometheus + Grafana + KEDA)
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Step 4: Install monitoring + autoscaling ==="

# Add Helm repos
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo add kedacore https://kedacore.github.io/charts 2>/dev/null || true
helm repo update

# Prometheus + Grafana (with Image Renderer for screenshots)
echo "Installing kube-prometheus-stack + Image Renderer..."
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
    -n monitoring --create-namespace \
    -f "${PHASE2_DIR}/k8s/monitoring/prometheus-values.yaml" \
    -f "${PHASE3_DIR}/k8s/grafana-image-renderer-values.yaml" \
    --wait --timeout 5m || echo "WARNING: Prometheus stack install may have issues"

# Grafana dashboard ConfigMap
echo "Deploying Grafana dashboard..."
kubectl apply -f "${PHASE2_DIR}/k8s/monitoring/grafana-dashboard-configmap.yaml"

# ServiceMonitors
echo "Deploying ServiceMonitors..."
kubectl apply -f "${PHASE2_DIR}/k8s/monitoring/" 2>/dev/null || true

# DCGM Exporter (GPU metrics)
echo "Deploying DCGM Exporter..."
kubectl apply -f "${PHASE2_DIR}/k8s/monitoring/dcgm-exporter.yaml" 2>/dev/null || true

# KEDA
echo "Installing KEDA..."
helm install keda kedacore/keda \
    -n keda --create-namespace \
    --wait --timeout 3m || echo "WARNING: KEDA install may have issues"

# KEDA ScaledObject + auth
echo "Deploying KEDA ScaledObject..."
kubectl apply -f "${PHASE2_DIR}/k8s/autoscaling/" || echo "WARNING: KEDA ScaledObject may need RabbitMQ auth"

echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Wait for GPU node
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Step 5: Waiting for GPU node + vLLM worker ==="
echo "  Karpenter → Spot launch ~90s → model load ~3 min"
echo "  Watch: kubectl -n $NAMESPACE get pods -w"
echo ""

# Wait up to 7 min
for i in $(seq 1 84); do
    WORKER_READY=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=vllm-worker \
        --no-headers 2>/dev/null | grep -c "2/2" || echo "0")
    echo -ne "\r  [$((i*5))s] vllm-worker 2/2: ${WORKER_READY}/1"
    if [ "$WORKER_READY" -ge 1 ]; then
        echo ""
        echo "  vLLM worker READY!"
        break
    fi
    sleep 5
done

echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Final status
# ──────────────────────────────────────────────────────────────────────────────
echo "============================================"
echo "  Phase 3 Cluster Ready!"
echo "============================================"
echo ""
echo "Cluster:   ${CLUSTER_NAME} (${AWS_REGION})"
echo "Namespace: ${NAMESPACE}"
echo ""
echo "Nodes:"
kubectl get nodes -o wide
echo ""
echo "Pods:"
kubectl -n "$NAMESPACE" get pods -o wide
echo ""
echo "Monitoring:"
kubectl -n monitoring get pods --no-headers | head -5
echo ""
echo "=== Next Steps ==="
echo "1. Port forward:   bash phase3/scripts/port-forward.sh"
echo "2. Run experiments: python3 phase3/tests/backpressure_comparison.py --host http://localhost:8080"
echo "3. Teardown:        bash phase3/scripts/teardown-cluster.sh"
