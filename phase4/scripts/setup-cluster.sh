#!/bin/bash
# setup-cluster.sh — Provision a separate EKS cluster for Phase 4 experiments.
#
# WHAT THIS DOES:
#   1. Terraform: VPC, EKS, Karpenter, IAM (reuses ECR + S3 from Phase 1)
#   2. kubectl: configure access to the new cluster
#   3. K8s: deploy namespace, GPU nodepool, NVIDIA plugin, RabbitMQ, Redis,
#           API gateway, vLLM worker (base from Phase 1)
#   4. Phase 4 overlays: hardened NodePool, ConfigMap, PDB, Deployment patch
#   5. Helm: KEDA + optional Prometheus/Grafana
#
# WHY separate cluster: Phase 4 previously deployed to Phase 3's cluster by
# mistake, causing experiment corruption. This cluster is fully isolated —
# its own VPC (10.4.0.0/16), its own IAM roles (inference-phase4-*), its own
# Karpenter discovery tags.
#
# COST: ~$0.23/hr idle ($0.10 EKS + $0.08 CPU + $0.045 NAT)
#        ~$0.55/hr with GPU ($0.16-0.50 Spot g4dn.xlarge)
#
# Usage: bash phase4/scripts/setup-cluster.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE4_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$PHASE4_DIR")"
PHASE1_DIR="${PROJECT_DIR}/phase1"
PHASE2_DIR="${PROJECT_DIR}/phase2"
TF_DIR="${PHASE4_DIR}/terraform"
K8S_DIR="${PHASE1_DIR}/k8s"
NAMESPACE="inference-system"

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

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Phase 4 — Scaling Policy Experiment Cluster${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Terraform
# WHY: creates the isolated VPC, EKS cluster, Karpenter controller, and IAM
# roles. S3 bucket and ECR repos are referenced as data sources (not created).
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Step 1: Terraform (VPC + EKS + Karpenter) ==="
cd "$TF_DIR"

terraform init
terraform apply -auto-approve

CLUSTER_NAME=$(terraform output -raw cluster_name)
AWS_REGION=$(terraform output -raw region)
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo ""
info "Cluster: ${CLUSTER_NAME}"
info "Region:  ${AWS_REGION}"
info "ECR:     ${ECR_URL}"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Configure kubectl for the NEW cluster
# WHY: after terraform creates the cluster, kubectl still points at whatever
# context was active before. This switches to inference-phase4.
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Step 2: Configure kubectl ==="
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$AWS_REGION"

# Safety check: verify we're pointing at the right cluster
CURRENT_CONTEXT=$(kubectl config current-context 2>/dev/null || echo "none")
if [[ "${CURRENT_CONTEXT}" != *"${CLUSTER_NAME}"* ]]; then
    error "kubectl context mismatch!"
    error "  Expected: ${CLUSTER_NAME}"
    error "  Got: ${CURRENT_CONTEXT}"
    exit 1
fi
success "kubectl configured for ${CLUSTER_NAME}"
kubectl cluster-info
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Deploy base K8s manifests (reuse Phase 1 manifests)
# WHY reuse: same architecture (namespace, GPU nodepool, NVIDIA plugin,
# RabbitMQ, Redis, API gateway, vLLM worker sidecar). Phase 4 overlays
# are applied on top in Step 4.
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Step 3: Deploy base K8s manifests ==="

# Create a temporary working copy of K8s manifests to sed without modifying originals
WORK_DIR=$(mktemp -d)
cp -r "$K8S_DIR"/* "$WORK_DIR/"
trap "rm -rf $WORK_DIR" EXIT

# Replace placeholders with actual values
info "Patching manifests with account ID and ECR URL..."
find "$WORK_DIR" -name "*.yaml" -exec \
    sed -i.bak "s|ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com|${ECR_URL}|g" {} \;
find "$WORK_DIR" -name "*.yaml" -exec \
    sed -i.bak "s|ACCOUNT_ID|${AWS_ACCOUNT_ID}|g" {} \;

# Replace ALL "inference-lab" references with the Phase 4 cluster name.
# WHY global replace: the GPU nodepool yaml has "inference-lab" in the IAM role
# name, subnet/SG discovery tags, and project tags — not just the Karpenter
# discovery tag. Missing any of these causes GPU nodes to launch in Phase 1's
# VPC or with the wrong IAM role, failing to register with Phase 4's cluster.
find "$WORK_DIR" -name "*.yaml" -exec \
    sed -i.bak "s|inference-lab|${CLUSTER_NAME}|g" {} \;
find "$WORK_DIR" -name "*.yaml.bak" -delete 2>/dev/null || true

# Also update the vLLM worker IRSA role ARN to the Phase 4 role
VLLM_ROLE_ARN=$(terraform -chdir="$TF_DIR" output -raw vllm_worker_role_arn 2>/dev/null || echo "")
if [ -n "$VLLM_ROLE_ARN" ]; then
    info "Patching vLLM worker ServiceAccount with role: ${VLLM_ROLE_ARN}"
    find "$WORK_DIR" -name "*.yaml" -exec \
        sed -i.bak "s|eks.amazonaws.com/role-arn:.*|eks.amazonaws.com/role-arn: ${VLLM_ROLE_ARN}|g" {} \;
    find "$WORK_DIR" -name "*.yaml.bak" -delete 2>/dev/null || true
fi

# Deploy in dependency order
info "Creating namespace..."
kubectl apply -f "$WORK_DIR/namespace.yaml"

info "Applying GPU NodePool (base — will be overridden by Phase 4 in Step 4)..."
kubectl apply -f "$WORK_DIR/gpu-nodepool.yaml"

info "Installing NVIDIA device plugin..."
kubectl apply -f "$WORK_DIR/nvidia-device-plugin.yaml"

info "Deploying RabbitMQ..."
kubectl apply -f "$WORK_DIR/rabbitmq/"

info "Deploying Redis..."
kubectl apply -f "$WORK_DIR/redis/"

info "Waiting for RabbitMQ..."
kubectl -n "$NAMESPACE" wait --for=condition=ready pod -l app.kubernetes.io/name=rabbitmq --timeout=180s || warn "RabbitMQ not ready yet"

info "Waiting for Redis..."
kubectl -n "$NAMESPACE" wait --for=condition=ready pod -l app.kubernetes.io/name=redis --timeout=120s || warn "Redis not ready yet"

info "Deploying API Gateway..."
kubectl apply -f "$WORK_DIR/api-gateway/"

info "Deploying vLLM Worker (base from Phase 1)..."
kubectl apply -f "$WORK_DIR/vllm-worker/"

success "Base K8s manifests deployed"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Apply Phase 4 overlays
# WHY separate step: Phase 4 replaces several Phase 1 resources with hardened
# versions (NodePool with higher limits, Deployment with init container + SOCI,
# PDB for disruption protection, ConfigMap for externalized scaling params).
# These must be applied AFTER the base manifests so we don't hit missing-
# dependency errors (e.g., the Deployment references the ConfigMap via envFrom).
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Step 4: Apply Phase 4 overlays ==="

info "Applying scaling ConfigMap..."
kubectl apply -f "${PHASE4_DIR}/k8s/scaling-config/configmap.yaml"

info "Applying hardened Karpenter NodePool (overrides Phase 1)..."
kubectl apply -f "${PHASE4_DIR}/k8s/karpenter/gpu-nodepool.yaml"

info "Applying PodDisruptionBudget..."
kubectl apply -f "${PHASE4_DIR}/k8s/worker/pdb.yaml"

info "Applying Phase 4 Deployment (init container + startup probe + envFrom)..."
kubectl apply -f "${PHASE4_DIR}/k8s/worker/deployment-patch.yaml"

success "Phase 4 overlays applied"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Install monitoring + KEDA
# WHY: KEDA provides autoscaling based on RabbitMQ queue depth and KV cache
# utilization. Prometheus + Grafana are optional but needed for the Phase 4
# dashboard (Row 5: scale-up health metrics).
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Step 5: Install KEDA + monitoring ==="

# Add Helm repos
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo add kedacore https://kedacore.github.io/charts 2>/dev/null || true
helm repo update

# KEDA (required for autoscaling)
info "Installing KEDA..."
helm install keda kedacore/keda \
    -n keda --create-namespace \
    --wait --timeout 3m || warn "KEDA install may have issues"

# Apply Phase 4 default KEDA ScaledObject (Policy A: queue-only baseline)
info "Applying KEDA ScaledObject (Policy A: queue-only)..."
kubectl apply -f "${PHASE4_DIR}/k8s/autoscaling/policy-a-queue-only.yaml" || warn "KEDA ScaledObject may need RabbitMQ auth"

# Prometheus + Grafana (optional but recommended for the Phase 4 dashboard)
info "Installing kube-prometheus-stack..."
if [ -f "${PHASE2_DIR}/k8s/monitoring/prometheus-values.yaml" ]; then
    HELM_ARGS=("-f" "${PHASE2_DIR}/k8s/monitoring/prometheus-values.yaml")
    # Add image renderer values if Phase 3 has them
    if [ -f "${PHASE4_DIR}/../phase3/k8s/grafana-image-renderer-values.yaml" ]; then
        HELM_ARGS+=("-f" "${PHASE4_DIR}/../phase3/k8s/grafana-image-renderer-values.yaml")
    fi
    helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
        -n monitoring --create-namespace \
        "${HELM_ARGS[@]}" \
        --wait --timeout 5m || warn "Prometheus stack install may have issues"

    # Grafana dashboards
    if [ -f "${PHASE2_DIR}/k8s/monitoring/grafana-dashboard-configmap.yaml" ]; then
        info "Deploying base Grafana dashboard..."
        kubectl apply -f "${PHASE2_DIR}/k8s/monitoring/grafana-dashboard-configmap.yaml"
    fi

    # Phase 4 dashboard patch (Row 5: scale-up health)
    if [ -f "${PHASE4_DIR}/k8s/monitoring/grafana-dashboard-patch.yaml" ]; then
        info "Applying Phase 4 Grafana dashboard patch..."
        kubectl apply -f "${PHASE4_DIR}/k8s/monitoring/grafana-dashboard-patch.yaml"
    fi

    # ServiceMonitors + DCGM Exporter
    kubectl apply -f "${PHASE2_DIR}/k8s/monitoring/" 2>/dev/null || true
    if [ -f "${PHASE2_DIR}/k8s/monitoring/dcgm-exporter.yaml" ]; then
        kubectl apply -f "${PHASE2_DIR}/k8s/monitoring/dcgm-exporter.yaml" 2>/dev/null || true
    fi
else
    warn "Phase 2 monitoring values not found — skipping Prometheus/Grafana"
fi

success "KEDA + monitoring installed"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Wait for GPU node + vLLM worker
# WHY: Karpenter needs to provision a Spot GPU instance (~90s), then the init
# container downloads the model from S3 (~20s), then vLLM loads onto GPU (~30s),
# then the startup probe passes and the worker warmup runs (~30s).
# Total: ~3-5 min for first readiness.
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Step 6: Waiting for GPU node + vLLM worker ==="
echo "  Karpenter -> Spot launch ~90s -> model load ~3 min"
echo "  Watch: kubectl -n $NAMESPACE get pods -w"
echo ""

# Wait up to 7 min
for i in $(seq 1 84); do
    WORKER_READY=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=vllm-worker \
        --no-headers 2>/dev/null | grep -c "2/2" || echo "0")
    echo -ne "\r  [$((i*5))s] vllm-worker 2/2: ${WORKER_READY}/1"
    if [ "$WORKER_READY" -ge 1 ]; then
        echo ""
        success "vLLM worker READY!"
        break
    fi
    sleep 5
done

echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Final status
# ──────────────────────────────────────────────────────────────────────────────
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Phase 4 Cluster Ready!${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo "Cluster:   ${CLUSTER_NAME} (${AWS_REGION})"
echo "Namespace: ${NAMESPACE}"
echo "VPC CIDR:  10.4.0.0/16"
echo ""
echo "Nodes:"
kubectl get nodes -o wide
echo ""
echo "Pods:"
kubectl -n "$NAMESPACE" get pods -o wide
echo ""
echo "Monitoring:"
kubectl -n monitoring get pods --no-headers 2>/dev/null | head -5 || echo "  (not installed)"
echo ""
echo "=== Next Steps ==="
echo "1. Port forward:      bash phase3/scripts/port-forward.sh"
echo "2. Switch policy:     bash phase4/scripts/deploy.sh policy-b"
echo "3. Run policy test:   python3 phase4/tests/scaling_policy_comparison.py --host http://localhost:8080"
echo "4. Teardown:          terraform -chdir=${TF_DIR} destroy"
echo ""
echo "=== Prerequisites (one-time, if not already done) ==="
echo "- Upload model to S3: bash phase4/scripts/upload-model-to-s3.sh"
echo "- Build AMI (optional): bash phase4/scripts/build-ami-soci.sh"
echo ""
