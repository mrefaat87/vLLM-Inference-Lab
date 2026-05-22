#!/bin/bash
# teardown-cluster.sh — Destroy the Phase 4 experiment cluster.
#
# Deletes: VPC, EKS, Karpenter, IAM roles, SQS, EventBridge rules.
# Preserves: ECR repos, S3 model cache (shared with other phases).
#
# Usage: bash phase4/scripts/teardown-cluster.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE4_DIR="$(dirname "$SCRIPT_DIR")"
TF_DIR="${PHASE4_DIR}/terraform"
NAMESPACE="inference-system"

CLUSTER_NAME=$(terraform -chdir="$TF_DIR" output -raw cluster_name 2>/dev/null || echo "inference-phase4")

echo "============================================"
echo "  Phase 4 — Teardown Cluster"
echo "============================================"
echo ""
echo "  Cluster: ${CLUSTER_NAME}"
echo "  This will DESTROY the entire Phase 4 EKS cluster."
echo "  ECR repos and S3 model cache are PRESERVED."
echo ""
read -p "  Type 'destroy' to confirm: " CONFIRM
if [ "$CONFIRM" != "destroy" ]; then
    echo "  Aborted."
    exit 0
fi
echo ""

# Step 1: Delete K8s namespaces (cascades to all resources)
echo "=== Deleting K8s namespaces ==="
kubectl delete namespace "$NAMESPACE" --timeout=120s 2>/dev/null || echo "  Namespace already gone"
kubectl delete namespace monitoring --timeout=60s 2>/dev/null || echo "  Monitoring namespace already gone"
kubectl delete namespace keda --timeout=60s 2>/dev/null || echo "  KEDA namespace already gone"

# Step 2: Wait for GPU nodes to terminate (Karpenter consolidation)
echo ""
echo "=== Waiting for GPU nodes to terminate ==="
for i in $(seq 1 36); do
    GPU_NODES=$(kubectl get nodes -l karpenter.sh/nodepool=gpu-pool --no-headers 2>/dev/null | wc -l | tr -d ' ')
    echo -ne "\r  [$((i*5))s] GPU nodes: ${GPU_NODES}"
    if [ "$GPU_NODES" -eq 0 ]; then
        echo ""
        echo "  GPU nodes terminated."
        break
    fi
    sleep 5
done
echo ""

# Step 3: Uninstall Helm releases
echo "=== Uninstalling Helm releases ==="
helm uninstall keda -n keda 2>/dev/null || true
helm uninstall kube-prometheus-stack -n monitoring 2>/dev/null || true
echo ""

# Step 4: Terraform destroy
echo "=== Terraform destroy ==="
cd "$TF_DIR"
terraform destroy -auto-approve

echo ""
echo "============================================"
echo "  Phase 4 Cluster Destroyed"
echo "============================================"
echo ""
echo "  Preserved:"
echo "    - ECR repos (inference-lab/api-gateway, inference-lab/worker, inference-lab/vllm-openai)"
echo "    - S3 model cache bucket"
echo ""
echo "  To recreate:"
echo "    cd phase4/terraform && terraform apply"
