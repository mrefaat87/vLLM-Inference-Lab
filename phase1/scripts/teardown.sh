#!/bin/bash
# teardown.sh — Destroy all Phase 1 infrastructure.
#
# WHY confirmation prompt: terraform destroy is irreversible. A single
# accidental run could delete the EKS cluster, all running workloads,
# and the VPC. This is the equivalent of `aws cloudformation delete-stack`
# with no undo — always require explicit human confirmation.
#
# WHAT'S PRESERVED: The S3 model cache bucket is retained by default
# (lifecycle policy in terraform). Model weights are expensive to re-download
# (7B models take ~10 min on a fast connection).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TERRAFORM_DIR="${PROJECT_DIR}/terraform"
NAMESPACE="inference-system"

echo "============================================"
echo "  Inference Lab — Phase 1 Teardown"
echo "============================================"
echo ""
echo "This will DESTROY:"
echo "  - All K8s resources in namespace '${NAMESPACE}'"
echo "  - EKS cluster and all nodes (including GPU Spot instances)"
echo "  - VPC, subnets, NAT Gateway, security groups"
echo "  - IAM roles (Karpenter, node, pod identity)"
echo "  - ECR repositories (images will be lost)"
echo ""
echo "This will PRESERVE:"
echo "  - S3 model cache bucket (contains downloaded model weights)"
echo "  - Terraform state file (for reference)"
echo ""

# ---------------------------------------------------------------------------
# Confirmation — ALWAYS require explicit yes
# WHY not --auto-approve here: this is destructive. Even in a learning lab,
# accidentally nuking infra wastes 30+ minutes re-provisioning.
# ---------------------------------------------------------------------------
read -p "Are you sure you want to destroy everything? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted. Nothing was destroyed."
    exit 0
fi

echo ""

# ---------------------------------------------------------------------------
# Step 1: Delete K8s namespace
# WHY delete namespace (not individual resources): deleting the namespace
# cascades to all resources within it — pods, services, configmaps, secrets,
# PVCs, etc. Like deleting a CloudFormation stack — one command, clean slate.
# ---------------------------------------------------------------------------
echo "=== Step 1: Delete K8s namespace ==="

# WHY check cluster connectivity first: if the cluster is already gone
# (e.g., from a previous partial teardown), kubectl commands will hang.
if kubectl cluster-info &>/dev/null; then
    echo "Deleting namespace '${NAMESPACE}'..."
    kubectl delete namespace "$NAMESPACE" --timeout=120s 2>/dev/null || {
        echo "WARNING: Namespace deletion timed out or already deleted"
    }

    # WHY wait for namespace deletion: K8s needs time to terminate pods,
    # release PVCs, and clean up finalizers. If we destroy the cluster
    # before this finishes, resources can get stuck in "Terminating" state
    # and leak (e.g., EBS volumes not deleted).
    echo "Waiting for namespace to be fully deleted..."
    kubectl wait --for=delete namespace/"$NAMESPACE" --timeout=120s 2>/dev/null || true

    echo "Namespace deleted"
else
    echo "Cluster not reachable — skipping K8s cleanup"
fi

echo ""

# ---------------------------------------------------------------------------
# Step 2: Wait for GPU nodes to terminate
# WHY wait: Karpenter-managed Spot nodes run outside the managed node group.
# Deleting the namespace removes the pods, which makes the nodes empty,
# which triggers Karpenter's consolidation (WhenEmpty + 60s). We wait for
# this to avoid orphaned EC2 instances that terraform doesn't track.
# ---------------------------------------------------------------------------
echo "=== Step 2: Wait for GPU nodes ==="

if kubectl cluster-info &>/dev/null; then
    echo "Checking for GPU nodes..."
    GPU_NODES=$(kubectl get nodes -l node-role=gpu --no-headers 2>/dev/null | wc -l | tr -d ' ')

    if [ "$GPU_NODES" -gt 0 ]; then
        echo "Found ${GPU_NODES} GPU node(s). Waiting for Karpenter to terminate them..."
        echo "(This may take up to 2 minutes — Karpenter waits 60s before consolidation)"

        # Poll until GPU nodes are gone or timeout after 3 minutes
        TIMEOUT=180
        ELAPSED=0
        while [ "$ELAPSED" -lt "$TIMEOUT" ]; do
            GPU_NODES=$(kubectl get nodes -l node-role=gpu --no-headers 2>/dev/null | wc -l | tr -d ' ')
            if [ "$GPU_NODES" -eq 0 ]; then
                echo "All GPU nodes terminated"
                break
            fi
            echo "  ${GPU_NODES} GPU node(s) remaining... (${ELAPSED}s/${TIMEOUT}s)"
            sleep 10
            ELAPSED=$((ELAPSED + 10))
        done

        if [ "$GPU_NODES" -gt 0 ]; then
            echo "WARNING: ${GPU_NODES} GPU node(s) still running after ${TIMEOUT}s"
            echo "They will be terminated when the cluster is destroyed by terraform"
        fi
    else
        echo "No GPU nodes found"
    fi
else
    echo "Cluster not reachable — skipping node check"
fi

echo ""

# ---------------------------------------------------------------------------
# Step 3: Terraform destroy
# WHY -auto-approve here: the user already confirmed at the top of the script.
# A second confirmation would be annoying.
# ---------------------------------------------------------------------------
echo "=== Step 3: Terraform destroy ==="
cd "$TERRAFORM_DIR"

echo "Running terraform destroy..."
terraform destroy -auto-approve

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Teardown Complete"
echo "============================================"
echo ""
echo "Destroyed:"
echo "  - EKS cluster and all nodes"
echo "  - VPC and networking"
echo "  - IAM roles and policies"
echo ""
echo "Preserved:"
echo "  - S3 model cache bucket (check AWS console)"
echo "  - Terraform state file (${TERRAFORM_DIR}/terraform.tfstate)"
echo "  - ECR images (if repos had lifecycle policies)"
echo ""
echo "To re-deploy: bash scripts/setup.sh"
