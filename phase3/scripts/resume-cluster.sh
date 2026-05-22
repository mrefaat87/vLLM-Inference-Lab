#!/bin/bash
# resume-cluster.sh — Scale up the EKS cluster from stopped state.
#
# WHY this script exists: cluster is kept at 0 nodes when not in use to save
# costs (~$0.45/hr while running). This script brings it back to operational
# state: CPU nodes for infra pods, GPU node for inference.
#
# SEQUENCE:
#   1. Scale CPU node group to 2 (infra: RabbitMQ, Redis, API Gateway, Prometheus)
#   2. Wait for CPU nodes to join and pods to schedule
#   3. GPU node is provisioned automatically by Karpenter when the vllm-worker pod
#      becomes schedulable (after infra pods are running)

set -euo pipefail

CLUSTER_NAME="inference-lab"
REGION=$(aws configure get region || echo "us-east-1")
NAMESPACE="inference-system"

echo "=== Resume Cluster ==="
echo "Cluster: ${CLUSTER_NAME}"
echo "Region:  ${REGION}"
echo ""

# Step 1: Scale CPU node group
echo "Scaling CPU node group to 2..."
# WHY 2 nodes: infra pods (RabbitMQ, Redis, API Gateway ×2, Prometheus stack)
# need ~4 CPU cores and ~4GB RAM. Two t3.medium (2 vCPU, 4GB each) provide
# enough capacity with some headroom for monitoring.
NODE_GROUP=$(aws eks list-nodegroups --cluster-name "$CLUSTER_NAME" --region "$REGION" --query 'nodegroups[0]' --output text)
aws eks update-nodegroup-config \
    --cluster-name "$CLUSTER_NAME" \
    --nodegroup-name "$NODE_GROUP" \
    --scaling-config minSize=2,maxSize=3,desiredSize=2 \
    --region "$REGION"
echo "Node group scaling initiated"

# Step 2: Wait for CPU nodes
echo ""
echo "Waiting for CPU nodes to be Ready (up to 3 min)..."
for i in $(seq 1 36); do
    READY_NODES=$(kubectl get nodes --no-headers 2>/dev/null | grep -c " Ready" || echo "0")
    echo -ne "\r  [$((i*5))s] Ready nodes: ${READY_NODES}/2"
    if [ "$READY_NODES" -ge 2 ]; then
        echo ""
        echo "  CPU nodes ready!"
        break
    fi
    sleep 5
done
echo ""

# Step 3: Check pod status
echo "Pod status:"
kubectl get pods -n "$NAMESPACE" -o wide 2>/dev/null || echo "  (namespace may not exist yet)"
echo ""
kubectl get pods -n monitoring -o wide 2>/dev/null || echo "  (monitoring namespace may not exist yet)"
echo ""

# Step 4: Wait for GPU node (Karpenter provisions when vllm-worker pod is pending)
echo "Waiting for GPU node + vLLM worker to be Ready (up to 7 min)..."
echo "  (Karpenter → Spot launch ~90s → model load ~3 min)"
for i in $(seq 1 84); do
    WORKER_READY=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=vllm-worker \
        --no-headers 2>/dev/null | grep -c "Running" || echo "0")
    echo -ne "\r  [$((i*5))s] vllm-worker Running: ${WORKER_READY}/1"
    if [ "$WORKER_READY" -ge 1 ]; then
        echo ""
        echo "  GPU worker ready!"
        break
    fi
    sleep 5
done
echo ""

echo "=== Cluster Status ==="
echo "Nodes:"
kubectl get nodes -o wide
echo ""
echo "Pods (inference-system):"
kubectl get pods -n "$NAMESPACE" -o wide
echo ""
echo "Pods (monitoring):"
kubectl get pods -n monitoring -o wide
echo ""

echo "Cluster is ready. Next steps:"
echo "  1. Build phase3 worker:  bash phase3/scripts/build_and_push.sh"
echo "  2. Deploy:               bash phase3/scripts/deploy.sh"
echo "  3. Port-forward:         bash phase3/scripts/port-forward.sh"
echo "  4. Run comparison test:  python3 phase3/tests/backpressure_comparison.py --host http://localhost:8080"
