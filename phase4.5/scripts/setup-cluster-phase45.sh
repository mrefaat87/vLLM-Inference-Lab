#!/usr/bin/env bash
# setup-cluster-phase45.sh — Post-terraform cluster setup for the Dynamo spike.
#
# Assumes: `terraform apply` already succeeded for phase4.5/terraform.
# Performs (in order):
#   1. Configure kubectl for inference-phase4-5
#   2. Apply the Karpenter GPU NodePool (g5.8xlarge + EFA-labeled)
#   3. Install EFA device plugin
#   4. Install Dynamo deps (etcd, NATS, postgres, minio)
#   5. Install Dynamo Operator (manual confirm step inside)
#
# Each step prints a clear OK / FAIL marker so we can resume mid-way if needed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TF_DIR="$ROOT/terraform"

step() { printf "\n=== %s ===\n" "$1"; }

step "1/5: kubectl context"
CLUSTER_NAME=$(terraform -chdir="$TF_DIR" output -raw cluster_name)
[[ "$CLUSTER_NAME" == "inference-phase4-5" ]] || {
  echo "unexpected cluster name from terraform: $CLUSTER_NAME" >&2; exit 1; }
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region us-east-1 \
  --kubeconfig "$ROOT/.kubeconfig"
export KUBECONFIG="$ROOT/.kubeconfig"
kubectl cluster-info
echo "OK"

step "2/5: Karpenter GPU NodePool"
kubectl apply -f "$ROOT/k8s/karpenter/gpu-nodepool.yaml"
kubectl get nodepool gpu-pool -o yaml | head -25
echo "OK (no nodes yet — Karpenter provisions on demand)"

step "3/5: EFA device plugin"
bash "$ROOT/k8s/efa-device-plugin/install.sh"

step "4/5: Dynamo dependencies"
bash "$ROOT/k8s/deps/install.sh"

step "5/5: Dynamo Operator"
echo "The operator install script has a manual confirmation step (the chart"
echo "path needs to be verified against the current Dynamo release). Run:"
echo "  bash $ROOT/k8s/dynamo-operator/install.sh"
echo
echo "Once the operator is up, validate with:"
echo "  kubectl get crds | grep dynamo"
echo "  kubectl -n dynamo-system get pods"
