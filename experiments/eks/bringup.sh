#!/usr/bin/env bash
# Bring up the inference-lab cluster end-to-end.
#   1. Terraform apply (cluster, VPC, IAM, ECR, S3 weights bucket)
#   2. Fetch kubeconfig (to ./eks/kubeconfig)
#   3. Install Karpenter via Helm
#   4. Apply Karpenter NodePool + EC2NodeClass
#   5. Install NVIDIA device plugin
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
REGION="${AWS_REGION:-us-east-1}"
CLUSTER="inference-lab"

./preflight.sh

echo "==> terraform init/apply"
terraform -chdir=terraform init -upgrade
terraform -chdir=terraform apply -auto-approve

echo "==> kubeconfig"
aws eks update-kubeconfig --region "$REGION" --name "$CLUSTER" \
  --alias "$CLUSTER" --kubeconfig "$HERE/kubeconfig"
export KUBECONFIG="$HERE/kubeconfig"

# Hard refuse to proceed if context isn't us. Matches the project's
# per-phase cluster guard convention.
CURRENT_CTX=$(kubectl config current-context)
if [[ "$CURRENT_CTX" != "$CLUSTER" ]]; then
  echo "ERROR: expected context '$CLUSTER', got '$CURRENT_CTX'" >&2
  exit 1
fi

echo "==> Karpenter helm install"
helm registry logout public.ecr.aws 2>/dev/null || true
KARPENTER_VERSION="${KARPENTER_VERSION:-1.0.6}"
helm upgrade --install karpenter oci://public.ecr.aws/karpenter/karpenter \
  --version "$KARPENTER_VERSION" \
  --namespace kube-system \
  --set "settings.clusterName=$CLUSTER" \
  --set "settings.interruptionQueue=Karpenter-$CLUSTER" \
  --set "serviceAccount.annotations.eks\.amazonaws\.com/role-arn=$(terraform -chdir=terraform output -raw karpenter_iam_role_arn)" \
  --wait

echo "==> NodePool"
kubectl apply -f manifests/karpenter-nodepool.yaml

echo "==> engines namespace (workload pods land here)"
kubectl create namespace engines --dry-run=client -o yaml | kubectl apply -f -

echo "==> NVIDIA device plugin"
# The plugin manifest moved under /deployments/static/ in v0.15.0 — the
# bare /nvidia-device-plugin.yml URL returns the README, not YAML.
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml

echo
echo "Bringup complete. Verify with:"
echo "  KUBECONFIG=$HERE/kubeconfig kubectl get nodes"
