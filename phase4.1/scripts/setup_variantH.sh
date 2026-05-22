#!/usr/bin/env bash
# setup_variantH.sh — Stages 3-6 of the Variant H plan, in one re-runnable script.
#
# 1. terraform apply for FSx Lustre + SG + Karpenter node IAM patch
# 2. Render PV manifest with terraform outputs and apply PV/PVC
# 3. Apply Variant H NodePool
# 4. Run the populate Job and wait for completion
#
# This is idempotent — safe to re-run if a step fails. Each step verifies
# state before mutating.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TF_DIR="${PHASE_DIR}/terraform"
K8S_DIR="${PHASE_DIR}/k8s"

export KUBECONFIG="${PHASE_DIR}/.kubeconfig"

step() { echo ""; echo "════════ $* ════════"; }

# ── Stage 3: Terraform — FSx, SG, IAM ───────────────────────────────────────
step "Stage 3: terraform apply (FSx + SG + Karpenter node IAM)"
cd "$TF_DIR"
terraform plan -target=aws_security_group.fsx_lustre \
  -target=aws_security_group_rule.node_from_fsx_988 \
  -target=aws_security_group_rule.node_from_fsx_ephemeral \
  -target=aws_fsx_lustre_file_system.phase41_variantH \
  -target=aws_iam_role_policy.karpenter_node_fsx_describe \
  -out=variantH.tfplan
echo ""
echo "Review plan above. Continue? [y/N]"
read -r confirm
[ "$confirm" = "y" ] || { echo "aborted"; exit 1; }

terraform apply variantH.tfplan
rm -f variantH.tfplan

FSX_ID=$(terraform output -raw fsx_id)
FSX_DNS=$(terraform output -raw fsx_dns_name)
FSX_MOUNT=$(terraform output -raw fsx_mount_name)
echo ""
echo "FSx ready: ${FSX_ID}"
echo "  DNS:   ${FSX_DNS}"
echo "  Mount: ${FSX_MOUNT}"

# ── Stage 4: PV / PVC ───────────────────────────────────────────────────────
step "Stage 4: render and apply PV/PVC"
sed "s|FSX_FS_ID|${FSX_ID}|; s|FSX_DNS_NAME|${FSX_DNS}|; s|FSX_MOUNT_NAME|${FSX_MOUNT}|" \
  "${K8S_DIR}/fsx-lustre-pv.yaml" | kubectl apply -f -
kubectl apply -f "${K8S_DIR}/fsx-lustre-pvc.yaml"

# Don't proceed until the PVC binds. If it stays Pending >2 min, almost
# certainly an SG / DNS issue (port 988 / 1018-1023). Stop and debug.
echo "Waiting for PVC to bind (timeout 2m)..."
if ! kubectl wait --for=jsonpath='{.status.phase}'=Bound \
  -n baseline pvc/model-weights-fsx --timeout=2m; then
  echo "ABORT: PVC did not bind. Run kubectl describe pvc -n baseline model-weights-fsx" >&2
  exit 1
fi

# ── Stage 6 (NodePool): apply BEFORE populate Job since the job needs the pool ─
step "Stage 6 (NodePool): apply Variant H Karpenter NodePool"
kubectl apply -f "${K8S_DIR}/karpenter/gpu-nodepool-prebaked-fsx.yaml"

# ── Stage 5: populate Job ───────────────────────────────────────────────────
step "Stage 5: run populate Job (copies Qwen weights to FSx)"
# Idempotent: delete any prior Job before re-applying.
kubectl delete job -n baseline fsx-populate --ignore-not-found
kubectl apply -f "${K8S_DIR}/fsx-populate-job.yaml"

echo "Waiting for Job to complete (timeout 15m)..."
if ! kubectl wait --for=condition=complete \
  -n baseline job/fsx-populate --timeout=15m; then
  echo "ABORT: populate Job did not complete." >&2
  echo "Diagnostics:" >&2
  kubectl logs -n baseline -l job-name=fsx-populate --tail=200 >&2
  exit 1
fi

echo ""
echo "Populate Job tail:"
kubectl logs -n baseline -l job-name=fsx-populate --tail=30

step "Setup complete — ready to run measurements"
echo "Next: bash scripts/run_variantH.sh"
