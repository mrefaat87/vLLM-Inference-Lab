#!/usr/bin/env bash
# Tear down everything created by bringup.sh.
#
# Order matters: NodePool first (so Karpenter releases nodes), then
# Karpenter chart, then Terraform destroy.
#
# Flags:
#   --yes      non-interactive (CI)
#   --dry-run  show what would be destroyed without doing it
#   --keep-state  don't delete the Terraform state bucket / DDB table
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

YES=0
DRY=0
KEEP_STATE=1   # default: leave backend alone (cheap, useful for re-bringup)
for arg in "$@"; do
  case "$arg" in
    --yes) YES=1 ;;
    --dry-run) DRY=1 ;;
    --delete-state) KEEP_STATE=0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

confirm() {
  [[ $YES -eq 1 ]] && return 0
  read -r -p "$1 [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]]
}

run() {
  if [[ $DRY -eq 1 ]]; then
    echo "DRYRUN: $*"
  else
    "$@"
  fi
}

REGION="${AWS_REGION:-us-east-1}"
CLUSTER="inference-lab"

echo "About to tear down cluster '$CLUSTER' in region '$REGION'."
confirm "Continue?" || { echo "aborted"; exit 1; }

export KUBECONFIG="$HERE/kubeconfig"

if [[ -f "$KUBECONFIG" ]]; then
  echo "==> Delete engine workloads"
  run kubectl --context "$CLUSTER" -n engines delete deploy,svc,job -l project=inference-lab --ignore-not-found=true --wait=false || true

  echo "==> Delete NodePool (lets Karpenter drain nodes)"
  run kubectl --context "$CLUSTER" delete -f manifests/karpenter-nodepool.yaml --ignore-not-found=true || true

  echo "==> Uninstall Karpenter"
  run helm --kubeconfig "$KUBECONFIG" uninstall karpenter -n kube-system || true
fi

echo "==> Terraform destroy"
run terraform -chdir=terraform destroy -auto-approve

if [[ $KEEP_STATE -eq 0 ]]; then
  echo "==> Delete Terraform state backend (irreversible)"
  confirm "Really delete the tfstate bucket + DDB lock table?" || { echo "kept state"; exit 0; }
  run aws s3 rb --force "s3://inference-lab-tfstate" || true
  run aws dynamodb delete-table --table-name inference-lab-tflock --region "$REGION" || true
fi

echo "==> Final verification"
run "$HERE/verify_teardown.sh"

echo "Teardown complete."
