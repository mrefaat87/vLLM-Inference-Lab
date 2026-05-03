#!/usr/bin/env bash
# install.sh — Install Dynamo control-plane dependencies into dynamo-system.
#
# What this provisions and why each is needed:
#   etcd       — service discovery for Dynamo workers (router watches it)
#   NATS       — KV-event bus (workers publish block_added/evicted; router subscribes)
#   PostgreSQL — Dynamo API Store backing
#   MinIO      — Dynamo artifact store backing
#
# All single-replica because spike. HA versions of these are Phase 7 territory.
set -euo pipefail

NS=dynamo-system
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
KC="$ROOT/.kubeconfig"
CTX=inference-phase4-5

K="kubectl --kubeconfig=$KC --context=$CTX"
H="helm --kubeconfig=$KC --kube-context=$CTX"

$K create namespace "$NS" --dry-run=client -o yaml | $K apply -f -

# Add Helm repos (idempotent — repos are stored in user homedir, not cluster).
helm repo add bitnami https://charts.bitnami.com/bitnami >/dev/null
helm repo add nats https://nats-io.github.io/k8s/helm/charts/ >/dev/null
helm repo update >/dev/null

$H upgrade --install etcd bitnami/etcd \
  --namespace "$NS" --values "$HERE/etcd-values.yaml" --wait --timeout 5m

$H upgrade --install nats nats/nats \
  --namespace "$NS" --values "$HERE/nats-values.yaml" --wait --timeout 5m

$H upgrade --install postgresql bitnami/postgresql \
  --namespace "$NS" --values "$HERE/postgresql-values.yaml" --wait --timeout 5m

$H upgrade --install minio bitnami/minio \
  --namespace "$NS" --values "$HERE/minio-values.yaml" --wait --timeout 5m

echo
echo "Verify:"
echo "  $K -n $NS get pods,svc"
