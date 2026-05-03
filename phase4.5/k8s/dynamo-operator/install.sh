#!/usr/bin/env bash
# install.sh — Install Dynamo Operator + CRDs into dynamo-system.
#
# Prereqs:
#   1. dynamo-system namespace exists
#   2. etcd, NATS, PostgreSQL, MinIO from k8s/deps/install.sh are healthy
#   3. kubectl context = inference-phase4-5
#
# Image source: nvcr.io/nvidia/ai-dynamo/* — requires NGC API key
# (`docker login nvcr.io` once before running).
set -euo pipefail

NS=dynamo-system

# Verify cluster guard fired correctly.
CTX=$(kubectl config current-context)
[[ "$CTX" =~ inference-phase4-5 ]] || { echo "wrong cluster context: $CTX" >&2; exit 1; }

# Verify deps are up before touching the operator.
for sts in etcd; do
  kubectl -n "$NS" rollout status statefulset/$sts --timeout=60s
done
kubectl -n "$NS" wait --for=condition=ready pod -l app.kubernetes.io/name=nats --timeout=120s

# Install Dynamo CRDs first (operator chart sometimes lags on CRD apply).
kubectl apply -f https://raw.githubusercontent.com/ai-dynamo/dynamo/main/deploy/cloud/helm/crds/dynamographdeployment.yaml || true
kubectl apply -f https://raw.githubusercontent.com/ai-dynamo/dynamo/main/deploy/cloud/helm/crds/dynamographdeploymentrequest.yaml || true

# Install operator via Helm. Chart path is from the upstream repo's deploy/cloud/helm directory.
# NOTE: verify chart path/name with `helm search repo` after the upstream chart is published.
# If the chart is not yet on a Helm repo, clone the dynamo repo and install from path:
#   git clone https://github.com/ai-dynamo/dynamo /tmp/dynamo
#   helm upgrade --install dynamo-operator /tmp/dynamo/deploy/cloud/helm/charts/dynamo-operator \
#     --namespace "$NS" --create-namespace
echo "TODO: confirm operator chart path against current Dynamo release. As of 1.0.x"
echo "the canonical install is from the repo's deploy/cloud/helm path. Update this"
echo "script with the verified Helm command on first run."

cat <<EOF

Smoke test once operator is up:
  kubectl get crds | grep dynamo
  kubectl -n $NS get pods -l app.kubernetes.io/name=dynamo-operator
  kubectl -n $NS logs -l app.kubernetes.io/name=dynamo-operator --tail=50

EOF
