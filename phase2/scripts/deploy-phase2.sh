#!/usr/bin/env bash
# deploy-phase2.sh — End-to-end deployment of Phase 2: Observability + Autoscaling.
#
# PREREQUISITES:
#   - Phase 1 fully deployed (namespace inference-system, all pods running)
#   - kubectl configured for the EKS cluster
#   - Helm 3 installed
#   - Docker + ECR access (for building API gateway v2 image)
#
# WHAT THIS DEPLOYS:
#   Part A — Observability:
#     - RabbitMQ Prometheus plugin (patches existing deployment)
#     - API gateway v2 with Prometheus metrics (builds + patches deployment)
#     - vLLM headless Service for metrics scraping
#     - DCGM Exporter DaemonSet (GPU hardware metrics)
#     - kube-prometheus-stack (Prometheus + Grafana + Operator)
#     - ServiceMonitors (vLLM, RabbitMQ, API gateway, DCGM)
#     - Grafana dashboard ConfigMap (14-panel inference dashboard)
#
#   Part B — Autoscaling:
#     - KEDA (event-driven autoscaler)
#     - ScaledObject watching RabbitMQ queue depth
#
# USAGE: bash phase2/scripts/deploy-phase2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PHASE2_DIR="$PROJECT_ROOT/phase2"
NAMESPACE="inference-system"

echo "================================================================"
echo "  Phase 2: Observability + Autoscaling Deployment"
echo "================================================================"
echo ""

# ---- Step 0: Verify Phase 1 is healthy ----
echo "[0/10] Verifying Phase 1 is healthy..."
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
    echo "ERROR: Namespace '$NAMESPACE' not found. Is Phase 1 deployed?"
    exit 1
fi

RUNNING_PODS=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l | tr -d ' ')
echo "  Found $RUNNING_PODS running pods in $NAMESPACE"
if [ "$RUNNING_PODS" -lt 3 ]; then
    echo "  WARNING: Expected at least 3 running pods (api-gateway, rabbitmq, redis)."
    echo "  Current pods:"
    kubectl get pods -n "$NAMESPACE" --no-headers
    echo ""
    read -p "  Continue anyway? (y/N): " -r
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi
echo ""

# ---- Step 1: Build and push API Gateway v2 image ----
echo "[1/10] Building API Gateway v2 (metrics-instrumented)..."
bash "$PHASE2_DIR/scripts/build-api-gateway-v2.sh"
echo ""

# ---- Step 2: Apply RabbitMQ patches (ConfigMap + Deployment + Service) ----
echo "[2/10] Patching RabbitMQ with Prometheus plugin..."
# WHY ConfigMap first: the Deployment references it as a volume.
# If we apply the Deployment first, the volume mount would fail.
kubectl apply -f "$PHASE2_DIR/k8s/patches/rabbitmq-plugins-configmap.yaml"
kubectl apply -f "$PHASE2_DIR/k8s/patches/rabbitmq-deployment.yaml"
kubectl apply -f "$PHASE2_DIR/k8s/patches/rabbitmq-service.yaml"
echo "  Waiting for RabbitMQ to restart with Prometheus plugin..."
kubectl rollout status deployment/rabbitmq -n "$NAMESPACE" --timeout=120s
echo ""

# ---- Step 3: Apply API Gateway patch ----
echo "[3/10] Deploying API Gateway v2 (with /metrics endpoint)..."
# WHY sed: replace placeholder account ID with actual account ID.
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
sed "s/019167255542/${AWS_ACCOUNT_ID}/g" \
    "$PHASE2_DIR/k8s/patches/api-gateway-deployment.yaml" | kubectl apply -f -
echo "  Waiting for API gateway rolling update..."
kubectl rollout status deployment/api-gateway -n "$NAMESPACE" --timeout=120s
echo ""

# ---- Step 4: Apply vLLM metrics headless Service + DCGM Exporter ----
echo "[4/10] Deploying vLLM metrics Service and DCGM Exporter..."
kubectl apply -f "$PHASE2_DIR/k8s/monitoring/vllm-metrics-service.yaml"
kubectl apply -f "$PHASE2_DIR/k8s/monitoring/dcgm-exporter.yaml"
echo ""

# ---- Step 5: Install kube-prometheus-stack via Helm ----
echo "[5/10] Installing kube-prometheus-stack (Prometheus + Grafana)..."
# WHY separate namespace: monitoring tools should be isolated from application
# workloads for RBAC and lifecycle management. Standard K8s convention.
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo update prometheus-community

# WHY --create-namespace: the monitoring namespace may not exist yet.
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --create-namespace \
    --values "$PHASE2_DIR/k8s/monitoring/prometheus-values.yaml" \
    --wait \
    --timeout 5m
echo ""

# ---- Step 6: Apply ServiceMonitors ----
echo "[6/10] Applying ServiceMonitors (vLLM, RabbitMQ, API gateway, DCGM)..."
kubectl apply -f "$PHASE2_DIR/k8s/monitoring/servicemonitor-vllm.yaml"
kubectl apply -f "$PHASE2_DIR/k8s/monitoring/servicemonitor-rabbitmq.yaml"
kubectl apply -f "$PHASE2_DIR/k8s/monitoring/servicemonitor-api-gateway.yaml"
kubectl apply -f "$PHASE2_DIR/k8s/monitoring/servicemonitor-dcgm.yaml"
echo ""

# ---- Step 7: Apply Grafana dashboard ----
echo "[7/10] Deploying Grafana inference dashboard..."
kubectl apply -f "$PHASE2_DIR/k8s/monitoring/grafana-dashboard-configmap.yaml"
echo "  Dashboard will auto-load via Grafana sidecar (label: grafana_dashboard=1)"
echo ""

# ---- Step 8: Install KEDA via Helm ----
echo "[8/10] Installing KEDA (event-driven autoscaler)..."
helm repo add kedacore https://kedacore.github.io/charts 2>/dev/null || true
helm repo update kedacore

helm upgrade --install keda kedacore/keda \
    --namespace keda \
    --create-namespace \
    --values "$PHASE2_DIR/k8s/autoscaling/keda-values.yaml" \
    --wait \
    --timeout 5m
echo ""

# ---- Step 9: Apply KEDA ScaledObject ----
echo "[9/10] Deploying KEDA ScaledObject for vLLM worker autoscaling..."
kubectl apply -f "$PHASE2_DIR/k8s/autoscaling/keda-rabbitmq-auth.yaml"
kubectl apply -f "$PHASE2_DIR/k8s/autoscaling/scaledobject-vllm.yaml"
echo ""

# ---- Step 10: Verify everything ----
echo "[10/10] Verifying deployment..."
echo ""
echo "--- Inference System Pods ---"
kubectl get pods -n "$NAMESPACE" -o wide
echo ""
echo "--- Monitoring Pods ---"
kubectl get pods -n monitoring -o wide
echo ""
echo "--- KEDA Pods ---"
kubectl get pods -n keda -o wide
echo ""
echo "--- ScaledObject Status ---"
kubectl get scaledobject -n "$NAMESPACE" -o wide 2>/dev/null || echo "  (ScaledObject may take a moment to appear)"
echo ""
echo "--- ServiceMonitors ---"
kubectl get servicemonitor -n "$NAMESPACE"
echo ""

echo "================================================================"
echo "  Phase 2 Deployment Complete!"
echo "================================================================"
echo ""
echo "Access dashboards via port-forward:"
echo "  bash phase2/scripts/port-forward-phase2.sh"
echo ""
echo "Or manually:"
echo "  Grafana:    kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n monitoring"
echo "  Prometheus: kubectl port-forward svc/kube-prometheus-stack-prometheus 9090:9090 -n monitoring"
echo "  API GW:     kubectl port-forward svc/api-gateway 8080:8080 -n inference-system"
echo "  RabbitMQ:   kubectl port-forward svc/rabbitmq 15672:15672 -n inference-system"
echo ""
echo "Grafana login: admin / inference-lab"
echo "Dashboard: Inference Platform (auto-provisioned)"
echo ""
echo "Run the autoscale load test:"
echo "  python3 phase2/tests/autoscale_load_test.py --host http://localhost:8080"
