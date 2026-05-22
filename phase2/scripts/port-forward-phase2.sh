#!/usr/bin/env bash
# port-forward-phase2.sh — Forward all Phase 2 services to localhost for testing.
#
# WHAT THIS FORWARDS:
#   localhost:8080  → API Gateway (inference requests)
#   localhost:15672 → RabbitMQ Management UI (queue inspection)
#   localhost:3000  → Grafana (dashboards)
#   localhost:9090  → Prometheus (raw metrics + targets)
#
# WHY port-forward: Phase 1-2 use ClusterIP Services (no external access).
# Port-forward creates a tunnel from your laptop through the K8s API server
# to the pod. For production access, you'd use an Ingress + NLB instead.
#
# NOTE: port-forward adds ~400ms latency (laptop → K8s API → pod).
# This affects client-side TTFT measurements but not server-side metrics.
#
# USAGE: bash phase2/scripts/port-forward-phase2.sh
# STOP:  Ctrl+C (kills all background port-forwards)

set -euo pipefail

NAMESPACE="inference-system"
MONITORING_NS="monitoring"

echo "=== Phase 2 Port Forwards ==="
echo ""

# Cleanup on exit — kill all background processes
cleanup() {
    echo ""
    echo "Stopping all port-forwards..."
    kill $(jobs -p) 2>/dev/null
    echo "Done."
}
trap cleanup EXIT

# API Gateway — inference requests and /metrics
echo "  API Gateway:     http://localhost:8080  (POST /generate, GET /stream, GET /metrics)"
kubectl port-forward svc/api-gateway 8080:8080 -n "$NAMESPACE" &

# RabbitMQ Management — queue inspection
echo "  RabbitMQ Mgmt:   http://localhost:15672 (login: inference / inference123)"
kubectl port-forward svc/rabbitmq 15672:15672 -n "$NAMESPACE" &

# Grafana — dashboards
echo "  Grafana:         http://localhost:3000  (login: admin / inference-lab)"
kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n "$MONITORING_NS" &

# Prometheus — raw metrics and target status
echo "  Prometheus:      http://localhost:9090  (check Status → Targets for scrape health)"
kubectl port-forward svc/kube-prometheus-stack-prometheus 9090:9090 -n "$MONITORING_NS" &

echo ""
echo "All port-forwards active. Press Ctrl+C to stop."
echo ""
echo "Quick verification:"
echo "  curl http://localhost:8080/health        # API gateway health"
echo "  curl http://localhost:8080/metrics        # Gateway Prometheus metrics"
echo "  curl http://localhost:9090/-/healthy      # Prometheus health"
echo ""

# Wait for all background processes — keeps the script alive until Ctrl+C
wait
