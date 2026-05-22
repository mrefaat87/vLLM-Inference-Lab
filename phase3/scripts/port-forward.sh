#!/bin/bash
# port-forward.sh — Forward API gateway and Grafana to localhost.
#
# WHY port-forward (not LoadBalancer Service):
#   1. No public IP exposure — cluster stays private
#   2. No additional AWS cost (NLB = ~$16/month)
#   3. Good enough for testing — production would use an Ingress controller
#
# Ports:
#   8080 → API Gateway (submit jobs, stream results)
#   3000 → Grafana (dashboards for GPU/KV/latency metrics)
#   15672 → RabbitMQ Management (queue depth monitoring)

set -euo pipefail

NAMESPACE="inference-system"
MONITORING_NS="monitoring"

echo "=== Phase 3 — Port Forward ==="
echo "Starting port-forwards (Ctrl+C to stop)..."
echo ""
echo "  API Gateway:  http://localhost:8080"
echo "  Grafana:      http://localhost:3000"
echo "  RabbitMQ Mgmt: http://localhost:15672"
echo ""

# Kill any existing port-forwards on these ports
for port in 8080 3000 15672; do
    lsof -ti:${port} 2>/dev/null | xargs kill -9 2>/dev/null || true
done

# Start port-forwards in background
kubectl port-forward svc/api-gateway -n "$NAMESPACE" 8080:8080 &
PID_API=$!

kubectl port-forward svc/kube-prometheus-stack-grafana -n "$MONITORING_NS" 3000:3000 &
PID_GRAFANA=$!

kubectl port-forward svc/rabbitmq -n "$NAMESPACE" 15672:15672 &
PID_RABBIT=$!

echo "Port-forwards started (PIDs: api=$PID_API grafana=$PID_GRAFANA rabbit=$PID_RABBIT)"
echo "Press Ctrl+C to stop all port-forwards"

# Cleanup on exit
cleanup() {
    echo ""
    echo "Stopping port-forwards..."
    kill $PID_API $PID_GRAFANA $PID_RABBIT 2>/dev/null || true
    echo "Done"
}
trap cleanup EXIT

# Wait for any background process to exit
wait
