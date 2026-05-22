#!/bin/bash
# port_forward.sh — Forward K8s services to localhost for local testing.
#
# WHY port-forward (not a LoadBalancer service): during development, we don't need
# an internet-facing ALB. Port-forward creates a secure tunnel through kubectl,
# avoiding the cost of an NLB (~$16/mo) and the security surface of a public endpoint.
# Same idea as SSH tunneling to an internal RDS instance.
#
# USAGE: ./scripts/port_forward.sh
#   Then test with: curl -X POST http://localhost:8080/generate \
#     -H "Content-Type: application/json" \
#     -d '{"prompt": "Hello, world!", "max_tokens": 50}'

set -euo pipefail

NAMESPACE="inference-system"

echo "=== Port Forwarding ==="
echo ""
echo "Starting port forwards in background..."
echo ""

# ---------------------------------------------------------------------------
# API Gateway — primary interface for inference requests
# WHY 8080→8080: matches the container port so curl commands in docs work
# without mental translation.
# ---------------------------------------------------------------------------
echo "API Gateway:          http://localhost:8080"
kubectl -n "$NAMESPACE" port-forward svc/api-gateway 8080:8080 &
API_GW_PID=$!

# ---------------------------------------------------------------------------
# RabbitMQ Management UI — for inspecting queue depth, message rates
# WHY 15672: RabbitMQ's standard management port. Useful for debugging
# queue backlog (analogous to CloudWatch SQS metrics dashboard).
# Default credentials: guest/guest
# ---------------------------------------------------------------------------
echo "RabbitMQ Management:  http://localhost:15672  (inference/inference123)"
kubectl -n "$NAMESPACE" port-forward svc/rabbitmq 15672:15672 &
RABBIT_PID=$!

echo ""
echo "=== Test Commands ==="
echo ""
echo "1. Health check:"
echo "   curl http://localhost:8080/health"
echo ""
echo "2. Submit inference job:"
echo "   curl -X POST http://localhost:8080/generate \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"prompt\": \"Explain KV cache in 3 sentences.\", \"max_tokens\": 100}'"
echo ""
echo "3. Stream results (replace JOB_ID):"
echo "   curl -N http://localhost:8080/stream/JOB_ID"
echo ""
echo "4. Check queue depth:"
echo "   Open http://localhost:15672 → Queues → inference_queue"
echo ""
echo "=== Press Ctrl+C to stop all port forwards ==="
echo ""

# WHY trap: ensures all background port-forward processes are killed when
# the user presses Ctrl+C. Without this, orphaned kubectl processes linger
# and hold the ports, forcing manual cleanup.
cleanup() {
    echo ""
    echo "Stopping port forwards..."
    kill "$API_GW_PID" "$RABBIT_PID" 2>/dev/null || true
    echo "Done"
}
trap cleanup EXIT INT TERM

# WHY wait: keeps the script running so the trap can fire on Ctrl+C.
# Without wait, the script exits immediately and the trap kills the
# background processes before they establish connections.
wait
