#!/usr/bin/env bash
# deploy.sh — Orchestrate Phase 4 inference platform rollout.
#
# WHAT THIS DOES (in order):
#   1. Apply ConfigMap (externalized scaling parameters)
#   2. Apply hardened Karpenter NodePool (wider limits, slower consolidation)
#   3. Apply PodDisruptionBudget (floor on running workers)
#   4. Apply KEDA ScaledObject (one of four scaling policies)
#   5. Apply Grafana dashboard patch (Row 5: scaling policy metrics)
#   6. Apply Deployment patch (init container + preStop + envFrom) — triggers rolling update
#   7. Wait for rollout to complete
#   8. Verify pods are running with correct Phase 4 config
#
# WHY THIS ORDER:
#   Resources that OTHER resources depend on go first. The Deployment references
#   the ConfigMap (envFrom), so the ConfigMap must exist before the Deployment
#   starts. The PDB must be in place before KEDA or Karpenter try to disrupt pods.
#   The Deployment goes LAST because it triggers the rolling update — everything
#   it depends on should already be in cluster.
#   Analogy: you create the launch template, scaling policy, and target group
#   before updating the ASG to use them. The ASG update triggers the instance refresh.
#
# USAGE:
#   bash phase4/scripts/deploy.sh                # deploy with Policy A (queue-only)
#   bash phase4/scripts/deploy.sh policy-b       # deploy with Policy B (queue-eager)
#   bash phase4/scripts/deploy.sh policy-c       # deploy with Policy C (composite-conservative)
#   bash phase4/scripts/deploy.sh policy-d       # deploy with Policy D (composite-aggressive)

set -euo pipefail

# ── Color Output ─────────────────────────────────────────────────────────
# WHY colors: scanning deploy output for failures in a wall of text is painful.
# Green = success, yellow = in progress, red = failure. Same idea as CloudFormation
# stack events using green/red/yellow for CREATE_COMPLETE/IN_PROGRESS/FAILED.
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Configuration ────────────────────────────────────────────────────────
NAMESPACE="inference-system"
DEPLOYMENT="vllm-worker"

# Accept optional policy argument; default to Policy A (queue-only baseline).
# WHY default to policy-a: it's the simplest (queue depth only, same as Phase 2
# but with higher maxReplicas). Starting simple lets us validate the deployment
# pipeline before testing composite triggers. Like starting with step scaling
# before enabling target tracking.
POLICY="${1:-policy-a}"

# Map the short policy name to the KEDA ScaledObject YAML file.
# WHY a case statement (not just string interpolation): validates the input
# and provides a clear error if the user passes an invalid policy name.
case "${POLICY}" in
    policy-a)
        KEDA_FILE="phase4/k8s/autoscaling/policy-a-queue-only.yaml"
        POLICY_DESC="Policy A: Queue-only (value=5)"
        ;;
    policy-b)
        KEDA_FILE="phase4/k8s/autoscaling/policy-b-queue-eager.yaml"
        POLICY_DESC="Policy B: Queue-eager (value=3)"
        ;;
    policy-c)
        KEDA_FILE="phase4/k8s/autoscaling/policy-c-composite-conservative.yaml"
        POLICY_DESC="Policy C: Composite-conservative (queue + KV cache)"
        ;;
    policy-d)
        KEDA_FILE="phase4/k8s/autoscaling/policy-d-composite-aggressive.yaml"
        POLICY_DESC="Policy D: Composite-aggressive (queue + KV cache, lower thresholds)"
        ;;
    *)
        error "Invalid policy: '${POLICY}'"
        echo "  Valid options: policy-a, policy-b, policy-c, policy-d"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Phase 4 — Inference Platform Deployment${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
info "Scaling Policy: ${POLICY_DESC}"
info "Namespace:      ${NAMESPACE}"
info "KEDA File:      ${KEDA_FILE}"
echo ""

# ── Cluster Context Safety Check ────────────────────────────────────────
# WHY this check: Phase 4 has its own EKS cluster (inference-phase4). Without
# this guard, running deploy.sh while pointed at inference-phase3 or inference-lab
# would deploy Phase 4 resources (NodePool, PDB, KEDA, Deployment) to the WRONG
# cluster — silently corrupting that cluster's state. Fail fast.
# Analogy: like checking you're deploying to the staging account, not production,
# before running `cdk deploy`.
EXPECTED_CLUSTER="inference-phase4"
CURRENT_CONTEXT=$(kubectl config current-context 2>/dev/null || echo "none")
if [[ "${CURRENT_CONTEXT}" != *"${EXPECTED_CLUSTER}"* ]]; then
    error "Wrong kubectl context!"
    error "  Current: ${CURRENT_CONTEXT}"
    error "  Expected context containing: ${EXPECTED_CLUSTER}"
    echo ""
    echo "  Fix: aws eks update-kubeconfig --name ${EXPECTED_CLUSTER} --region us-east-1"
    exit 1
fi
success "Cluster context verified: ${EXPECTED_CLUSTER}"
echo ""

# ── Pre-flight Checks ───────────────────────────────────────────────────
# WHY check cluster access first: failing 4 steps into a deploy because kubectl
# isn't configured is frustrating. Fail fast, like validating CloudFormation
# parameters before starting the stack create.
info "Pre-flight: checking cluster access..."
if ! kubectl get namespace "${NAMESPACE}" &>/dev/null; then
    error "Namespace '${NAMESPACE}' not found. Is the cluster running and Phase 1 deployed?"
    echo "  Try: kubectl get nodes"
    exit 1
fi
success "Cluster access confirmed"
echo ""

# ── Step 1: Apply ConfigMap ──────────────────────────────────────────────
# WHY first: the Deployment's envFrom references this ConfigMap. If the
# ConfigMap doesn't exist when the pod starts, all envFrom keys become empty
# strings — silent misconfiguration. Creating it first prevents that.
# Analogy: you write the SSM Parameter Store values before the instances
# that read them launch.
info "[1/7] Applying scaling ConfigMap..."
kubectl apply -f phase4/k8s/scaling-config/configmap.yaml
success "ConfigMap 'scaling-config' applied"
echo ""

# ── Step 2: Apply Hardened Karpenter NodePool ────────────────────────────
# WHY second: the NodePool defines WHERE and HOW new GPU nodes are provisioned.
# If KEDA scales to 2 replicas and Karpenter doesn't have the updated limits
# (32 CPU / 128Gi for 4 nodes), the new pod stays Pending forever.
# Analogy: updating the ASG's max capacity and launch template before triggering
# the instance refresh.
info "[2/7] Applying hardened Karpenter NodePool..."
kubectl apply -f phase4/k8s/karpenter/gpu-nodepool.yaml
success "Karpenter NodePool 'gpu-inference' updated"
echo ""

# ── Step 3: Apply PodDisruptionBudget ────────────────────────────────────
# WHY before KEDA and Deployment: PDB protects running pods from voluntary
# disruption. If Karpenter tries to consolidate during the rolling update,
# PDB prevents it from evicting the last worker. Must be in place before
# any disruption source (KEDA scale-down, Karpenter consolidation) is active.
# Analogy: enabling instance protection on the ASG before modifying scaling policies.
info "[3/7] Applying PodDisruptionBudget..."
kubectl apply -f phase4/k8s/worker/pdb.yaml
success "PDB 'vllm-worker-pdb' applied (minAvailable=1)"
echo ""

# ── Step 4: Apply KEDA ScaledObject ──────────────────────────────────────
# WHY before Deployment: KEDA watches the Deployment's replica count. Applying
# the ScaledObject first means KEDA is ready to react as soon as the new pods
# come up. If we applied the Deployment first, there's a window where scale-up
# decisions could be made against stale (Phase 2) KEDA config.
# Analogy: attaching the new scaling policy to the ASG before the instance refresh
# replaces instances.
info "[4/7] Applying KEDA ScaledObject (${POLICY})..."
kubectl apply -f "${KEDA_FILE}"
success "ScaledObject applied: ${POLICY_DESC}"
echo ""

# ── Step 5: Apply Grafana Dashboard Patch ────────────────────────────────
# WHY before Deployment: the dashboard should be ready to show Phase 4 metrics
# the moment the new pods start emitting them. Non-blocking — if the patch
# doesn't exist yet (still in development), warn and continue.
# Analogy: updating the CloudWatch dashboard before deploying, so you can
# watch the rollout in real time.
DASHBOARD_FILE="phase4/k8s/monitoring/grafana-dashboard-patch.yaml"
info "[5/7] Applying Grafana dashboard patch..."
if [ -f "${DASHBOARD_FILE}" ]; then
    kubectl apply -f "${DASHBOARD_FILE}"
    success "Grafana dashboard patch applied"
else
    warn "Dashboard file not found: ${DASHBOARD_FILE}"
    warn "Skipping — Phase 4 dashboard is still in development (task #9)"
fi
echo ""

# ── Step 6: Apply Deployment Patch ───────────────────────────────────────
# WHY last: this is the "big bang" step. Applying the Deployment triggers a
# rolling update that replaces running pods with the Phase 4 spec (init container,
# preStop hook, envFrom ConfigMap, phase4 worker image). Every dependency
# (ConfigMap, NodePool, PDB, KEDA) is already in cluster.
# Analogy: this is the "start instance refresh" API call. Everything it references
# (launch template, scaling policy, target group) must already exist.
info "[6/7] Applying Deployment patch (triggers rolling update)..."
kubectl apply -f phase4/k8s/worker/deployment-patch.yaml
success "Deployment '${DEPLOYMENT}' updated — rolling update started"
echo ""

# ── Step 7: Wait for Rollout ─────────────────────────────────────────────
# WHY 420s timeout (down from 600s): Phase 4v3 eliminated the fixed 120s
# initialDelaySeconds on readiness. The startup probe detects vLLM health
# the moment it happens (~35s), then worker warmup runs (~30s), then the
# file-based readiness probe passes (~5s). With SOCI, image unpack drops
# from 110s to ~15s.
# New worst case: node provision (90s) + SOCI image start (15s) + init (20s)
# + GPU load (30s) + startup probe (5s) + warmup (30s) + readiness (5s)
# = ~195s. 420s gives 3+ min headroom.
# Without SOCI: add 95s for full image unpack = ~290s. Still under 420s.
# Analogy: waiting for an ASG instance refresh to reach 100% healthy.
info "[7/7] Waiting for rollout to complete (timeout: 420s)..."
echo "  This includes: init container (S3 model download) + vLLM GPU load + startup probe + warmup"
if kubectl rollout status "deployment/${DEPLOYMENT}" -n "${NAMESPACE}" --timeout=420s; then
    echo ""
    success "Rollout complete!"
else
    echo ""
    error "Rollout timed out or failed."
    error "Debug: kubectl describe deployment ${DEPLOYMENT} -n ${NAMESPACE}"
    error "Debug: kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/name=${DEPLOYMENT}"
    exit 1
fi

# ── Verification ─────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Verification${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

# Print pod status — quick visual check that pods are Running, not CrashLoopBackOff.
info "Pod status:"
kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/name="${DEPLOYMENT}" -o wide
echo ""

# Check worker logs for Phase 4 startup message and strategy name.
# WHY: the Phase 4 worker prints its configuration on startup. Seeing "Phase 4"
# and the correct strategy name confirms the new code is actually running,
# not a cached old image.
info "Checking worker logs for Phase 4 startup..."
WORKER_POD=$(kubectl get pods -n "${NAMESPACE}" \
    -l app.kubernetes.io/name="${DEPLOYMENT}" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -n "${WORKER_POD}" ]; then
    # Grab the last 30 lines from queue-worker container — startup messages appear early.
    LOGS=$(kubectl logs "${WORKER_POD}" -n "${NAMESPACE}" -c queue-worker --tail=30 2>/dev/null || echo "")
    if echo "${LOGS}" | grep -qi "phase 4\|Phase4\|phase_4\|scaling.config\|ADMISSION_STRATEGY"; then
        success "Phase 4 worker confirmed in logs"
        echo "${LOGS}" | grep -i "phase\|strategy\|config\|admission\|drain" | head -5 | sed 's/^/  /'
    else
        warn "Could not find 'Phase 4' in worker logs (worker may still be starting)"
        echo "  Check manually: kubectl logs ${WORKER_POD} -n ${NAMESPACE} -c queue-worker"
    fi
else
    warn "No worker pod found — rollout may still be in progress"
fi

# Check if SOCI lazy loading is active by examining node events/labels
echo ""
info "Checking if SOCI lazy loading is active..."
NODE_NAME=$(kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/name="${DEPLOYMENT}" \
    -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || echo "")
if [ -n "${NODE_NAME}" ]; then
    # Check if the node was provisioned with a SOCI AMI
    AMI_LABEL=$(kubectl get node "${NODE_NAME}" \
        -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}' 2>/dev/null || echo "unknown")
    success "Worker running on ${NODE_NAME} (${AMI_LABEL})"
    # WHY check events: if SOCI is working, containerd reports the image as
    # "pulled" much faster than the 110s baseline. A pull under 30s strongly
    # suggests SOCI lazy loading is active.
    PULL_EVENT=$(kubectl get events -n "${NAMESPACE}" \
        --field-selector "reason=Pulled,involvedObject.name=${WORKER_POD}" \
        -o jsonpath='{.items[0].message}' 2>/dev/null || echo "no pull event found")
    info "Image pull event: ${PULL_EVENT}"
else
    warn "Could not determine node name for SOCI check"
fi

echo ""
info "Active KEDA ScaledObject:"
kubectl get scaledobject -n "${NAMESPACE}" -o wide 2>/dev/null || warn "No ScaledObject found"
echo ""

info "PDB status:"
kubectl get pdb -n "${NAMESPACE}" 2>/dev/null || warn "No PDB found"
echo ""

echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Phase 4 Deployment Complete${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo "  Policy:    ${POLICY_DESC}"
echo "  Namespace: ${NAMESPACE}"
echo ""
echo "  Next steps:"
echo "    1. Port forward:      bash phase3/scripts/port-forward.sh"
echo "    2. Switch policy:     bash phase4/scripts/deploy.sh policy-b"
echo "    3. Run policy test:   python3 phase4/tests/scaling_policy_comparison.py --host http://localhost:8080"
echo "    4. Rollback:          bash phase4/scripts/rollback.sh"
echo ""
echo "  Prerequisites (one-time, if not already done):"
echo "    - Build SOCI AMI:     bash phase4/scripts/build-ami-soci.sh (or build-ami.sh without SOCI)"
echo "    - Setup SOCI:         bash phase4/scripts/setup-soci.sh (push vLLM image + index to ECR)"
echo "    - Upload model to S3: bash phase4/scripts/upload-model-to-s3.sh"
echo ""
