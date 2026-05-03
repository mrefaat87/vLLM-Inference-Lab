#!/usr/bin/env bash
# rollback.sh — Revert Phase 4 inference platform back to Phase 2/3 state.
#
# WHAT THIS DOES (in order):
#   1. Reapply Phase 2 KEDA ScaledObject (queue-only, maxReplicas=2)
#   2. Reapply Phase 1 Karpenter NodePool (smaller limits, faster consolidation)
#   3. Delete Phase 4 PodDisruptionBudget
#   4. Delete Phase 4 ConfigMap (scaling-config)
#   5. Roll back Deployment to previous revision (kubectl rollout undo)
#   6. Wait for rollout to complete
#   7. Verify pods are running with pre-Phase-4 config
#
# WHY THIS ORDER:
#   Reverse of deploy.sh. First, revert the external resources that the Deployment
#   depends on (KEDA, NodePool). Then remove Phase 4-only resources (PDB, ConfigMap).
#   The Deployment rollback goes second-to-last because it triggers the rolling update.
#   This way, when the new (old) pods start, they find the correct Phase 2 KEDA
#   and Phase 1 NodePool already in place.
#
#   Analogy: rolling back an instance refresh. First restore the old scaling policy
#   and launch template, then cancel the refresh. The instances that come up use
#   the restored config.
#
# WHY rollout undo (not kubectl apply -f phase1/...):
#   kubectl rollout undo uses Kubernetes' built-in revision history — it knows
#   the exact previous Deployment spec, including the correct image tag. This is
#   safer than manually applying an old manifest that might be stale.
#   Analogy: ASG rollback to previous launch template version (a known-good snapshot)
#   vs manually recreating the old launch template from memory.
#
# USAGE:
#   bash phase4.5/scripts/rollback.sh         # interactive (asks for confirmation)
#   bash phase4.5/scripts/rollback.sh --yes   # skip confirmation (for CI/automation)

set -euo pipefail

# ── Color Output ─────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Configuration ────────────────────────────────────────────────────────
NAMESPACE="inference-system"
DEPLOYMENT="vllm-worker"
SKIP_CONFIRM="${1:-}"

echo ""
echo -e "${RED}================================================================${NC}"
echo -e "${RED}  Phase 4 — Rollback to Phase 2/3${NC}"
echo -e "${RED}================================================================${NC}"
echo ""

# ── Cluster Context Safety Check ────────────────────────────────────────
# WHY this check: Phase 4 has its own EKS cluster (inference-phase4-5). Without
# this guard, running rollback.sh while pointed at inference-phase3 or inference-lab
# would roll back the WRONG cluster's deployment. Fail fast.
EXPECTED_CLUSTER="inference-phase4-5"
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

# ── Pre-flight: Show Current State ───────────────────────────────────────
# WHY show state before rollback: the operator should see what they're reverting
# FROM. Prevents accidental rollbacks when already on the correct version.
# Like checking the current ASG launch template version before rolling back.
info "Current deployment state:"
kubectl get deployment "${DEPLOYMENT}" -n "${NAMESPACE}" -o wide 2>/dev/null || true
echo ""
info "Current pods:"
kubectl get pods -n "${NAMESPACE}" -l "app.kubernetes.io/name=${DEPLOYMENT}" -o wide 2>/dev/null || true
echo ""
info "Current KEDA ScaledObject:"
kubectl get scaledobject -n "${NAMESPACE}" -o wide 2>/dev/null || true
echo ""

# ── Confirmation Prompt ──────────────────────────────────────────────────
# WHY confirm: rollback is destructive — it replaces running pods, changes KEDA
# config, and deletes the PDB. An accidental rollback during a load test would
# corrupt results. The --yes flag skips this for automation.
# Analogy: the "Type the stack name to confirm deletion" prompt in CloudFormation.
if [ "${SKIP_CONFIRM}" != "--yes" ]; then
    echo -e "${YELLOW}This will revert the inference platform to Phase 2/3 state.${NC}"
    echo "  - KEDA ScaledObject -> Phase 2 (queue-only, maxReplicas=2)"
    echo "  - Karpenter NodePool -> Phase 1 (8 CPU / 32Gi limits)"
    echo "  - PDB and ConfigMap will be deleted"
    echo "  - Deployment will rollback to previous revision"
    echo ""
    read -p "Proceed with rollback? (y/N): " CONFIRM
    if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
        info "Rollback cancelled."
        exit 0
    fi
    echo ""
fi

# ── Step 1: Reapply Phase 2 KEDA ScaledObject ───────────────────────────
# WHY first: KEDA might try to scale based on the Phase 4 composite triggers
# during rollback. Reverting to Phase 2's simpler queue-only trigger prevents
# KEDA from making unexpected scaling decisions while we're changing other
# resources. Like disabling the aggressive scaling policy before modifying the ASG.
info "[1/6] Reverting KEDA ScaledObject to Phase 2..."
if [ -f "phase2/k8s/autoscaling/scaledobject-vllm.yaml" ]; then
    kubectl apply -f phase2/k8s/autoscaling/scaledobject-vllm.yaml
    success "KEDA ScaledObject reverted to Phase 2 (queue-only, maxReplicas=2)"
else
    warn "Phase 2 ScaledObject not found. Deleting Phase 4 ScaledObject instead."
    # WHY delete instead of error: if the Phase 2 file was moved/renamed, deleting
    # the ScaledObject disables autoscaling entirely — safer than leaving Phase 4
    # triggers active during a rollback.
    kubectl delete scaledobject -n "${NAMESPACE}" --all --ignore-not-found
    warn "All ScaledObjects deleted. Manual KEDA reconfig needed."
fi
echo ""

# ── Step 2: Reapply Phase 1 Karpenter NodePool ──────────────────────────
# WHY revert NodePool: Phase 4 raised limits to 32 CPU / 128Gi (4 GPU nodes).
# Rolling back to Phase 1 limits (8 CPU / 32Gi = 1 node) prevents Karpenter
# from provisioning extra nodes that the Phase 2 KEDA config (maxReplicas=2)
# can't fill. Wasted nodes = wasted Spot spend.
# Analogy: reducing ASG MaxCapacity back to 1 when the scaling policy only
# targets 1-2 instances.
info "[2/6] Reverting Karpenter NodePool to Phase 1..."
if [ -f "phase1/k8s/gpu-nodepool.yaml" ]; then
    kubectl apply -f phase1/k8s/gpu-nodepool.yaml
    success "Karpenter NodePool reverted to Phase 1 limits"
else
    warn "Phase 1 NodePool not found at phase1/k8s/gpu-nodepool.yaml"
    warn "Skipping — Karpenter NodePool unchanged"
fi
echo ""

# ── Step 3: Delete PodDisruptionBudget ───────────────────────────────────
# WHY delete (not update): PDB doesn't exist in Phase 1/2. Leaving an orphaned
# PDB could block future node drains or Karpenter consolidation unexpectedly.
# Clean deletion is safer than leaving stale resources.
# Analogy: removing instance protection when downgrading from a multi-AZ HA
# setup back to single-instance.
info "[3/6] Deleting Phase 4 PodDisruptionBudget..."
kubectl delete pdb vllm-worker-pdb -n "${NAMESPACE}" --ignore-not-found
# WHY --ignore-not-found: idempotent — safe to run even if PDB was already deleted
# or never created. Like "delete stack if exists" pattern.
success "PDB deleted (or didn't exist)"
echo ""

# ── Step 4: Delete ConfigMap ─────────────────────────────────────────────
# WHY delete: Phase 1/2 workers don't use envFrom scaling-config. The ConfigMap
# would be harmless (orphaned, unused), but cleaning it up prevents confusion
# when inspecting the cluster state.
info "[4/6] Deleting Phase 4 ConfigMap..."
kubectl delete configmap scaling-config -n "${NAMESPACE}" --ignore-not-found
success "ConfigMap 'scaling-config' deleted (or didn't exist)"
echo ""

# ── Step 5: Rollback Deployment ──────────────────────────────────────────
# WHY rollout undo: Kubernetes stores Deployment revision history (default: 10
# revisions). rollout undo reverts to the immediately previous revision — the
# exact spec that was running before Phase 4. This is more reliable than
# applying phase1/k8s/vllm-worker/deployment.yaml, which might not match
# the actual running state (e.g., Phase 3 patches applied on top).
#
# Analogy: ASG "rollback to previous launch template version" — it uses the
# exact known-good config, not a potentially stale file on disk.
info "[5/6] Rolling back Deployment to previous revision..."
kubectl rollout undo "deployment/${DEPLOYMENT}" -n "${NAMESPACE}"
success "Deployment rollback initiated"
echo ""

# ── Step 6: Wait for Rollout ─────────────────────────────────────────────
# WHY 600s: same reasoning as deploy.sh — vLLM model load + readiness probe
# can take several minutes. The rollback might also trigger a new node if the
# previous revision has different resource requests.
info "[6/6] Waiting for rollout to complete (timeout: 600s)..."
if kubectl rollout status "deployment/${DEPLOYMENT}" -n "${NAMESPACE}" --timeout=600s; then
    echo ""
    success "Rollback complete!"
else
    echo ""
    error "Rollback timed out or failed."
    error "Debug: kubectl describe deployment ${DEPLOYMENT} -n ${NAMESPACE}"
    error "Debug: kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/name=${DEPLOYMENT}"
    exit 1
fi

# ── Verification ─────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Rollback Verification${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

info "Pod status:"
kubectl get pods -n "${NAMESPACE}" -l "app.kubernetes.io/name=${DEPLOYMENT}" -o wide
echo ""

info "Deployment revision history:"
kubectl rollout history "deployment/${DEPLOYMENT}" -n "${NAMESPACE}" 2>/dev/null | tail -5
echo ""

info "Active KEDA ScaledObject:"
kubectl get scaledobject -n "${NAMESPACE}" -o wide 2>/dev/null || warn "No ScaledObject found"
echo ""

# Confirm no Phase 4 resources remain.
info "Checking for orphaned Phase 4 resources..."
ORPHANS=0
if kubectl get configmap scaling-config -n "${NAMESPACE}" &>/dev/null; then
    warn "ConfigMap 'scaling-config' still exists"
    ORPHANS=$((ORPHANS + 1))
fi
if kubectl get pdb vllm-worker-pdb -n "${NAMESPACE}" &>/dev/null; then
    warn "PDB 'vllm-worker-pdb' still exists"
    ORPHANS=$((ORPHANS + 1))
fi
if [ "${ORPHANS}" -eq 0 ]; then
    success "No orphaned Phase 4 resources found"
fi

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Rollback to Phase 2/3 Complete${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo "  The inference platform is now running Phase 2/3 configuration."
echo "  KEDA: queue-only trigger, maxReplicas=2"
echo "  Karpenter: Phase 1 limits"
echo ""
echo "  To redeploy Phase 4:"
echo "    bash phase4.5/scripts/deploy.sh [policy-a|policy-b|policy-c|policy-d]"
echo ""
