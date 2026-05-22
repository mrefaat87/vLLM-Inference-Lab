# karpenter.tf — Install Karpenter via Helm and create namespace
#
# Karpenter is to Kubernetes what a custom ASG scaling policy is to EC2 — but smarter.
# Instead of scaling a node group and hoping pods fit, Karpenter looks at pending pods,
# finds the cheapest instance type that satisfies their resource requests, and launches it.
# For GPU workloads this is critical: a g4dn.xlarge costs $0.53/hr vs g5.xlarge at $1.01/hr.

# Create the karpenter namespace before installing the Helm chart
resource "kubernetes_namespace" "karpenter" {
  metadata {
    name = "karpenter"

    labels = {
      # Exempt from Kyverno/OPA policies if we add them later
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }

  depends_on = [module.eks]
}

# Install Karpenter controller using Helm
resource "helm_release" "karpenter" {
  namespace  = kubernetes_namespace.karpenter.metadata[0].name
  name       = "karpenter"

  # OCI registry — Karpenter moved from the old Helm repo to OCI in v0.32+
  repository = "oci://public.ecr.aws/karpenter"
  chart      = "karpenter"
  version    = "0.37.0" # Pinned — compatible with EKS 1.30

  # Wait for all pods to be ready before marking the release as successful.
  # Without this, terraform might proceed before Karpenter can process NodePools.
  wait = true
  timeout = 600 # 10 minutes — image pull can be slow on first deploy

  # ── Core settings ──────────────────────────────────────────────────────────

  # Cluster identity — Karpenter needs these to configure kubelet on new nodes
  set {
    name  = "settings.clusterName"
    value = module.eks.cluster_name
  }

  set {
    name  = "settings.clusterEndpoint"
    value = module.eks.cluster_endpoint
  }

  # Instance profile for nodes Karpenter launches (the "launch template" equivalent)
  set {
    name  = "settings.aws.defaultInstanceProfile"
    value = aws_iam_instance_profile.karpenter_node.name
  }

  # Interruption queue — Karpenter drains Spot nodes before termination
  # (like a lifecycle hook on an ASG that gracefully stops in-flight requests)
  set {
    name  = "settings.aws.interruptionQueueName"
    value = var.cluster_name
  }

  # IRSA — annotate the ServiceAccount so the pod gets Karpenter controller permissions
  set {
    name  = "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = module.karpenter_controller_irsa.iam_role_arn
  }

  # ── Scheduling ─────────────────────────────────────────────────────────────

  # Force Karpenter to run on CPU nodes only. It can't schedule itself onto nodes
  # it manages (chicken-and-egg problem). This is like pinning your scaling controller
  # to a dedicated On-Demand instance instead of letting it run on Spot.
  set {
    name  = "affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].key"
    value = "role"
  }

  set {
    name  = "affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].operator"
    value = "In"
  }

  set {
    name  = "affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].values[0]"
    value = "cpu"
  }

  # ── Replicas ───────────────────────────────────────────────────────────────

  # Single replica is fine for learning. In production you'd run 2+ for HA.
  set {
    name  = "replicas"
    value = "1"
  }

  depends_on = [
    module.eks,
    kubernetes_namespace.karpenter,
    aws_iam_instance_profile.karpenter_node,
    module.karpenter_controller_irsa,
  ]
}

# ─── SQS Queue for Spot Interruption Handling ────────────────────────────────────
# When AWS needs to reclaim a Spot GPU instance, it sends a 2-minute warning to this queue.
# Karpenter listens, cordons the node, and starts draining pods — like an ASG lifecycle hook
# that gives your app time to finish in-flight inference requests.

resource "aws_sqs_queue" "karpenter_interruption" {
  name                      = var.cluster_name
  message_retention_seconds = 300    # 5 min — interruption notices are time-sensitive
  sqs_managed_sse_enabled   = true   # Encrypt at rest (good practice, zero extra cost)

  tags = {
    Name = "${var.cluster_name}-karpenter-interruption"
  }
}

# Allow EventBridge to send Spot interruption events to this queue
resource "aws_sqs_queue_policy" "karpenter_interruption" {
  queue_url = aws_sqs_queue.karpenter_interruption.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowEventBridge"
        Effect = "Allow"
        Principal = {
          Service = ["events.amazonaws.com", "sqs.amazonaws.com"]
        }
        Action   = "sqs:SendMessage"
        Resource = aws_sqs_queue.karpenter_interruption.arn
      },
    ]
  })
}

# ── EventBridge Rules ────────────────────────────────────────────────────────────
# Capture EC2 events that affect Spot instances and route them to the SQS queue

resource "aws_cloudwatch_event_rule" "spot_interruption" {
  name        = "${var.cluster_name}-spot-interruption"
  description = "Capture EC2 Spot Instance interruption warnings"

  event_pattern = jsonencode({
    source      = ["aws.ec2"]
    detail-type = ["EC2 Spot Instance Interruption Warning"]
  })
}

resource "aws_cloudwatch_event_target" "spot_interruption" {
  rule      = aws_cloudwatch_event_rule.spot_interruption.name
  target_id = "karpenter-interruption-queue"
  arn       = aws_sqs_queue.karpenter_interruption.arn
}

resource "aws_cloudwatch_event_rule" "instance_rebalance" {
  name        = "${var.cluster_name}-instance-rebalance"
  description = "Capture EC2 Instance Rebalance Recommendation — early warning before interruption"

  event_pattern = jsonencode({
    source      = ["aws.ec2"]
    detail-type = ["EC2 Instance Rebalance Recommendation"]
  })
}

resource "aws_cloudwatch_event_target" "instance_rebalance" {
  rule      = aws_cloudwatch_event_rule.instance_rebalance.name
  target_id = "karpenter-interruption-queue"
  arn       = aws_sqs_queue.karpenter_interruption.arn
}

resource "aws_cloudwatch_event_rule" "instance_state_change" {
  name        = "${var.cluster_name}-instance-state-change"
  description = "Capture instance state changes (stopping, terminated) for Karpenter cleanup"

  event_pattern = jsonencode({
    source      = ["aws.ec2"]
    detail-type = ["EC2 Instance State-change Notification"]
  })
}

resource "aws_cloudwatch_event_target" "instance_state_change" {
  rule      = aws_cloudwatch_event_rule.instance_state_change.name
  target_id = "karpenter-interruption-queue"
  arn       = aws_sqs_queue.karpenter_interruption.arn
}
