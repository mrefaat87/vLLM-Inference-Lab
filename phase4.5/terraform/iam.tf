# iam.tf — IAM roles and policies for Karpenter, S3 CSI, and vLLM workers
#
# Mental model: Each component gets the minimum permissions it needs (least privilege).
# IRSA (IAM Roles for Service Accounts) = a pod-level IAM role. Like giving each
# microservice its own IAM role instead of one big instance profile for the whole fleet.

# ─── Data Sources ────────────────────────────────────────────────────────────────

data "aws_caller_identity" "current" {} # Gets account ID for ARN construction
data "aws_partition" "current" {}       # "aws" in commercial, "aws-cn" in China, etc.
data "aws_region" "current" {}          # Resolves to var.aws_region

# ─── Karpenter Node IAM Role ────────────────────────────────────────────────────
# This role is assumed by EC2 instances that Karpenter launches (GPU nodes).
# It's the instance profile role — like the IAM role on an ASG launch template.

resource "aws_iam_role" "karpenter_node" {
  name = "${var.cluster_name}-karpenter-node"

  # Trust policy: only EC2 can assume this role (it's an instance profile)
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

# Attach the standard EKS worker node policies — same ones any managed node group gets
resource "aws_iam_role_policy_attachment" "karpenter_node_eks_worker" {
  role       = aws_iam_role.karpenter_node.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "karpenter_node_cni" {
  role       = aws_iam_role.karpenter_node.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "karpenter_node_ecr" {
  role       = aws_iam_role.karpenter_node.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "karpenter_node_ssm" {
  # SSM lets you shell into nodes without SSH — useful for debugging GPU driver issues
  role       = aws_iam_role.karpenter_node.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Instance profile wraps the role so EC2 instances can assume it
resource "aws_iam_instance_profile" "karpenter_node" {
  name = "${var.cluster_name}-karpenter-node"
  role = aws_iam_role.karpenter_node.name
}

# ─── Karpenter Controller IRSA Role ─────────────────────────────────────────────
# The Karpenter controller pod needs AWS permissions to launch/terminate EC2 instances.
# IRSA scopes this to just the karpenter ServiceAccount (not all pods on the node).

module "karpenter_controller_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${var.cluster_name}-karpenter-controller"

  # Bind this role to the karpenter ServiceAccount in the karpenter namespace
  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["karpenter:karpenter"]
    }
  }
}

# Karpenter controller policy — everything it needs to manage the GPU fleet
resource "aws_iam_role_policy" "karpenter_controller" {
  name = "${var.cluster_name}-karpenter-controller"
  role = module.karpenter_controller_irsa.iam_role_name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # EC2 fleet management — Karpenter's core job is launching and terminating instances.
        # This is equivalent to what an ASG needs, but Karpenter makes decisions itself
        # instead of relying on a fixed scaling policy.
        Sid    = "EC2Management"
        Effect = "Allow"
        Action = [
          "ec2:RunInstances",
          "ec2:TerminateInstances",
          "ec2:DescribeInstances",
          "ec2:DescribeInstanceTypes", # Needed to calculate instance fitness for pods
          "ec2:DescribeImages",        # Find the right EKS-optimized AMI
          "ec2:DescribeSecurityGroups",
          "ec2:DescribeSubnets",
          "ec2:DescribeLaunchTemplates",
          "ec2:CreateFleet", # Spot fleet requests for GPU instances
          "ec2:CreateLaunchTemplate",
          "ec2:CreateTags", # Tag instances for cost tracking
          "ec2:DeleteLaunchTemplate",
          "ec2:DescribeAvailabilityZones",
          "ec2:DescribeSpotPriceHistory",      # Optimal Spot instance selection
          "ec2:DescribeInstanceTypeOfferings", # Karpenter needs this to resolve instance types per AZ
        ]
        Resource = "*" # EC2 Describe actions require * — can't scope to specific resources
      },
      {
        # SSM Parameter Store — Karpenter reads the EKS-optimized AMI ID from here
        # instead of hardcoding AMIs (like using SSM public parameters for latest AL2 AMI)
        Sid      = "SSMGetParameter"
        Effect   = "Allow"
        Action   = ["ssm:GetParameter"]
        Resource = "arn:${data.aws_partition.current.partition}:ssm:${data.aws_region.current.name}:*:parameter/aws/service/*"
      },
      {
        # Pricing API — Karpenter uses this to choose cost-optimal instance types
        # (e.g., g5.xlarge vs g4dn.xlarge based on current pricing)
        Sid      = "PricingAPI"
        Effect   = "Allow"
        Action   = ["pricing:GetProducts"]
        Resource = "*" # Pricing API doesn't support resource-level permissions
      },
      {
        # SQS — Karpenter listens for Spot interruption warnings and rebalance recommendations.
        # Like a circuit breaker pattern: gracefully drain the node before AWS reclaims it.
        Sid    = "SQSInterruptionHandling"
        Effect = "Allow"
        Action = [
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes",
          "sqs:GetQueueUrl",
          "sqs:ReceiveMessage",
        ]
        Resource = "arn:${data.aws_partition.current.partition}:sqs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:${var.cluster_name}"
      },
      {
        # IAM — Karpenter needs to manage instance profiles and pass the node role.
        Sid    = "IAMManagement"
        Effect = "Allow"
        Action = [
          "iam:PassRole",                 # Assign node role to launched instances
          "iam:GetInstanceProfile",       # Check if instance profile exists
          "iam:CreateInstanceProfile",    # Create instance profile for nodes
          "iam:DeleteInstanceProfile",    # Cleanup on node termination
          "iam:TagInstanceProfile",       # Tag for discovery
          "iam:AddRoleToInstanceProfile", # Attach node role to profile
          "iam:RemoveRoleFromInstanceProfile",
        ]
        Resource = "*"
      },
      {
        # EKS DescribeCluster — Karpenter reads cluster info (endpoint, CA cert, etc.)
        # to configure kubelet on new nodes
        Sid      = "EKSDescribeCluster"
        Effect   = "Allow"
        Action   = ["eks:DescribeCluster"]
        Resource = module.eks.cluster_arn
      },
    ]
  })
}

# ─── S3 CSI Driver IRSA Role ────────────────────────────────────────────────────
# Mountpoint for S3 CSI driver needs S3 access to mount buckets as filesystems.
# This lets GPU pods read model weights directly from S3 without downloading first.

module "s3_csi_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${var.cluster_name}-s3-csi-driver"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:s3-csi-driver-sa"]
    }
  }
}

resource "aws_iam_role_policy" "s3_csi_driver" {
  name = "${var.cluster_name}-s3-csi-driver"
  role = module.s3_csi_irsa.iam_role_name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3MountAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject",
          "s3:GetBucketLocation",
          "s3:AbortMultipartUpload",
        ]
        Resource = [
          data.aws_s3_bucket.model_cache.arn,
          "${data.aws_s3_bucket.model_cache.arn}/*",
        ]
      },
    ]
  })
}

# ─── vLLM Worker IRSA Role ──────────────────────────────────────────────────────
# Worker pods running vLLM need direct S3 access for model downloads and caching.
# This is separate from the S3 CSI role — defense in depth (different SA, different perms).

module "vllm_worker_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${var.cluster_name}-vllm-worker"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["inference-system:vllm-worker"]
    }
  }
}

resource "aws_iam_role_policy" "vllm_worker_s3" {
  name = "${var.cluster_name}-vllm-worker-s3"
  role = module.vllm_worker_irsa.iam_role_name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ModelCacheAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",    # Download cached model weights
          "s3:PutObject",    # Upload newly downloaded models to cache
          "s3:ListBucket",   # Check if model is already cached
          "s3:DeleteObject", # Clean up old model versions
        ]
        Resource = [
          data.aws_s3_bucket.model_cache.arn,
          "${data.aws_s3_bucket.model_cache.arn}/*",
        ]
      },
    ]
  })
}
