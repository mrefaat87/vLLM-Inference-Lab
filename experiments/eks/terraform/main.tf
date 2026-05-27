################################################################################
# inference-lab EKS cluster (isolated from sibling phase* stacks)
################################################################################
provider "aws" {
  region = var.region
  default_tags { tags = var.tags }
}

data "aws_caller_identity" "current" {}

locals {
  name          = var.cluster_name
  weight_bucket = "${var.cluster_name}-models-${data.aws_caller_identity.current.account_id}"
}

# ----- VPC ------------------------------------------------------------------
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.7"

  name = "${local.name}-vpc"
  cidr = var.vpc_cidr
  azs  = var.azs

  private_subnets = [for i, az in var.azs : cidrsubnet(var.vpc_cidr, 8, i)]
  public_subnets  = [for i, az in var.azs : cidrsubnet(var.vpc_cidr, 8, i + 32)]

  enable_nat_gateway      = true
  single_nat_gateway      = true
  enable_dns_hostnames    = true
  enable_dns_support      = true
  map_public_ip_on_launch = false

  public_subnet_tags = { "kubernetes.io/role/elb" = 1 }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
    "karpenter.sh/discovery"          = local.name
  }
}

# ----- EKS ------------------------------------------------------------------
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.13"

  cluster_name             = local.name
  cluster_version          = var.k8s_version
  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access = true

  enable_cluster_creator_admin_permissions = true

  cluster_addons = {
    coredns    = { most_recent = true }
    kube-proxy = { most_recent = true }
    vpc-cni    = { most_recent = true }
    # EBS CSI needs its own IRSA role — scoping the EBS policy onto the
    # node role (the manual workaround used during the May bring-up)
    # gives every Karpenter node EBS write access, which is much
    # broader than the CSI controller actually needs. The role minted
    # below is bound only to the kube-system/ebs-csi-controller-sa
    # service account.
    aws-ebs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.ebs_csi_irsa.iam_role_arn
    }
    eks-pod-identity-agent = { most_recent = true }
  }

  eks_managed_node_groups = {
    system = {
      ami_type       = "AL2_x86_64"
      instance_types = ["t3.medium"]
      min_size       = 2
      max_size       = 4
      desired_size   = 2
      labels         = { role = "system" }
      taints = [{
        key    = "CriticalAddonsOnly"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }

  node_security_group_tags = {
    "karpenter.sh/discovery" = local.name
  }
}

# ----- EBS CSI IRSA ---------------------------------------------------------
# Dedicated role for the EBS CSI controller service account. Attaches
# only AmazonEBSCSIDriverPolicy and trusts only the OIDC-mapped
# kube-system/ebs-csi-controller-sa SA — no node-role overreach.
module "ebs_csi_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.39"

  role_name             = "${local.name}-ebs-csi"
  attach_ebs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }

  tags = var.tags
}

# ----- Karpenter IAM (NodePool itself is applied as YAML post-bringup) ------
module "karpenter" {
  source  = "terraform-aws-modules/eks/aws//modules/karpenter"
  version = "~> 20.13"

  cluster_name                    = module.eks.cluster_name
  enable_pod_identity             = true
  create_pod_identity_association = true

  # Distinct IAM names so we never overlap with phase* Karpenter installs.
  iam_role_name                 = "${local.name}-karpenter-controller"
  iam_role_use_name_prefix      = false
  node_iam_role_name            = "${local.name}-karpenter-node"
  node_iam_role_use_name_prefix = false

  enable_v1_permissions = true
  tags                  = var.tags
}

# ----- ECR repositories (one per engine + driver) ---------------------------
locals {
  ecr_repos = [
    "${local.name}/vllm-runner",
    "${local.name}/sglang-runner",
    "${local.name}/trtllm-runner",
    "${local.name}/trtllm-builder",
    "${local.name}/experiment-driver",
  ]
}

resource "aws_ecr_repository" "engines" {
  for_each             = toset(local.ecr_repos)
  name                 = each.value
  image_tag_mutability = "IMMUTABLE"
  encryption_configuration { encryption_type = "AES256" }
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_lifecycle_policy" "engines" {
  for_each   = aws_ecr_repository.engines
  repository = each.value.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the 10 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ----- S3 weight cache ------------------------------------------------------
resource "aws_s3_bucket" "weights" {
  bucket        = local.weight_bucket
  force_destroy = false
}

resource "aws_s3_bucket_server_side_encryption_configuration" "weights" {
  bucket = aws_s3_bucket.weights.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "weights" {
  bucket                  = aws_s3_bucket.weights.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "weights" {
  bucket = aws_s3_bucket.weights.id
  versioning_configuration { status = "Enabled" }
}

# Allow Karpenter-launched nodes to pull weights at boot.
resource "aws_iam_role_policy" "node_weight_read" {
  name = "${local.name}-node-weight-read"
  role = module.karpenter.node_iam_role_name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "WeightsRead"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.weights.arn,
          "${aws_s3_bucket.weights.arn}/*",
        ]
      }
    ]
  })
}
