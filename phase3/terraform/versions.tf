# versions.tf — Pin all provider versions and configure backend
# Why pin versions: Terraform providers change frequently. Pinning prevents
# surprise breaking changes during `terraform init` (same reason you'd pin
# AMI IDs in a launch template).

terraform {
  required_version = ">= 1.5.0" # Need 1.5+ for moved blocks and import support

  # Local backend — fine for a learning project. In production you'd use S3 + DynamoDB
  # for state locking (like a distributed lock to prevent concurrent modifications).
  backend "local" {
    path = "terraform.tfstate"
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0" # Major version 5 — includes EKS Pod Identity support
    }

    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12" # Needed for OCI registry support (Karpenter chart)
    }

    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25" # Aligns with EKS 1.29 API compatibility
    }

    kubectl = {
      source  = "gavinbunney/kubectl"
      version = "~> 1.14" # For applying raw YAML manifests (Karpenter CRDs, NodePools)
    }

    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0" # Used for OIDC thumbprint calculation
    }
  }
}

# Configure the AWS provider — region comes from variables
provider "aws" {
  region = var.aws_region

  # Tag every resource created by this config for cost tracking and ownership
  default_tags {
    tags = {
      Project   = "inference-lab"
      ManagedBy = "terraform"
      Stage     = "phase3"
    }
  }
}

# Helm provider authenticates to EKS cluster to install charts (Karpenter, etc.)
# Think of this like an SSM agent that needs cluster credentials to do its job.
provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

    # Use short-lived token from AWS CLI instead of storing long-lived creds
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# Kubernetes provider — same auth pattern as Helm, needed for K8s resources
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

# kubectl provider — for raw YAML manifests that the kubernetes provider can't handle
provider "kubectl" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  load_config_file       = false # Don't read ~/.kube/config — use explicit auth

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}
