# vpc.tf — Network foundation for the EKS cluster
# Analogy: The VPC is your data center. Public subnets are the DMZ (load balancers),
# private subnets are the internal network (worker nodes, GPU instances).

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.cluster_name}-vpc"
  cidr = var.vpc_cidr

  azs = var.azs

  # Private subnets: where EKS nodes live (no direct internet access, egress via NAT)
  # /19 per AZ → 4 AZs gives Karpenter real choice for the Spot-capacity AZ pin.
  private_subnets = [
    for i, az in var.azs : cidrsubnet(var.vpc_cidr, 3, i) # /19 each, indices 0..3
  ]

  # Public subnets: for NAT Gateway and any internet-facing load balancers
  public_subnets = [
    for i, az in var.azs : cidrsubnet(var.vpc_cidr, 4, i + 8) # /20 each, indices 8..11
  ]

  # Single NAT Gateway saves ~$32/mo vs one per AZ. Acceptable risk for learning —
  # if the NAT AZ goes down, private subnet egress breaks. In production you'd use one per AZ
  # (same trade-off as single-AZ vs multi-AZ for an RDS instance).
  enable_nat_gateway   = true
  single_nat_gateway   = true
  one_nat_gateway_per_az = false

  # DNS support is required for EKS service discovery and VPC endpoints
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Tags that EKS and Karpenter use for subnet discovery.
  # Karpenter reads "karpenter.sh/discovery" to know which subnets it can launch nodes into.
  # This is like tagging ASG subnets so the scaling controller knows where to place instances.
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1                       # AWS LB controller: use these for internal ALBs
    "karpenter.sh/discovery"          = var.cluster_name         # Karpenter: launch GPU nodes here
  }

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1 # AWS LB controller: use these for internet-facing ALBs
  }

  tags = {
    "karpenter.sh/discovery" = var.cluster_name # VPC-level tag for Karpenter discovery
  }
}
