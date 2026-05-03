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
  # /19 gives ~8k IPs per subnet — needed because VPC CNI assigns a pod IP per ENI slot
  private_subnets = [
    cidrsubnet(var.vpc_cidr, 3, 0), # 10.0.0.0/19
    cidrsubnet(var.vpc_cidr, 3, 1), # 10.0.32.0/19
  ]

  # Public subnets: for NAT Gateway and any internet-facing load balancers
  # /20 gives ~4k IPs — more than enough for LBs
  public_subnets = [
    cidrsubnet(var.vpc_cidr, 4, 8), # 10.0.128.0/20
    cidrsubnet(var.vpc_cidr, 4, 9), # 10.0.144.0/20
  ]

  # Single NAT Gateway saves ~$32/mo vs one per AZ. Acceptable risk for learning —
  # if the NAT AZ goes down, private subnet egress breaks. In production you'd use one per AZ
  # (same trade-off as single-AZ vs multi-AZ for an RDS instance).
  enable_nat_gateway     = true
  single_nat_gateway     = true
  one_nat_gateway_per_az = false

  # DNS support is required for EKS service discovery and VPC endpoints
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Tags that EKS and Karpenter use for subnet discovery.
  # Karpenter reads "karpenter.sh/discovery" to know which subnets it can launch nodes into.
  # This is like tagging ASG subnets so the scaling controller knows where to place instances.
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1                # AWS LB controller: use these for internal ALBs
    "karpenter.sh/discovery"          = var.cluster_name # Karpenter: launch GPU nodes here
  }

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1 # AWS LB controller: use these for internet-facing ALBs
  }

  tags = {
    "karpenter.sh/discovery" = var.cluster_name # VPC-level tag for Karpenter discovery
  }
}
