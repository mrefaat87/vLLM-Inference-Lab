# eks.tf — EKS cluster and CPU node group
# The cluster is the control plane (managed by AWS). Node groups are the data plane
# (EC2 instances running kubelet). Karpenter will manage GPU nodes separately —
# this node group only runs system workloads (CoreDNS, Karpenter itself, API gateway).

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = var.cluster_version

  # Place control plane ENIs in private subnets (API server communication stays internal)
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Endpoint access: public lets you run kubectl from your laptop,
  # private lets pods talk to the API server without leaving the VPC.
  # In production you'd disable public and use a bastion or VPN.
  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  # OIDC provider enables IRSA (IAM Roles for Service Accounts) — this is how
  # pods get AWS permissions without instance profiles. Like assuming a role
  # scoped to a single microservice instead of giving the whole EC2 fleet permissions.
  enable_irsa = true

  # Let the Terraform caller (you) be a cluster admin. Without this, you'd be locked out
  # because EKS no longer auto-maps the cluster creator in v20+ of this module.
  enable_cluster_creator_admin_permissions = true

  # Allow Karpenter-launched GPU nodes to register with the cluster.
  # WHY: EKS module v20 uses access entries (not aws-auth ConfigMap). Managed node
  # groups get auto-registered, but Karpenter uses a custom IAM role that must be
  # explicitly added. Without this, GPU nodes launch but kubelet can't authenticate.
  # Analogy: adding the ASG instance role to the cluster's RBAC allow-list.
  access_entries = {
    karpenter_node = {
      principal_arn = aws_iam_role.karpenter_node.arn
      type          = "EC2_LINUX"
    }
  }

  # Disable KMS key creation — avoids needing KMS permissions for a learning lab.
  # In production you'd enable this for secrets encryption at rest.
  create_kms_key            = false
  cluster_encryption_config = {}

  # Cluster addons — managed by AWS, auto-updated with the cluster
  cluster_addons = {
    coredns = {
      # DNS resolution inside the cluster. Every service lookup goes through CoreDNS.
      # Without it, pods can't resolve service names (like not having Route53 in your VPC).
      most_recent = true
    }
    kube-proxy = {
      # Maintains iptables rules on each node for Service → Pod routing.
      # Analogous to the ENI routing tables that VPC CNI manages.
      most_recent = true
    }
    vpc-cni = {
      # AWS VPC CNI assigns real VPC IPs to pods (not overlay network IPs).
      # This means pods are directly routable from the VPC — critical for GPU node networking.
      most_recent = true
    }
    aws-ebs-csi-driver = {
      # Enables dynamic EBS volume provisioning for PersistentVolumeClaims.
      # Used for model caching on local EBS when S3 mount isn't appropriate.
      most_recent = true
      service_account_role_arn = module.ebs_csi_irsa.iam_role_arn
    }
  }

  # CPU node group — the "always-on" fleet for system workloads.
  # GPU nodes are managed by Karpenter on-demand (like Spot fleet scaling).
  eks_managed_node_groups = {
    cpu-system = {
      instance_types = [var.cpu_instance_type]
      ami_type       = "AL2_x86_64" # Amazon Linux 2 — well-tested with EKS

      desired_size = var.cpu_node_count
      min_size     = 1   # At least 1 node to keep CoreDNS and Karpenter alive
      max_size     = 3   # Allow scaling up for system workload spikes

      labels = {
        role = "cpu" # Used for node affinity — Karpenter and API pods target this label
      }

      # Attach the SSH security group so we can debug nodes if needed
      vpc_security_group_ids = [aws_security_group.node_ssh.id]
    }
  }

  # Allow all egress from nodes — they need to pull container images, talk to S3, etc.
  node_security_group_additional_rules = {
    egress_all = {
      description = "Allow all outbound traffic from nodes"
      protocol    = "-1"       # All protocols
      from_port   = 0
      to_port     = 0
      type        = "egress"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }

  tags = {
    "karpenter.sh/discovery" = var.cluster_name # Karpenter uses this to find the cluster's SGs
  }
}

# IRSA role for the EBS CSI driver — lets it call ec2:CreateVolume etc.
# Without this, PersistentVolumeClaims would hang in Pending state.
module "ebs_csi_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "${var.cluster_name}-ebs-csi-driver"
  attach_ebs_csi_policy = true # Attaches the AWS-managed AmazonEBSCSIDriverPolicy

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }
}
