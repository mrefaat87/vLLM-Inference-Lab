# security_groups.tf — Network access controls for cluster nodes
#
# Security groups in K8s context: the EKS module creates a cluster SG and a node SG
# automatically. We add an SSH rule scoped to your IP for debugging.
# In production you'd skip SSH entirely and use SSM Session Manager instead.

resource "aws_security_group" "node_ssh" {
  name_prefix = "${var.cluster_name}-node-ssh-"
  description = "Allow SSH access to EKS nodes from operator IP"
  vpc_id      = module.vpc.vpc_id

  # Inbound: SSH only from your IP. The /32 CIDR means exactly one IP address.
  # This is like a security group on a bastion host — never open port 22 to 0.0.0.0/0.
  ingress {
    description = "SSH from operator IP"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  # No egress rules here — the EKS node security group handles egress.
  # Adding egress here would create duplicate rules. This SG is additive (attached
  # alongside the EKS-managed node SG).

  tags = {
    Name = "${var.cluster_name}-node-ssh"
  }

  # Use name_prefix + create_before_destroy so Terraform can replace the SG
  # without breaking existing nodes (like a blue-green deployment for SGs)
  lifecycle {
    create_before_destroy = true
  }
}
