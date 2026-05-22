# fsx.tf — Variant H: shared FSx for Lustre filesystem holding Qwen2.5-7B-AWQ
# weights for all inference pods.
#
# WHY THIS EXISTS
#   Variants A-G use either S3 (init container fetch) or AMI-baked weights.
#   Both approaches scale per-model: N models = N AMIs or N init-container paths.
#   FSx Lustre is the multi-model production answer: one shared filesystem, all
#   pods mount it via PVC, model swaps become path changes (no infra rebuild).
#   This file provisions the FS for a single-experiment measurement window —
#   tagged auto-delete=true so a leak is easy to find.
#
# COST SHAPE
#   SCRATCH_2 SSD, 1.2 TiB minimum, ~$0.075-0.145/GB-month tier-dependent.
#   Per-second prorated, no minimum hour. For a ~3 hour measurement window:
#   ~$0.72. The HARD risk is leaving the FS up after the experiment — that's
#   what `auto-delete=true` + the manual teardown handoff doc are for.
#
# AZ
#   Single-AZ Lustre FS. us-east-1a to match the rest of Phase 4.1 (AMIs, FSR,
#   Karpenter NodePools). var.azs[0] is us-east-1a per variables.tf.
#
# NETWORK
#   FSx Lustre client traffic uses TCP 988 + ephemeral 1018-1023 between the
#   FSx ENIs and the worker node SG. The CSI mount hangs silently if these
#   aren't open — verified bidirectionally. FSx is in private subnet[0]
#   (us-east-1a) where the GPU nodes also land.

# ─── FSx security group ──────────────────────────────────────────────────────
# A dedicated SG attached to the FSx ENIs. Lets us scope inbound to ONLY the
# EKS node SG (no 0.0.0.0/0 anywhere). Cleaner than reusing the cluster SG.
resource "aws_security_group" "fsx_lustre" {
  name_prefix = "${var.cluster_name}-fsx-lustre-"
  description = "FSx Lustre filesystem ENIs - Variant H multi-model weights"
  vpc_id      = module.vpc.vpc_id

  # Inbound: Lustre client traffic from the EKS worker node SG.
  # Port 988 is the Lustre routing/network port; 1018-1023 is the ephemeral
  # range Lustre's connection-establishment uses.
  ingress {
    description     = "Lustre 988 from EKS worker nodes"
    from_port       = 988
    to_port         = 988
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  ingress {
    description     = "Lustre ephemeral 1018-1023 from EKS worker nodes"
    from_port       = 1018
    to_port         = 1023
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  # Self-referential rules — FSx Lustre's pre-flight network check requires
  # the SG to permit port 988 from itself (FSx ENI to FSx ENI in the same
  # filesystem). Without this, CreateFileSystem fails with InvalidNetworkSettings.
  ingress {
    description = "Lustre 988 self-referential (FSx pre-flight check)"
    from_port   = 988
    to_port     = 988
    protocol    = "tcp"
    self        = true
  }

  ingress {
    description = "Lustre 1018-1023 self-referential (FSx pre-flight check)"
    from_port   = 1018
    to_port     = 1023
    protocol    = "tcp"
    self        = true
  }

  # Egress: Lustre return traffic back to the worker nodes. The protocol is
  # symmetric — same ports.
  egress {
    description     = "Lustre 988 return traffic to EKS worker nodes"
    from_port       = 988
    to_port         = 988
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  egress {
    description     = "Lustre ephemeral 1018-1023 return traffic to EKS worker nodes"
    from_port       = 1018
    to_port         = 1023
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  tags = {
    Name       = "${var.cluster_name}-fsx-lustre"
    Project    = "inference-lab"
    Stage      = "phase4.1"
    Variant    = "fsx-lustre"
    experiment = "phase4.1-variantH"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# ─── Reciprocal ingress on the EKS node SG ───────────────────────────────────
# The Lustre client running on a worker node has to ACCEPT inbound from the
# FSx ENIs on the same ports (replies to the client's outbound connect). The
# EKS node SG default-allows traffic from itself, but the FSx ENIs are in a
# DIFFERENT SG, so we need explicit rules here too.
resource "aws_security_group_rule" "node_from_fsx_988" {
  description              = "Lustre 988 from FSx ENIs"
  type                     = "ingress"
  from_port                = 988
  to_port                  = 988
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.fsx_lustre.id
  security_group_id        = module.eks.node_security_group_id
}

resource "aws_security_group_rule" "node_from_fsx_ephemeral" {
  description              = "Lustre 1018-1023 from FSx ENIs"
  type                     = "ingress"
  from_port                = 1018
  to_port                  = 1023
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.fsx_lustre.id
  security_group_id        = module.eks.node_security_group_id
}

# ─── FSx Lustre filesystem ───────────────────────────────────────────────────
# SCRATCH_2 SSD: per-second prorated, deletes ephemerally (data lost when FS
# goes away — fine for a benchmark, not for production). Persistent_2 was
# considered and rejected for this experiment as overkill for a ~3 hour run.
#
# TAGGING IS LOAD-BEARING
#   The operator IAM policy phase41-variantH-fsx scopes mutations on the
#   experiment=phase4.1-variantH RequestTag. If those tags get omitted from a
#   create call, the create itself will fail with AccessDenied. Belt and
#   suspenders + makes leaked FSes easy to find.
resource "aws_fsx_lustre_file_system" "phase41_variantH" {
  storage_capacity            = 1200 # TiB minimum for SSD SCRATCH_2
  storage_type                = "SSD"
  deployment_type             = "SCRATCH_2"
  subnet_ids                  = [module.vpc.private_subnets[0]] # us-east-1a (var.azs[0])
  security_group_ids          = [aws_security_group.fsx_lustre.id]
  copy_tags_to_backups        = false # SCRATCH_2 doesn't support backups anyway

  tags = {
    Name        = "${var.cluster_name}-variantH-weights"
    Project     = "inference-lab"
    Stage       = "phase4.1"
    Variant     = "fsx-lustre"
    experiment  = "phase4.1-variantH"
    auto-delete = "true"
    ManagedBy   = "terraform"
  }
}

# ─── Outputs — used by phase4.1/k8s/fsx-lustre-pv.yaml at apply time ─────────
output "fsx_id" {
  description = "FSx for Lustre filesystem ID — referenced by the static PV manifest."
  value       = aws_fsx_lustre_file_system.phase41_variantH.id
}

output "fsx_dns_name" {
  description = "FSx DNS name — fed into PV.spec.csi.volumeAttributes.dnsname."
  value       = aws_fsx_lustre_file_system.phase41_variantH.dns_name
}

output "fsx_mount_name" {
  description = "FSx mount name (random 8-char suffix) — fed into PV.spec.csi.volumeAttributes.mountname."
  value       = aws_fsx_lustre_file_system.phase41_variantH.mount_name
}
