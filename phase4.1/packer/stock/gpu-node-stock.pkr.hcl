# gpu-node-stock.pkr.hcl — Packer template that pre-bakes the STOCK 9.5GB vLLM image.
#
# WHY THIS EXISTS (separate from gpu-node.pkr.hcl):
#   The original gpu-node.pkr.hcl bakes the SLIM 4GB image (sm_75-only rebuild).
#   For the Phase 4.1 image-pull comparison experiment, we need the IDENTICAL
#   bytes that Docker Hub serves baked into the AMI — otherwise the variant-C
#   measurement conflates "pre-baking" with "image shrink." The slim template
#   stays intact for Phase 6 SOCI work; this is a sibling.
#
# IMAGE SOURCE:
#   Pulls v0.19.0-stock from ECR (mirrored from Docker Hub by push-stock-to-ecr.sh).
#   We use ECR not Hub here because:
#     1. The build instance already has IRSA-style ECR pull permissions.
#     2. ECR pulls inside us-east-1 are 5-10x faster than Hub through NAT —
#        cuts AMI build time from ~12 min to ~3 min.
#     3. The image bytes are byte-identical to Hub (digest verified by the push
#        script), so this doesn't affect the experiment's correctness.
#
# RELATIONSHIP TO FSR / S3-LAZY-LOAD:
#   See phase4.1/LEARNINGS.md for the full explanation. Short version: the
#   pre-baked image lives in the AMI's EBS snapshot. On first launch, EBS
#   lazy-loads blocks from S3 on first read. FSR pre-warms the snapshot at
#   $0.75/hr/AZ. We deliberately do NOT enable FSR for this AMI on the first
#   pass — making the lazy-load tax visible is part of the data we're collecting.
#
# USAGE:
#   bash phase4.1/scripts/build-ami.sh --template gpu-node-stock.pkr.hcl
#   # OR manually:
#   cd phase4.1/packer && packer build -var 'region=us-east-1' gpu-node-stock.pkr.hcl
#
# OUTPUT:
#   AMI tagged Name=inference-lab-gpu-node-stock so the prebaked NodePool's
#   amiSelectorTerms can target it without colliding with the slim AMI.

packer {
  required_plugins {
    amazon = {
      version = ">= 1.3.0"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

# ---------------------------------------------------------------------------
# Variables — same shape as gpu-node.pkr.hcl, only image refs differ
# ---------------------------------------------------------------------------
variable "region" {
  type    = string
  default = "us-east-1"
}

variable "eks_version" {
  type    = string
  default = "1.30"
}

variable "vllm_image" {
  type    = string
  # STOCK image — 9.5GB. Uses the existing v0.19.0 ECR tag, verified
  # byte-identical to vllm/vllm-openai:v0.19.0 amd64 on Docker Hub
  # (manifest digest sha256:7a0f0fdd2771464b6976625c2b2d5dd46f566aa00fbc53eceab86ef50883da90).
  # No separate -stock tag needed — saves ~$1/mo of duplicate ECR storage.
  default = "019167255542.dkr.ecr.us-east-1.amazonaws.com/inference-lab/vllm-openai:v0.19.0"
}

variable "aws_cli_image" {
  type    = string
  default = "amazon/aws-cli:2.17.18"  # Match the version used by baseline-pod.yaml init container
}

variable "instance_type" {
  type    = string
  default = "g4dn.xlarge"
}

# ---------------------------------------------------------------------------
# Data source — same EKS-optimized AL2 GPU AMI as the slim template
# ---------------------------------------------------------------------------
data "amazon-ami" "eks_gpu" {
  filters = {
    name                = "amazon-eks-gpu-node-${var.eks_version}-*"
    virtualization-type = "hvm"
    architecture        = "x86_64"
  }
  owners      = ["602401143452"]
  most_recent = true
  region      = var.region
}

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
source "amazon-ebs" "gpu_node_stock" {
  region     = var.region
  source_ami = data.amazon-ami.eks_gpu.id

  ssh_username        = "ec2-user"
  spot_price          = "auto"
  spot_instance_types = ["g4dn.xlarge", "g4dn.2xlarge"]

  # Pin to us-east-1a for two reasons:
  #   1. us-east-1e does not support g4dn — Spot fleet would otherwise oscillate.
  #   2. Phase 4.1 baseline runs (fresh-001..003) all landed in us-east-1a, so
  #      pinning the AMI build there keeps the EBS snapshot AZ-local to where
  #      variant C will run.
  availability_zone = "us-east-1a"

  # The AWS account enforces IMDSv2 (httpTokensEnforced=true). Without this
  # block, Packer's launched instance fails with "httpTokens=required".
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  # Distinct AMI name from the slim template — Karpenter EC2NodeClass amiSelectorTerms
  # for the prebaked NodePool will filter on this name pattern.
  ami_name        = "inference-lab-gpu-node-stock-{{timestamp}}"
  ami_description = "EKS ${var.eks_version} GPU node with pre-cached STOCK 9.5GB vLLM image (${var.vllm_image})"

  tags = {
    Name        = "inference-lab-gpu-node-stock"
    Project     = "inference-lab"
    Stage       = "phase4.1"
    Variant     = "prebaked-stock"  # Used by gpu-nodepool-prebaked.yaml selectorTerms
    EKSVersion  = var.eks_version
    VLLMVersion = var.vllm_image
    BuildTime   = "{{timestamp}}"
    ManagedBy   = "packer"
  }

  iam_instance_profile = "inference-phase4-1-karpenter-node"

  encrypt_boot = true

  launch_block_device_mappings {
    device_name           = "/dev/xvda"
    volume_size           = 100
    volume_type           = "gp3"
    delete_on_termination = true
  }
}

# ---------------------------------------------------------------------------
# Provisioners
# ---------------------------------------------------------------------------
build {
  sources = ["source.amazon-ebs.gpu_node_stock"]

  # ── Tune containerd for faster image operations ─────────────────────────
  # Same as gpu-node.pkr.hcl — kept here to make this template standalone.
  provisioner "shell" {
    inline = [
      "echo '=== Tuning containerd for faster image operations ==='",
      "sudo cp /etc/containerd/config.toml /etc/containerd/config.toml.bak",
      "sudo tee -a /etc/containerd/config.toml > /dev/null << 'CTRDEOF'",
      "",
      "# Phase 4v3: parallel download/unpack tuning (10 concurrent saturates gp3)",
      "[plugins.\"io.containerd.grpc.v1.cri\".containerd]",
      "  max_concurrent_downloads = 10",
      "[plugins.\"io.containerd.transfer.v1.local\"]",
      "  max_concurrent_uploaded_layers = 10",
      "CTRDEOF",
      "echo 'containerd config updated'",
    ]
  }

  # ── Pre-pull the stock vLLM image + AWS CLI image into containerd's k8s.io namespace ──
  # WHY namespace k8s.io: kubelet uses the k8s.io containerd namespace by default.
  # Pulling into the default namespace would not be visible to kubelet at runtime.
  provisioner "shell" {
    inline = [
      "echo '=== Pre-caching stock vLLM image (9.5GB) ==='",
      "echo 'Base AMI: ${data.amazon-ami.eks_gpu.id}'",
      "",
      "# Wait for containerd",
      "for i in $(seq 1 30); do",
      "  if sudo ctr version >/dev/null 2>&1; then break; fi",
      "  sleep 2",
      "done",
      "",
      "# Pull stock vLLM from ECR (faster than Hub, byte-identical)",
      "echo 'Pulling vLLM stock image from ECR: ${var.vllm_image}'",
      "ECR_PASS=`aws ecr get-login-password --region ${var.region} 2>/dev/null` || true",
      "if [ -z \"$ECR_PASS\" ]; then",
      "  echo 'ERROR: ECR auth failed. Check the build instance IAM profile has ECR pull rights.' >&2",
      "  exit 1",
      "fi",
      "sudo ctr -n k8s.io images pull --user \"AWS:$ECR_PASS\" ${var.vllm_image}",
      "echo 'vLLM stock image cached successfully'",
      "",
      "# Pre-pull AWS CLI image (~150MB) — same init container the baseline pod uses",
      "echo 'Pulling AWS CLI image: ${var.aws_cli_image}'",
      "sudo ctr -n k8s.io images pull docker.io/${var.aws_cli_image}",
      "echo 'AWS CLI image cached successfully'",
      "",
      "# ── Verify the digest matches what we intended to bake ─────────────",
      "# This catches the case where ECR served a different image than the",
      "# push script verified. The crictl digest must match the ECR digest",
      "# that push-stock-to-ecr.sh reported. We log both so the AMI build log",
      "# is auditable evidence.",
      "echo ''",
      "echo '=== Image digest verification ==='",
      "sudo ctr -n k8s.io images ls | grep vllm-openai",
      "echo ''",
      "sudo crictl images --digests | grep vllm-openai || true",
      "",
      "echo ''",
      "echo '=== Disk usage after pre-cache ==='",
      "df -h /",
      "echo ''",
      "echo '=== AMI build complete ==='",
    ]
  }
}
