# gpu-node.pkr.hcl — Packer template for pre-baked GPU node AMI.
#
# WHAT THIS DOES:
#   Starts from the EKS-optimized AL2 AMI (NVIDIA drivers + kubelet pre-installed),
#   pre-pulls the vLLM container image (~8GB) into containerd's image store, and
#   saves the result as a custom AMI. When Karpenter provisions a node using this
#   AMI, containerd finds the image locally — no ECR/Docker Hub pull needed.
#
# WHY PRE-BAKE:
#   Phase 2 showed 5-min cold starts. Breakdown:
#     ~90s  container image pull (vLLM is 8GB, Docker Hub through NAT gateway)
#     ~60s  model download from HuggingFace (solved by S3 init container)
#     ~30s  GPU model loading + CUDA compilation
#     ~120s infrastructure (Spot launch + node join + kubelet + device plugin)
#
#   Pre-baking the AMI eliminates the 90s image pull. Combined with the S3 init
#   container, total cold start drops from ~300s to ~165s (45% improvement).
#
# CRITICAL: EBS FAST SNAPSHOT RESTORE (FSR)
#   The AMI bakes images into the EBS root volume snapshot. But EBS uses LAZY
#   LOADING — first access to each block fetches from S3, which is as slow as
#   a regular image pull. FSR pre-initializes the volume so all blocks are
#   immediately available on launch.
#
#   Without FSR: image is in containerd metadata but layers are lazily loaded
#     → pull still takes ~107s (same as network pull)
#   With FSR: volume is fully initialized → pull takes ~0s (local disk read)
#
#   Enable FSR after AMI build:
#     aws ec2 enable-fast-snapshot-restores \
#       --availability-zones us-east-1a us-east-1b \
#       --source-snapshot-ids <ami-root-snapshot-id>
#
#   FSR costs $0.75/hr per AZ while enabled. Disable when not testing.
#   The build-ami.sh script enables FSR automatically.
#
# ANALOGY: This is like creating a Golden AMI for an ASG with EBS pre-warming.
#   Without pre-warming, the first read from each block goes to S3 (cold).
#   With FSR, the volume is fully materialized on local disk before launch.
#
# USAGE:
#   bash phase4.5/scripts/build-ami.sh
#   # OR manually:
#   cd phase4.5/packer && packer build -var 'region=us-east-1' gpu-node.pkr.hcl
#
# REBUILD WHEN:
#   - vLLM version changes (e.g., v0.4.1 → v0.5.0)
#   - Worker image changes (new phase tag)
#   - EKS version upgrades (new base AMI)
#   The AMI does NOT contain model weights — those come from S3 at runtime.

packer {
  required_plugins {
    amazon = {
      version = ">= 1.3.0"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
variable "region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region for the AMI build"
}

variable "eks_version" {
  type        = string
  default     = "1.30"
  description = "EKS version — must match your cluster for kubelet compatibility"
}

variable "vllm_image" {
  type        = string
  default     = "019167255542.dkr.ecr.us-east-1.amazonaws.com/inference-lab/vllm-openai:v0.19.0-slim"
  description = "vLLM container image to pre-cache on the AMI (slim = sm_75 only, ~4GB vs 9.5GB stock)"
}

variable "worker_image" {
  type        = string
  default     = "019167255542.dkr.ecr.us-east-1.amazonaws.com/inference-lab/worker:phase4v3"
  description = "Queue worker container image to pre-cache"
}

variable "aws_cli_image" {
  type        = string
  default     = "amazon/aws-cli:2.15.0"
  description = "AWS CLI image used by the S3 model download init container"
}

variable "instance_type" {
  type        = string
  default     = "g4dn.xlarge"
  description = "Instance type for building the AMI (needs GPU for NVIDIA driver validation)"
}

# ---------------------------------------------------------------------------
# Data source: find the latest EKS-optimized AL2 GPU AMI
# ---------------------------------------------------------------------------
# WHY SSM parameter: AWS publishes the latest EKS AMI ID here. This is the
# same mechanism Karpenter uses internally when amiFamily=AL2. By reading the
# same SSM parameter, our custom AMI inherits the exact same base.
data "amazon-ami" "eks_gpu" {
  filters = {
    name                = "amazon-eks-gpu-node-${var.eks_version}-*"
    virtualization-type = "hvm"
    architecture        = "x86_64"
  }
  owners      = ["602401143452"] # Amazon EKS AMI account
  most_recent = true
  region      = var.region
}

# ---------------------------------------------------------------------------
# Builder: create the AMI
# ---------------------------------------------------------------------------
source "amazon-ebs" "gpu_node" {
  region        = var.region
  source_ami    = data.amazon-ami.eks_gpu.id

  ssh_username = "ec2-user"

  # WHY spot_instance_types (not instance_type): saves ~70% on build cost.
  # Can't specify both instance_type and spot_instance_types in the same block.
  spot_price    = "auto"
  spot_instance_types = ["g4dn.xlarge", "g4dn.2xlarge"]

  ami_name        = "inference-lab-gpu-node-{{timestamp}}"
  ami_description = "EKS ${var.eks_version} GPU node with pre-cached vLLM image (${var.vllm_image})"

  # WHY these tags: Karpenter discovers the AMI via amiSelectorTerms.
  # The tag "inference-lab-gpu" is what EC2NodeClass will filter on.
  tags = {
    Name        = "inference-lab-gpu-node"
    Project     = "inference-lab"
    EKSVersion  = var.eks_version
    VLLMVersion = var.vllm_image
    BuildTime   = "{{timestamp}}"
    ManagedBy   = "packer"
  }

  # WHY IAM instance profile: the build instance needs ECR pull permissions
  # to pull the slim vLLM image from ECR during AMI baking.
  iam_instance_profile = "inference-phase4-5-karpenter-node"

  # WHY encrypt: defense-in-depth for model weights cached on disk
  encrypt_boot = true

  # WHY 100Gi: same as the Karpenter EC2NodeClass blockDeviceMapping.
  # The pre-cached images consume ~10GB; remaining space is for runtime.
  launch_block_device_mappings {
    device_name           = "/dev/xvda"
    volume_size           = 100
    volume_type           = "gp3"
    delete_on_termination = true
  }
}

# ---------------------------------------------------------------------------
# Provisioner: pre-pull container images
# ---------------------------------------------------------------------------
build {
  sources = ["source.amazon-ebs.gpu_node"]

  # ── Phase 4v3: Tune containerd for faster image operations ──────────────
  # WHY: containerd defaults to 3 concurrent unpacks per image. The vLLM image
  # has ~30 layers; at 3 concurrent, decompression is serialized. Bumping to 10
  # yields 60% faster image operations (60s → 24s on fresh nodes per Tensorfuse benchmarks).
  # ANALOGY: like increasing ASG DesiredCapacity from 3 to 10 for a batch job —
  # more parallel workers finish faster.
  provisioner "shell" {
    inline = [
      "echo '=== Tuning containerd for faster image operations ==='",
      "",
      "# Back up original config",
      "sudo cp /etc/containerd/config.toml /etc/containerd/config.toml.bak",
      "",
      "# Append parallel download/unpack settings to containerd config",
      "sudo tee -a /etc/containerd/config.toml > /dev/null << 'CTRDEOF'",
      "",
      "# Phase 4v3: Parallel image download and unpack tuning",
      "# Default: 3 concurrent downloads, 3 concurrent unpacks",
      "# Tuned: 10 concurrent each — saturates gp3 EBS throughput (125 MB/s)",
      "[plugins.\"io.containerd.grpc.v1.cri\".containerd]",
      "  max_concurrent_downloads = 10",
      "[plugins.\"io.containerd.transfer.v1.local\"]",
      "  max_concurrent_uploaded_layers = 10",
      "CTRDEOF",
      "",
      "echo 'containerd config updated with parallel tuning'",
    ]
  }

  # WHY shell provisioner: containerd is already installed on the EKS AMI.
  # We use ctr (containerd CLI) to pull images directly into containerd's
  # image store. When kubelet starts pods, it checks containerd first —
  # if the image exists locally, no network pull needed.
  provisioner "shell" {
    inline = [
      "echo '=== Pre-caching container images for inference platform ==='",
      "echo 'Base AMI: ${data.amazon-ami.eks_gpu.id}'",
      "",
      "# Wait for containerd to be ready (EKS AMI starts it on boot)",
      "echo 'Waiting for containerd...'",
      "for i in $(seq 1 30); do",
      "  if sudo ctr version >/dev/null 2>&1; then break; fi",
      "  sleep 2",
      "done",
      "",
      "# Pre-pull vLLM image (slim ~7GB — cu130, x86_64 only)",
      "echo 'Pulling vLLM image: ${var.vllm_image}'",
      "echo 'This is ~7GB (cu130, x86_64 only) and takes 3-5 minutes...'",
      "",
      "# WHY --user 'AWS:pass' (colon-separated): ctr --user AWS alone opens",
      "# /dev/tty for password prompt, which panics in non-interactive Packer.",
      "# Inline colon-separated credentials bypass the prompt entirely.",
      "ECR_PASS=`aws ecr get-login-password --region us-east-1 2>/dev/null` || true",
      "if [ -n \"$ECR_PASS\" ]; then",
      "  sudo ctr -n k8s.io images pull --user \"AWS:$ECR_PASS\" ${var.vllm_image}",
      "else",
      "  echo WARNING: ECR auth failed. Image will be pulled at runtime.",
      "fi",
      "echo 'vLLM image cached successfully'",
      "",
      "# Pre-pull AWS CLI image (~150MB — used by S3 init container)",
      "echo 'Pulling AWS CLI image: ${var.aws_cli_image}'",
      "sudo ctr -n k8s.io images pull docker.io/${var.aws_cli_image}",
      "echo 'AWS CLI image cached successfully'",
      "",
      "# Pre-pull worker image from ECR (requires IAM permissions on build instance)",
      "# NOTE: This may fail if the build instance doesn't have ECR access.",
      "# The worker image is small (~50MB), so skipping it is acceptable.",
      "echo 'Attempting to pull worker image: ${var.worker_image}'",
      "if sudo ctr -n k8s.io images pull ${var.worker_image} 2>/dev/null; then",
      "  echo 'Worker image cached successfully'",
      "else",
      "  echo 'WARNING: Could not pull worker image (ECR auth may not be configured)'",
      "  echo 'This is OK — the worker image is small (~50MB) and pulls in <5s'",
      "fi",
      "",
      "# Verify cached images",
      "echo ''",
      "echo '=== Cached images ==='",
      "sudo ctr -n k8s.io images ls | grep -E 'vllm|aws-cli|worker'",
      "",
      "# Report disk usage",
      "echo ''",
      "echo '=== Disk usage ==='",
      "df -h /",
      "",
      "echo ''",
      "echo '=== AMI build complete ==='",
    ]
  }
}
