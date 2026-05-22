# gpu-node-lustre.pkr.hcl — Variant H: prebake stock vLLM image AND install the
# Lustre kernel module so worker nodes can mount FSx for Lustre filesystems.
#
# RELATIONSHIP TO STOCK TEMPLATE
#   Identical to phase4.1/packer/stock/gpu-node-stock.pkr.hcl with ONE addition:
#   `amazon-linux-extras install -y lustre` runs as a provisioner step. The
#   kernel module ships with the AMI; first-boot kubelet → CSI driver → mount
#   path doesn't need any privileged init container at runtime.
#
# WHY BAKE THE LUSTRE CLIENT (vs install via FSx CSI daemonset)
#   The FSx CSI driver assumes the lustre client kernel module is already
#   available on the host kernel — its node DaemonSet does NOT load the module.
#   Two options:
#     (a) bake into the AMI (this template) — clean, no privileged runtime init,
#         module is in EBS snapshot so first-pod-on-node mount latency is bounded.
#     (b) privileged init container that runs `modprobe lustre` per pod — works
#         but adds 5-15s and requires capability escalation.
#   We pick (a) since Packer is already wired up for Phase 4.1 and the runtime
#   stage breakdown stays clean.
#
# WHAT THIS DOESN'T BAKE
#   Model weights are NOT in this AMI. They live on the FSx Lustre filesystem
#   provisioned by phase4.1/terraform/fsx.tf. The runtime pod mounts the FSx
#   PVC at /mnt/fsx and vLLM reads weights from there. Multi-model workloads
#   change models by changing the PVC path, NOT by rebaking AMIs (that's the
#   whole point of variant H).
#
# AMI / SNAPSHOT SIZE
#   Same as stock prebake (~14 GB). The Lustre client kernel module + userspace
#   tools add ~30 MB — negligible vs the 9.5 GB vLLM image. FSR pre-warm time
#   matches D's (~7 min observed in Variant G).
#
# USAGE
#   bash phase4.1/scripts/build-ami.sh --template gpu-node-lustre.pkr.hcl
#
# OUTPUT
#   AMI tagged Variant=prebaked-fsx so gpu-nodepool-prebaked-fsx.yaml's
#   amiSelectorTerms can target it.

packer {
  required_plugins {
    amazon = {
      version = ">= 1.3.0"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

# ---------------------------------------------------------------------------
# Variables — same shape as gpu-node-stock.pkr.hcl
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
  type = string
  # Same byte-identical stock vLLM image as variant D — Variant H runs vLLM's
  # standard weight loader (not the streamer); weights come from the FSx mount.
  default = "019167255542.dkr.ecr.us-east-1.amazonaws.com/inference-lab/vllm-openai:v0.19.0"
}

variable "aws_cli_image" {
  type    = string
  default = "amazon/aws-cli:2.17.18"
}

variable "instance_type" {
  type    = string
  default = "g4dn.xlarge"
}

# ---------------------------------------------------------------------------
# Data source — same EKS-optimized AL2 GPU AMI as the other prebake templates.
# AL2 ships `amazon-linux-extras` with a `lustre` topic — see provisioner below.
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
source "amazon-ebs" "gpu_node_lustre" {
  region     = var.region
  source_ami = data.amazon-ami.eks_gpu.id

  ssh_username        = "ec2-user"
  spot_price          = "auto"
  spot_instance_types = ["g4dn.xlarge", "g4dn.2xlarge"]

  # us-east-1a — same AZ as D/E/F/G (g4dn unavailable in 1e; FSR is per-AZ;
  # FSx Lustre is also per-AZ and we provision it in 1a).
  availability_zone = "us-east-1a"

  # Account enforces IMDSv2.
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  ami_name        = "inference-lab-gpu-node-lustre-{{timestamp}}"
  ami_description = "EKS ${var.eks_version} GPU node with pre-cached stock vLLM image (${var.vllm_image}) AND Lustre client kernel module for FSx mounting"

  tags = {
    Name        = "inference-lab-gpu-node-lustre"
    Project     = "inference-lab"
    Stage       = "phase4.1"
    Variant     = "prebaked-fsx" # Used by gpu-nodepool-prebaked-fsx.yaml selectorTerms
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
# Provisioners — copy of gpu-node-stock.pkr.hcl + ONE additional step
# ---------------------------------------------------------------------------
build {
  sources = ["source.amazon-ebs.gpu_node_lustre"]

  # ── Tune containerd for faster image operations (same as stock) ─────────
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

  # ── THE ONE THING THAT MAKES THIS VARIANT H ──────────────────────────────
  # Install the Lustre client kernel module + userspace tools. AL2 ships
  # `lustre` as an amazon-linux-extras topic. Two flavors are typically available
  # (lustre2.10 / lustre2.12); we take whichever the topic resolves to (latest).
  #
  # IMPORTANT: we DO NOT modprobe here. The module is on disk, but loading it
  # requires the kernel that will boot on the FINAL launched instance, not the
  # one Packer's build instance is currently running. modprobe may fail at
  # build time with "module not found" because the kernel uname differs after
  # snapshot. The CSI driver's `mount.lustre` invocation at runtime triggers
  # autoload of the module via /etc/modules-load.d.
  provisioner "shell" {
    inline = [
      "echo '=== Installing Lustre client (Variant H specific) ==='",
      "amazon-linux-extras list 2>&1 | grep -i lustre || true",
      "sudo amazon-linux-extras enable lustre",
      "sudo yum install -y lustre-client",
      "",
      "# Audit the install — log the package + kernel module presence so the AMI",
      "# build log is auditable evidence that the module shipped in the snapshot.",
      "echo '--- Lustre install audit ---'",
      "rpm -qa | grep -i lustre || echo 'no lustre rpms found'",
      "ls -la /lib/modules/$(uname -r)/extra/lustre-client/ 2>&1 || true",
      "find /lib/modules -name 'lustre.ko*' 2>&1 || true",
      "",
      "# Persist autoload on every boot so kubelet/CSI doesn't need privileged init.",
      "echo 'lustre' | sudo tee /etc/modules-load.d/lustre.conf",
      "cat /etc/modules-load.d/lustre.conf",
    ]
  }

  # ── Pre-pull the stock vLLM image + AWS CLI image (same as stock) ─────────
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
      "# Pull stock vLLM from ECR",
      "echo 'Pulling vLLM stock image from ECR: ${var.vllm_image}'",
      "ECR_PASS=`aws ecr get-login-password --region ${var.region} 2>/dev/null` || true",
      "if [ -z \"$ECR_PASS\" ]; then",
      "  echo 'ERROR: ECR auth failed. Check the build instance IAM profile has ECR pull rights.' >&2",
      "  exit 1",
      "fi",
      "sudo ctr -n k8s.io images pull --user \"AWS:$ECR_PASS\" ${var.vllm_image}",
      "echo 'vLLM stock image cached successfully'",
      "",
      "# Pre-pull AWS CLI image — kept for parity with D's AMI even though",
      "# Variant H's pod has no init container (no functional dependency).",
      "echo 'Pulling AWS CLI image: ${var.aws_cli_image}'",
      "sudo ctr -n k8s.io images pull docker.io/${var.aws_cli_image}",
      "echo 'AWS CLI image cached successfully'",
      "",
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
      "echo '=== Variant H AMI build complete ==='",
    ]
  }
}
