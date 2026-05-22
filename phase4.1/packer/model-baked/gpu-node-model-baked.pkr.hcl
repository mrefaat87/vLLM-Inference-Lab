# gpu-node-model-baked.pkr.hcl — Variant G: prebake STOCK image AND model weights into the AMI.
#
# RELATIONSHIP TO STOCK TEMPLATE
#   This is variant G's AMI for the Phase 4.1 cold-start comparison. It is
#   identical to phase4.1/packer/stock/gpu-node-stock.pkr.hcl with ONE addition:
#   a provisioner step that aws-s3-cps the Qwen2.5-7B-Instruct-AWQ weights into
#   /opt/models/Qwen2.5-7B-Instruct-AWQ on the root volume, where they remain at
#   snapshot time (unlike Variant F, which deletes them before snapshot). The
#   runtime pod hostPath-mounts that path read-only and skips the S3 init container.
#
# WHAT THIS BUYS US
#   - Eliminates the ~37s S3-download init container that Variant D pays.
#   - Eliminates the K8s init→main container handoff (the gap that gave Variant E
#     its 30s edge over D *despite* E's weight load being slower).
#   - With FSR enabled on the resulting snapshot, weight reads from /opt/models
#     hit FSR-warmed gp3 (~1 GB/s) — same speed as Variant D's emptyDir reads.
#
# WHY NOT FORK FROM VARIANT F (stock-warmed)
#   F also stages weights to /opt/models, but then runs vLLM during build to warm
#   JIT caches AND `rm -rf /opt/models` before snapshot. We want neither side
#   effect — we want weights to PERSIST in the snapshot, and we don't want the
#   warmup-induced AMI-build cost (extra 20 min + need GPU on the build host).
#
# AMI / SNAPSHOT SIZE
#   Stock prebake snapshot is ~14 GB. Adding 5.2 GiB of safetensors → ~20 GB
#   snapshot. Root volume bumped 100Gi → 120Gi to leave headroom for
#   /var/log, /tmp, etc. on a freshly-launched node. FSR cost is per-AZ flat
#   regardless of snapshot size, but pre-warm time scales with size — expect
#   30-45 min for `enabled` state vs ~20 min for D's 14 GB snapshot.
#
# USAGE
#   bash phase4.1/scripts/build-ami.sh --template gpu-node-model-baked.pkr.hcl
#
# OUTPUT
#   AMI tagged Name=inference-lab-gpu-node-model-baked and Variant=prebaked-model-baked
#   so gpu-nodepool-prebaked-model-baked.yaml's amiSelectorTerms can target it
#   without colliding with D's, E's, or F's AMIs.

packer {
  required_plugins {
    amazon = {
      version = ">= 1.3.0"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

# ---------------------------------------------------------------------------
# Variables — same shape as gpu-node-stock.pkr.hcl + model_s3_uri
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
  # STOCK image — byte-identical to vllm/vllm-openai:v0.19.0 amd64 on Docker Hub.
  # Same image as Variant D so the runtime container has the standard vLLM weight
  # loader (no streamer needed; we read weights from local disk).
  default = "019167255542.dkr.ecr.us-east-1.amazonaws.com/inference-lab/vllm-openai:v0.19.0"
}

variable "aws_cli_image" {
  type    = string
  default = "amazon/aws-cli:2.17.18"
}

variable "model_s3_uri" {
  type = string
  # Same Qwen2.5-7B-Instruct-AWQ that Variant D's runtime init container fetches.
  # Stage 0 pre-flight confirmed: 12 objects, 5.2 GiB total, 2 safetensors shards.
  # Variant G bakes it into the AMI permanently (no rm-rf before snapshot).
  default = "s3://inference-lab-model-cache/Qwen/Qwen2.5-7B-Instruct-AWQ/"
}

variable "instance_type" {
  type    = string
  default = "g4dn.xlarge"
}

# ---------------------------------------------------------------------------
# Data source — same EKS-optimized AL2 GPU AMI as the other prebake templates
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
source "amazon-ebs" "gpu_node_model_baked" {
  region     = var.region
  source_ami = data.amazon-ami.eks_gpu.id

  ssh_username        = "ec2-user"
  spot_price          = "auto"
  spot_instance_types = ["g4dn.xlarge", "g4dn.2xlarge"]

  # Pin to us-east-1a (same as D/E/F):
  #   - g4dn unsupported in us-east-1e
  #   - All Phase 4.1 baseline runs land in 1a → AMI snapshot must be AZ-local
  #     for FSR (per-AZ enablement).
  availability_zone = "us-east-1a"

  # Account enforces IMDSv2.
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  # Distinct AMI name from D/E/F — Karpenter EC2NodeClass for Variant G filters
  # by Variant tag, but the Name pattern keeps human listings readable.
  ami_name        = "inference-lab-gpu-node-model-baked-{{timestamp}}"
  ami_description = "EKS ${var.eks_version} GPU node with pre-cached vLLM ${var.vllm_image} AND Qwen2.5-7B-AWQ weights in /opt/models"

  tags = {
    Name        = "inference-lab-gpu-node-model-baked"
    Project     = "inference-lab"
    Stage       = "phase4.1"
    Variant     = "prebaked-model-baked" # Used by gpu-nodepool-prebaked-model-baked.yaml selectorTerms
    EKSVersion  = var.eks_version
    VLLMVersion = var.vllm_image
    ModelURI    = var.model_s3_uri
    BuildTime   = "{{timestamp}}"
    ManagedBy   = "packer"
  }

  iam_instance_profile = "inference-phase4-1-karpenter-node"

  encrypt_boot = true

  launch_block_device_mappings {
    device_name = "/dev/xvda"
    # Bumped 100→120Gi vs Variant D: stock image (~14 GB) + Qwen weights (~5.2 GiB)
    # + /var/log/tmp headroom for a freshly-launched node still leaves >90 GB free.
    volume_size           = 120
    volume_type           = "gp3"
    delete_on_termination = true
  }
}

# ---------------------------------------------------------------------------
# Provisioners — copy of gpu-node-stock.pkr.hcl + ONE additional step (S3 cp)
# ---------------------------------------------------------------------------
build {
  sources = ["source.amazon-ebs.gpu_node_model_baked"]

  # ── Tune containerd for faster image operations ─────────────────────────
  # Same as gpu-node-stock.pkr.hcl. The contract with the runtime pod is that
  # the image is already cached in containerd's k8s.io namespace AND the
  # parallelism config is in place so any in-band re-pull benefits from it.
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
  # Same as gpu-node-stock.pkr.hcl. ECR pull is faster than Hub through NAT
  # AND the build instance's IAM profile already has ECR pull rights.
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
      "# Variant G's pod has no init container (no functional dependency).",
      "echo 'Pulling AWS CLI image: ${var.aws_cli_image}'",
      "sudo ctr -n k8s.io images pull docker.io/${var.aws_cli_image}",
      "echo 'AWS CLI image cached successfully'",
      "",
      "echo ''",
      "echo '=== Image digest verification ==='",
      "sudo ctr -n k8s.io images ls | grep vllm-openai",
      "echo ''",
      "sudo crictl images --digests | grep vllm-openai || true",
    ]
  }

  # ── THE ONE THING THAT MAKES THIS VARIANT G ──────────────────────────────
  # Bake the model weights to /opt/models/Qwen2.5-7B-Instruct-AWQ on the root
  # volume. NO cleanup — these bytes stay in the snapshot. The runtime pod
  # hostPath-mounts /opt/models read-only.
  #
  # Lifted from packer/stock-warmed/gpu-node-stock-warmed.pkr.hcl:194-200,
  # MINUS the warmup container and MINUS the rm-rf cleanup.
  #
  # IAM: build host runs under inference-phase4-1-karpenter-node, which has
  # s3:GetObject on inference-lab-model-cache (verified — D's runtime init
  # container reads the same path with the same role).
  provisioner "shell" {
    inline = [
      "echo '=== Baking model weights into AMI (Variant G specific) ==='",
      "sudo mkdir -p /opt/models/Qwen2.5-7B-Instruct-AWQ",
      "sudo aws s3 cp --recursive --no-progress \\",
      "  ${var.model_s3_uri} \\",
      "  /opt/models/Qwen2.5-7B-Instruct-AWQ/",
      "",
      "# Verify size & file count — catches a partial cp that would silently",
      "# corrupt the AMI. Expect ~5.2 GiB and 12 files (per Stage 0 pre-flight).",
      "echo '--- Baked model audit ---'",
      "sudo du -sh /opt/models/Qwen2.5-7B-Instruct-AWQ",
      "sudo find /opt/models/Qwen2.5-7B-Instruct-AWQ -type f | wc -l",
      "sudo ls -la /opt/models/Qwen2.5-7B-Instruct-AWQ/",
      "",
      "# Permissions: vLLM in the runtime container runs as a non-root UID,",
      "# but reads via a hostPath mount that the kubelet exposes. World-readable",
      "# is the simplest contract that works. The AMI is single-tenant per pod",
      "# (g4dn.xlarge, one pod per node) — no cross-tenant exposure concern.",
      "sudo chmod -R a+rX /opt/models",
      "",
      "echo ''",
      "echo '=== Disk usage after model bake ==='",
      "df -h /",
      "echo ''",
      "echo '=== Variant G AMI build complete ==='",
    ]
  }
}
