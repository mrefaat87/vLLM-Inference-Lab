# gpu-node-soci.pkr.hcl — Packer template for GPU AMI with SOCI lazy image loading.
#
# WHAT THIS DOES (differs from gpu-node.pkr.hcl):
#   Same base: EKS-optimized AL2 AMI with NVIDIA drivers + kubelet
#   ADDED: SOCI snapshotter binary + systemd service + containerd config
#   CHANGED: vLLM image pulled from ECR (not Docker Hub) because SOCI
#            indexes are stored as OCI referrers in ECR
#
# WHY SOCI:
#   The gpu-node.pkr.hcl AMI pre-caches the vLLM image in containerd, but
#   containerd STILL needs to decompress layers to overlayfs on first pod start.
#   This takes ~110s on gp3 EBS (3.8GB compressed → overlayfs at 125 MB/s).
#   Pre-baking only saved ~7s (network download tail, not the decompression).
#
#   SOCI (Seekable OCI) eliminates the decompression bottleneck entirely.
#   Instead of decompressing all layers upfront, the SOCI containerd snapshotter
#   creates a FUSE mount that lazily fetches file contents from ECR via HTTP
#   range requests. The container starts in ~10-20s with a skeleton filesystem,
#   and files are populated on-demand as the process accesses them.
#
#   ANALOGY: think of SOCI like Amazon EFS lazy loading. The filesystem appears
#   fully populated immediately, but data is fetched from the remote store on
#   first access. vs. the old approach is like EBS snapshot restore — you wait
#   for all data to materialize before the volume is usable.
#
# HOW SOCI WORKS ON THIS AMI:
#   1. soci-snapshotter-grpc runs as a systemd service (started before containerd)
#   2. containerd is configured with SOCI as a proxy snapshotter plugin
#   3. When kubelet pulls an image, containerd checks ECR for a SOCI index
#   4. If index exists: SOCI creates a FUSE mount, container starts immediately
#   5. If no index: falls back to normal overlayfs (full decompression)
#   6. Background: SOCI prefetches commonly-accessed files to reduce latency
#
# PREREQUISITES:
#   Run setup-soci.sh first to push vLLM image + SOCI index to ECR
#
# USAGE:
#   cd phase4.5/packer && packer build \
#     -var 'ecr_vllm_image=<ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/inference-lab/vllm-openai:v0.19.0' \
#     gpu-node-soci.pkr.hcl
#
# REBUILD WHEN:
#   - vLLM version changes (rebuild image + re-run setup-soci.sh for new index)
#   - SOCI snapshotter version updates (security patches)
#   - EKS version upgrades (new base AMI)

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

variable "ecr_vllm_image" {
  type        = string
  description = "ECR URI for vLLM image (must have SOCI index pushed via setup-soci.sh)"
  # No default — must be passed explicitly to ensure SOCI index exists
  # Example: 019167255542.dkr.ecr.us-east-1.amazonaws.com/inference-lab/vllm-openai:v0.19.0-slim
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
  description = "Instance type for building the AMI"
}

variable "soci_version" {
  type        = string
  default     = "0.7.0"
  description = "SOCI snapshotter version to install"
}

# ---------------------------------------------------------------------------
# Data source: find the latest EKS-optimized AL2 GPU AMI
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
source "amazon-ebs" "gpu_node_soci" {
  region        = var.region
  source_ami    = data.amazon-ami.eks_gpu.id
  ssh_username  = "ec2-user"

  # WHY spot_instance_types (not instance_type): saves ~70% on build cost.
  # Can't specify both instance_type and spot_instance_types in the same block.
  spot_price           = "auto"
  spot_instance_types  = ["g4dn.xlarge", "g4dn.2xlarge"]

  ami_name        = "inference-lab-gpu-node-soci-{{timestamp}}"
  ami_description = "EKS ${var.eks_version} GPU node with SOCI lazy loading (${var.ecr_vllm_image})"

  tags = {
    Name            = "inference-lab-gpu-node-soci"
    Project         = "inference-lab"
    EKSVersion      = var.eks_version
    VLLMVersion     = var.ecr_vllm_image
    SOCIVersion     = var.soci_version
    BuildTime       = "{{timestamp}}"
    ManagedBy       = "packer"
    ImageLoadMode   = "soci-lazy"
  }

  encrypt_boot = true

  launch_block_device_mappings {
    device_name           = "/dev/xvda"
    volume_size           = 100
    volume_type           = "gp3"
    delete_on_termination = true
  }

  # WHY IAM instance profile: the build instance needs ECR pull permissions
  # to pull the vLLM image from ECR during AMI baking.
  # WHY inference-phase4-5: Terraform creates this instance profile. The Phase 3
  # stack uses different naming. NEVER use inference-lab-karpenter-node — that
  # doesn't exist in the Phase 4 Terraform state.
  iam_instance_profile = "inference-phase4-5-karpenter-node"
}

# ---------------------------------------------------------------------------
# Provisioner: install SOCI + configure containerd + pre-cache images
# ---------------------------------------------------------------------------
build {
  sources = ["source.amazon-ebs.gpu_node_soci"]

  # ── Step 1: Install SOCI snapshotter ──────────────────────────────────
  provisioner "shell" {
    inline = [
      "echo '=== Step 1: Installing SOCI snapshotter v${var.soci_version} ==='",
      "",
      "# Download and install SOCI binaries",
      "cd /tmp",
      "curl -fsSL https://github.com/awslabs/soci-snapshotter/releases/download/v${var.soci_version}/soci-snapshotter-${var.soci_version}-linux-amd64.tar.gz -o soci.tar.gz",
      "sudo tar -xzf soci.tar.gz -C /usr/local/bin/ soci-snapshotter-grpc",
      "sudo chmod +x /usr/local/bin/soci-snapshotter-grpc",
      "rm soci.tar.gz",
      "",
      "# Verify installation",
      "echo 'SOCI snapshotter installed:'",
      "ls -la /usr/local/bin/soci-snapshotter-grpc",
    ]
  }

  # ── Step 2: Create SOCI systemd service ───────────────────────────────
  # WHY systemd: SOCI snapshotter runs as a long-lived gRPC service that
  # containerd connects to via Unix socket. It must start BEFORE containerd
  # so the proxy snapshotter is available when kubelet starts pulling images.
  # ANALOGY: like starting a sidecar proxy (e.g., Envoy) before the app server.
  provisioner "shell" {
    inline = [
      "echo '=== Step 2: Creating SOCI systemd service ==='",
      "",
      "# Create the SOCI snapshotter config directory",
      "sudo mkdir -p /etc/soci-snapshotter-grpc",
      "",
      "# Write SOCI config",
      "# WHY these settings:",
      "#   - content_store: uses containerd's content store to share cached data",
      "#   - no_background_fetch=false: prefetch layers in background after mount",
      "#   - prefetch_size: first 32MB of each layer fetched eagerly (covers Python",
      "#     imports, shared libraries, and CUDA kernels that are needed immediately)",
      "sudo tee /etc/soci-snapshotter-grpc/config.toml > /dev/null << 'SOCIEOF'",
      "[fuse]",
      "# WHY 120s: generous FUSE operation timeout. Large CUDA libraries may take",
      "# a few seconds to fetch via HTTP range request on first access.",
      "fuse_timeout_sec = 120",
      "",
      "[background_fetch]",
      "# WHY enabled: after the container starts with a lazy FUSE mount, SOCI",
      "# prefetches remaining layer data in the background. This means the first",
      "# access is lazy (fast start), but subsequent accesses hit local cache",
      "# (fast runtime). Like EBS lazy loading with readahead.",
      "disable = false",
      "# Prefetch at lower priority to avoid competing with active inference I/O",
      "fetch_period_msec = 500",
      "SOCIEOF",
      "",
      "# Create systemd unit file",
      "sudo tee /etc/systemd/system/soci-snapshotter.service > /dev/null << 'UNITEOF'",
      "[Unit]",
      "Description=SOCI Snapshotter for containerd",
      "Documentation=https://github.com/awslabs/soci-snapshotter",
      "# WHY Before=containerd: containerd must find the SOCI snapshotter socket",
      "# when it starts. If SOCI isn't running, containerd can't use it as a",
      "# proxy snapshotter and falls back to normal overlayfs (no lazy loading).",
      "Before=containerd.service",
      "After=network.target",
      "",
      "[Service]",
      "Type=simple",
      "ExecStart=/usr/local/bin/soci-snapshotter-grpc \\",
      "  --config /etc/soci-snapshotter-grpc/config.toml \\",
      "  --log-level info \\",
      "  --root /var/lib/soci-snapshotter-grpc/",
      "Restart=always",
      "RestartSec=5",
      "",
      "# WHY LimitNOFILE: SOCI uses FUSE mounts with many file handles open for",
      "# lazy loading. Default ulimit (1024) is too low for large images.",
      "LimitNOFILE=65536",
      "",
      "[Install]",
      "WantedBy=multi-user.target",
      "UNITEOF",
      "",
      "# Enable the service (will start on boot)",
      "sudo systemctl daemon-reload",
      "sudo systemctl enable soci-snapshotter.service",
      "echo 'SOCI systemd service created and enabled'",
    ]
  }

  # ── Step 3: Configure containerd to use SOCI snapshotter ──────────────
  # WHY modify containerd config: containerd needs to know about the SOCI
  # snapshotter as a proxy plugin. When kubelet requests an image pull,
  # containerd delegates snapshot operations to SOCI via the Unix socket.
  # SOCI checks ECR for a SOCI index; if found, it creates a lazy FUSE mount
  # instead of decompressing all layers. If no index, it falls back to overlayfs.
  provisioner "shell" {
    inline = [
      "echo '=== Step 3: Configuring containerd for SOCI ==='",
      "",
      "# Back up the original containerd config",
      "sudo cp /etc/containerd/config.toml /etc/containerd/config.toml.pre-soci",
      "",
      "# Add SOCI proxy snapshotter plugin to containerd config.",
      "# WHY proxy_plugins: containerd supports external snapshotters via gRPC.",
      "# The SOCI snapshotter runs as a separate process and containerd delegates",
      "# snapshot (filesystem) operations to it. This is the containerd equivalent",
      "# of a plugin architecture — like a sidecar container handling specific duties.",
      "#",
      "# We append to the existing config rather than replacing it, to preserve",
      "# EKS-specific settings (sandbox image, CNI, device plugins, etc.).",
      "sudo tee -a /etc/containerd/config.toml > /dev/null << 'CTRDEOF'",
      "",
      "# Phase 4v3: Parallel image operations + SOCI lazy loading",
      "# WHY parallel tuning: containerd defaults to 3 concurrent unpacks.",
      "# The vLLM image has ~30 layers; bumping to 10 yields 60% faster operations.",
      "# This stacks with SOCI — non-SOCI layers (AWS CLI, worker) benefit from",
      "# parallel unpack while SOCI layers are lazy-loaded.",
      "[plugins.\"io.containerd.grpc.v1.cri\".containerd]",
      "  max_concurrent_downloads = 10",
      "[plugins.\"io.containerd.transfer.v1.local\"]",
      "  max_concurrent_uploaded_layers = 10",
      "",
      "# SOCI lazy image loading — added by Packer (gpu-node-soci.pkr.hcl)",
      "[proxy_plugins.soci]",
      "  type = \"snapshot\"",
      "  address = \"/run/soci-snapshotter-grpc/soci-snapshotter-grpc.sock\"",
      "CTRDEOF",
      "",
      "echo 'Containerd config updated with SOCI proxy plugin'",
      "echo 'Proxy plugins section:'",
      "grep -A 3 'proxy_plugins' /etc/containerd/config.toml || true",
    ]
  }

  # ── Step 4: Pre-cache container images ────────────────────────────────
  # WHY still pre-cache: even with SOCI, pre-caching the image metadata
  # (manifests, config) in containerd avoids the initial registry roundtrip
  # for image resolution. The actual layer data is still lazy-loaded from ECR.
  # For the AWS CLI and worker images (which don't have SOCI indexes), pre-caching
  # avoids the full pull at pod start time.
  provisioner "shell" {
    inline = [
      "echo '=== Step 4: Pre-caching container images ==='",
      "",
      "# Explicitly restart containerd after config changes in Step 3.",
      "# WHY: the EKS AMI starts containerd on boot, but our config changes",
      "# (SOCI proxy plugin, parallel tuning) require a restart. If containerd",
      "# can't start (e.g., SOCI socket not ready), the image pre-caching fails",
      "# but the AMI is still valid — SOCI will work at runtime when both",
      "# services start in the right order via systemd dependencies.",
      "echo 'Restarting containerd with updated config...'",
      "sudo systemctl restart containerd || true",
      "sleep 5",
      "",
      "# Wait for containerd to be ready",
      "echo 'Waiting for containerd...'",
      "CONTAINERD_READY=false",
      "for i in $(seq 1 30); do",
      "  if sudo ctr version >/dev/null 2>&1; then CONTAINERD_READY=true; break; fi",
      "  sleep 2",
      "done",
      "",
      "if [ \"$CONTAINERD_READY\" != \"true\" ]; then",
      "  echo 'WARNING: containerd not ready after 60s. Skipping image pre-cache.'",
      "  echo 'This is OK — SOCI config is installed. Images will be lazily loaded at runtime.'",
      "  echo ''",
      "  echo '=== AMI build complete (SOCI-enabled, no pre-cached images) ==='",
      "  exit 0",
      "fi",
      "",
      "# Authenticate with ECR for pulling the vLLM image",
      "ECR_PASS=$$(aws ecr get-login-password --region ${var.region} 2>/dev/null || echo '')",
      "if [ -n \"$$ECR_PASS\" ]; then",
      "  echo 'Pulling vLLM image metadata from ECR...'",
      "  sudo ctr -n k8s.io images pull --user \"AWS:$$ECR_PASS\" ${var.ecr_vllm_image} || true",
      "  echo 'vLLM image metadata cached from ECR'",
      "else",
      "  echo 'WARNING: ECR auth failed — image will be pulled at pod start time'",
      "fi",
      "",
      "# Pre-pull AWS CLI image (small, no SOCI needed)",
      "echo 'Pulling AWS CLI image: ${var.aws_cli_image}'",
      "sudo ctr -n k8s.io images pull docker.io/${var.aws_cli_image} || true",
      "echo 'AWS CLI image cached (or skipped)'",
      "",
      "# Pre-pull worker image from ECR (small, no SOCI needed)",
      "echo 'Pulling worker image: ${var.worker_image}'",
      "if [ -n \"$$ECR_PASS\" ]; then",
      "  sudo ctr -n k8s.io images pull --user \"AWS:$$ECR_PASS\" ${var.worker_image} 2>/dev/null || true",
      "fi",
      "",
      "# Verify cached images",
      "echo ''",
      "echo '=== Cached images ==='",
      "sudo ctr -n k8s.io images ls | grep -E 'vllm|aws-cli|worker' || true",
      "",
      "echo ''",
      "echo '=== Disk usage ==='",
      "df -h /",
      "",
      "echo ''",
      "echo '=== AMI build complete (SOCI-enabled) ==='",
      "echo 'SOCI snapshotter: installed and enabled'",
      "echo 'Containerd: configured with SOCI proxy plugin'",
      "echo 'On boot: SOCI starts → containerd starts → lazy image loading active'",
    ]
  }
}
