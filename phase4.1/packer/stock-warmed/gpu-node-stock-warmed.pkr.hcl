# gpu-node-stock-warmed.pkr.hcl — Variant F: stock prebake + multi-compiler JIT cache.
#
# WHAT THIS DOES (vs gpu-node-stock.pkr.hcl):
#   1. Pre-pulls the same 9.5GB stock vLLM image (variant D parity).
#   2. Stages model weights from S3 to host disk (build-time only; deleted before AMI snapshot).
#   3. Runs vLLM ONCE inside containerd against the staged model, then sends two real
#      `/v1/completions` requests to trigger every JIT compile path: torch.compile,
#      Triton kernel codegen, FlashInfer attention kernels, vLLM FX graph cache.
#   4. Snapshots the resulting cache directories into /opt/vllm-cache on the AMI's root volume.
#   5. Deletes /opt/models so the AMI doesn't carry duplicate weights (runtime pod has its
#      own init-container model fetch — same as variant D).
#
# WHY THIS EXISTS (separate from gpu-node-stock.pkr.hcl):
#   Per phase4.1/COLD_START_RESEARCH.md §R2.1a (Doubleword Q1-2026), torch.compile is only
#   ONE of several JIT compilers vLLM runs at startup. Caching all of them targets ~10-15s
#   of `vllm_init_cuda_ctx`, vs ~5-10s if we only cached torch.compile.
#
# COMPILE-CACHE COMPATIBILITY CONTRACT (READ BEFORE REBUILDING):
#   The caches in /opt/vllm-cache are TIGHTLY COUPLED to:
#     - GPU compute capability (sm_75 — T4. WILL NOT WORK on g5/A10G/H100/etc.)
#     - NVIDIA driver version baked into the base EKS GPU AMI (currently v20251209)
#     - CUDA runtime version inside vllm-openai:v0.19.0 image
#     - torch + triton + flashinfer + vllm versions (frozen by image digest)
#   ANY of these changing requires a fresh AMI build. The base EKS GPU AMI periodically
#   bumps the NVIDIA driver — re-build whenever `data.amazon-ami.eks_gpu` resolves to a
#   new ID. vLLM may log "cache invalidated due to driver version mismatch" if a stale
#   AMI is used; that's the contract enforcing itself.
#
# USAGE:
#   bash phase4.1/scripts/build-ami.sh --template stock-warmed/gpu-node-stock-warmed.pkr.hcl
#   # OR manually:
#   cd phase4.1/packer/stock-warmed && packer init . && \
#     packer build -var 'region=us-east-1' gpu-node-stock-warmed.pkr.hcl
#
# OUTPUT:
#   AMI tagged Variant=prebaked-stock-warmed (distinct from prebaked-stock so the
#   gpu-nodes-prebaked-warmed EC2NodeClass selector doesn't collide with variant D's).

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
  type    = string
  default = "us-east-1"
}

variable "eks_version" {
  type    = string
  default = "1.30"
}

variable "vllm_image" {
  type = string
  # Same image as variant D — byte-identical to Hub v0.19.0 (sha256:7a0f0fdd…)
  default = "019167255542.dkr.ecr.us-east-1.amazonaws.com/inference-lab/vllm-openai:v0.19.0"
}

variable "aws_cli_image" {
  type    = string
  default = "amazon/aws-cli:2.17.18"
}

variable "model_s3_uri" {
  type = string
  # The model variant D's init container fetches at runtime. We stage it during AMI build
  # only to drive the JIT warmup; it's deleted before snapshot.
  default = "s3://inference-lab-model-cache/Qwen/Qwen2.5-7B-Instruct-AWQ/"
}

variable "instance_type" {
  type    = string
  default = "g4dn.xlarge"
}

# ---------------------------------------------------------------------------
# Data source — same EKS GPU AMI base as variants C/D/E
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
source "amazon-ebs" "gpu_node_stock_warmed" {
  region     = var.region
  source_ami = data.amazon-ami.eks_gpu.id

  ssh_username        = "ec2-user"
  spot_price          = "auto"
  spot_instance_types = ["g4dn.xlarge", "g4dn.2xlarge"]

  # Pin to us-east-1a for AZ-locality with FSR (same rationale as variant D).
  availability_zone = "us-east-1a"

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  ami_name        = "inference-lab-gpu-node-stock-warmed-{{timestamp}}"
  ami_description = "EKS ${var.eks_version} GPU node + 9.5GB stock vLLM image + pre-warmed JIT compile caches (torch.compile + Triton + FlashInfer + vLLM)"

  tags = {
    Name        = "inference-lab-gpu-node-stock-warmed"
    Project     = "inference-lab"
    Stage       = "phase4.1"
    Variant     = "prebaked-stock-warmed"  # distinct from prebaked-stock — see compatibility contract above
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
  sources = ["source.amazon-ebs.gpu_node_stock_warmed"]

  # ── Tune containerd for faster image operations (parity with stock template) ──
  provisioner "shell" {
    inline = [
      "echo '=== Tuning containerd ==='",
      "sudo cp /etc/containerd/config.toml /etc/containerd/config.toml.bak",
      "sudo tee -a /etc/containerd/config.toml > /dev/null << 'CTRDEOF'",
      "",
      "# Phase 4v3: parallel download/unpack tuning (10 concurrent saturates gp3)",
      "[plugins.\"io.containerd.grpc.v1.cri\".containerd]",
      "  max_concurrent_downloads = 10",
      "[plugins.\"io.containerd.transfer.v1.local\"]",
      "  max_concurrent_uploaded_layers = 10",
      "CTRDEOF",
    ]
  }

  # ── Pre-pull vLLM + AWS CLI images into k8s.io namespace (parity with stock template) ──
  provisioner "shell" {
    inline = [
      "echo '=== Pre-caching stock vLLM image (9.5GB) ==='",
      "echo 'Base AMI: ${data.amazon-ami.eks_gpu.id}'",
      "for i in $(seq 1 30); do",
      "  if sudo ctr version >/dev/null 2>&1; then break; fi",
      "  sleep 2",
      "done",
      "ECR_PASS=`aws ecr get-login-password --region ${var.region} 2>/dev/null` || true",
      "if [ -z \"$ECR_PASS\" ]; then",
      "  echo 'ERROR: ECR auth failed.' >&2",
      "  exit 1",
      "fi",
      "sudo ctr -n k8s.io images pull --user \"AWS:$ECR_PASS\" ${var.vllm_image}",
      "sudo ctr -n k8s.io images pull docker.io/${var.aws_cli_image}",
      "echo '=== Image digests ==='",
      "sudo ctr -n k8s.io images ls | grep vllm-openai",
    ]
  }

  # ── Stage model weights to host disk (deleted later) ──
  # Why on host disk and not inside the container: the warmup container bind-mounts
  # /opt/models read-only, so we keep the model on the host where we can rm -rf it
  # before snapshot. Putting weights inside an emptyDir-equivalent inside ctr would
  # work too but complicates the cleanup step.
  provisioner "shell" {
    inline = [
      "echo '=== Staging model weights from S3 (build-time only) ==='",
      "sudo mkdir -p /opt/models/Qwen2.5-7B-Instruct-AWQ",
      "sudo aws s3 cp --recursive --no-progress \\",
      "  ${var.model_s3_uri} \\",
      "  /opt/models/Qwen2.5-7B-Instruct-AWQ/",
      "sudo du -sh /opt/models/Qwen2.5-7B-Instruct-AWQ",
    ]
  }

  # ── THE WARMUP STEP: run vLLM, fire real inference, snapshot caches ──
  # Per COLD_START_RESEARCH.md §R2.1a, /health alone does NOT trigger FlashInfer's
  # attention-kernel JIT or Triton's per-shape kernels — those need a real forward pass.
  # Two requests with different prompt lengths widen Triton cache key coverage.
  provisioner "shell" {
    inline = [
      "echo '=== Warmup: running vLLM to populate JIT caches ==='",
      "sudo mkdir -p /opt/vllm-cache/{torchinductor,triton,flashinfer,vllm_compile,hf}",
      "sudo touch /tmp/build-start  # marker for the leakage discovery step",
      "",
      "# Verify GPU is visible to ctr before launching vLLM (cheap fail-fast)",
      "sudo nvidia-smi || { echo 'ERROR: GPU not available on build host.' >&2; exit 1; }",
      "",
      "# Run vLLM in background. Pin all JIT caches to /opt/vllm-cache via env vars,",
      "# bind-mounted at the SAME absolute path so cache keys with embedded paths stay valid.",
      "# --gpus 0 = all visible GPUs; --net-host so we can curl localhost:8000.",
      "# NOTE: --rm and -d are mutually exclusive in ctr (unlike docker). We use -d",
      "# and clean up the container manually in the audit step below.",
      "sudo ctr -n k8s.io run --gpus 0 --net-host -d \\",
      "  --env TORCHINDUCTOR_CACHE_DIR=/opt/vllm-cache/torchinductor \\",
      "  --env TRITON_CACHE_DIR=/opt/vllm-cache/triton \\",
      "  --env FLASHINFER_JIT_DIR=/opt/vllm-cache/flashinfer \\",
      "  --env VLLM_CACHE_ROOT=/opt/vllm-cache/vllm_compile \\",
      "  --env HF_HOME=/opt/vllm-cache/hf \\",
      "  --mount type=bind,src=/opt/vllm-cache,dst=/opt/vllm-cache,options=rbind:rw \\",
      "  --mount type=bind,src=/opt/models,dst=/models,options=rbind:ro \\",
      "  ${var.vllm_image} \\",
      "  vllm-warmup-build python3 -m vllm.entrypoints.openai.api_server \\",
      "    --model=/models/Qwen2.5-7B-Instruct-AWQ \\",
      "    --quantization=awq --dtype=half \\",
      "    --gpu-memory-utilization=0.85 --max-model-len=4096 \\",
      "    --host=127.0.0.1 --port=8000",
      "",
      "# Wait for /health (up to 240s — vLLM cold start including all JIT compiles)",
      "echo '=== Waiting for vLLM /health ==='",
      "READY=0",
      "for i in $(seq 1 240); do",
      "  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then",
      "    echo \"vLLM ready after $${i}s\"; READY=1; break",
      "  fi",
      "  sleep 1",
      "done",
      "if [ $READY -ne 1 ]; then",
      "  echo 'ERROR: vLLM never became ready.' >&2",
      "  sudo ctr -n k8s.io tasks ls",
      "  exit 1",
      "fi",
      "",
      "echo '=== Sending warmup inference requests (triggers FlashInfer + Triton JITs) ==='",
      "curl -sf http://127.0.0.1:8000/v1/completions \\",
      "  -H 'Content-Type: application/json' \\",
      "  -d '{\"model\":\"/models/Qwen2.5-7B-Instruct-AWQ\",\"prompt\":\"Hello\",\"max_tokens\":8,\"temperature\":0}' \\",
      "  | sudo tee /tmp/warmup-response.json",
      "echo ''",
      "curl -sf http://127.0.0.1:8000/v1/completions \\",
      "  -H 'Content-Type: application/json' \\",
      "  -d '{\"model\":\"/models/Qwen2.5-7B-Instruct-AWQ\",\"prompt\":\"The quick brown fox jumps over the lazy dog. Tell me a story about it.\",\"max_tokens\":16,\"temperature\":0}' \\",
      "  | sudo tee /tmp/warmup-response-2.json",
      "echo ''",
      "",
      "# Validate both completions returned text",
      "if ! grep -q '\"text\"' /tmp/warmup-response.json || ! grep -q '\"text\"' /tmp/warmup-response-2.json; then",
      "  echo 'ERROR: warmup inference did not return valid completions.' >&2",
      "  exit 1",
      "fi",
      "",
      "# Grace for any deferred cache writes to flush",
      "sleep 5",
      "",
      "# SIGTERM cleanly, then SIGKILL if needed",
      "sudo ctr -n k8s.io tasks kill --signal SIGTERM vllm-warmup-build || true",
      "sleep 10",
      "sudo ctr -n k8s.io tasks kill --signal SIGKILL vllm-warmup-build 2>/dev/null || true",
      "sudo ctr -n k8s.io containers rm vllm-warmup-build 2>/dev/null || true",
    ]
  }

  # ── Audit + leakage discovery ──
  # If anything significant landed outside /opt/vllm-cache, we need to know now —
  # otherwise variant F will under-perform for non-obvious reasons. The audit output
  # is the single most important artifact in the build log.
  provisioner "shell" {
    inline = [
      "echo '=== CACHE AUDIT (pinned dirs) ==='",
      "sudo du -sh /opt/vllm-cache/* 2>/dev/null || true",
      "echo ''",
      "echo '=== Per-dir file counts ==='",
      "for d in torchinductor triton flashinfer vllm_compile hf; do",
      "  echo \"$d: $(sudo find /opt/vllm-cache/$d -type f 2>/dev/null | wc -l) files\"",
      "done",
      "echo ''",
      "echo \"Total cache files: $(sudo find /opt/vllm-cache -type f 2>/dev/null | wc -l)\"",
      "echo \"Total cache size:  $(sudo du -sh /opt/vllm-cache | cut -f1)\"",
      "echo ''",
      "echo '=== LEAKAGE CHECK (compile artifacts outside /opt/vllm-cache) ==='",
      "# Anything compile-shaped that landed in /root, /home, /tmp, or containerd's overlay",
      "# in the last few minutes. If this list is non-trivial, the next AMI build needs to",
      "# capture more paths.",
      "sudo find /root /home /tmp /var/lib/containerd -newer /tmp/build-start \\",
      "  \\( -name '*.cubin' -o -name '*.ptx' -o -name 'fx_graph_cache*' \\",
      "     -o \\( -name '*.so' -path '*torch_extensions*' \\) \\) \\",
      "  2>/dev/null | head -50 || true",
      "echo ''",
      "echo '=== CLEANUP: remove staged model (runtime pod fetches its own) ==='",
      "sudo rm -rf /opt/models",
      "df -h /",
      "echo ''",
      "echo '=== AMI build complete ==='",
    ]
  }
}
