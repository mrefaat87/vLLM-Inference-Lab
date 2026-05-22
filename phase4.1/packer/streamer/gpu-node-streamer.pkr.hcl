# gpu-node-streamer.pkr.hcl — Pre-bake the STREAMER-augmented vLLM image.
#
# RELATIONSHIP TO STOCK TEMPLATE
#   This is variant E's AMI for the Phase 4.1 cold-start comparison. It is
#   identical to phase4.1/packer/stock/gpu-node-stock.pkr.hcl in every way
#   except the image reference: this template pulls
#   inference-lab/vllm-openai:v0.19.0-streamer (= v0.19.0 + RunAI Model
#   Streamer pip install) instead of v0.19.0. Variant=prebaked-streamer is
#   the new tag; the prebaked-streamer NodePool selects on it.
#
# WHY A SEPARATE AMI (not just a different image at runtime)
#   The whole point of the prebaked variants is "image bytes already on
#   disk, no pull at pod start." Variant D's prebaked-stock AMI only has
#   the v0.19.0 image cached — pulling :v0.19.0-streamer at runtime would
#   re-introduce a network pull and erase the prebake's effect. Variant E
#   needs its own AMI with the streamer image cached.
#
# WHAT'S DIFFERENT FROM STOCK ON DISK
#   The streamer image shares 27 of 28 layers (digests byte-identical) with
#   v0.19.0. Only one new ~1MB layer is added. So this AMI's snapshot is
#   nearly identical in size to the prebaked-stock snapshot — most of the
#   bake-time pull is layer dedup, not redownload.
#
# FSR
#   Like the stock template, this does NOT enable FSR at build time. The
#   variant-E run procedure enables FSR on the resulting snapshot in
#   us-east-1a only for the measurement window, then disables it.
#
# USAGE
#   bash phase4.1/scripts/build-ami.sh --template gpu-node-streamer.pkr.hcl

packer {
  required_plugins {
    amazon = {
      version = ">= 1.3.0"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

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
  # STREAMER image — v0.19.0 + runai-model-streamer + s3 extras.
  # Built by phase4.1/scripts/launch-streamer-build.sh on 2026-05-01.
  # Digest at bake time: sha256:f4b7c5cc5b49131bcc2733fee664c5c9ac9034ed894261a10e8ed44e0955b099
  default = "019167255542.dkr.ecr.us-east-1.amazonaws.com/inference-lab/vllm-openai:v0.19.0-streamer"
}

variable "instance_type" {
  type    = string
  default = "g4dn.xlarge"
}

# Same EKS-optimized AL2 GPU AMI as the slim/stock templates — keeps the
# OS layer constant across variants so node_bootstrap differences are
# attributable only to the cached image, not the base AMI.
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

source "amazon-ebs" "gpu_node_streamer" {
  region     = var.region
  source_ami = data.amazon-ami.eks_gpu.id

  ssh_username        = "ec2-user"
  spot_price          = "auto"
  spot_instance_types = ["g4dn.xlarge", "g4dn.2xlarge"]

  # Pin to us-east-1a — same AZ as variants A-D ran in, so the EBS snapshot
  # is co-located with where variant E will run.
  availability_zone = "us-east-1a"

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  ami_name        = "inference-lab-gpu-node-streamer-{{timestamp}}"
  ami_description = "EKS ${var.eks_version} GPU node with pre-cached STREAMER vLLM image (${var.vllm_image})"

  tags = {
    Name        = "inference-lab-gpu-node-streamer"
    Project     = "inference-lab"
    Stage       = "phase4.1"
    Variant     = "prebaked-streamer"  # gpu-nodepool-prebaked-streamer.yaml selects on this
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

build {
  sources = ["source.amazon-ebs.gpu_node_streamer"]

  # Containerd parallelism tuning — same values as stock/slim templates so
  # the node_bootstrap stage is comparable across variants C/D/E.
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

  # Pre-pull the streamer-augmented vLLM image into containerd's k8s.io
  # namespace. This is the variable under test for variant E vs D.
  provisioner "shell" {
    inline = [
      "echo '=== Pre-caching streamer vLLM image (~9.5GB) ==='",
      "echo 'Base AMI: ${data.amazon-ami.eks_gpu.id}'",
      "",
      "for i in $(seq 1 30); do",
      "  if sudo ctr version >/dev/null 2>&1; then break; fi",
      "  sleep 2",
      "done",
      "",
      "echo 'Pulling vLLM streamer image from ECR: ${var.vllm_image}'",
      "ECR_PASS=`aws ecr get-login-password --region ${var.region} 2>/dev/null` || true",
      "if [ -z \"$ECR_PASS\" ]; then",
      "  echo 'ERROR: ECR auth failed. Check the build instance IAM profile has ECR pull rights.' >&2",
      "  exit 1",
      "fi",
      "sudo ctr -n k8s.io images pull --user \"AWS:$ECR_PASS\" ${var.vllm_image}",
      "echo 'vLLM streamer image cached successfully'",
      "",
      # Variant E removes the model-download init container, so we do NOT
      # pre-pull aws-cli here. That image was only ever used by variant
      # C/D's init container — variant E reads weights via the streamer
      # directly from the vLLM main container.
      "",
      "echo ''",
      "echo '=== Image digest verification ==='",
      "sudo ctr -n k8s.io images ls | grep vllm-openai",
      "echo ''",
      "sudo crictl images --digests | grep vllm-openai || true",
      "",
      # Verify the streamer package actually loads from the cached image.
      # This is the GPU-side verification we deferred from Stage 1 (where
      # the build host was CPU-only). On a g4dn the image runs with full
      # CUDA — if vllm + runai_model_streamer can both import here, we
      # know variant E's pod will not crash on imports at runtime.
      "echo ''",
      "echo '=== GPU-side import smoke test ==='",
      "sudo ctr -n k8s.io run --rm --gpus 0 \\",
      "  ${var.vllm_image} streamer-smoke \\",
      "  python3 -c 'import runai_model_streamer; from runai_model_streamer import SafetensorsStreamer; import vllm; print(\"imports ok\", vllm.__version__)' \\",
      "  || (echo 'IMPORT TEST FAILED — abort the bake' && exit 1)",
      "",
      "echo ''",
      "echo '=== Disk usage after pre-cache ==='",
      "df -h /",
      "echo ''",
      "echo '=== AMI build complete ==='",
    ]
  }
}
