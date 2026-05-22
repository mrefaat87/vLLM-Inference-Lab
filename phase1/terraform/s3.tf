# s3.tf — S3 model cache bucket and Mountpoint CSI driver
#
# Why S3 for model caching: GPU model weights are 2-30GB+. Downloading from HuggingFace
# every time a pod starts is slow and wastes bandwidth. S3 acts as a regional cache —
# download once from HF, serve many times. Mountpoint CSI mounts the bucket as a filesystem
# so vLLM can read model files with standard file I/O (no SDK needed in the container).
# Think of it like an EFS mount, but backed by S3 pricing (~$0.023/GB/mo vs $0.30/GB/mo).

# ─── S3 Bucket ───────────────────────────────────────────────────────────────────

resource "aws_s3_bucket" "model_cache" {
  # Append account ID for global uniqueness (S3 bucket names are globally unique)
  bucket = "${var.model_cache_bucket_prefix}-${data.aws_caller_identity.current.account_id}"

  # force_destroy = false — prevent accidental deletion of cached models during terraform destroy.
  # You'd need to empty the bucket manually first. This is intentional — re-downloading
  # a 14GB Llama model takes 20+ minutes.
  force_destroy = false

  tags = {
    Name = "${var.cluster_name}-model-cache"
  }
}

# Block ALL public access — model weights should never be publicly accessible
resource "aws_s3_bucket_public_access_block" "model_cache" {
  bucket = aws_s3_bucket.model_cache.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable versioning — allows rollback if a corrupted model gets uploaded
resource "aws_s3_bucket_versioning" "model_cache" {
  bucket = aws_s3_bucket.model_cache.id

  versioning_configuration {
    status = "Enabled"
  }
}

# ─── Mountpoint for S3 CSI Driver ────────────────────────────────────────────────
# EKS addon that lets pods mount S3 buckets as local filesystems.
# Karpenter-launched GPU nodes will use this to access model weights without
# baking them into the container image (which would make images 10GB+).

resource "aws_eks_addon" "s3_csi_driver" {
  cluster_name             = module.eks.cluster_name
  addon_name               = "aws-mountpoint-s3-csi-driver"
  service_account_role_arn = module.s3_csi_irsa.iam_role_arn

  # OVERWRITE preserves the addon even if someone manually edited it
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"

  depends_on = [module.eks]
}
