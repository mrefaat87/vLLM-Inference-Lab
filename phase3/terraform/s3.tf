# s3.tf — Reference existing S3 model cache + install Mountpoint CSI driver.
#
# WHY reference (not create): the model cache bucket already has Qwen2.5-7B-AWQ
# downloaded (~4GB). Re-downloading wastes time and bandwidth. The bucket is
# shared across phases; only the CSI driver addon is cluster-specific.

data "aws_s3_bucket" "model_cache" {
  bucket = "${var.model_cache_bucket_prefix}-${data.aws_caller_identity.current.account_id}"
}

# Mountpoint for S3 CSI Driver — must be installed per-cluster
resource "aws_eks_addon" "s3_csi_driver" {
  cluster_name             = module.eks.cluster_name
  addon_name               = "aws-mountpoint-s3-csi-driver"
  service_account_role_arn = module.s3_csi_irsa.iam_role_arn

  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"

  depends_on = [module.eks]
}
