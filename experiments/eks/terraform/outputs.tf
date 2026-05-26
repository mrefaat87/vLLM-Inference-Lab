output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "kubeconfig_command" {
  value = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name} --alias ${module.eks.cluster_name} --kubeconfig ./eks/kubeconfig"
}

output "weights_bucket" {
  value = aws_s3_bucket.weights.bucket
}

output "ecr_registry" {
  value = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.region}.amazonaws.com"
}

output "karpenter_iam_role_arn" {
  value = module.karpenter.iam_role_arn
}

output "karpenter_node_iam_role_name" {
  value = module.karpenter.node_iam_role_name
}
