# outputs.tf — Values needed for subsequent phases and daily operations
# These outputs are your "connection info" — like the endpoint URLs you'd get
# after creating an RDS instance or an API Gateway.

output "cluster_name" {
  description = "EKS cluster name — used in kubectl commands and Karpenter config"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS API server endpoint — kubectl talks to this URL"
  value       = module.eks.cluster_endpoint
}

output "kubeconfig_command" {
  description = "Run this to configure kubectl for the cluster"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

output "ecr_api_gateway_url" {
  description = "ECR repository URL for the API gateway image"
  value       = aws_ecr_repository.api_gateway.repository_url
}

output "ecr_worker_url" {
  description = "ECR repository URL for the vLLM worker image"
  value       = aws_ecr_repository.worker.repository_url
}

output "model_cache_bucket_name" {
  description = "S3 bucket name for cached model weights"
  value       = data.aws_s3_bucket.model_cache.id
}

output "karpenter_node_role_name" {
  description = "IAM role name for Karpenter-managed nodes — needed for aws-auth ConfigMap"
  value       = aws_iam_role.karpenter_node.name
}

output "karpenter_node_role_arn" {
  description = "IAM role ARN for Karpenter-managed nodes — used in EC2NodeClass"
  value       = aws_iam_role.karpenter_node.arn
}

output "karpenter_node_instance_profile_name" {
  description = "Instance profile name for Karpenter-launched nodes"
  value       = aws_iam_instance_profile.karpenter_node.name
}

output "region" {
  description = "AWS region where everything is deployed"
  value       = var.aws_region
}

output "vpc_id" {
  description = "VPC ID — useful for debugging network issues"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "Private subnet IDs — where GPU nodes are launched"
  value       = module.vpc.private_subnets
}

output "vllm_worker_role_arn" {
  description = "IRSA role ARN for vLLM worker pods — annotate the ServiceAccount with this"
  value       = module.vllm_worker_irsa.iam_role_arn
}

output "oidc_provider_arn" {
  description = "OIDC provider ARN — needed to create additional IRSA roles"
  value       = module.eks.oidc_provider_arn
}
