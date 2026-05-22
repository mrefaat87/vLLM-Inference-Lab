# variables.tf — All configurable inputs for the inference platform
# Defaults are tuned for cost-effective learning. Override in terraform.tfvars.

variable "aws_region" {
  description = "AWS region to deploy into. us-east-1 has the most GPU instance availability."
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name. Also used as the Karpenter discovery tag value."
  type        = string
  default     = "inference-lab"
}

variable "cluster_version" {
  description = "Kubernetes version for EKS. 1.29 is latest stable with good Karpenter v0.33 support."
  type        = string
  default     = "1.29"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC. /16 gives 65k IPs — plenty for pods with VPC CNI."
  type        = string
  default     = "10.0.0.0/16"
}

variable "azs" {
  description = "Availability zones. Two AZs balance HA vs cost (NAT Gateway per AZ is $32/mo each)."
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

variable "cpu_instance_type" {
  description = "Instance type for CPU node group. t3.medium is cheapest option that runs system pods."
  type        = string
  default     = "t3.medium"
}

variable "cpu_node_count" {
  description = "Desired number of CPU nodes. 2 ensures HA for system workloads (CoreDNS, Karpenter)."
  type        = number
  default     = 2
}

variable "my_ip" {
  description = "Your public IP in CIDR notation (e.g., '203.0.113.42/32'). Used to restrict SSH access. Find yours: curl ifconfig.me"
  type        = string
  # No default — force explicit setting so we never accidentally open SSH to 0.0.0.0/0
}

variable "model_cache_bucket_prefix" {
  description = "Prefix for the S3 bucket storing cached model weights. Account ID is appended for uniqueness."
  type        = string
  default     = "inference-lab-model-cache"
}

variable "hf_token" {
  description = "HuggingFace API token for downloading gated models (e.g., Llama). Stored as K8s secret later."
  type        = string
  default     = ""
  sensitive   = true # Prevents token from appearing in plan output or state diffs
}
