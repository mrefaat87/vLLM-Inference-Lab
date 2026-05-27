terraform {
  # Matches sibling phase4* stacks. 1.5+ covers the moved/import blocks
  # this stack uses; 1.6 was aspirational and blocked CI runners that
  # ship 1.5.7.
  required_version = ">= 1.5.0"
  required_providers {
    aws        = { source = "hashicorp/aws", version = "~> 5.50" }
    kubernetes = { source = "hashicorp/kubernetes", version = "~> 2.30" }
    helm       = { source = "hashicorp/helm", version = "~> 2.13" }
    random     = { source = "hashicorp/random", version = "~> 3.6" }
  }

  # State is isolated from sibling phase* stacks.
  # Run `./eks/bootstrap_backend.sh` once per AWS account to create the
  # S3 bucket and DynamoDB lock table before the first apply.
  backend "s3" {
    bucket         = "inference-lab-tfstate"
    key            = "cluster/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "inference-lab-tflock"
    encrypt        = true
  }
}
