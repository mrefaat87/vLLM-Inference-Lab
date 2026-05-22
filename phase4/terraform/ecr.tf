# ecr.tf — Reference existing ECR repos from Phase 1 (shared, not duplicated).
#
# WHY data sources instead of resources: ECR repos are account-level, not
# cluster-level. Phase 1 already created them and they contain the worker:phase4v3
# image we need. Creating new repos would require re-pushing images.

data "aws_ecr_repository" "api_gateway" {
  name = "inference-lab/api-gateway"
}

data "aws_ecr_repository" "worker" {
  name = "inference-lab/worker"
}
