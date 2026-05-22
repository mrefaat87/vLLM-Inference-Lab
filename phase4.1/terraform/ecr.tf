# ecr.tf — ECR repos for the vLLM worker and API gateway images.
#
# Phase 4 originally referenced these as data sources (Phase 1 created them).
# After the 2026-04-30 teardown, they no longer exist, so Phase 4.1 owns them.
# Account-level resources: kept under the "inference-lab/" prefix so future
# phases can reference them as data if Phase 4.1 is destroyed first (i.e. the
# teardown order is: phase4.1 cluster ≠ ECR repos — ECR survives).

resource "aws_ecr_repository" "api_gateway" {
  name                 = "inference-lab/api-gateway"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "worker" {
  name                 = "inference-lab/worker"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}
