# ecr.tf — Container registries for the inference platform images
#
# Two repositories: one for the API gateway (routes requests, manages queue)
# and one for the vLLM worker (runs the model). Separating them means you can
# update the gateway without rebuilding the 10GB+ worker image.

resource "aws_ecr_repository" "api_gateway" {
  name = "inference-lab/api-gateway"

  # MUTABLE tags let you push "latest" repeatedly during development.
  # In production you'd use IMMUTABLE to prevent overwriting a deployed tag.
  image_tag_mutability = "MUTABLE"

  # force_delete = true — allows terraform destroy to clean up even if images exist.
  # Safe for a learning project; in production you'd protect against accidental deletion.
  force_delete = true

  # Scan images for CVEs on push — free tier covers this
  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "${var.cluster_name}-api-gateway"
  }
}

resource "aws_ecr_repository" "worker" {
  name = "inference-lab/worker"

  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "${var.cluster_name}-worker"
  }
}

# Lifecycle policy — auto-delete untagged images after 7 days to save storage costs.
# ECR charges $0.10/GB/month and vLLM worker images can be 5-10GB each.
resource "aws_ecr_lifecycle_policy" "api_gateway" {
  repository = aws_ecr_repository.api_gateway.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Expire untagged images after 7 days"
      selection = {
        tagStatus   = "untagged"
        countType   = "sinceImagePushed"
        countUnit   = "days"
        countNumber = 7
      }
      action = {
        type = "expire"
      }
    }]
  })
}

resource "aws_ecr_lifecycle_policy" "worker" {
  repository = aws_ecr_repository.worker.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Expire untagged images after 7 days"
      selection = {
        tagStatus   = "untagged"
        countType   = "sinceImagePushed"
        countUnit   = "days"
        countNumber = 7
      }
      action = {
        type = "expire"
      }
    }]
  })
}
