variable "region" {
  type    = string
  default = "us-east-1"
}

variable "cluster_name" {
  type    = string
  default = "inference-lab"

  validation {
    condition     = !can(regex("inference-phase", var.cluster_name))
    error_message = "Cluster name must not contain 'inference-phase' — that namespace belongs to sibling stacks."
  }
}

variable "k8s_version" {
  type    = string
  default = "1.30"
}

variable "vpc_cidr" {
  type    = string
  default = "10.42.0.0/16"
}

variable "azs" {
  type    = list(string)
  default = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "tags" {
  type = map(string)
  default = {
    Project = "inference-lab"
    Owner   = "mohamed"
  }
}
