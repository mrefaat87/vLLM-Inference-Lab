#!/usr/bin/env bash
# Post-teardown assertion. Exits non-zero if any inference-lab AWS
# resource still exists. Safe to run any time as a no-op sanity check.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
FAIL=0

check() {
  local label="$1"; shift
  if "$@" 2>/dev/null | grep -q .; then
    echo "STILL PRESENT: $label" >&2
    FAIL=1
  else
    echo "ok: $label gone"
  fi
}

# EKS cluster
check "EKS cluster" aws eks list-clusters --region "$REGION" --query "clusters[?@=='inference-lab']" --output text

# Running EC2 instances tagged Project=inference-lab
check "EC2 instances (Project=inference-lab)" \
  aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:Project,Values=inference-lab" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text

# ECR repositories
check "ECR repos" aws ecr describe-repositories --region "$REGION" \
  --query "repositories[?starts_with(repositoryName, 'inference-lab')].repositoryName" --output text

# S3 weight bucket
ACCT=$(aws sts get-caller-identity --query Account --output text)
check "S3 weight bucket" aws s3api list-buckets \
  --query "Buckets[?Name=='inference-lab-models-${ACCT}'].Name" --output text

# IAM roles
check "IAM roles" aws iam list-roles \
  --query "Roles[?starts_with(RoleName, 'inference-lab-')].RoleName" --output text

if [[ $FAIL -ne 0 ]]; then
  echo
  echo "Teardown incomplete — see above. Run ./eks/teardown.sh again or remove manually." >&2
  exit 1
fi
echo "All clean."
