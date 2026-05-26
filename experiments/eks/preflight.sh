#!/usr/bin/env bash
# Pre-flight checks before running ./eks/bringup.sh.
# Verifies AWS credentials, region quota for spot G/VT vCPU, and that no
# existing inference-lab resources will be clobbered.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
NEEDED_VCPU="${NEEDED_VCPU:-48}"  # g5.12xlarge

red()    { printf '\033[31m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }

echo "== AWS identity =="
aws sts get-caller-identity --output table

echo
echo "== Existing cluster check =="
if aws eks describe-cluster --region "$REGION" --name inference-lab >/dev/null 2>&1; then
  yellow "Cluster 'inference-lab' already exists in $REGION. bringup.sh will be a no-op for cluster create."
else
  green "No existing 'inference-lab' cluster — clean."
fi

echo
echo "== Spot G/VT vCPU quota (need >= $NEEDED_VCPU for g5.12xlarge TP=4) =="
QUOTA_CODE="L-3819A6DF"  # All G and VT Spot Instance Requests
VALUE=$(aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code "$QUOTA_CODE" \
  --region "$REGION" \
  --query 'Quota.Value' \
  --output text 2>/dev/null || echo "0")
echo "current quota: $VALUE"
# Use awk for float compare; bash can't natively.
if awk "BEGIN {exit !($VALUE < $NEEDED_VCPU)}"; then
  red  "Spot G/VT quota ($VALUE) is below required $NEEDED_VCPU."
  red  "Request an increase via AWS Service Quotas, or fall back to on-demand."
  exit 1
else
  green "Quota OK."
fi

echo
echo "== Terraform backend bucket =="
ACCT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="inference-lab-tfstate"
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  green "Backend bucket exists."
else
  yellow "Backend bucket '$BUCKET' missing — run ./eks/bootstrap_backend.sh once."
fi

echo
echo "Pre-flight done. Acct=$ACCT Region=$REGION"
