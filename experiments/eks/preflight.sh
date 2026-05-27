#!/usr/bin/env bash
# Pre-flight checks before running ./eks/bringup.sh.
#
# Verifies AWS credentials, region quota for spot G/VT vCPU, that no
# existing inference-lab resources will be clobbered, AND that the IAM
# principal actually has every permission bringup.sh needs. The original
# version swallowed quota-API AccessDenied as a literal zero, which made
# permission gaps look like a low quota — multiple failed bringups were
# chasing that ghost.
#
# Exit code: 0 = ready to bring up; non-zero = fix the reported issues
# before retrying.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
NEEDED_VCPU="${NEEDED_VCPU:-48}"  # g5.12xlarge
CLUSTER="${CLUSTER:-inference-lab}"

red()    { printf '\033[31m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }

# Collect failures and surface them all at the end so the user sees the
# full picture per run, not the first-blocker-then-stop loop the old
# script forced.
ERRORS=()
fail() { red "FAIL: $*"; ERRORS+=("$*"); }
ok()   { green "OK:   $*"; }

echo "== AWS identity =="
aws sts get-caller-identity --output table

echo
echo "== Existing cluster check =="
if aws eks describe-cluster --region "$REGION" --name "$CLUSTER" >/dev/null 2>&1; then
  yellow "Cluster '$CLUSTER' already exists in $REGION. bringup.sh will be a no-op for cluster create."
else
  green "No existing '$CLUSTER' cluster — clean."
fi

echo
echo "== Spot G/VT vCPU quota (need >= $NEEDED_VCPU for g5.12xlarge TP=4) =="
QUOTA_CODE="L-3819A6DF"  # All G and VT Spot Instance Requests
# Do NOT swallow AccessDenied as "0" here — distinguishing "quota call
# failed" from "quota is zero" was the source of an hours-long red herring.
if VALUE=$(aws service-quotas get-service-quota \
    --service-code ec2 \
    --quota-code "$QUOTA_CODE" \
    --region "$REGION" \
    --query 'Quota.Value' \
    --output text 2>&1); then
  echo "current quota: $VALUE"
  # Use awk for float compare; bash can't natively.
  if awk "BEGIN {exit !($VALUE < $NEEDED_VCPU)}"; then
    fail "Spot G/VT quota ($VALUE) below required $NEEDED_VCPU. Request an increase via AWS Service Quotas, or fall back to on-demand."
  else
    ok "Quota OK."
  fi
else
  fail "Could not read Spot G/VT quota (servicequotas:GetServiceQuota failed): $VALUE"
fi

echo
echo "== Terraform backend bucket =="
ACCT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="inference-lab-tfstate"
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  ok "Backend bucket exists."
else
  yellow "Backend bucket '$BUCKET' missing — run ./eks/bootstrap_backend.sh once."
fi

echo
echo "== IAM permission probes =="
# Each probe targets a permission bringup.sh needs. The pattern: make a
# READ-ONLY call that touches the same action — if it returns
# AccessDenied / AccessDeniedException / UnauthorizedOperation we know
# the principal can't do the matching write either. Errors that aren't
# auth-related (NotFound, ValidationException) are GOOD here — they mean
# auth passed but the resource doesn't exist, which is expected.
probe_iam() {
  local desc="$1"; shift
  local out; local rc
  set +e
  out=$("$@" 2>&1); rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    ok "$desc"
    return
  fi
  # Match the common AWS auth-failure markers. Anything else is treated
  # as "auth probably fine, call just naturally errored" so a probe
  # against a non-existent resource doesn't false-positive.
  if echo "$out" | grep -qE "AccessDenied|UnauthorizedOperation|not authorized|expired|InvalidClientTokenId"; then
    fail "$desc — $(echo "$out" | head -n 1)"
  else
    ok "$desc"
  fi
}

probe_iam "ec2:DescribeVpcs"               aws ec2 describe-vpcs --region "$REGION" --max-results 5
probe_iam "ec2:DescribeSecurityGroups"     aws ec2 describe-security-groups --region "$REGION" --max-results 5
probe_iam "eks:ListClusters"               aws eks list-clusters --region "$REGION"
probe_iam "iam:ListRoles"                  aws iam list-roles --max-items 1
probe_iam "iam:GetRole (PassRole proxy)"   aws iam get-role --role-name "__probe_does_not_exist_${RANDOM}"
probe_iam "ecr:DescribeRepositories"       aws ecr describe-repositories --region "$REGION" --max-results 1
probe_iam "s3:ListAllMyBuckets"            aws s3api list-buckets
probe_iam "s3:HeadBucket on weights bucket" aws s3api head-bucket --bucket "inference-lab-models-$ACCT"
probe_iam "dynamodb:DescribeTable (tflock)" aws dynamodb describe-table --table-name "inference-lab-tflock" --region "$REGION"
probe_iam "kms:ListKeys"                   aws kms list-keys --region "$REGION" --limit 1
probe_iam "servicequotas:GetServiceQuota"  aws service-quotas get-service-quota --service-code ec2 --quota-code "$QUOTA_CODE" --region "$REGION"

echo
if (( ${#ERRORS[@]} > 0 )); then
  red "Pre-flight FAILED with ${#ERRORS[@]} issue(s):"
  for e in "${ERRORS[@]}"; do
    red "  - $e"
  done
  exit 1
fi
green "Pre-flight passed. Acct=$ACCT Region=$REGION"
