# Phase 4.1 Variant H — Manual Teardown (run if Claude session ends before Stage 9)

If the Claude session ended early or the auto-cleanup cron didn't fire, run these
commands in order to make sure nothing leaks billable FSx capacity.

## 1. Check what's still up

```bash
aws fsx describe-file-systems --region us-east-1 \
  --query 'FileSystems[?Tags[?Key==`experiment`&&Value==`phase4.1-variantH`]].{Id:FileSystemId,State:Lifecycle}' \
  --output table
```

If the table is empty: nothing to clean up.

## 2. Destroy FSx + SG

```bash
cd /Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/phase4.1/terraform
terraform destroy \
  -target=aws_fsx_lustre_file_system.phase41_variantH \
  -target=aws_security_group.fsx_lustre \
  -auto-approve
```

Filesystem deletion takes 5–10 min. Re-run the describe call until the table is empty.

## 3. Disable FSR on the H AMI snapshot (if FSR was enabled in this session)

```bash
# Find the snapshot ID
SNAP=$(aws ec2 describe-images --region us-east-1 \
  --filters "Name=tag:Variant,Values=prebaked-fsx" \
  --query 'Images[0].BlockDeviceMappings[0].Ebs.SnapshotId' \
  --output text)
aws ec2 disable-fast-snapshot-restores --region us-east-1 \
  --availability-zones us-east-1a --source-snapshot-ids $SNAP
```

## 4. Detach IAM permissions added for this experiment

```bash
# Karpenter node role temp policy (added in Stage 3)
aws iam delete-role-policy \
  --role-name inference-phase4-1-karpenter-node \
  --policy-name fsx-describe-temp 2>/dev/null || echo "already detached"

# Operator user policy (added in Stage 0)
aws iam delete-user-policy \
  --user-name vLLM-spot-lab \
  --policy-name phase41-variantH-fsx
```

## 5. Verify zero leak

```bash
aws fsx describe-file-systems --region us-east-1 --query 'FileSystems[].FileSystemId' --output text
# Empty output = clean.
```
