# `inference-lab` EKS cluster

This directory provisions a clean, isolated EKS cluster named
**`inference-lab`** that hosts the empirical inference experiments. It
does not share VPC, IAM, ECR, S3, or Karpenter NodePool with any other
`phase*` stack in this repository — see [Non-interference](#non-interference)
below for the full list of separation guarantees.

## Layout

```
terraform/         # cluster, VPC, IAM, ECR, S3 weight cache
manifests/         # Karpenter, engine Deployments (per engine), driver Job
bringup.sh         # apply Terraform + install Karpenter + bootstrap NodePool
teardown.sh        # reverse, with --dry-run support
preflight.sh       # quota + auth check (run before bringup)
```

## Prerequisites

- **Terraform ≥ 1.5.0** — pinned in `terraform/versions.tf`. We need
  ≥ 1.5 for the `moved`/`import` blocks this stack uses, but stay below
  the aspirational 1.6 floor that broke CI runners shipping 1.5.7. The
  CI `terraform validate` step runs on 1.5.7 as the canary.
- **AWS CLI v2** with credentials for an account that has at least the
  permissions probed by `./preflight.sh` (IAM/EKS/EC2/ECR/S3/DynamoDB/KMS
  + the `servicequotas:GetServiceQuota` action).
- **Service-quotas headroom:** ≥ 48 vCPU for "All G and VT Spot Instance
  Requests" in the target region (default `us-east-1`).

## Quickstart

```bash
# 1. Verify AWS creds + quotas (spot G/VT vCPU >= 48 for g5.12xlarge TP=4)
./eks/preflight.sh

# 2. Provision (Terraform + Karpenter + NodePool)
./eks/bringup.sh

# 3. Build & push engine images (does NOT touch phase* ECRs)
./eks/build_images.sh vllm sglang trtllm

# 4. Verify isolation guard
KUBECONFIG=./eks/kubeconfig kubectl --context inference-lab cluster-info

# 5. Run a sweep (from project root)
exp run --engine vllm --workload chatbot --rate 8 --duration 300

# 6. Tear down
./eks/teardown.sh           # interactive
./eks/teardown.sh --yes     # CI / non-interactive
```

## Non-interference

| Aspect | This stack | Sibling phase* stacks |
| --- | --- | --- |
| Cluster name | `inference-lab` | `inference-phase*` |
| Kubeconfig | `experiments/eks/kubeconfig` | Their own |
| Terraform state | S3 `inference-lab-tfstate-<acct>` + DDB `inference-lab-tflock` | Distinct |
| VPC | New `10.42.0.0/16` (CIDR collides with nothing) | Distinct |
| IAM roles | `inference-lab-*` | `inference-phase*-*` |
| ECR repos | `inference-lab/*` | Distinct |
| S3 weight cache | `inference-lab-models-<acct>` | Distinct |
| Karpenter NodePool | `inference-lab-gpu` | Distinct |
| Tag | `Project=inference-lab` on every resource | Distinct |

A CI build test ([`tests/build/test_no_phase_collisions.py`](../tests/build/test_no_phase_collisions.py))
greps every YAML / TF / shell artifact in this directory and fails the
build if the literal `inference-phase` appears anywhere outside the
allowlist — catches accidental copy-paste from phase4.5 manifests.

## Cost notes

Default NodePool: `g5.12xlarge` (4×A10G, 48 vCPU, 192 GB), Spot preferred,
OnDemand fallback. **Empty NodePool = $0/hr** thanks to Karpenter
consolidating idle nodes. Run `./eks/teardown.sh` when you're done — leaving
the control plane up is $0.10/hr but nodes are the real cost.
