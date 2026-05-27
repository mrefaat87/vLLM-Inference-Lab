"""Contract tests for the EKS bring-up assets.

These tests pin the bringup.sh / preflight.sh / nodepool / terraform
invariants that the May bring-up debugging session uncovered. They run
without a live cluster (shell parse-check + YAML structure + Terraform
file-level validation), so they fit a normal pytest pass.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.contract


_REPO = Path(__file__).resolve().parents[3]
_EKS = _REPO / "experiments" / "eks"


def _read(path: Path) -> str:
    return path.read_text()


# ---------------------------------------------------------------------------
# Shell scripts: parse-check with bash -n and pin the critical invariants.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("script", ["bringup.sh", "preflight.sh", "teardown.sh"])
def test_shell_script_parses(script: str) -> None:
    """`bash -n` parses the script without executing it."""
    path = _EKS / script
    r = subprocess.run(
        ["bash", "-n", str(path)], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, f"{script} failed to parse:\n{r.stderr}"


def test_bringup_uses_prefixed_interruption_queue() -> None:
    # Terraform mints the SQS queue as "Karpenter-<cluster>"; Helm must
    # be told the same name or Karpenter silently won't drain spot
    # interruptions.
    text = _read(_EKS / "bringup.sh")
    assert "interruptionQueue=Karpenter-$CLUSTER" in text, (
        "bringup.sh must point Karpenter at the Karpenter-<cluster> SQS queue"
    )


def test_bringup_creates_engines_namespace() -> None:
    # Drivers apply vllm.yaml into the `engines` namespace; if it
    # doesn't exist the apply fails with NotFound.
    text = _read(_EKS / "bringup.sh")
    assert "create namespace engines" in text


def test_bringup_uses_correct_nvidia_plugin_url() -> None:
    # The NVIDIA plugin manifest moved under /deployments/static/ in
    # v0.15.0 — the bare URL serves the README, not YAML, and `kubectl
    # apply` happily applies nothing.
    text = _read(_EKS / "bringup.sh")
    assert "/deployments/static/nvidia-device-plugin.yml" in text


def test_preflight_does_not_swallow_quota_failures() -> None:
    # Old preflight used `|| echo "0"` which made AccessDenied
    # indistinguishable from a real quota of 0. That cost hours.
    text = _read(_EKS / "preflight.sh")
    assert '|| echo "0"' not in text
    # And the probe scaffolding must be present.
    assert "probe_iam" in text
    assert "servicequotas:GetServiceQuota" in text


def test_preflight_probes_critical_permissions() -> None:
    """Every action bringup.sh needs must show up as a probe."""
    text = _read(_EKS / "preflight.sh")
    required_actions = [
        "ec2:DescribeVpcs",
        "eks:ListClusters",
        "iam:ListRoles",
        "ecr:DescribeRepositories",
        "s3:ListAllMyBuckets",
        "dynamodb:DescribeTable",
        "kms:ListKeys",
        "servicequotas:GetServiceQuota",
    ]
    missing = [a for a in required_actions if a not in text]
    assert not missing, f"preflight.sh missing probes for: {missing}"


# ---------------------------------------------------------------------------
# NodePool / EC2NodeClass: cleanup pins.
# ---------------------------------------------------------------------------
def _load_yaml_docs(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(_read(path)) if d]


def test_nodepool_has_no_literal_gpu_label() -> None:
    """The old `gpu: a10g` label mislabeled g4dn (T4) and g6 (L4) nodes."""
    docs = _load_yaml_docs(_EKS / "manifests" / "karpenter-nodepool.yaml")
    nodepool = next(d for d in docs if d["kind"] == "NodePool")
    labels = (
        nodepool["spec"]["template"]["metadata"].get("labels", {})
    )
    assert "gpu" not in labels, (
        "NodePool must not stamp a fixed `gpu:` label — pods select on "
        "karpenter.k8s.aws/instance-gpu-name instead"
    )


def test_nodepool_allows_multiple_gpu_families() -> None:
    docs = _load_yaml_docs(_EKS / "manifests" / "karpenter-nodepool.yaml")
    nodepool = next(d for d in docs if d["kind"] == "NodePool")
    reqs = nodepool["spec"]["template"]["spec"]["requirements"]
    family_req = next(r for r in reqs if r["key"] == "karpenter.k8s.aws/instance-family")
    # Must allow at least g4dn (T4) and g5 (A10G); the rendered vllm
    # Deployment pins the exact type so widening here is safe.
    assert "g4dn" in family_req["values"]
    assert "g5" in family_req["values"]


def test_ec2nodeclass_sets_imds_hop_limit_2() -> None:
    """Without hopLimit=2 the fetch-weights init container can't reach IMDS."""
    docs = _load_yaml_docs(_EKS / "manifests" / "karpenter-nodepool.yaml")
    nodeclass = next(d for d in docs if d["kind"] == "EC2NodeClass")
    md = nodeclass["spec"].get("metadataOptions")
    assert md is not None, "EC2NodeClass must declare metadataOptions"
    assert md.get("httpPutResponseHopLimit") == 2


# ---------------------------------------------------------------------------
# Terraform: version pin + EBS CSI IRSA wiring.
# ---------------------------------------------------------------------------
def test_terraform_version_pin_is_attainable() -> None:
    """1.6.0 was aspirational; CI runners ship 1.5.7."""
    text = _read(_EKS / "terraform" / "versions.tf")
    # Allow either ">= 1.5.0" or any version that includes 1.5.
    assert ">= 1.5" in text, "terraform required_version must allow 1.5.x"
    assert ">= 1.6" not in text, (
        "terraform required_version must not require 1.6+ (CI ships 1.5.7)"
    )


def test_main_tf_wires_ebs_csi_irsa() -> None:
    """The manual node-role policy attach was a workaround. Real fix: IRSA."""
    text = _read(_EKS / "terraform" / "main.tf")
    # The module mints a dedicated role; the addon block references it.
    assert "ebs_csi_irsa" in text, "main.tf must declare an ebs_csi_irsa module"
    assert "service_account_role_arn = module.ebs_csi_irsa.iam_role_arn" in text, (
        "aws-ebs-csi-driver addon must consume the IRSA role ARN"
    )


@pytest.mark.skipif(
    not shutil.which("terraform"), reason="terraform binary not installed in CI"
)
def test_terraform_fmt_check_passes() -> None:
    """`terraform fmt -check` catches the kind of single-line-block style
    drift that the May session left behind across multiple HCL files.
    """
    r = subprocess.run(
        ["terraform", "fmt", "-check", "-recursive", str(_EKS / "terraform")],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, f"terraform fmt drift:\n{r.stdout}\n{r.stderr}"


# The vLLM template's vLLM-0.7 invocation + ${AWS_ACCOUNT} → {{AWS_ACCOUNT}}
# fix lives in the sibling `lab-driver-honor-instance-and-portforward` PR.
# Its own contract test covers those invariants.
