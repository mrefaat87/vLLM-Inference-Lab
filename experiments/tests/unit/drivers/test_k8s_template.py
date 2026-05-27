"""Tests for the K8s template substitution / Karpenter affinity wiring."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from experiments.drivers.k8s_common import _vars_for_cfg, _substitute
from experiments.drivers.vllm_driver import VLLMDriver
from experiments.runner.schema import EngineConfig

pytestmark = pytest.mark.unit


def _cfg(instance: str | None = "g5.12xlarge", gpu: str | None = "A10G") -> EngineConfig:
    return EngineConfig(
        name="vllm",
        image="vllm/vllm-openai:latest",
        model="meta-llama/Llama-3-70B-Instruct-AWQ",
        quantization="awq",
        tensor_parallel=4,
        max_model_len=8192,
        instance=instance,
        gpu=gpu,
        n_gpu=4,
    )


def test_vars_for_cfg_splits_instance_into_family_size() -> None:
    vars_ = _vars_for_cfg(_cfg(instance="g5.12xlarge"), "vllm")
    assert vars_["INSTANCE_FAMILY"] == "g5"
    assert vars_["INSTANCE_SIZE"] == "12xlarge"


def test_vars_for_cfg_gpu_name_is_lowercased() -> None:
    vars_ = _vars_for_cfg(_cfg(gpu="A10G"), "vllm")
    # Karpenter labels are lowercase: a10g, t4, h100. Mixing case here
    # would make the nodeAffinity match silently fail.
    assert vars_["GPU_NAME"] == "a10g"


def test_vars_for_cfg_sizes_memory_to_instance() -> None:
    # The old hardcoded "32Gi request / 96Gi limit" doesn't fit a g4dn.xlarge
    # (16 GiB). Verify we now scale down.
    vars_ = _vars_for_cfg(_cfg(instance="g4dn.xlarge"), "vllm")
    mem_req = int(vars_["MEMORY_REQUEST"].rstrip("Gi"))
    mem_lim = int(vars_["MEMORY_LIMIT"].rstrip("Gi"))
    assert mem_req < 16, f"memory request {mem_req}Gi won't fit on g4dn.xlarge (16 GiB)"
    assert mem_lim <= 16


def test_vars_for_cfg_unknown_instance_falls_back_without_raising() -> None:
    vars_ = _vars_for_cfg(_cfg(instance="g99.42xlarge"), "vllm")
    # Unknown instance: family/size end up empty but render still completes.
    # The driver logs a warning; the test asserts we don't crash mid-run.
    assert vars_["INSTANCE_FAMILY"] == ""
    assert vars_["INSTANCE_SIZE"] == ""
    assert vars_["MEMORY_REQUEST"]  # populated with fallback


def test_vars_for_cfg_no_instance_emits_empty_selectors() -> None:
    vars_ = _vars_for_cfg(_cfg(instance=None, gpu=None), "vllm")
    assert vars_["INSTANCE_FAMILY"] == ""
    assert vars_["INSTANCE_SIZE"] == ""
    assert vars_["GPU_NAME"] == ""


def test_rendered_vllm_manifest_uses_serve_command_and_positional_weights() -> None:
    driver = VLLMDriver()
    rendered = driver.render(_cfg())
    docs = list(yaml.safe_load_all(rendered))
    deployment = next(d for d in docs if d["kind"] == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    # vLLM ≥ 0.7 wants "serve" + positional model path. The old --model=
    # flag was removed; verify we're not regressing.
    assert container["command"] == ["vllm", "serve"]
    assert container["args"][0] == "/weights"
    # The served-model-name keeps the HF id as the API handle even though
    # the on-disk path is /weights.
    assert any(
        a.startswith("--served-model-name=") for a in container["args"]
    ), "served-model-name flag missing — clients will have to query '/weights'"


def test_rendered_vllm_manifest_does_not_include_disable_log_requests() -> None:
    rendered = VLLMDriver().render(_cfg())
    # --disable-log-requests was a 0.6-era flag that doesn't exist on 0.7+.
    # If we accidentally re-add it the serve loop won't start.
    assert "--disable-log-requests" not in rendered


def test_rendered_vllm_manifest_carries_karpenter_affinity() -> None:
    rendered = VLLMDriver().render(_cfg(instance="g5.12xlarge", gpu="A10G"))
    docs = list(yaml.safe_load_all(rendered))
    deployment = next(d for d in docs if d["kind"] == "Deployment")
    affinity = deployment["spec"]["template"]["spec"]["affinity"]["nodeAffinity"]
    terms = affinity["requiredDuringSchedulingIgnoredDuringExecution"][
        "nodeSelectorTerms"
    ]
    flat_keys = {expr["key"]: expr["values"] for expr in terms[0]["matchExpressions"]}
    assert flat_keys["karpenter.k8s.aws/instance-family"] == ["g5"]
    assert flat_keys["karpenter.k8s.aws/instance-size"] == ["12xlarge"]
    assert flat_keys["karpenter.k8s.aws/instance-gpu-name"] == ["a10g"]


def test_rendered_vllm_aws_account_is_substituted_not_literal_var() -> None:
    rendered = VLLMDriver().render(_cfg())
    # The old template used shell-style ${AWS_ACCOUNT}, which never gets
    # expanded because there's no shell at apply time. We now substitute
    # at render time. Verify no literal ${AWS_ACCOUNT} survives.
    assert "${AWS_ACCOUNT}" not in rendered
    assert "{{AWS_ACCOUNT}}" not in rendered


def test_substitute_replaces_only_known_keys() -> None:
    out = _substitute("hello {{FOO}} {{BAR}}", {"FOO": "world"})
    # Unknown keys must be left untouched so a template typo surfaces
    # in the kubectl apply, not silently render to empty.
    assert out == "hello world {{BAR}}"
