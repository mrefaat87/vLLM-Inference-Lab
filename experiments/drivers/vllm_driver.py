"""vLLM driver.

Wraps the K8s lifecycle around a vLLM OpenAI-compatible server. The
template at eks/manifests/engines/vllm.yaml renders to a Deployment +
Service. The engine speaks the OpenAI chat/completions API natively.
"""

from __future__ import annotations

from pathlib import Path

from experiments.drivers.k8s_common import K8sEngineDriver

_HERE = Path(__file__).resolve().parent
_TEMPLATE_PATH = _HERE.parent / "eks" / "manifests" / "engines" / "vllm.yaml"


class VLLMDriver(K8sEngineDriver):
    name = "vllm"
    template_path = _TEMPLATE_PATH
    health_path = "/health"
    service_port = 8000
