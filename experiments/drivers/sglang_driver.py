"""SGLang driver.

SGLang exposes an OpenAI-compatible API on port 30000 by default and a
``/health_generate`` endpoint that confirms the model is loaded and the
runtime is ready (different from ``/health``, which only confirms the
HTTP server is up).
"""

from __future__ import annotations

from pathlib import Path

from experiments.drivers.k8s_common import K8sEngineDriver

_HERE = Path(__file__).resolve().parent
_TEMPLATE_PATH = _HERE.parent / "eks" / "manifests" / "engines" / "sglang.yaml"


class SGLangDriver(K8sEngineDriver):
    name = "sglang"
    template_path = _TEMPLATE_PATH
    health_path = "/health_generate"
    service_port = 30000
