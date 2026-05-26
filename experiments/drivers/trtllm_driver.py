"""TRT-LLM driver.

Two-stage:
  1. **Engine build** (one-shot K8s Job) compiles a TensorRT-LLM engine
     for the (model, quantization, TP) tuple and writes it to the
     S3 weights bucket under ``engines/<model>/<quant>/<tp>/``.
  2. **Serve** (Deployment) is a Triton Inference Server with the
     ``tensorrt_llm`` backend pointing at the built engine. Exposes a
     ``/v2/models/ensemble/generate`` Triton endpoint and an OpenAI-compat
     ``/v1/chat/completions`` endpoint via the trtllm-serve front-end.

If the engine artifact already exists in S3 for the requested config,
the build Job is skipped.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from experiments.drivers.k8s_common import K8sEngineDriver
from experiments.runner.schema import EngineConfig

_HERE = Path(__file__).resolve().parent
_SERVE_TEMPLATE = _HERE.parent / "eks" / "manifests" / "engines" / "trtllm-serve.yaml"
_BUILD_TEMPLATE = _HERE.parent / "eks" / "manifests" / "engines" / "trtllm-build.yaml"


class TRTLLMDriver(K8sEngineDriver):
    name = "trtllm"
    template_path = _SERVE_TEMPLATE
    health_path = "/v2/health/ready"
    service_port = 8000

    def __init__(self, *, builder_image: str | None = None, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self._builder_image = builder_image

    def start(self, cfg: EngineConfig) -> object:
        # Ensure the engine artifact exists before deploying the server.
        self._ensure_engine_built(cfg)
        return super().start(cfg)

    def _ensure_engine_built(self, cfg: EngineConfig) -> None:
        """Submit a one-shot Job if the engine isn't already in S3.

        The Job is named with the config hash so re-runs are idempotent —
        if the artifact already exists, the Job's first step exits 0.
        """
        if not _BUILD_TEMPLATE.exists():
            # Tests can run without the build template if engine is pre-built.
            return
        text = _BUILD_TEMPLATE.read_text()
        text = text.replace("{{MODEL}}", cfg.model)
        text = text.replace("{{QUANTIZATION}}", cfg.quantization)
        text = text.replace("{{TENSOR_PARALLEL}}", str(cfg.tensor_parallel))
        text = text.replace("{{MAX_MODEL_LEN}}", str(cfg.max_model_len))
        text = text.replace("{{BUILDER_IMAGE}}", self._builder_image or cfg.image)
        args = [*self._kubectl.base_args(), "apply", "-f", "-"]
        subprocess.run(args, input=text, text=True, check=True)
        wait_args = [
            *self._kubectl.base_args(),
            "wait",
            "job/trtllm-build",
            "--for=condition=Complete",
            "--timeout=30m",
        ]
        subprocess.run(wait_args, check=True)
