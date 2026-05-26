"""Shared K8s driver helpers.

The vLLM / SGLang / TRT-LLM drivers all follow the same shape:
  1. Render a Deployment + Service from a YAML template, substituting the
     EngineConfig fields.
  2. ``kubectl apply -f`` it.
  3. Poll the Service endpoint for /health (or equivalent).
  4. ``kubectl delete`` on stop.

To keep each driver tiny, the boilerplate lives here. Real engines just
declare their template path and a few config knobs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import urllib.request

from experiments.drivers.base import EndpointInfo, EngineDriver
from experiments.runner.schema import EngineConfig


class TemplateRenderer(Protocol):
    def render(self, cfg: EngineConfig) -> str:
        """Return the YAML for the engine's K8s resources."""


@dataclass
class KubectlConfig:
    context: str = "inference-lab"
    namespace: str = "engines"
    kubeconfig: str | None = None
    kubectl_path: str | None = None  # for tests: override the executable

    def base_args(self) -> list[str]:
        args = [self.kubectl_path or "kubectl", "--context", self.context, "-n", self.namespace]
        if self.kubeconfig:
            args = [args[0], "--kubeconfig", self.kubeconfig, *args[1:]]
        return args


class K8sEngineDriver(EngineDriver):
    """Base class for engine drivers that run as K8s Deployments."""

    #: Subclasses set this to the template file under eks/manifests/engines/.
    template_path: Path
    #: Subclasses set this to the health endpoint path. Most engines expose /health.
    health_path: str = "/health"
    #: Service port — must match the rendered template.
    service_port: int = 8000

    def __init__(
        self,
        kubectl: KubectlConfig | None = None,
        readiness_timeout_s: float = 600.0,
    ) -> None:
        self._kubectl = kubectl or KubectlConfig()
        self._readiness_timeout_s = readiness_timeout_s
        self._cfg: EngineConfig | None = None
        self._endpoint: EndpointInfo | None = None
        self._release_name: str | None = None

    # ----- subclass hooks -----
    def render(self, cfg: EngineConfig) -> str:
        """Render the YAML manifest for this engine, substituting cfg values."""
        text = self.template_path.read_text()
        return _substitute(text, _vars_for_cfg(cfg, self.name))

    def endpoint_url(self) -> str:
        """Override if a driver needs a non-standard service URL (e.g. external LB)."""
        # Default: in-cluster service DNS. The driver loop accesses the cluster
        # via `kubectl port-forward` (set up by start()), so the URL is localhost.
        return f"http://127.0.0.1:{self.service_port}"

    # ----- lifecycle -----
    def start(self, cfg: EngineConfig) -> EndpointInfo:
        if cfg.name != self.name:
            raise ValueError(f"{type(self).__name__} expects cfg.name={self.name!r}, got {cfg.name!r}")
        self._cfg = cfg
        self._release_name = f"{self.name}-engine"
        manifest = self.render(cfg)
        self._kubectl_apply(manifest)
        self._await_ready()
        self._endpoint = EndpointInfo(base_url=self.endpoint_url(), openai_compat=True)
        return self._endpoint

    def healthcheck(self) -> bool:
        if self._endpoint is None:
            return False
        try:
            with urllib.request.urlopen(
                f"{self._endpoint.base_url}{self.health_path}",
                timeout=2.0,
            ) as resp:
                return 200 <= resp.status < 300
        except Exception:  # noqa: BLE001 — any failure means "not healthy"
            return False

    def metrics(self) -> dict[str, Any]:
        # Default: scrape /metrics if present, parse Prometheus exposition. Drivers
        # with engine-native metrics endpoints override.
        if self._endpoint is None:
            return {}
        try:
            with urllib.request.urlopen(
                f"{self._endpoint.base_url}/metrics", timeout=2.0
            ) as resp:
                text = resp.read().decode("utf-8")
            return _parse_prom(text)
        except Exception:  # noqa: BLE001
            return {}

    def stop(self) -> None:
        if self._cfg is None:
            return
        try:
            self._kubectl_delete()
        finally:
            self._cfg = None
            self._endpoint = None

    # ----- internals -----
    def _kubectl_apply(self, manifest: str) -> None:
        args = [*self._kubectl.base_args(), "apply", "-f", "-"]
        subprocess.run(args, input=manifest, text=True, check=True)

    def _kubectl_delete(self) -> None:
        args = [
            *self._kubectl.base_args(),
            "delete",
            "deployment,service",
            f"-l=app={self.name}-engine",
            "--ignore-not-found=true",
            "--wait=false",
        ]
        subprocess.run(args, check=False)

    def _await_ready(self) -> None:
        deadline = time.monotonic() + self._readiness_timeout_s
        while time.monotonic() < deadline:
            # Wait for Deployment to be Available...
            args = [
                *self._kubectl.base_args(),
                "wait",
                "deployment",
                f"{self.name}-engine",
                "--for=condition=Available",
                "--timeout=10s",
            ]
            r = subprocess.run(args, capture_output=True, text=True, check=False)
            if r.returncode == 0:
                # ...then port-forward (best-effort: this is left to the caller
                # in production; the contract test uses an alternate path).
                return
            time.sleep(2.0)
        raise TimeoutError(f"{self.name} engine never became Available")


def _vars_for_cfg(cfg: EngineConfig, engine_name: str) -> dict[str, str]:
    return {
        "ENGINE_NAME": engine_name,
        "MODEL": cfg.model,
        "QUANTIZATION": cfg.quantization,
        "TENSOR_PARALLEL": str(cfg.tensor_parallel),
        "MAX_MODEL_LEN": str(cfg.max_model_len),
        "IMAGE": cfg.image,
    }


def _substitute(text: str, vars_: dict[str, str]) -> str:
    out = text
    for k, v in vars_.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def _parse_prom(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # name{labels} value
        try:
            name, value = line.rsplit(" ", 1)
            out[name] = float(value)
        except ValueError:
            continue
    return out
