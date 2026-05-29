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

import json
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import urllib.request

from experiments.drivers.base import EndpointInfo, EngineDriver
from experiments.drivers.instance_specs import InstanceSpec, parse_instance
from experiments.runner.schema import EngineConfig

logger = logging.getLogger(__name__)


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
        readiness_timeout_s: float = 1200.0,
        port_forward: bool = True,
    ) -> None:
        self._kubectl = kubectl or KubectlConfig()
        self._readiness_timeout_s = readiness_timeout_s
        # Caller can disable port-forward for in-cluster runs (where the
        # driver pod already has Service DNS reachability).
        self._port_forward_enabled = port_forward
        self._cfg: EngineConfig | None = None
        self._endpoint: EndpointInfo | None = None
        self._release_name: str | None = None
        self._port_forward_proc: subprocess.Popen[bytes] | None = None

    # ----- subclass hooks -----
    def render(self, cfg: EngineConfig) -> str:
        """Render the YAML manifest for this engine, substituting cfg values."""
        text = self.template_path.read_text()
        return _substitute(text, _vars_for_cfg(cfg, self.name))

    def endpoint_url(self) -> str:
        """Override if a driver needs a non-standard service URL (e.g. external LB)."""
        # Default: localhost via the port-forward set up in start(). When
        # port_forward is disabled the caller is responsible for plumbing
        # in-cluster Service DNS via a subclass override.
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
        self._verify_realized_instance(cfg)
        if self._port_forward_enabled:
            self._start_port_forward()
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
            self._stop_port_forward()
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
                return
            time.sleep(2.0)
        raise TimeoutError(f"{self.name} engine never became Available")

    def _verify_realized_instance(self, cfg: EngineConfig) -> None:
        """Read the Pod's node and warn loudly if Karpenter parked it on a
        different instance type than the user asked for.

        Without this check the engine starts on whatever Karpenter
        happened to scale up (often a much larger/more expensive type),
        and the user only notices on the AWS bill.
        """
        if not cfg.instance:
            # No requested instance to compare against — skip silently.
            return
        node_name = self._pod_node_name()
        if not node_name:
            logger.warning(
                "could not read pod's node name; skipping realized-instance check",
            )
            return
        realized = self._node_instance_type(node_name)
        if not realized:
            logger.warning(
                "could not read instance-type label on node %s; "
                "skipping realized-instance check",
                node_name,
            )
            return
        if realized != cfg.instance:
            logger.warning(
                "REALIZED INSTANCE MISMATCH: requested %s but scheduled on %s "
                "(node=%s). Check Karpenter NodePool allowed instance-family / "
                "instance-size selectors.",
                cfg.instance,
                realized,
                node_name,
            )

    def _pod_node_name(self) -> str | None:
        args = [
            *self._kubectl.base_args(),
            "get", "pod",
            "-l", f"app={self.name}-engine",
            "-o", "jsonpath={.items[0].spec.nodeName}",
        ]
        r = subprocess.run(args, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            return None
        return r.stdout.strip() or None

    def _node_instance_type(self, node_name: str) -> str | None:
        args = [
            *self._kubectl.base_args(),
            "get", "node", node_name,
            "-o",
            # node.kubernetes.io/instance-type is the canonical label;
            # Karpenter and the AWS cloud-provider both populate it.
            "jsonpath={.metadata.labels.node\\.kubernetes\\.io/instance-type}",
        ]
        r = subprocess.run(args, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            return None
        return r.stdout.strip() or None

    def _start_port_forward(self) -> None:
        """Spawn ``kubectl port-forward`` to expose the engine on localhost.

        Held on the driver so ``stop()`` can terminate it cleanly. The
        load driver hits ``endpoint_url()`` which resolves to localhost.
        """
        args = [
            *self._kubectl.base_args(),
            "port-forward",
            f"svc/{self.name}-engine",
            f"{self.service_port}:{self.service_port}",
        ]
        # Capture stderr so a failed forward shows up in the run log;
        # stdout is just "Forwarding from …" noise.
        self._port_forward_proc = subprocess.Popen(  # noqa: S603 — inputs are driver-internal
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        # Give the forward a beat to bind the local port before the load
        # driver fires its first request. 2s is conservative for an
        # already-Ready Service; a slower path will surface as a healthcheck
        # retry loop, not a hard failure here.
        time.sleep(2.0)

    def _stop_port_forward(self) -> None:
        proc = self._port_forward_proc
        if proc is None:
            return
        self._port_forward_proc = None
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        except Exception:  # noqa: BLE001 — cleanup must not raise
            logger.warning("port-forward cleanup raised; ignoring")


def _vars_for_cfg(cfg: EngineConfig, engine_name: str) -> dict[str, str]:
    """Build the template-substitution table for an EngineConfig.

    Karpenter scheduling vars (INSTANCE_FAMILY / INSTANCE_SIZE) and
    resource-sizing vars are derived from ``cfg.instance``. When the
    caller didn't pass an instance (e.g. mock driver in tests), the
    Karpenter selectors fall through to empty strings — the template
    must guard against that with ``existsIn`` / a default block.
    """
    instance = cfg.instance or ""
    spec: InstanceSpec | None = None
    if instance:
        try:
            spec = parse_instance(instance)
        except KeyError:
            logger.warning(
                "instance %r not in instance_specs.py; falling back to "
                "conservative defaults (8 GiB memory request)",
                instance,
            )
    # GPU count: prefer the explicit n_gpu, fall back to the tp setting.
    n_gpu = cfg.n_gpu or cfg.tensor_parallel
    # Memory request derives from the instance row; default 8 GiB is
    # enough for a 7B FP16 model but not larger — surface a warning above
    # when we have to fall back.
    mem_req_gib = spec.pod_memory_request_gib if spec else 8
    mem_limit_gib = spec.allocatable_memory_gib if spec else 12
    # CPU request: scale to mem_gib//8 (rough proxy for typical core
    # count). Floor at 3 — Karpenter reserves ~210m for DaemonSets, so
    # requesting 4 on a 4-vCPU box (g4dn.xlarge) is unschedulable.
    # Floor of 3 leaves ~790m headroom on the smallest GPU instance.
    cpu_req = "3" if not spec else str(max(3, spec.allocatable_memory_gib // 8))
    return {
        "ENGINE_NAME": engine_name,
        "MODEL": cfg.model,
        "QUANTIZATION": cfg.quantization,
        "TENSOR_PARALLEL": str(cfg.tensor_parallel),
        "N_GPU": str(n_gpu),
        "MAX_MODEL_LEN": str(cfg.max_model_len),
        "MAX_NUM_SEQS": str(cfg.max_num_seqs),
        "IMAGE": cfg.image,
        "INSTANCE_FAMILY": spec.family if spec else "",
        "INSTANCE_SIZE": spec.size if spec else "",
        # Lowercased GPU name for ``karpenter.k8s.aws/instance-gpu-name``
        # (the label Karpenter exports — e.g. ``a10g``, ``t4``, ``h100``).
        "GPU_NAME": (cfg.gpu or "").lower(),
        "MEMORY_REQUEST": f"{mem_req_gib}Gi",
        "MEMORY_LIMIT": f"{mem_limit_gib}Gi",
        "CPU_REQUEST": cpu_req,
        # AWS account is needed by the weights init-container's s3 sync.
        # Look it up once at render time via STS so the same template
        # works across accounts without hardcoding.
        "AWS_ACCOUNT": _aws_account_id(),
    }


def _aws_account_id() -> str:
    """Return the caller's AWS account ID, or empty string if STS is unreachable.

    Cached for the process lifetime — STS calls are slow and idempotent
    within a session.
    """
    if not hasattr(_aws_account_id, "_cached"):
        _aws_account_id._cached = _query_aws_account_id()  # type: ignore[attr-defined]
    return _aws_account_id._cached  # type: ignore[attr-defined]


def _query_aws_account_id() -> str:
    if not shutil.which("aws"):
        return ""
    try:
        r = subprocess.run(
            ["aws", "sts", "get-caller-identity", "--output", "json"],
            capture_output=True, text=True, timeout=10.0, check=False,
        )
        if r.returncode != 0:
            return ""
        return str(json.loads(r.stdout).get("Account", ""))
    except Exception:  # noqa: BLE001
        return ""


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
