"""Engine driver interface.

Every engine (vLLM, SGLang, TRT-LLM, mock) implements this same shape so
the sweep runner is engine-agnostic. The swap policy (stop/start vs
all-up routing) is left to the concrete implementation — the interface
only requires that ``start`` returns an endpoint and ``stop`` is
idempotent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from experiments.runner.schema import EngineConfig


@dataclass(frozen=True)
class EndpointInfo:
    """Where the engine is reachable, and what dialect it speaks."""

    base_url: str            # e.g. "http://1.2.3.4:8000"
    openai_compat: bool      # True for vLLM/SGLang; True for TRT-LLM via its OAI shim
    extra: dict[str, Any] | None = None


class EngineDriver(ABC):
    """Abstract base for all engine drivers.

    Lifecycle (one shot per run):

        cfg -> driver.start(cfg) -> EndpointInfo
        driver.healthcheck() -> bool          (poll until ready)
        ...                                   (sweep runner sends traffic)
        driver.metrics() -> dict              (snapshot)
        driver.stop()                         (idempotent)

    Subclasses must set ``name`` to the canonical engine identifier.
    """

    #: Canonical engine identifier; matches EngineConfig.name.
    name: str = ""

    @abstractmethod
    def start(self, cfg: EngineConfig) -> EndpointInfo:
        """Bring the engine to a ready state and return its endpoint.

        Implementations must be idempotent: calling start on an already
        running engine should either return the existing endpoint or
        rotate cleanly.
        """

    @abstractmethod
    def healthcheck(self) -> bool:
        """Return True iff the endpoint is ready to accept requests."""

    @abstractmethod
    def metrics(self) -> dict[str, Any]:
        """Snapshot engine-native metrics (KV usage, queue depth, ...)."""

    @abstractmethod
    def stop(self) -> None:
        """Shut the engine down. Must be safe to call multiple times."""

    # ----- context manager sugar so callers can `with driver: ...` -----
    def __enter__(self) -> EngineDriver:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
