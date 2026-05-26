"""Workload generator interface.

A WorkloadGenerator yields a *deterministic* sequence of Request objects
when given the same seed. Determinism is non-negotiable — empirical
comparisons across engines hinge on every engine seeing the same traffic.

The arrival timeline is encoded in ``Request.arrival_offset_s`` (seconds
since the start of the run). The driver loop is responsible for sleeping
until the requested time before dispatching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Request:
    """A single LLM request to be issued."""

    request_id: str
    prompt: str
    prompt_tokens: int       # tokenizer-counted; generators estimate; drivers do not re-tokenize
    max_new_tokens: int
    arrival_offset_s: float  # seconds from t0
    label: str               # workload-defined, e.g. "chatbot.turn2"
    conversation_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class WorkloadGenerator(ABC):
    """Abstract base for workload generators.

    Implementations should accept a seed in __init__ and use it for every
    randomness source. The same seed must yield the same stream.
    """

    #: Canonical workload identifier.
    name: str = ""

    @abstractmethod
    async def requests(self, duration_s: float) -> AsyncIterator[Request]:
        """Yield requests until ``duration_s`` of simulated time elapses.

        This is an async generator — implementations should ``yield`` and
        not ``return`` a list, so that very long runs do not balloon
        memory.

        The yielded requests' ``arrival_offset_s`` must be monotonically
        non-decreasing and < duration_s.
        """
