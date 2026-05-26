"""MixWorkload merge correctness."""

from __future__ import annotations

import pytest

from experiments.workloads.agentic_coding import AgenticCodingWorkload, AgenticParams
from experiments.workloads.chatbot import ChatbotParams, ChatbotWorkload
from experiments.workloads.mix import MixWorkload, WeightedChild

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_arrivals_monotonic_after_merge() -> None:
    mix = MixWorkload(
        [
            WeightedChild(ChatbotWorkload(seed=1, params=ChatbotParams(rate_rps=10.0))),
            WeightedChild(
                AgenticCodingWorkload(seed=2, params=AgenticParams(rate_rps=3.0))
            ),
        ]
    )
    prev = -1.0
    n = 0
    async for r in mix.requests(duration_s=4.0):
        assert r.arrival_offset_s >= prev
        prev = r.arrival_offset_s
        n += 1
    assert n > 10


@pytest.mark.asyncio
async def test_combined_rate_is_sum_of_children() -> None:
    rate_a = 20.0
    rate_b = 5.0
    duration = 4.0
    mix = MixWorkload(
        [
            WeightedChild(ChatbotWorkload(seed=1, params=ChatbotParams(rate_rps=rate_a))),
            WeightedChild(
                AgenticCodingWorkload(seed=2, params=AgenticParams(rate_rps=rate_b))
            ),
        ]
    )
    n = 0
    async for _ in mix.requests(duration_s=duration):
        n += 1
    expected = (rate_a + rate_b) * duration
    assert expected * 0.6 <= n <= expected * 1.5  # Poisson sum tolerance


@pytest.mark.asyncio
async def test_labels_indicate_source() -> None:
    mix = MixWorkload(
        [
            WeightedChild(ChatbotWorkload(seed=1, params=ChatbotParams(rate_rps=10.0))),
            WeightedChild(
                AgenticCodingWorkload(seed=2, params=AgenticParams(rate_rps=3.0))
            ),
        ]
    )
    sources: set[str] = set()
    async for r in mix.requests(duration_s=3.0):
        assert r.metadata.get("source") in {"chatbot", "agentic_coding"}
        sources.add(r.metadata["source"])
    # We should see both sources in a 3-second window.
    assert sources == {"chatbot", "agentic_coding"}


def test_empty_children_rejected() -> None:
    with pytest.raises(ValueError):
        MixWorkload([])
