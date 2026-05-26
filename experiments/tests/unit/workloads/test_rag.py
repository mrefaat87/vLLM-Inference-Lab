"""RAG workload: distribution + anti-cache invariants."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from experiments.workloads.rag import RagParams, RagWorkload

pytestmark = pytest.mark.unit


async def _drain(w: RagWorkload, duration_s: float) -> list[tuple[int, str]]:
    return [(r.prompt_tokens, r.prompt) async for r in w.requests(duration_s=duration_s)]


@pytest.mark.asyncio
async def test_prompt_length_is_approximately_lognormal() -> None:
    params = RagParams(rate_rps=500.0, prompt_median_tokens=8000.0, prompt_sigma=0.5)
    samples = await _drain(RagWorkload(seed=1, params=params), duration_s=4.0)
    # 500 rps * 4 s = ~2000 samples.
    assert len(samples) > 500
    log_lens = np.log([n for n, _ in samples])
    assert abs(float(np.mean(log_lens)) - math.log(8000.0)) < 0.1
    # KS test against the declared log-normal — very loose, alpha=0.001.
    ks_stat, _ = stats.kstest(log_lens, "norm", args=(math.log(8000.0), 0.5))
    assert ks_stat < 0.1, f"prompt-length KS stat too high: {ks_stat}"


@pytest.mark.asyncio
async def test_consecutive_prompts_differ_in_prefix() -> None:
    """The defining property of RAG: prefix-cache hit rate ≈ 0.

    Anti-cache invariant: the leading ``head_chars`` of any two consecutive
    requests must differ. We assert it across 100 consecutive pairs.
    """
    params = RagParams(rate_rps=100.0, chunk_pool_size=100_000)
    pairs = await _drain(RagWorkload(seed=1, params=params), duration_s=2.0)
    assert len(pairs) >= 100
    head_chars = 64
    different = 0
    for (_, a), (_, b) in zip(pairs[:100], pairs[1:101], strict=False):
        if a[:head_chars] != b[:head_chars]:
            different += 1
    # Allow a single coincidental collision against ~astronomical odds, but
    # anything more than that means the chunk pool is too small or the
    # synthesizer leaks shared boilerplate at the head.
    assert different >= 99, f"only {different}/100 consecutive heads differed"


@pytest.mark.asyncio
async def test_output_length_smaller_than_prompt() -> None:
    """RAG outputs are short answers; they must be smaller than prompts on average."""
    params = RagParams(rate_rps=100.0)
    w = RagWorkload(seed=3, params=params)
    out_tokens = []
    in_tokens = []
    async for r in w.requests(duration_s=2.0):
        in_tokens.append(r.prompt_tokens)
        out_tokens.append(r.max_new_tokens)
    assert np.mean(in_tokens) > 10 * np.mean(out_tokens)


@pytest.mark.asyncio
async def test_zero_duration_yields_nothing() -> None:
    out = await _drain(RagWorkload(seed=1), duration_s=0.0)
    assert out == []
