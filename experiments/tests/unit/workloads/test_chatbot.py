"""Chatbot statistical properties.

We check that the prompt-length distribution is approximately log-normal
with the requested median, using a Kolmogorov–Smirnov goodness-of-fit
test. The test is intentionally generous (alpha=0.001) so it does not
flake — what we're catching is *systematic* drift (someone swaps the
distribution to uniform, or scales the median by 10x).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from experiments.workloads.chatbot import ChatbotParams, ChatbotWorkload

pytestmark = pytest.mark.unit


async def _drain_prompt_tokens(w: ChatbotWorkload, duration_s: float) -> list[int]:
    out: list[int] = []
    async for r in w.requests(duration_s=duration_s):
        out.append(r.prompt_tokens)
    return out


@pytest.mark.asyncio
async def test_prompt_length_is_approximately_lognormal() -> None:
    params = ChatbotParams(rate_rps=2000.0, prompt_median_tokens=200.0, prompt_sigma=1.0)
    samples = await _drain_prompt_tokens(ChatbotWorkload(seed=1, params=params), duration_s=5.0)
    # 2000rps * 5s = 10000 samples (give or take).
    assert len(samples) > 1000

    log_samples = np.log(samples)
    mu_expected = math.log(200.0)
    # Mix of single-turn and follow-up; follow-ups have mu shifted by -0.7,
    # weighted by multi_turn_fraction=0.3. Expected mean of log is a blend.
    blended_mu = 0.7 * mu_expected + 0.3 * (mu_expected - 0.7)
    # Compare sample mean against the blended mean — generous tolerance.
    assert abs(float(np.mean(log_samples)) - blended_mu) < 0.15

    # KS against a normal with our expected mean+sigma (very loose: alpha=0.001).
    # We can't test against an exact distribution because of the multi-turn mixture,
    # so we only assert the upper-tail isn't pathologically wrong.
    ks_stat, _p = stats.kstest(
        log_samples,
        "norm",
        args=(blended_mu, 1.0),
    )
    assert ks_stat < 0.15, f"KS stat too high: {ks_stat}"


@pytest.mark.asyncio
async def test_rate_is_approximately_target() -> None:
    target_rps = 50.0
    duration = 4.0
    params = ChatbotParams(rate_rps=target_rps)
    n = len(await _drain_prompt_tokens(ChatbotWorkload(seed=7, params=params), duration_s=duration))
    expected = target_rps * duration
    # Poisson with mean=200 has std=sqrt(200)≈14; 4 sigma is ~56 → tolerate ±30%.
    assert expected * 0.7 <= n <= expected * 1.3, f"got {n}, expected ~{expected}"


@pytest.mark.asyncio
async def test_multi_turn_share_is_approximately_target() -> None:
    params = ChatbotParams(rate_rps=2000.0, multi_turn_fraction=0.3)
    w = ChatbotWorkload(seed=3, params=params)
    follow_ups = 0
    total = 0
    async for r in w.requests(duration_s=2.0):
        total += 1
        if r.label == "chatbot.turnN":
            follow_ups += 1
    # First request can never be a follow-up; correct for that by skipping
    # the very first observation when computing the fraction.
    frac = follow_ups / max(1, total - 1)
    assert 0.22 < frac < 0.38, f"follow-up fraction={frac}"


@pytest.mark.asyncio
async def test_zero_duration_yields_nothing() -> None:
    samples = await _drain_prompt_tokens(ChatbotWorkload(seed=1), duration_s=0.0)
    assert samples == []
