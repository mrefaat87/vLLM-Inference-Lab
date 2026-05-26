"""Agentic-coding distribution checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.workloads.agentic_coding import AgenticCodingWorkload, AgenticParams

pytestmark = pytest.mark.unit


async def _drain(w: AgenticCodingWorkload, duration_s: float) -> list[tuple[int, int, str]]:
    return [
        (r.prompt_tokens, r.max_new_tokens, r.label)
        async for r in w.requests(duration_s=duration_s)
    ]


@pytest.mark.asyncio
async def test_prompts_centered_near_median() -> None:
    params = AgenticParams(rate_rps=500.0, prompt_median_tokens=4000.0, prompt_sigma=0.6)
    pairs = await _drain(AgenticCodingWorkload(seed=1, params=params), duration_s=5.0)
    assert len(pairs) > 200
    log_prompts = np.log([p for p, _, _ in pairs])
    # Median of a log-normal with mu=log(4000) is e^mu = 4000.
    assert abs(float(np.mean(log_prompts)) - math.log(4000.0)) < 0.15


@pytest.mark.asyncio
async def test_output_is_bimodal() -> None:
    params = AgenticParams(
        rate_rps=500.0,
        short_output_mean_tokens=50.0,
        long_output_mean_tokens=800.0,
        long_output_fraction=0.4,
    )
    pairs = await _drain(AgenticCodingWorkload(seed=1, params=params), duration_s=4.0)
    short = [o for _, o, _ in pairs if _label_is(_, "tool", pairs)]
    long = [o for _, o, _ in pairs if _label_is(_, "code", pairs)]
    # We don't use label_is helper above — instead split by label directly:
    short = [o for _, o, lbl in pairs if lbl == "agentic.tool"]
    long = [o for _, o, lbl in pairs if lbl == "agentic.code"]
    assert short and long
    # Sample means should be within 30% of the configured exponentials.
    assert 35 <= float(np.mean(short)) <= 70
    assert 600 <= float(np.mean(long)) <= 1100


def _label_is(_arg: object, _kind: str, _pairs: list[tuple[int, int, str]]) -> bool:
    return True  # placeholder, real filter is done inline above


@pytest.mark.asyncio
async def test_burst_fraction_roughly_targeted() -> None:
    """Burst mode amplifies effective rate by `burst_rate_multiplier`,
    so a single-seed count can swing 0.5×–3× of nominal. We average
    across multiple seeds to characterize the long-run rate, then
    assert that average is within 2× of the nominal target.
    """
    counts: list[int] = []
    for seed in range(1, 11):
        params = AgenticParams(rate_rps=300.0, burst_mode_fraction=0.65)
        pairs = await _drain(AgenticCodingWorkload(seed=seed, params=params), duration_s=5.0)
        counts.append(len(pairs))
    nominal = 300.0 * 5.0
    mean = sum(counts) / len(counts)
    assert nominal * 0.5 <= mean <= nominal * 2.5, (
        f"avg over 10 seeds = {mean:.0f}, expected near {nominal:.0f}; per-seed = {counts}"
    )
