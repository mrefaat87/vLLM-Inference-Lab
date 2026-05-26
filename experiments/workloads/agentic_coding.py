"""Agentic-coding workload.

Profile (matches Claude Code / Cursor agentic traffic):
  - Low effective RPS (default 2/sec) because each session sustains long
    multi-step reasoning.
  - Very large prompts: 2k–8k tokens (system + tool definitions + history).
    Modeled as log-normal with median 4000, sigma 0.6.
  - **Bimodal** output lengths: tool-call decisions are ~50 tokens; code
    generations are ~800 tokens. Mixture coefficient 0.6 / 0.4 by default.
  - Bursty arrivals: agents pause for "thinking" time. Inter-arrival is
    drawn from a mixture of a short-burst exponential and a long-pause
    exponential.

The mixture structure is what makes this workload interesting for engine
comparison — prefix-caching engines see big wins on the long shared
system prompts, but only if the arrival pattern keeps the cache hot.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np

from experiments.workloads.base import Request, WorkloadGenerator


@dataclass(frozen=True)
class AgenticParams:
    """Default medians anchored to published agentic-coding telemetry
    (Cursor, Claude Code, Devin per-step traces — Vantage benchmark
    article + vendor engineering posts, 2024-2025). Per-step prompts
    typically land 20 000–80 000 tokens (system + tools + history +
    file context). Outputs are strongly bimodal: short tool-call
    decisions (~50 tok) vs long code generations (~1 500 tok). The
    calc exposes the two output modes as separate presets
    ("agentic_step" 20 000 / 500 and "agentic_edit" 40 000 / 1 500);
    here we model the mixture with a single generator whose median
    sits between the two.
    """

    rate_rps: float = 2.0
    prompt_median_tokens: float = 25000.0     # was 4 000; Cursor/Claude Code per-step typical
    prompt_sigma: float = 0.6
    short_output_mean_tokens: float = 50.0    # tool-call branch unchanged
    long_output_mean_tokens: float = 1500.0   # was 800; Claude Code code-edit median
    long_output_fraction: float = 0.4
    burst_mode_fraction: float = 0.65  # fraction of arrivals coming from "burst" mode
    burst_rate_multiplier: float = 6.0  # bursts arrive 6x as fast as the long-pause mode


class AgenticCodingWorkload(WorkloadGenerator):
    name = "agentic_coding"

    def __init__(self, seed: int = 1, params: AgenticParams | None = None) -> None:
        self._seed = seed
        self._params = params or AgenticParams()
        self._rng = np.random.default_rng(seed)

    async def requests(self, duration_s: float) -> AsyncIterator[Request]:
        p = self._params
        rng = self._rng
        # Decompose target_rps into two Poisson processes, then thin/merge.
        # lambda_burst + lambda_slow = rate_rps
        # lambda_burst / (lambda_burst + lambda_slow) = burst_mode_fraction
        # => lambda_burst = rate_rps * burst_mode_fraction
        lam_burst = p.rate_rps * p.burst_mode_fraction
        lam_slow = p.rate_rps - lam_burst
        # We mimic burstiness by amplifying lam_burst into a higher-rate
        # process that fires for short windows, with idle gaps in between.
        lam_burst_eff = lam_burst * p.burst_rate_multiplier
        lam_pause = lam_slow  # used during idle windows

        mu_prompt = math.log(p.prompt_median_tokens)
        next_id = 0
        t = 0.0
        # Bernoulli switch: are we in burst or pause mode at t=0?
        in_burst = bool(rng.random() < 0.5)
        mode_remaining = float(rng.exponential(scale=2.0))  # avg 2s per mode

        while True:
            lam = lam_burst_eff if in_burst else lam_pause
            if lam <= 0:
                # Degenerate: just exit if no rate.
                return
            delta = float(rng.exponential(scale=1.0 / lam))
            t += delta
            mode_remaining -= delta
            if mode_remaining <= 0:
                in_burst = not in_burst
                mode_remaining = float(rng.exponential(scale=2.0))
            if t >= duration_s:
                return

            next_id += 1
            prompt_tokens = max(64, int(math.exp(rng.normal(mu_prompt, p.prompt_sigma))))
            # Bimodal output.
            is_long = float(rng.random()) < p.long_output_fraction
            if is_long:
                out_tokens = max(1, int(rng.exponential(scale=p.long_output_mean_tokens)))
                label = "agentic.code"
            else:
                out_tokens = max(1, int(rng.exponential(scale=p.short_output_mean_tokens)))
                label = "agentic.tool"
            yield Request(
                request_id=f"agent-{self._seed}-{next_id}",
                prompt=_synthetic_long_prompt(prompt_tokens, rng),
                prompt_tokens=prompt_tokens,
                max_new_tokens=out_tokens,
                arrival_offset_s=t,
                label=label,
            )


def _synthetic_long_prompt(n_tokens: int, rng: np.random.Generator) -> str:
    """Long shared-prefix-friendly prompt followed by a salt suffix.

    Engines that prefix-cache (SGLang RadixAttention, vLLM prefix caching)
    should see high reuse on the shared head; the salt prevents trivially
    identical requests so we don't accidentally measure 100% cache hits.
    """
    head = (
        "You are a senior software engineer. Use the tools available to read "
        "files, edit them, and run tests. Always think step by step. "
    )
    body = " ".join(["context"] * max(0, n_tokens - 32))
    salt = int(rng.integers(0, 1_000_000_000))
    return f"{head}{body} salt={salt}"
