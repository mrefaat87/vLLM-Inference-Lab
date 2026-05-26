"""Chatbot workload.

Profile (matches public chat-traffic shape):
  - Poisson arrivals at rate ``rate_rps``.
  - Prompt length: log-normal, default median ~200 tokens (mu=log(200), sigma=1.0).
    Captures the heavy tail of long contexts without dwarfing typical-len traffic.
  - Output length: log-normal, default median ~150 tokens (mu=log(150), sigma=0.7).
  - 30% of arrivals are turn-2+ of an existing conversation. Their prompt is the
    previous turn's prompt + a short follow-up, so KV-cache prefix reuse is realistic.

All randomness is driven by a single ``numpy.random.Generator`` seeded from
the constructor so the same seed reproduces the stream byte-for-byte.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np

from experiments.workloads.base import Request, WorkloadGenerator


@dataclass(frozen=True)
class ChatbotParams:
    """Default medians anchored to the Azure LLM Inference Trace (Splitwise,
    ISCA 2024): conversation-service median prompt ≈ 1 020 tokens, median
    completion ≈ 130 tokens. We use a slightly smaller prompt (500) because
    the calc's "Chatbot · turn 1" preset assumes a fresh conversation
    before transcript growth; multi-turn growth is modeled by the
    follow-up branch.
    """

    rate_rps: float = 8.0
    prompt_median_tokens: float = 500.0       # was 200; Azure conv. median ≈ 1k, turn-1 anchor 500
    prompt_sigma: float = 1.0
    output_median_tokens: float = 200.0       # was 150; ShareGPT/Azure completion median
    output_sigma: float = 0.7
    multi_turn_fraction: float = 0.30
    max_conversations: int = 256


class ChatbotWorkload(WorkloadGenerator):
    name = "chatbot"

    def __init__(self, seed: int = 1, params: ChatbotParams | None = None) -> None:
        self._seed = seed
        self._params = params or ChatbotParams()
        self._rng = np.random.default_rng(seed)
        # Bounded ring buffer of recent conversation ids for turn-2+ selection.
        self._conversations: deque[str] = deque(maxlen=self._params.max_conversations)

    async def requests(self, duration_s: float) -> AsyncIterator[Request]:
        p = self._params
        rng = self._rng
        t = 0.0
        next_id = 0
        mu_prompt = math.log(p.prompt_median_tokens)
        mu_output = math.log(p.output_median_tokens)
        while True:
            # Poisson inter-arrival = exponential(1/lambda)
            delta = float(rng.exponential(scale=1.0 / p.rate_rps))
            t += delta
            if t >= duration_s:
                return
            next_id += 1
            req_id = f"chat-{self._seed}-{next_id}"

            # Decide whether this is a follow-up turn.
            is_followup = (
                len(self._conversations) > 0
                and float(rng.random()) < p.multi_turn_fraction
            )
            if is_followup:
                conv_id = self._conversations[
                    int(rng.integers(0, len(self._conversations)))
                ]
                # Follow-up: shorter prompt added to a (notional) growing context.
                prompt_tokens = max(1, int(math.exp(rng.normal(mu_prompt - 0.7, p.prompt_sigma))))
                label = "chatbot.turnN"
            else:
                conv_id = f"conv-{self._seed}-{next_id}"
                self._conversations.append(conv_id)
                prompt_tokens = max(1, int(math.exp(rng.normal(mu_prompt, p.prompt_sigma))))
                label = "chatbot.turn1"

            output_tokens = max(1, int(math.exp(rng.normal(mu_output, p.output_sigma))))
            yield Request(
                request_id=req_id,
                prompt=_synthetic_prompt(prompt_tokens, rng),
                prompt_tokens=prompt_tokens,
                max_new_tokens=output_tokens,
                arrival_offset_s=t,
                label=label,
                conversation_id=conv_id,
            )


def _synthetic_prompt(n_tokens: int, rng: np.random.Generator) -> str:
    """Build a synthetic prompt of roughly n_tokens (whitespace-token proxy).

    We don't tokenize for real — engines do their own tokenization, and the
    prompt_tokens field is the authoritative count for analysis. The prompt
    text is intentionally repetitive so engines that prefix-cache see
    realistic reuse.
    """
    # Choose from a small bank of words for variety with reproducibility.
    words = ["the", "model", "should", "answer", "the", "user", "request", "clearly"]
    if n_tokens <= len(words):
        return " ".join(words[:n_tokens])
    repeats = (n_tokens // len(words)) + 1
    # Inject a salt so different requests still hash differently (no realism gain
    # from token-by-token random choice, but caches need diverse suffixes).
    salt = int(rng.integers(0, 1_000_000))
    return " ".join(words * repeats)[: n_tokens * 5] + f" salt{salt}"
