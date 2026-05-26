"""Multi-turn workload with growing conversation context.

Unlike ``ChatbotWorkload``, which merely *labels* 30% of arrivals as
follow-ups, this workload models real sessions where turn N's prompt
carries the full transcript of turns 1..N-1. That makes prompt length —
and therefore KV-cache occupancy — grow strictly within a session, which
is exactly the regime where engine prefix-caching / KV-eviction
strategies diverge.

Sessions:
  * At any time, ``concurrent_sessions`` sessions are active (default 32).
    When one ends, a new one is spawned so the population stays constant.
  * Each session has a length drawn from a geometric distribution with
    mean ``mean_session_turns`` (default 6), capped at ``max_session_turns``.
  * Within a session, user turns are Poisson-spaced. Across sessions,
    arrivals are merged on the absolute timeline.

Prompt growth:
  * Turn 1: a single user message (log-normal, median 80 tokens).
  * Turn N: prior user messages + synthetic prior assistant messages +
    new user message. The workload does not see the engine's actual
    responses (the generator interface is closed-loop-free), so the
    assistant turns are *synthetic*: a fixed-length placeholder drawn
    from another log-normal (median 150 tokens). This is documented in
    the test (``test_multi_turn.py``) so it's not a hidden assumption.

Conversation IDs:
  * Each session has a real ``conversation_id`` set on every Request, so
    KV-aware routers and prefix-caching engines can leverage it.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import numpy as np

from experiments.workloads.base import Request, WorkloadGenerator


@dataclass(frozen=True)
class MultiTurnParams:
    """Per-turn medians anchored to the Azure LLM Inference Trace
    (Splitwise) and ShareGPT conversation distributions: typical user
    turn ~200 tokens, typical assistant response ~300 tokens, mean
    session length ~6 turns.

    With defaults below, the cumulative prompt at turn 6 is
        200 + 5 × (200 + 300) ≈ 2 700 tokens
    which lines up with the calc's "Chatbot · mid-session" preset
    (2 000 / 200) and with Azure's mid-session conversation lengths.
    """

    rate_rps: float = 8.0
    concurrent_sessions: int = 32
    mean_session_turns: float = 6.0
    max_session_turns: int = 20
    user_turn_median_tokens: float = 200.0       # was 80; Azure conv. per-turn median
    user_turn_sigma: float = 0.6
    assistant_turn_median_tokens: float = 300.0  # was 150; reflects real GPT-4/Claude reply lengths
    assistant_turn_sigma: float = 0.6
    output_median_tokens: float = 200.0          # was 150; chatbot-aligned
    output_sigma: float = 0.5


@dataclass
class _Session:
    """Mutable per-session state. Internal to MultiTurnWorkload."""

    conversation_id: str
    turns_remaining: int
    cumulative_tokens: int = 0
    turn_index: int = 0  # 0 before the first turn fires
    pending_user_messages: list[str] = field(default_factory=list)


class MultiTurnWorkload(WorkloadGenerator):
    name = "multi_turn"

    def __init__(self, seed: int = 1, params: MultiTurnParams | None = None) -> None:
        self._seed = seed
        self._params = params or MultiTurnParams()
        self._rng = np.random.default_rng(seed)

    async def requests(self, duration_s: float) -> AsyncIterator[Request]:
        p = self._params
        rng = self._rng

        # Per-session arrival rate so the aggregate matches rate_rps.
        per_session_rate = p.rate_rps / max(1, p.concurrent_sessions)

        # Heap of (next_arrival_t, session_index). Lets us merge per-session
        # streams in O(log K) per event.
        sessions: list[_Session] = []
        heap: list[tuple[float, int, int]] = []  # (t, session_idx, tiebreak)
        tiebreak = 0
        next_session_id = 0

        def _spawn_session(start_t: float) -> int:
            nonlocal next_session_id, tiebreak
            length = min(
                p.max_session_turns,
                max(1, int(rng.geometric(p=1.0 / p.mean_session_turns))),
            )
            sess = _Session(
                conversation_id=f"conv-{self._seed}-{next_session_id}",
                turns_remaining=length,
            )
            next_session_id += 1
            sessions.append(sess)
            idx = len(sessions) - 1
            first_t = start_t + float(rng.exponential(scale=1.0 / per_session_rate))
            tiebreak += 1
            heapq.heappush(heap, (first_t, idx, tiebreak))
            return idx

        # Bootstrap the population.
        for _ in range(p.concurrent_sessions):
            _spawn_session(start_t=0.0)

        while heap:
            t, idx, _tb = heapq.heappop(heap)
            if t >= duration_s:
                # No more arrivals will be earlier than this — finished.
                # (heap is min-ordered, so any remaining popped events also
                # land >= duration_s; but they might be from sessions we
                # haven't iterated yet, so just keep draining the heap.)
                # Actually break — heap is sorted, everything else is later.
                return
            sess = sessions[idx]
            if sess.turns_remaining <= 0:
                # Session expired between scheduling and firing; spawn a
                # replacement and continue.
                new_idx = _spawn_session(start_t=t)
                _ = new_idx
                continue

            # Draw the user turn size and grow the cumulative context.
            user_tokens = max(
                1,
                int(
                    math.exp(
                        rng.normal(
                            math.log(p.user_turn_median_tokens), p.user_turn_sigma
                        )
                    )
                ),
            )
            # On turns >= 2, prepend the synthetic prior assistant turn.
            if sess.turn_index >= 1:
                assistant_tokens = max(
                    1,
                    int(
                        math.exp(
                            rng.normal(
                                math.log(p.assistant_turn_median_tokens),
                                p.assistant_turn_sigma,
                            )
                        )
                    ),
                )
                sess.cumulative_tokens += assistant_tokens

            sess.cumulative_tokens += user_tokens
            sess.turn_index += 1
            sess.turns_remaining -= 1

            output_tokens = max(
                1,
                int(
                    math.exp(
                        rng.normal(math.log(p.output_median_tokens), p.output_sigma)
                    )
                ),
            )
            label = "multi_turn.turn1" if sess.turn_index == 1 else "multi_turn.turnN"

            yield Request(
                request_id=f"mt-{self._seed}-{sess.conversation_id}-{sess.turn_index}",
                prompt=_synthesize_transcript_prompt(sess.cumulative_tokens, sess.turn_index),
                prompt_tokens=sess.cumulative_tokens,
                max_new_tokens=output_tokens,
                arrival_offset_s=t,
                label=label,
                conversation_id=sess.conversation_id,
            )

            # Schedule the next turn of this session, or retire it.
            if sess.turns_remaining > 0:
                next_t = t + float(rng.exponential(scale=1.0 / per_session_rate))
                tiebreak += 1
                heapq.heappush(heap, (next_t, idx, tiebreak))
            else:
                # Replace this session with a fresh one so the population
                # stays at concurrent_sessions.
                _spawn_session(start_t=t)

        # Heap drained — no more events possible.
        return


def _synthesize_transcript_prompt(total_tokens: int, turn_index: int) -> str:
    """Build a prompt whose size approximates the cumulative transcript.

    The exact text is irrelevant — prompt_tokens on the Request is the
    authoritative count for downstream analysis. We just need *some*
    text of roughly the right length so the engine has work to do.
    """
    # 4 chars/token is a coarse proxy.
    payload = "user_assistant_turn " * max(1, total_tokens // 4)
    return f"[turn={turn_index}] {payload}"
