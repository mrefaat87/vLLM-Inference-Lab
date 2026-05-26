"""Multi-turn workload: monotonic growth per session + structural invariants.

Note: this workload uses *synthetic* assistant turn lengths to grow the
context — the WorkloadGenerator interface is open-loop and does not see
the engine's real outputs. The unit test below explicitly asserts the
intended invariant (prompt size grows strictly within a session) without
relying on real engine responses.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from experiments.workloads.base import Request
from experiments.workloads.multi_turn import MultiTurnParams, MultiTurnWorkload

pytestmark = pytest.mark.unit


async def _drain(w: MultiTurnWorkload, duration_s: float) -> list[Request]:
    return [r async for r in w.requests(duration_s=duration_s)]


@pytest.mark.asyncio
async def test_prompt_grows_strictly_within_a_session() -> None:
    params = MultiTurnParams(rate_rps=30.0, concurrent_sessions=8, mean_session_turns=8.0)
    reqs = await _drain(MultiTurnWorkload(seed=1, params=params), duration_s=4.0)
    by_conv: dict[str, list[Request]] = defaultdict(list)
    for r in reqs:
        assert r.conversation_id is not None
        by_conv[r.conversation_id].append(r)
    # At least some sessions should have produced 2+ turns in a 4-second window.
    multi_turn_sessions = [v for v in by_conv.values() if len(v) >= 2]
    assert multi_turn_sessions, "no multi-turn sessions observed; tune the test"
    for turns in multi_turn_sessions:
        for prev, cur in zip(turns, turns[1:], strict=False):
            assert cur.prompt_tokens > prev.prompt_tokens, (
                f"prompt_tokens did not grow in session {cur.conversation_id}: "
                f"{prev.prompt_tokens} -> {cur.prompt_tokens}"
            )


@pytest.mark.asyncio
async def test_session_length_distribution_is_in_range() -> None:
    """Drawn session lengths should hover near mean_session_turns and never exceed the cap."""
    params = MultiTurnParams(
        rate_rps=200.0, concurrent_sessions=4, mean_session_turns=6.0, max_session_turns=20
    )
    reqs = await _drain(MultiTurnWorkload(seed=2, params=params), duration_s=8.0)
    by_conv: dict[str, int] = defaultdict(int)
    for r in reqs:
        assert r.conversation_id is not None
        by_conv[r.conversation_id] += 1
    # Only count sessions that finished (their turn count == drawn length).
    # We approximate "finished" as "second-most-recent or earlier" sessions —
    # i.e. take all but the 4 currently-active ones at end-of-window.
    counts = sorted(by_conv.values())
    finished = counts[:-4] if len(counts) > 4 else counts
    assert finished, "not enough completed sessions to measure"
    assert max(finished) <= 20
    mean_len = sum(finished) / len(finished)
    # Generous ±40% bound — geometric has high variance and we're sampling
    # a smallish number of sessions.
    assert 3.6 <= mean_len <= 8.4, f"mean session length {mean_len} out of range"


@pytest.mark.asyncio
async def test_concurrent_sessions_population_stays_bounded() -> None:
    """At any instant the *active* session population should equal
    `concurrent_sessions`. We sample the population at a few instants
    by counting sessions whose first arrival precedes the sample and
    whose last arrival follows it. (A sliding-window over conv_ids
    would overcount because retired sessions are replaced with new
    ones — those distinct ids are sequential, not concurrent.)
    """
    cap = 6
    params = MultiTurnParams(
        rate_rps=60.0, concurrent_sessions=cap, mean_session_turns=6.0
    )
    reqs = await _drain(MultiTurnWorkload(seed=3, params=params), duration_s=3.0)

    # Group arrivals by conv_id; each conv spans [first, last].
    spans: dict[str, tuple[float, float]] = {}
    for r in reqs:
        assert r.conversation_id is not None
        cur = spans.get(r.conversation_id)
        if cur is None:
            spans[r.conversation_id] = (r.arrival_offset_s, r.arrival_offset_s)
        else:
            spans[r.conversation_id] = (cur[0], r.arrival_offset_s)

    # Sample population at fixed instants; an instant "inside" a span
    # counts the session as active there.
    sample_ts = [0.5, 1.0, 1.5, 2.0, 2.5]
    populations = [
        sum(1 for (lo, hi) in spans.values() if lo <= t <= hi)
        for t in sample_ts
    ]
    # Allow a small slack — between turn-N retirement and turn-1 of the
    # replacement, the population briefly dips.
    assert max(populations) <= cap, f"max active {max(populations)} > cap {cap}"
    assert max(populations) >= cap - 2, f"populations never near cap: {populations}"


@pytest.mark.asyncio
async def test_first_turn_label_assigned_correctly() -> None:
    params = MultiTurnParams(rate_rps=20.0, concurrent_sessions=4)
    reqs = await _drain(MultiTurnWorkload(seed=4, params=params), duration_s=2.0)
    by_conv: dict[str, list[Request]] = defaultdict(list)
    for r in reqs:
        assert r.conversation_id is not None
        by_conv[r.conversation_id].append(r)
    for turns in by_conv.values():
        assert turns[0].label == "multi_turn.turn1"
        for r in turns[1:]:
            assert r.label == "multi_turn.turnN"
