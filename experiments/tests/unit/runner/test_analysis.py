"""Unit tests for analysis.py."""

from __future__ import annotations

import pytest

from experiments.runner.analysis import analyze
from experiments.runner.schema import RequestRecord

pytestmark = pytest.mark.unit


def _r(
    rid: str = "r",
    submit: float = 1.0,
    ttft: float | None = 0.05,
    e2e: float | None = 0.5,
    err: str | None = None,
    comp: int = 50,
    prompt: int = 100,
) -> RequestRecord:
    return RequestRecord(
        request_id=rid,
        label="chatbot.turn1",
        submit_offset_s=submit,
        prompt_tokens=prompt,
        max_new_tokens=comp,
        completion_tokens=comp,
        ttft_s=ttft,
        end_to_end_s=e2e,
        error=err,
    )


def test_excludes_warmup_from_percentiles_but_counts_throughput() -> None:
    records = [
        _r("a", submit=0.5, ttft=0.10),
        _r("b", submit=0.6, ttft=0.20),
        _r("c", submit=2.0, ttft=0.30),
        _r("d", submit=3.0, ttft=0.40),
    ]
    a = analyze(records, duration_s=4.0, warmup_s=1.0)
    assert a.steady_state_requests == 2
    assert a.ttft_s is not None
    assert a.ttft_s.n == 2
    # Throughput counts ALL records, not just steady.
    assert a.throughput.total_completion_tokens == 200


def test_failed_requests_counted() -> None:
    records = [_r("ok"), _r("bad", err="boom", ttft=None, e2e=None)]
    a = analyze(records, duration_s=2.0)
    assert a.failed_requests == 1
    assert a.ttft_s is not None
    assert a.ttft_s.n == 1


def test_percentiles_handle_singleton() -> None:
    a = analyze([_r("only", ttft=0.42)], duration_s=1.0)
    assert a.ttft_s is not None
    assert a.ttft_s.p50 == pytest.approx(0.42)
    assert a.ttft_s.p99 == pytest.approx(0.42)


def test_empty_records() -> None:
    a = analyze([], duration_s=1.0)
    assert a.steady_state_requests == 0
    assert a.failed_requests == 0
    assert a.throughput.total_completion_tokens == 0
    assert a.ttft_s is None
