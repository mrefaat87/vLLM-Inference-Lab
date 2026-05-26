"""Post-run analysis: turn a list of RequestRecords into an Analysis summary."""

from __future__ import annotations

from statistics import mean

from experiments.runner.schema import Analysis, Percentiles, RequestRecord, ThroughputStats


def analyze(
    records: list[RequestRecord],
    *,
    duration_s: float,
    warmup_s: float = 0.0,
) -> Analysis:
    """Compute summary stats from raw records.

    Requests with ``submit_offset_s < warmup_s`` are excluded from the steady-state
    percentiles but still counted in throughput totals — same convention as
    standard inference benchmarks (vLLM benchmark_serving.py).
    """
    steady = [r for r in records if r.submit_offset_s >= warmup_s and r.error is None]
    failed = [r for r in records if r.error is not None]

    ttft_pcts = _pcts([r.ttft_s for r in steady if r.ttft_s is not None])
    tbt_pcts = _pcts([r.tbt_p50_s for r in steady if r.tbt_p50_s is not None])
    e2e_pcts = _pcts([r.end_to_end_s for r in steady if r.end_to_end_s is not None])

    total_completion = sum(r.completion_tokens for r in records)
    total_prompt = sum(r.prompt_tokens for r in records)
    elapsed = max(duration_s, 1e-9)
    # ISL = prompt-token p50 across STEADY records (warmup excluded so a
    # ramp doesn't drag the median). OSL = completion-token p50 across
    # steady records that actually produced output (completion_tokens > 0)
    # so failed / aborted requests don't poison the percentile with zeros.
    isl_p50 = _int_pct50([r.prompt_tokens for r in steady])
    osl_p50 = _int_pct50([r.completion_tokens for r in steady if r.completion_tokens > 0])
    return Analysis(
        steady_state_requests=len(steady),
        failed_requests=len(failed),
        ttft_s=ttft_pcts,
        tbt_s=tbt_pcts,
        e2e_s=e2e_pcts,
        throughput=ThroughputStats(
            total_completion_tokens=total_completion,
            total_prompt_tokens=total_prompt,
            tok_per_sec_avg=total_completion / elapsed,
            requests_per_sec_avg=len(records) / elapsed,
        ),
        isl_tokens_p50=isl_p50,
        osl_tokens_p50=osl_p50,
    )


def _pcts(xs: list[float]) -> Percentiles | None:
    if not xs:
        return None
    s = sorted(xs)
    return Percentiles(
        p50=_pct(s, 0.50),
        p95=_pct(s, 0.95),
        p99=_pct(s, 0.99),
        mean=float(mean(s)),
        n=len(s),
    )


def _int_pct50(xs: list[int]) -> int | None:
    """Median of an int list, returning None on empty.

    Used for ISL/OSL where the consumer (calc bridge) wants a whole-token
    count rather than a fractional interpolated value.
    """
    if not xs:
        return None
    s = sorted(xs)
    return int(_pct([float(x) for x in s], 0.50))


def _pct(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    # Linear interpolation between two nearest ranks.
    rank = q * (len(sorted_xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = rank - lo
    return float(sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac)
