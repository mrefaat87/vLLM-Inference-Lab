#!/usr/bin/env python3
"""load-test.py — Phase 4.5 Dynamo experiment driver.

Adapted from phase3's load-tester. Differences:
  - Talks to a Dynamo Frontend (OpenAI-compatible) instead of the Phase 3 gateway
  - Reads workloads from phase4.5/scripts/workloads/{w1,w2,w3,w4}_*.py
  - Captures Dynamo-specific metrics: router cache hit rate, NIXL transfer
    latency histogram, per-worker request distribution
  - JSON output schema matches phase3 so existing analysis tools transfer

Output JSON: NEVER overwritten on re-run. Caller picks the path; tool refuses
to overwrite an existing file (per feedback_sweep_data_preservation.md).

Usage:
  python3 load-test.py \
      --endpoint http://dynamo-frontend.dynamo-system.svc:8000 \
      --workload w2_shared_prefix \
      --rate 1.5 \
      --duration 90 \
      --warmup 20 \
      --drain 10 \
      --metrics-endpoint http://dynamo-frontend.dynamo-system.svc:9090/metrics \
      --output ../tests/exp-B-W2-results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# Allow `python3 load-test.py` from phase4.5/scripts/ without packaging.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class RequestResult:
    request_id: int
    label: str
    submit_offset_sec: float
    client_ttft_sec: float | None
    client_total_sec: float | None
    tokens_out: int
    client_avg_tbt_sec: float | None
    error: str | None = None
    # Dynamo-specific (populated from /metrics correlation if available):
    routed_worker: str | None = None
    cache_hit_blocks: int | None = None
    nixl_transfer_ms: float | None = None


@dataclass
class RunConfig:
    endpoint: str
    metrics_endpoint: str | None
    workload: str
    rate: float       # requests per second
    duration: float
    warmup: float
    drain: float
    seed: int


@dataclass
class RunReport:
    strategy: str        # the experiment label (e.g., "exp-B-kvrouter")
    workload: str
    config: dict[str, Any]
    analysis: dict[str, Any]
    raw_results: list[dict[str, Any]] = field(default_factory=list)
    dynamo_metrics: dict[str, Any] = field(default_factory=dict)


def _load_workload(name: str, n: int, seed: int) -> list:
    """Import the named workload module dynamically and call generate(n, seed).

    W4 (cold_start) ignores n — it always emits a single probe."""
    if name == "w1_random":
        from workloads.w1_random import generate
        return generate(n, seed)
    if name == "w2_shared_prefix":
        from workloads.w2_shared_prefix import generate
        return generate(n, seed)
    if name == "w3_bursty":
        from workloads.w3_bursty import generate
        return generate(n, seed)
    if name == "w4_cold_start":
        from workloads.w4_cold_start import generate
        return generate()
    raise ValueError(f"unknown workload: {name}")


async def _send_one(session, endpoint: str, prompt, request_id: int,
                    submit_offset: float) -> RequestResult:
    """Issue one streaming chat completion. Records client-observed TTFT and
    total time, plus per-token receipt times for TBT estimation."""
    import aiohttp
    body = {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [{"role": "user", "content": prompt.text}],
        "stream": True,
        "max_tokens": prompt.max_tokens,
    }
    t_start = time.monotonic()
    ttft = None
    token_times: list[float] = []
    tokens_out = 0
    try:
        async with session.post(f"{endpoint}/v1/chat/completions",
                                json=body, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                return RequestResult(request_id, prompt.label, submit_offset,
                                     None, None, 0, None,
                                     error=f"http_{resp.status}")
            async for line in resp.content:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                if ttft is None:
                    ttft = time.monotonic() - t_start
                token_times.append(time.monotonic())
                tokens_out += 1
        total = time.monotonic() - t_start
        avg_tbt = None
        if len(token_times) >= 2:
            diffs = [token_times[i] - token_times[i - 1]
                     for i in range(1, len(token_times))]
            avg_tbt = sum(diffs) / len(diffs)
        return RequestResult(request_id, prompt.label, submit_offset,
                             ttft, total, tokens_out, avg_tbt)
    except Exception as e:
        return RequestResult(request_id, prompt.label, submit_offset,
                             None, None, tokens_out, None, error=str(e))


async def _run(cfg: RunConfig) -> RunReport:
    import aiohttp
    n = max(int((cfg.duration + cfg.warmup) * cfg.rate), 1)
    prompts = _load_workload(cfg.workload, n, cfg.seed)
    interval = 1.0 / cfg.rate if cfg.rate > 0 else 0
    t_zero = time.monotonic()
    tasks = []

    async with aiohttp.ClientSession() as session:
        for i, p in enumerate(prompts):
            target = t_zero + i * interval
            now = time.monotonic()
            if target > now:
                await asyncio.sleep(target - now)
            tasks.append(asyncio.create_task(
                _send_one(session, cfg.endpoint, p, i, time.monotonic() - t_zero)
            ))
        results: list[RequestResult] = await asyncio.gather(*tasks)

    # Drain window: skip warmup and last `drain` seconds for steady-state.
    steady = [r for r in results
              if cfg.warmup <= r.submit_offset_sec <= (cfg.warmup + cfg.duration - cfg.drain)
              and r.error is None]

    def _pct(values, p):
        if not values:
            return None
        s = sorted(values)
        k = max(0, min(len(s) - 1, int(len(s) * p / 100)))
        return round(s[k], 4)

    ttfts = [r.client_ttft_sec for r in steady if r.client_ttft_sec is not None]
    tbts = [r.client_avg_tbt_sec for r in steady if r.client_avg_tbt_sec is not None]
    totals = [r.client_total_sec for r in steady if r.client_total_sec is not None]
    total_tokens = sum(r.tokens_out for r in steady)

    analysis = {
        "steady_state_requests": len(steady),
        "steady_state_errors": sum(1 for r in results
                                   if r.error is not None
                                   and cfg.warmup <= r.submit_offset_sec <= cfg.warmup + cfg.duration - cfg.drain),
        "total_requests": len(results),
        "total_errors": sum(1 for r in results if r.error is not None),
        "ttft": {"p50": _pct(ttfts, 50), "p95": _pct(ttfts, 95), "p99": _pct(ttfts, 99)},
        "tbt_client": {"p50": _pct(tbts, 50), "p95": _pct(tbts, 95), "p99": _pct(tbts, 99)},
        "total_sec": {"p50": _pct(totals, 50), "p95": _pct(totals, 95), "p99": _pct(totals, 99)},
        "throughput": {
            "total_tokens": total_tokens,
            "tok_per_sec_avg": round(total_tokens / cfg.duration, 1) if cfg.duration > 0 else 0,
        },
    }

    # Dynamo metrics scrape: best-effort. If endpoint is None or unreachable,
    # we still write the JSON — just without the dynamo block.
    dynamo_metrics = {}
    if cfg.metrics_endpoint:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.get(cfg.metrics_endpoint, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    text = await resp.text()
            # Crude line filter — store everything mentioning dynamo or nixl.
            lines = [l for l in text.splitlines()
                     if ("dynamo" in l.lower() or "nixl" in l.lower())
                     and not l.startswith("#")]
            dynamo_metrics["raw_metric_lines"] = lines[:200]  # cap to keep JSON small
        except Exception as e:
            dynamo_metrics["scrape_error"] = str(e)

    return RunReport(
        strategy=os.environ.get("PHASE45_EXPERIMENT_LABEL", "unknown"),
        workload=cfg.workload,
        config=asdict(cfg),
        analysis=analysis,
        raw_results=[asdict(r) for r in results],
        dynamo_metrics=dynamo_metrics,
    )


def _main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4.5 Dynamo load tester.")
    ap.add_argument("--endpoint", required=True, help="Dynamo Frontend base URL (without /v1).")
    ap.add_argument("--metrics-endpoint", default=None,
                    help="Prometheus /metrics URL on the frontend; optional.")
    ap.add_argument("--workload", required=True,
                    choices=["w1_random", "w2_shared_prefix", "w3_bursty", "w4_cold_start"])
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--warmup", type=float, default=20.0)
    ap.add_argument("--drain", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True, help="JSON path; refuses to overwrite.")
    args = ap.parse_args()

    out = Path(args.output)
    if out.exists():
        print(f"[load-test] refusing to overwrite existing file: {out}", file=sys.stderr)
        return 2

    cfg = RunConfig(
        endpoint=args.endpoint.rstrip("/"),
        metrics_endpoint=args.metrics_endpoint,
        workload=args.workload,
        rate=args.rate,
        duration=args.duration,
        warmup=args.warmup,
        drain=args.drain,
        seed=args.seed,
    )

    try:
        report = asyncio.run(_run(cfg))
    except KeyboardInterrupt:
        print("[load-test] interrupted", file=sys.stderr)
        return 130

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2))
    print(f"[load-test] wrote {out}")
    print(f"  steady_state: {report.analysis['steady_state_requests']} requests, "
          f"{report.analysis['steady_state_errors']} errors")
    print(f"  TTFT p50/p99: {report.analysis['ttft']['p50']}/{report.analysis['ttft']['p99']} sec")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
