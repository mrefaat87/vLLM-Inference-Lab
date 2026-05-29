"""Async driver loop.

Given an endpoint and a workload iterator, dispatch requests according to
their ``arrival_offset_s`` schedule and collect per-request TTFT, TBT,
and end-to-end latency. The loop is engine-agnostic — it speaks OpenAI
chat-completions SSE and trusts the EndpointInfo from the driver.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import aiohttp

from experiments.drivers.base import EndpointInfo
from experiments.runner.schema import RequestRecord
from experiments.workloads.base import Request

# How long a single request is allowed to hang before we mark it failed.
DEFAULT_REQUEST_TIMEOUT_S = 120.0


@dataclass
class LoopConfig:
    endpoint: EndpointInfo
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    model: str = "mock-model"
    stream: bool = True
    concurrency_cap: int = 1024


async def run_loop(
    requests: AsyncIterator[Request],
    cfg: LoopConfig,
    t0: float | None = None,
    wall_clock_cap_s: float | None = None,
) -> list[RequestRecord]:
    """Run the driver loop to completion and return the per-request records.

    When ``wall_clock_cap_s`` is set (calibration probes only), in-flight
    requests that haven't completed by ``t0 + wall_clock_cap_s`` are cancelled
    and recorded with ``error="ProbeCutoff"`` so the saturation rule can count
    them as completion-lag without blocking the calibration phase on a slow
    engine's residence time. The measurement window passes ``None`` and waits
    for every request to finish normally.
    """
    t0 = t0 if t0 is not None else time.monotonic()
    sem = asyncio.Semaphore(cfg.concurrency_cap)
    timeout = aiohttp.ClientTimeout(total=cfg.request_timeout_s)
    connector = aiohttp.TCPConnector(limit=0)
    records: list[RequestRecord] = []

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Track (request, task) pairs so we can build cutoff records for
        # tasks we have to cancel — RequestRecord needs the originating
        # Request's fields and the task itself doesn't carry them.
        pairs: list[tuple[Request, asyncio.Task[RequestRecord]]] = []
        async for req in requests:
            # Sleep until the request's scheduled arrival time. Negative sleeps
            # are clamped to 0 — if we fell behind, fire immediately and let
            # the analysis spot the slip.
            now = time.monotonic() - t0
            delay = req.arrival_offset_s - now
            if delay > 0:
                await asyncio.sleep(delay)
            task = asyncio.create_task(_dispatch_one(session, req, cfg, t0, sem))
            pairs.append((req, task))
        if wall_clock_cap_s is None:
            # Measurement path: wait for every in-flight request to finish.
            for _, t in pairs:
                records.append(await t)
        else:
            # Calibration path: bounded wait, then cancel + record cutoffs.
            remaining = max(0.0, (t0 + wall_clock_cap_s) - time.monotonic())
            task_to_req = {t: req for (req, t) in pairs}
            done, pending = await asyncio.wait(
                [t for (_, t) in pairs],
                timeout=remaining,
            )
            for t in done:
                try:
                    records.append(t.result())
                except Exception as exc:  # noqa: BLE001 — surface as error record
                    req = task_to_req[t]
                    records.append(_cutoff_record(req, f"{type(exc).__name__}: {exc}"))
            for t in pending:
                t.cancel()
                req = task_to_req[t]
                records.append(_cutoff_record(req, "ProbeCutoff: in-flight at probe wall-clock cap"))
            # Let cancellations settle so the session can close cleanly.
            for t in pending:
                try:
                    await t
                except (asyncio.CancelledError, aiohttp.ClientError):
                    pass
    return records


def _cutoff_record(req: Request, error: str) -> RequestRecord:
    """Synthetic record for a request still in-flight at the probe cutoff."""
    return RequestRecord(
        request_id=req.request_id,
        label=req.label,
        # Arrival offset is the closest thing to "when this entered the system"
        # we have at this point; it's >= 0 and bounded by the probe window.
        submit_offset_s=req.arrival_offset_s,
        prompt_tokens=req.prompt_tokens,
        max_new_tokens=req.max_new_tokens,
        error=error,
        conversation_id=req.conversation_id,
    )


async def _dispatch_one(
    session: aiohttp.ClientSession,
    req: Request,
    cfg: LoopConfig,
    t0: float,
    sem: asyncio.Semaphore,
) -> RequestRecord:
    submit = time.monotonic()
    submit_offset = submit - t0
    payload = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_new_tokens,
        "stream": cfg.stream,
        "metadata": {"request_id": req.request_id, "label": req.label},
    }
    url = f"{cfg.endpoint.base_url}/v1/chat/completions"
    async with sem:
        try:
            return await _do_request(session, url, payload, req, submit_offset, cfg)
        except (TimeoutError, aiohttp.ClientError) as exc:
            return RequestRecord(
                request_id=req.request_id,
                label=req.label,
                submit_offset_s=submit_offset,
                prompt_tokens=req.prompt_tokens,
                max_new_tokens=req.max_new_tokens,
                error=f"{type(exc).__name__}: {exc}",
                conversation_id=req.conversation_id,
            )


async def _do_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, object],
    req: Request,
    submit_offset: float,
    cfg: LoopConfig,
) -> RequestRecord:
    start = time.monotonic()
    if not cfg.stream:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            body = await resp.json()
            end = time.monotonic()
        completion_tokens = int(body.get("usage", {}).get("completion_tokens", 0))
        return RequestRecord(
            request_id=req.request_id,
            label=req.label,
            submit_offset_s=submit_offset,
            prompt_tokens=req.prompt_tokens,
            max_new_tokens=req.max_new_tokens,
            completion_tokens=completion_tokens,
            ttft_s=None,
            tbt_p50_s=None,
            end_to_end_s=end - start,
            conversation_id=req.conversation_id,
        )

    # Streaming path.
    ttft: float | None = None
    inter_token: list[float] = []
    last_token_t: float | None = None
    n_tokens = 0
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                evt = json.loads(data)
            except json.JSONDecodeError:
                continue
            # An SSE chunk with a delta is one token (close enough for mock; real
            # engines may chunk multiple tokens — accounted for in analysis).
            now = time.monotonic()
            if ttft is None:
                ttft = now - start
                last_token_t = now
            else:
                if last_token_t is not None:
                    inter_token.append(now - last_token_t)
                last_token_t = now
            _ = evt  # not used further; presence is enough to count a chunk
            n_tokens += 1
    end = time.monotonic()
    tbt_p50 = _median(inter_token) if inter_token else None
    return RequestRecord(
        request_id=req.request_id,
        label=req.label,
        submit_offset_s=submit_offset,
        prompt_tokens=req.prompt_tokens,
        max_new_tokens=req.max_new_tokens,
        completion_tokens=n_tokens,
        ttft_s=ttft,
        tbt_p50_s=tbt_p50,
        end_to_end_s=end - start,
        conversation_id=req.conversation_id,
    )


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])
