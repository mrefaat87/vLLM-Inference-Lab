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
) -> list[RequestRecord]:
    """Run the driver loop to completion and return the per-request records."""
    t0 = t0 if t0 is not None else time.monotonic()
    sem = asyncio.Semaphore(cfg.concurrency_cap)
    timeout = aiohttp.ClientTimeout(total=cfg.request_timeout_s)
    connector = aiohttp.TCPConnector(limit=0)
    records: list[RequestRecord] = []

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks: list[asyncio.Task[RequestRecord]] = []
        async for req in requests:
            # Sleep until the request's scheduled arrival time. Negative sleeps
            # are clamped to 0 — if we fell behind, fire immediately and let
            # the analysis spot the slip.
            now = time.monotonic() - t0
            delay = req.arrival_offset_s - now
            if delay > 0:
                await asyncio.sleep(delay)
            task = asyncio.create_task(_dispatch_one(session, req, cfg, t0, sem))
            tasks.append(task)
        # Wait for in-flight requests to finish.
        for t in tasks:
            records.append(await t)
    return records


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
