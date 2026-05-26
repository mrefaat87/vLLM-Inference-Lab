"""In-process OpenAI-compatible stub server.

Used by MockEngineDriver and by integration tests to exercise the full
sweep runner pipeline with zero GPU dependency. Supports SSE streaming
on /v1/chat/completions so the driver loop's TTFT and TBT logic is
genuinely exercised, not stubbed out.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def _build_app() -> FastAPI:
    app = FastAPI()
    counters: dict[str, int] = {"requests": 0, "stream_chunks": 0}

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    # SGLang uses /health_generate; TRT-LLM Triton uses /v2/health/ready.
    @app.get("/health_generate")
    async def health_generate() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/v2/health/ready")
    async def triton_ready() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/metrics")
    async def metrics() -> JSONResponse:
        return JSONResponse({"requests": counters["requests"], "stream_chunks": counters["stream_chunks"]})

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(req: Request) -> StreamingResponse | JSONResponse:
        body = await req.json()
        counters["requests"] += 1
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_tokens", body.get("max_new_tokens", 16)))
        # Simulate prefill latency proportional to prompt size to make TTFT non-zero.
        prompt_len = _approx_prompt_len(body)
        prefill_s = 0.001 + 0.00001 * prompt_len  # 1ms + 10us/token
        decode_s = 0.002                          # 2ms/token

        if not stream:
            await asyncio.sleep(prefill_s + decode_s * max_tokens)
            return JSONResponse(
                {
                    "id": f"mock-{counters['requests']}",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok " * max_tokens},
                            "finish_reason": "stop",
                            "index": 0,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_len,
                        "completion_tokens": max_tokens,
                        "total_tokens": prompt_len + max_tokens,
                    },
                }
            )

        async def gen() -> Any:
            await asyncio.sleep(prefill_s)
            for i in range(max_tokens):
                counters["stream_chunks"] += 1
                chunk = {
                    "id": f"mock-{counters['requests']}",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"delta": {"content": "ok "}, "finish_reason": None, "index": 0}
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(decode_s)
            # Final "done" frame mirrors OpenAI's SSE termination.
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _approx_prompt_len(body: dict[str, Any]) -> int:
    msgs = body.get("messages", [])
    total = 0
    for m in msgs:
        c = m.get("content", "")
        if isinstance(c, str):
            total += max(1, len(c.split()))
    return total or 1


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class StubServer:
    """Spawn a uvicorn server in a background thread.

    Owns its event loop so callers don't have to. Designed to be cheap
    (start in <100ms on a normal laptop) so tests can spin one up
    per-test.
    """

    def __init__(self, *, port: int = 0) -> None:
        self._port = port or _free_port()
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._running = False
        self._metrics: dict[str, int] = {}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, timeout_s: float = 5.0) -> None:
        if self._running:
            return
        cfg = uvicorn.Config(
            _build_app(),
            host="127.0.0.1",
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        # Wait for readiness.
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._server.started:
                self._running = True
                return
            time.sleep(0.02)
        raise RuntimeError("StubServer did not start within timeout")

    def stop(self) -> None:
        if not self._running:
            return
        assert self._server is not None
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._running = False

    def metrics(self) -> dict[str, int]:
        # Lightweight in-process snapshot. The /metrics endpoint is also
        # available to remote callers.
        return dict(self._metrics)
