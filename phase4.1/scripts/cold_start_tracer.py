#!/usr/bin/env python3
"""Cold-start tracer wrapper (NOT a K8s probe — in-process instrumentation) around the vLLM OpenAI server.

Why: stages 6-8 of the waterfall (vLLM init / weight load / CUDA graph capture +
warmup) are internal to the vLLM process. K8s events and pod status only see
"container started" and "Ready" — they can't tell us when CUDA context allocation
finished vs when weights finished loading. We need timestamps emitted from inside
the process.

How: wrap the vLLM entrypoint as a subprocess. Tee its stdout through a regex
matcher that watches for known phase boundaries in vLLM's log output. Emit JSONL
events to stdout AND to a hostPath file that survives pod deletion.

Output paths:
  - stdout: every event also goes to stdout so kubectl logs picks it up
  - $JSONL_DIR/run-<id>.jsonl: hostPath-mounted file (collector reads via kubectl debug node)

Event schema:
  {"event": "<name>", "epoch_ms": <int>, "run_id": "<str>"}

Stages emitted (from observable signals):
  process_start         — wrapper entrypoint reached
  vllm_import_done      — `import vllm` returned (pure Python, no GPU)
  cuda_context_ready    — first torch.cuda op forced; CUDA context exists
  weight_load_start     — log line "Starting to load model" matched
  weight_load_done      — log line "Loading model weights took" matched
  graph_capture_start   — log line "Capturing CUDA graphs" matched
  warmup_done           — log line "Graph capturing finished" matched
  server_ready          — vLLM /health returns 200 (polled in background)

The log-scraping markers are vLLM v0.19.0-specific. If patterns break in a future
vLLM version, the collector flags missing events rather than silently dropping
a stage — see collect_cold_start_run.py validation.

Usage:
  cold_start_tracer.py --run-id fresh-001 -- \
    python -m vllm.entrypoints.openai.api_server --model <...> [args...]

Everything after "--" is exec'd as the inference server.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
import signal
from typing import Optional

DEFAULT_JSONL_DIR = "/var/log/cold_start"

# vLLM v0.19.0 log patterns. These were chosen because they're stable strings
# that appear once per server start. Each pattern → event name.
LOG_PATTERNS = [
    (re.compile(r"Starting to load model"), "weight_load_start"),
    (re.compile(r"Loading (?:model )?weights took"), "weight_load_done"),
    (re.compile(r"Capturing CUDA graph"), "graph_capture_start"),
    (re.compile(r"Graph capturing finished"), "warmup_done"),
]

# Each event fires AT MOST ONCE per run — vLLM occasionally re-logs similar
# strings during retries; we don't want stage timings polluted.
_FIRED: set[str] = set()


class EventEmitter:
    """Single-writer JSONL emitter. Thread-safe (log scraper + readiness watcher
    can both emit)."""

    def __init__(self, run_id: str, jsonl_path: pathlib.Path):
        self.run_id = run_id
        self.jsonl_path = jsonl_path
        self._lock = threading.Lock()
        self._jsonl_file = None
        try:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_file = open(jsonl_path, "a", buffering=1)
        except OSError as e:
            print(f"[tracer] cannot open jsonl at {jsonl_path}: {e}", file=sys.stderr)

    def emit(self, event: str, **extra) -> None:
        # Dedupe: the same event name fires only once per run.
        with self._lock:
            if event in _FIRED:
                return
            _FIRED.add(event)
        rec = {
            "event": event,
            "epoch_ms": int(time.time() * 1000),
            "run_id": self.run_id,
        }
        rec.update(extra)
        line = json.dumps(rec)
        with self._lock:
            print(line, flush=True)
            if self._jsonl_file:
                self._jsonl_file.write(line + "\n")
                self._jsonl_file.flush()

    def close(self) -> None:
        with self._lock:
            if self._jsonl_file:
                self._jsonl_file.close()
                self._jsonl_file = None


def scan_line_for_events(line: str, emitter: EventEmitter) -> None:
    """Match a single log line against the event patterns and emit on first hit.
    Pure function (modulo the emitter side-effect) so it's easy to unit test.

    Implicit-close fallback: graph capture cannot start before weights have
    finished loading. If `graph_capture_start` matches and `weight_load_done`
    was never emitted (e.g. RunAI Model Streamer doesn't log vLLM's stock
    "Loading weights took" message), close the weight-load stage at the
    moment graph capture begins. This keeps the per-stage waterfall whole
    for streamer-like loaders without faking a measurement we don't have."""
    for pat, ev in LOG_PATTERNS:
        if pat.search(line):
            if ev == "graph_capture_start" and "weight_load_done" not in _FIRED:
                emitter.emit("weight_load_done")
            emitter.emit(ev)
            # Don't break — a line that matches start could in principle also
            # contain another marker. In practice patterns are exclusive, but
            # being defensive costs nothing.


def force_cuda_context(emitter: EventEmitter) -> None:
    """Allocate a 1-byte CUDA tensor to force context creation. Stage 6 boundary."""
    try:
        import torch  # type: ignore

        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        emitter.emit("cuda_context_ready")
    except Exception as e:
        print(f"[tracer] cuda context init failed: {e}", file=sys.stderr)


def wait_for_server_ready(
    emitter: EventEmitter,
    host: str = "127.0.0.1",
    port: int = 8000,
    timeout: float = 1200.0,
) -> None:
    """Poll /health until 200. Runs as a background thread."""
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1.0) as r:
                if r.status == 200:
                    emitter.emit("server_ready")
                    return
        except Exception:
            pass
        time.sleep(0.5)
    emitter.emit("server_ready_timeout")


def _stream_subprocess(proc: subprocess.Popen, emitter: EventEmitter) -> None:
    """Read subprocess stdout line-by-line, forward to our stdout, scan for events."""
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        # Forward unchanged so kubectl logs still sees vLLM's native output.
        sys.stdout.write(line)
        sys.stdout.flush()
        scan_line_for_events(line, emitter)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--jsonl-dir", default=os.environ.get("TRACER_JSONL_DIR", DEFAULT_JSONL_DIR))
    parser.add_argument("--ready-host", default="127.0.0.1")
    parser.add_argument("--ready-port", type=int, default=8000)
    parser.add_argument("--skip-cuda-init", action="store_true",
                        help="Skip torch CUDA init (for unit tests / dry runs).")
    args, rest = parser.parse_known_args()

    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        print("usage: cold_start_tracer.py --run-id ID -- <vllm command...>",
              file=sys.stderr)
        return 2

    jsonl_path = pathlib.Path(args.jsonl_dir) / f"run-{args.run_id}.jsonl"
    emitter = EventEmitter(args.run_id, jsonl_path)
    emitter.emit("process_start")

    # Pure Python import — measures framework init upper bound (no GPU yet).
    try:
        import importlib
        importlib.import_module("vllm")
        emitter.emit("vllm_import_done")
    except Exception as e:
        print(f"[tracer] vllm import failed: {e}", file=sys.stderr)

    if not args.skip_cuda_init:
        force_cuda_context(emitter)

    # Run vLLM as a subprocess so we can scrape its stdout. Combine stderr into
    # stdout so we don't miss patterns that vLLM logs to stderr.
    proc = subprocess.Popen(
        rest,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    # Forward signals so kubelet's SIGTERM reaches the actual server.
    def _forward(signum, _frame):
        proc.send_signal(signum)

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)

    # Background readiness watcher.
    threading.Thread(
        target=wait_for_server_ready,
        args=(emitter, args.ready_host, args.ready_port),
        daemon=True,
    ).start()

    # Stream vLLM output (blocks until the process exits).
    try:
        _stream_subprocess(proc, emitter)
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGTERM)

    rc = proc.wait()
    emitter.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
