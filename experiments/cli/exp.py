"""`exp` CLI — start runs, list manifests, build the portal.

Usage:
    exp run --engine vllm --workload chatbot --rate 8 --duration 300 \\
        --model meta-llama/Llama-3-70B-Instruct-AWQ --quant awq --tp 4
    exp list
    exp build-portal --out _site
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import click

from experiments.drivers.base import EngineDriver
from experiments.drivers.mock import MockEngineDriver
from experiments.drivers.sglang_driver import SGLangDriver
from experiments.drivers.trtllm_driver import TRTLLMDriver
from experiments.drivers.vllm_driver import VLLMDriver
from experiments.portal.build import BuildInputs, build as build_portal
from experiments.runner.calc_bridge import CalcBridge, CalcInputs
from experiments.runner.manifest_store import ManifestStore
from experiments.runner.schema import (
    EngineConfig,
    HardwareSpec,
    ModelSpec,
    Prediction,
    RooflineLink,
    WorkloadConfig,
)
from experiments.runner.sweep import SweepRunner, _precision_from_quant
from experiments.workloads.agentic_coding import AgenticCodingWorkload
from experiments.workloads.base import WorkloadGenerator
from experiments.workloads.chatbot import ChatbotParams, ChatbotWorkload
from experiments.workloads.mix import MixWorkload, WeightedChild
from experiments.workloads.multi_turn import MultiTurnParams, MultiTurnWorkload
from experiments.workloads.rag import RagParams, RagWorkload


DRIVERS: dict[str, type[EngineDriver]] = {
    "mock": MockEngineDriver,
    "vllm": VLLMDriver,
    "sglang": SGLangDriver,
    "trtllm": TRTLLMDriver,
}

ENGINE_IMAGES: dict[str, str] = {
    "mock": "n/a",
    "vllm": "vllm/vllm-openai:latest",
    "sglang": "lmsysorg/sglang:latest",
    "trtllm": "nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3",
}


def _workload(name: str, *, seed: int, rate: float) -> WorkloadGenerator:
    if name == "chatbot":
        return ChatbotWorkload(seed=seed, params=ChatbotParams(rate_rps=rate))
    if name == "agentic_coding":
        return AgenticCodingWorkload(seed=seed)
    if name == "rag":
        return RagWorkload(seed=seed, params=RagParams(rate_rps=rate))
    if name == "multi_turn":
        return MultiTurnWorkload(seed=seed, params=MultiTurnParams(rate_rps=rate))
    if name == "mix":
        return MixWorkload(
            [
                WeightedChild(ChatbotWorkload(seed=seed, params=ChatbotParams(rate_rps=rate * 0.7))),
                WeightedChild(AgenticCodingWorkload(seed=seed + 1)),
            ]
        )
    raise click.BadParameter(f"unknown workload: {name}")


@click.group()
def cli() -> None:
    """Empirical Inference Lab CLI."""


def _run_options(f):
    """Apply the shared --engine / --workload / … option set.

    Used by both `exp run` and `exp launch`. Kept as a stacked-decorator
    function (not @click.pass_context) so the wrapped functions receive
    the options as plain keyword args.
    """
    for opt in reversed([
        click.option("--engine", type=click.Choice(list(DRIVERS)), required=True),
        click.option(
            "--workload",
            type=click.Choice(["chatbot", "agentic_coding", "rag", "multi_turn", "mix"]),
            required=True,
        ),
        click.option("--rate", type=float, default=8.0, help="target arrival RPS"),
        click.option("--duration", type=float, default=300.0, help="run duration (s)"),
        click.option("--warmup", type=float, default=10.0),
        click.option("--seed", type=int, default=1),
        click.option("--model", default="meta-llama/Llama-3-70B-Instruct-AWQ"),
        click.option("--quant", default="awq"),
        click.option("--tp", type=int, default=4),
        click.option("--instance", default="g5.12xlarge"),
        click.option("--gpu", default="A10G"),
        click.option("--n-gpu", type=int, default=4),
        click.option(
            "--model-ref",
            default="llama-3-70b",
            help="join key into calculators/sizing_calc/src/data/models.json (the model's `key` field)",
        ),
        click.option(
            "--hw-ref",
            default="A10G",
            help="join key into calculators/sizing_calc/src/data/hardware.json (the hardware's `key` field)",
        ),
        click.option("--tbt-target-ms", type=float, default=50.0,
                     help="target inter-token latency (used by the calc bridge for predicted curves)"),
        click.option(
            "--preflight",
            type=click.Choice(["off", "advisory", "strict"]),
            default="advisory",
            help="consult the sizing calculator before launching: advisory (default) prints warnings, strict exits non-zero",
        ),
        click.option("--results-dir", type=click.Path(path_type=Path), default=Path("results")),
        click.option("--notes", default=None),
    ]):
        f = opt(f)
    return f


@cli.command("run")
@_run_options
def run_cmd(**kwargs) -> None:
    """Run one (engine, workload, rate) experiment."""
    path = _do_run(**kwargs)
    click.echo(f"wrote {path}")


def _do_run(
    *,
    engine: str,
    workload: str,
    rate: float,
    duration: float,
    warmup: float,
    seed: int,
    model: str,
    quant: str,
    tp: int,
    instance: str,
    gpu: str,
    n_gpu: int,
    model_ref: str,
    hw_ref: str,
    tbt_target_ms: float,
    preflight: str,
    results_dir: Path,
    notes: str | None,
) -> Path:
    """Body of `exp run`, returning the result-JSON path so other
    subcommands (e.g. `exp launch`) can chain off it.
    """
    driver_cls = DRIVERS[engine]
    driver = driver_cls()
    engine_cfg = EngineConfig(
        name=engine,
        image=ENGINE_IMAGES[engine],
        model=model,
        quantization=quant,
        tensor_parallel=tp,
    )
    workload_cfg = WorkloadConfig(
        name=workload, rate_rps=rate, duration_s=duration, warmup_s=warmup, seed=seed
    )
    bridge = CalcBridge()

    if preflight != "off":
        ok = _do_preflight(
            bridge=bridge,
            model_ref=model_ref,
            hw_ref=hw_ref,
            quant=quant,
            n_gpu=n_gpu,
            tbt_target_ms=tbt_target_ms,
            strict=(preflight == "strict"),
        )
        if not ok:
            sys.exit(1)

    runner = SweepRunner(results_dir=results_dir, calc_bridge=bridge)

    async def _go() -> Path:
        return await runner.run_one(
            engine=driver,
            engine_cfg=engine_cfg,
            workload=_workload(workload, seed=seed, rate=rate),
            workload_cfg=workload_cfg,
            model=ModelSpec(name=model, quant=quant, tp=tp),
            hardware=HardwareSpec(instance=instance, gpu=gpu, n_gpu=n_gpu),
            roofline_link=RooflineLink(model_ref=model_ref, hw_ref=hw_ref),
            notes=notes,
            tbt_target_ms=tbt_target_ms,
        )

    return asyncio.run(_go())


def _do_preflight(
    *,
    bridge: CalcBridge,
    model_ref: str,
    hw_ref: str,
    quant: str,
    n_gpu: int,
    tbt_target_ms: float,
    strict: bool,
) -> bool:
    """Consult the calc for KV-fit + roofline regime hints.

    Returns True if the run should proceed (advisory mode always returns
    True after printing; strict returns False on error-level warnings).
    """
    weight_prec, kv_prec, act_prec = _precision_from_quant(quant)
    # ISL/OSL for pre-flight default to the lab's chatbot medians since
    # we don't yet know what the workload will actually produce. Used
    # only to populate compute(); the real prediction snapshot at
    # submit-result time uses measured values.
    pf_inputs = CalcInputs(
        model_key=model_ref,
        hw_key=hw_ref,
        weight_prec=weight_prec,
        kv_prec=kv_prec,
        act_prec=act_prec,
        isl=200, osl=150,
        ngpus=n_gpu, tbt_ms=tbt_target_ms,
    )
    pred = bridge.predict(pf_inputs)
    if pred is None:
        click.echo("pre-flight: calc bridge unavailable; skipping.", err=True)
        return True  # not a failure — degraded mode

    click.echo(
        f"pre-flight: b_crit={pred.b_crit} b_slo={pred.b_slo} b_kv={pred.b_kv} "
        f"y_max={pred.y_max} recommended_batch={pred.recommended_batch}"
    )
    errors = [w for w in pred.warnings if w.get("level") == "error"]
    warns = [w for w in pred.warnings if w.get("level") == "warn"]
    for w in pred.warnings:
        click.echo(f"  [{w.get('level')}] {w.get('msg')}", err=True)
    if errors:
        if strict:
            click.echo("pre-flight: errors found, aborting (strict mode).", err=True)
            return False
        click.echo("pre-flight: errors above are advisory; run continues.", err=True)
    if warns and strict:
        click.echo("pre-flight: warnings found (strict mode).", err=True)
        # Warnings don't block even in strict mode — only errors do.
    return True


@cli.command("list")
@click.option("--results-dir", type=click.Path(path_type=Path), default=Path("results"))
def list_cmd(results_dir: Path) -> None:
    """List runs and their statuses."""
    store = ManifestStore(results_dir / "manifests")
    for m in store.list_all():
        click.echo(
            f"{m.status.value:8s}  {m.run_id}  engine={m.engine_name} workload={m.workload_name}"
        )


@cli.command("build-portal")
@click.option("--results-dir", type=click.Path(path_type=Path), default=Path("results"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("_site"))
@click.option(
    "--calc-bridge",
    type=click.Path(path_type=Path),
    default=None,
    help="also write a calc-side bridge directory (index.json + runs/*.json) to this path",
)
def build_portal_cmd(results_dir: Path, out: Path, calc_bridge: Path | None) -> None:
    """Build the static portal pages."""
    _do_build_portal(results_dir=results_dir, out=out, calc_bridge=calc_bridge)


def _do_build_portal(*, results_dir: Path, out: Path, calc_bridge: Path | None) -> None:
    build_portal(BuildInputs(results_dir=results_dir, out_dir=out, calc_bridge=calc_bridge))
    click.echo(f"built {out}")
    if calc_bridge:
        click.echo(f"calc bridge → {calc_bridge}")


def _default_calc_bridge(results_dir: Path) -> Path:
    """Resolve the default ``--calc-bridge`` path: a sibling
    ``calculators/sizing_calc/lab_runs`` directory of the experiments
    package. This matches where the calc's Validation panel looks.
    """
    # results_dir is typically `experiments/results`. The repo layout
    # places `calculators/` as a sibling of `experiments/`.
    repo_root = results_dir.resolve().parent.parent
    return repo_root / "calculators" / "sizing_calc" / "lab_runs"


def _port_in_use(host: str, port: int) -> bool:
    """Return True if something is already listening on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0


def _start_portal_server(portal_dir: Path, port: int) -> ThreadingHTTPServer:
    """Spin up a static file server rooted at ``portal_dir``.

    Returns the server (caller is responsible for ``serve_forever`` or
    ``shutdown``). Bound to 127.0.0.1 — this is dev ergonomics, not a
    deployment target.
    """
    handler = partial(SimpleHTTPRequestHandler, directory=str(portal_dir))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    return server


@cli.command("launch")
@_run_options
@click.option("--portal-dir", type=click.Path(path_type=Path), default=Path("_site"),
              help="where to build the static portal")
@click.option("--calc-bridge", type=click.Path(path_type=Path), default=None,
              help="bridge dir for the calc's Validation panel (default: ../calculators/sizing_calc/lab_runs)")
@click.option("--serve/--no-serve", default=True,
              help="start a local http.server so the browser can load the portal (default on)")
@click.option("--open/--no-open", "open_browser", default=True,
              help="open the default browser to the result URL (default on)")
@click.option("--port", type=int, default=8765,
              help="port for the static server; auto-bumps to next free if taken")
def launch_cmd(
    portal_dir: Path,
    calc_bridge: Path | None,
    serve: bool,
    open_browser: bool,
    port: int,
    **run_kwargs,
) -> None:
    """One-shot: run an experiment, rebuild the portal, open the browser.

    Designed for the pasted command coming out of the sizing calculator's
    ``[ COPY EXP RUN ]`` button — everything from launch to "tab open on
    measured-vs-predicted overlay" in a single invocation.
    """
    results_dir: Path = run_kwargs["results_dir"]
    # 1. Run the experiment (blocks until result JSON is written).
    result_path = _do_run(**run_kwargs)
    run_id = result_path.stem
    click.echo(f"wrote {result_path}")

    # 2. Rebuild portal + bridge dir. Default bridge path is the calc's
    # `lab_runs/` sibling unless the caller overrode it.
    bridge = calc_bridge if calc_bridge is not None else _default_calc_bridge(results_dir)
    _do_build_portal(results_dir=results_dir, out=portal_dir, calc_bridge=bridge)

    # 3. Pick a port. If our preferred port is in use, assume an existing
    # `exp serve` is already up and reuse it (don't spawn a competing server).
    reused = False
    chosen_port = port
    if serve and _port_in_use("127.0.0.1", chosen_port):
        reused = True
        click.echo(f"port {chosen_port} already in use — reusing existing server")

    url = f"http://127.0.0.1:{chosen_port}/results_explorer.html#{run_id}"
    click.echo(f"open: {url}")

    if open_browser:
        webbrowser.open(url)

    if serve and not reused:
        server = _start_portal_server(portal_dir, chosen_port)
        click.echo(f"serving {portal_dir} at http://127.0.0.1:{chosen_port}/ (Ctrl-C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            click.echo("stopping server")
        finally:
            server.shutdown()
            server.server_close()


@cli.command("serve")
@click.option("--results-dir", type=click.Path(path_type=Path), default=Path("results"))
@click.option("--portal-dir", type=click.Path(path_type=Path), default=Path("_site"))
@click.option("--calc-bridge", type=click.Path(path_type=Path), default=None,
              help="bridge dir for the calc's Validation panel (default: ../calculators/sizing_calc/lab_runs)")
@click.option("--port", type=int, default=8765)
@click.option("--watch/--no-watch", default=True,
              help="rebuild portal whenever a result/bridge file changes (requires watchdog)")
def serve_cmd(
    results_dir: Path,
    portal_dir: Path,
    calc_bridge: Path | None,
    port: int,
    watch: bool,
) -> None:
    """Long-running static server + rebuild-on-change.

    Run this in a separate terminal once per session. Subsequent
    `exp launch` invocations will detect the running server and reuse
    it instead of spawning a competing one on the same port.
    """
    bridge = calc_bridge if calc_bridge is not None else _default_calc_bridge(results_dir)

    # Build once up front so the portal is browsable before any new run.
    _do_build_portal(results_dir=results_dir, out=portal_dir, calc_bridge=bridge)

    observer = None
    if watch:
        observer = _try_start_watcher(
            results_dir=results_dir, portal_dir=portal_dir, bridge=bridge
        )

    server = _start_portal_server(portal_dir, port)
    click.echo(f"serving {portal_dir} at http://127.0.0.1:{port}/ (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("stopping server")
    finally:
        server.shutdown()
        if observer is not None:
            observer.stop()
            observer.join(timeout=2.0)


def _try_start_watcher(
    *, results_dir: Path, portal_dir: Path, bridge: Path | None
):
    """Start a watchdog Observer that rebuilds the portal on changes.

    Returns the Observer (already started) or ``None`` if watchdog is
    not installed — that's an optional dev dep, so missing it degrades
    to "static serve, no auto-rebuild" rather than aborting.
    """
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        click.echo(
            "watchdog not installed — running without auto-rebuild. "
            "Install with `pip install 'inference-lab[dev]'` to enable.",
            err=True,
        )
        return None

    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Debounce: filesystem events arrive in bursts (e.g. write + flush +
    # rename). Coalesce anything within 200ms into one rebuild.
    debounce_lock = threading.Lock()
    pending = {"timer": None}

    def rebuild() -> None:
        with debounce_lock:
            pending["timer"] = None
        try:
            _do_build_portal(results_dir=results_dir, out=portal_dir, calc_bridge=bridge)
        except Exception as exc:  # noqa: BLE001 — keep the watcher alive past one bad rebuild
            click.echo(f"rebuild failed: {exc}", err=True)

    class _Handler(FileSystemEventHandler):
        def on_any_event(self, event) -> None:  # type: ignore[override]
            if event.is_directory:
                return
            with debounce_lock:
                if pending["timer"] is not None:
                    pending["timer"].cancel()
                t = threading.Timer(0.2, rebuild)
                t.daemon = True
                pending["timer"] = t
                t.start()

    observer = Observer()
    observer.schedule(_Handler(), str(runs_dir), recursive=False)
    observer.start()
    click.echo(f"watching {runs_dir} for changes")
    return observer


@cli.command("plan")
@click.option("--model-ref", required=True, help="models.json key, e.g. 'llama-3-70b'")
@click.option("--hw-ref", required=True, help="hardware.json key, e.g. 'A10G'")
@click.option("--quant", default="awq")
@click.option("--n-gpu", type=int, default=4)
@click.option("--isl", type=int, default=200)
@click.option("--osl", type=int, default=150)
@click.option("--tbt-target-ms", type=float, default=50.0)
@click.option("--rows", type=int, default=5, help="number of grid rows to emit")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="write JSON to this path; defaults to stdout")
def plan_cmd(
    model_ref: str,
    hw_ref: str,
    quant: str,
    n_gpu: int,
    isl: int,
    osl: int,
    tbt_target_ms: float,
    rows: int,
    out: Path | None,
) -> None:
    """Emit a suggested run grid centered on the calc's b_crit.

    Output rows are JSON objects with ``batch_target``, ``rate_rps``,
    and ``predicted_tps``. Use ``exp run --plan <file>`` (TODO) or
    iterate them manually.
    """
    weight_prec, kv_prec, act_prec = _precision_from_quant(quant)
    bridge = CalcBridge()
    pred = bridge.predict(CalcInputs(
        model_key=model_ref, hw_key=hw_ref,
        weight_prec=weight_prec, kv_prec=kv_prec, act_prec=act_prec,
        isl=isl, osl=osl, ngpus=n_gpu, tbt_ms=tbt_target_ms,
    ))
    if pred is None:
        raise click.ClickException(
            "calc bridge unavailable; cannot plan. Build "
            "calculators/sizing_calc/predictions/grid.json or install Node."
        )
    grid = _build_run_grid(pred, rows=rows, tbt_ms=tbt_target_ms)
    blob = json.dumps(grid, indent=2)
    if out is None:
        click.echo(blob)
    else:
        out.write_text(blob + "\n")
        click.echo(f"wrote {out}")


def _build_run_grid(pred: Prediction, *, rows: int, tbt_ms: float) -> list[dict[str, float]]:
    """Pick `rows` batch targets in geometric steps around b_crit.

    The lab's load driver controls *arrival rate*, not batch. So we map
    each batch target to a rate via Little's Law: rate ≈ batch / tbt_s.
    The result is an advisory grid: at this rate, you should observe a
    steady-state batch near the target.
    """
    if not pred.curve:
        return []
    # Center on recommended_batch when available, else b_crit, else the
    # middle of the curve.
    center = (
        pred.recommended_batch
        or pred.b_crit
        or pred.curve[len(pred.curve) // 2].batch
    )
    center = max(1.0, float(center))
    # Geometric span: factor 4× either side, log-spaced.
    lo = max(1.0, center / 4.0)
    hi = center * 4.0
    tbt_s = max(tbt_ms / 1000.0, 1e-3)
    batches: list[int] = []
    for i in range(rows):
        # Log spacing in [lo, hi].
        frac = i / max(rows - 1, 1)
        b = lo * (hi / lo) ** frac
        batches.append(max(1, int(round(b))))
    # Look up predicted tps at each batch (nearest point on the curve).
    out: list[dict[str, float]] = []
    for b in batches:
        p = min(pred.curve, key=lambda pt: abs(pt.batch - b))
        out.append({
            "batch_target": float(b),
            "rate_rps": round(b / tbt_s, 3),
            "predicted_tps": round(p.tps, 2),
            "predicted_step_ms": round(p.step_ms, 3),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    try:
        cli.main(args=argv, standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
