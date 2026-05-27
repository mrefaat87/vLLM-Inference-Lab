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
import sys
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


@cli.command("run")
@click.option("--engine", type=click.Choice(list(DRIVERS)), required=True)
@click.option(
    "--workload",
    type=click.Choice(["chatbot", "agentic_coding", "rag", "multi_turn", "mix"]),
    required=True,
)
@click.option("--rate", type=float, default=8.0, help="target arrival RPS")
@click.option("--duration", type=float, default=300.0, help="run duration (s)")
@click.option("--warmup", type=float, default=10.0)
@click.option("--seed", type=int, default=1)
@click.option("--model", default="meta-llama/Llama-3-70B-Instruct-AWQ")
@click.option("--quant", default="awq")
@click.option("--tp", type=int, default=4)
@click.option("--instance", default="g5.12xlarge")
@click.option("--gpu", default="A10G")
@click.option("--n-gpu", type=int, default=4)
@click.option(
    "--model-ref",
    default="llama-3-70b",
    help="join key into calculators/sizing_calc/src/data/models.json (the model's `key` field)",
)
@click.option(
    "--hw-ref",
    default="A10G",
    help="join key into calculators/sizing_calc/src/data/hardware.json (the hardware's `key` field)",
)
@click.option("--tbt-target-ms", type=float, default=50.0,
              help="target inter-token latency (used by the calc bridge for predicted curves)")
@click.option(
    "--preflight",
    type=click.Choice(["off", "advisory", "strict"]),
    default="advisory",
    help="consult the sizing calculator before launching: advisory (default) prints warnings, strict exits non-zero",
)
@click.option("--results-dir", type=click.Path(path_type=Path), default=Path("results"))
@click.option("--notes", default=None)
def run_cmd(
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
) -> None:
    """Run one (engine, workload, rate) experiment."""
    driver_cls = DRIVERS[engine]
    driver = driver_cls()
    engine_cfg = EngineConfig(
        name=engine,
        image=ENGINE_IMAGES[engine],
        model=model,
        quantization=quant,
        tensor_parallel=tp,
        instance=instance,
        gpu=gpu,
        n_gpu=n_gpu,
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

    path = asyncio.run(_go())
    click.echo(f"wrote {path}")


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
    build_portal(BuildInputs(results_dir=results_dir, out_dir=out, calc_bridge=calc_bridge))
    click.echo(f"built {out}")
    if calc_bridge:
        click.echo(f"calc bridge → {calc_bridge}")


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
