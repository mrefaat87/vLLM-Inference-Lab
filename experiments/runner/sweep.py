"""Sweep runner.

A single 'run' = (engine, workload, rate, seed). The sweep runner orchestrates
N runs sequentially, owning the manifest lifecycle and result persistence.
Concurrency across runs is intentionally NOT supported — engines fight for
GPU, results would be apples-to-bananas, and the manifest concept assumes
serial execution.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from collections.abc import Callable

import click

from experiments.drivers.base import EngineDriver
from experiments.runner.analysis import analyze
from experiments.runner.calc_bridge import CalcBridge, CalcInputs
from experiments.runner.calibration import (
    DEFAULT_START_RATE,
    DEFAULT_TTFT_SLO_MS,
    PROBE_S,
    PROBE_WALL_CLOCK_FACTOR,
    calibrate,
    explicit_calibration,
)
from experiments.runner.driver_loop import LoopConfig, run_loop
from experiments.runner.manifest_store import ManifestStore
from experiments.runner.schema import (
    Calibration,
    EngineConfig,
    HardwareSpec,
    ModelSpec,
    Prediction,
    RequestRecord,
    RooflineLink,
    RunManifest,
    RunResult,
    RunStatus,
    WorkloadConfig,
)
from experiments.workloads.base import Request, WorkloadGenerator

# Quantization → calc precision-tuple. Anything not in the table falls
# back to BF16 and emits a warning on the resulting Prediction so the
# user knows the prediction shape is approximate for that engine config.
_QUANT_TO_PRECISION: dict[str, tuple[str, str, str]] = {
    "none":       ("BF16", "FP16", "BF16"),
    "fp16":       ("BF16", "FP16", "BF16"),
    "bf16":       ("BF16", "FP16", "BF16"),
    "fp8":        ("FP8",  "FP16", "FP8"),
    "awq":        ("INT4", "FP16", "BF16"),
    "awq-int4":   ("INT4", "FP16", "BF16"),
    "gptq":       ("INT4", "FP16", "BF16"),
    "gptq-int4":  ("INT4", "FP16", "BF16"),
    "int8":       ("INT8", "FP16", "BF16"),
}


def _precision_from_quant(quant: str) -> tuple[str, str, str]:
    return _QUANT_TO_PRECISION.get(quant.lower(), ("BF16", "FP16", "BF16"))


# Warmup constants. vLLM captures CUDA graphs lazily per-batch-size: the
# first request at batch=1 triggers a ~30s compile, batch=2 triggers another,
# etc. A single Poisson-sampled request only warms batch=1, so the first
# probe at rate>1 still pays the compile tax mid-probe and false-positives
# saturation. Instead we dispatch _WARMUP_CONCURRENCY requests all at
# arrival_offset_s=0 so vLLM compiles the batch={1..N} graphs the probe
# sweep will exercise. Records are discarded.
_WARMUP_CONCURRENCY: int = 8
_WARMUP_PROMPT_TOKENS: int = 64
_WARMUP_MAX_NEW_TOKENS: int = 32


async def _warmup_requests():
    """Yield ``_WARMUP_CONCURRENCY`` requests all at arrival_offset_s=0.

    Forces vLLM to compile CUDA graphs for batch={1..N} before the probe
    sweep starts. Prompt + OSL are kept small so warmup stays fast.
    """
    prompt = "say hi"
    for i in range(_WARMUP_CONCURRENCY):
        yield Request(
            request_id=f"warmup-{i}",
            prompt=prompt,
            prompt_tokens=_WARMUP_PROMPT_TOKENS,
            max_new_tokens=_WARMUP_MAX_NEW_TOKENS,
            arrival_offset_s=0.0,
            label="warmup",
        )


class SweepRunner:
    """Single-run orchestrator.

    Despite the name, today this is a 1-run executor. The cartesian product
    of (engines × workloads × rates) is exposed via ``run_many``.
    """

    def __init__(
        self,
        *,
        results_dir: Path | str,
        manifest_store: ManifestStore | None = None,
        calc_bridge: CalcBridge | None = None,
    ) -> None:
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._manifests = manifest_store or ManifestStore(self._results_dir / "manifests")
        # None disables prediction capture entirely; otherwise the bridge
        # is consulted once per run after analysis (so ISL/OSL reflect what
        # the workload actually produced).
        self._calc_bridge = calc_bridge

    async def run_one(
        self,
        *,
        engine: EngineDriver,
        engine_cfg: EngineConfig,
        workload: WorkloadGenerator | None = None,
        workload_cfg: WorkloadConfig,
        model: ModelSpec,
        hardware: HardwareSpec,
        roofline_link: RooflineLink,
        run_id: str | None = None,
        notes: str | None = None,
        tbt_target_ms: float = 50.0,
        ttft_slo_ms: float = DEFAULT_TTFT_SLO_MS,
        workload_factory: Callable[[float, int], WorkloadGenerator] | None = None,
        probe_start_rate: float = DEFAULT_START_RATE,
    ) -> Path:
        """Execute one run end-to-end and return the result JSON path."""
        run_id = run_id or _new_run_id(engine_cfg.name, workload_cfg.name)
        manifest = RunManifest(
            run_id=run_id,
            status=RunStatus.PLANNED,
            engine_name=engine_cfg.name,
            workload_name=workload_cfg.name,
        )
        self._manifests.write(manifest)

        try:
            endpoint = engine.start(engine_cfg)
            # Wait for readiness (up to 5 min). Mock comes up in <100ms; real
            # engines may take much longer.
            self._await_ready(engine, timeout_s=300.0)
            self._manifests.mark_running(run_id, log_path=None)

            loop_cfg = LoopConfig(endpoint=endpoint, model=engine_cfg.model)

            # Calibration phase: when a workload_factory is supplied we probe
            # the engine to find its real capacity ceiling and override the
            # placeholder rate_rps on workload_cfg. Explicit-rate runs (caller
            # passes a constructed workload) bypass the probe sweep and just
            # record the explicit rate so downstream readers have a uniform
            # shape.
            if workload_factory is not None:
                calibration_block = await self._calibrate(
                    loop_cfg=loop_cfg,
                    workload_factory=workload_factory,
                    workload_cfg=workload_cfg,
                    ttft_slo_ms=ttft_slo_ms,
                    probe_start_rate=probe_start_rate,
                )
                workload_cfg = workload_cfg.model_copy(
                    update={"rate_rps": calibration_block.selected_rate}
                )
                workload = workload_factory(
                    calibration_block.selected_rate, workload_cfg.seed
                )
            else:
                if workload is None:
                    raise ValueError(
                        "run_one requires either workload or workload_factory"
                    )
                calibration_block = explicit_calibration(workload_cfg.rate_rps)

            t0 = time.monotonic()
            started_at = datetime.now(timezone.utc)
            records = await run_loop(
                workload.requests(duration_s=workload_cfg.duration_s),
                loop_cfg,
                t0=t0,
            )
            finished_at = datetime.now(timezone.utc)
            analysis = analyze(
                records,
                duration_s=workload_cfg.duration_s,
                warmup_s=workload_cfg.warmup_s,
            )
            prediction = self._snapshot_prediction(
                roofline_link=roofline_link,
                model=model,
                hardware=hardware,
                engine_cfg=engine_cfg,
                analysis_isl=analysis.isl_tokens_p50,
                analysis_osl=analysis.osl_tokens_p50,
                tbt_target_ms=tbt_target_ms,
            )
            result = RunResult(
                run_id=run_id,
                started_at=started_at,
                finished_at=finished_at,
                engine=engine_cfg,
                model=model,
                hardware=hardware,
                workload=workload_cfg,
                analysis=analysis,
                roofline_link=roofline_link,
                prediction=prediction,
                calibration=calibration_block,
                raw_results=records,
                engine_metrics=engine.metrics(),
                notes=notes,
            )
            path = self._write_result(result)
            self._manifests.mark_done(run_id, result_path=str(path))
            return path
        except Exception as exc:  # noqa: BLE001 — top-level orchestrator catches everything
            self._manifests.mark_failed(run_id, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            engine.stop()

    def _snapshot_prediction(
        self,
        *,
        roofline_link: RooflineLink,
        model: ModelSpec,
        hardware: HardwareSpec,
        engine_cfg: EngineConfig,
        analysis_isl: int | None,
        analysis_osl: int | None,
        tbt_target_ms: float,
    ) -> Prediction | None:
        """Best-effort capture of the calc's prediction for this run.

        Uses measured ISL/OSL p50 so the prediction reflects what the
        engine actually saw, not the workload's nominal medians. Returns
        None on any bridge failure — the lab continues with
        `prediction: null` and a notes-line documenting the gap.
        """
        if self._calc_bridge is None:
            return None
        if analysis_isl is None or analysis_osl is None:
            return None
        weight_prec, kv_prec, act_prec = _precision_from_quant(engine_cfg.quantization)
        # Look up the calc's default $/hr for this hw row and pass it through
        # so Prediction.inputs.price_per_hour_usd is non-null. The Results
        # Explorer uses this to compute measured cost-per-Mtok without having
        # to re-load the hardware.json client-side. None means clone-only-lab
        # (calc data unavailable) — cost panels then degrade to predicted-only.
        price = self._calc_bridge.lookup_hw_price(roofline_link.hw_ref)
        inputs = CalcInputs(
            model_key=roofline_link.model_ref,
            hw_key=roofline_link.hw_ref,
            weight_prec=weight_prec,
            kv_prec=kv_prec,
            act_prec=act_prec,
            isl=int(analysis_isl),
            osl=int(analysis_osl),
            ngpus=hardware.n_gpu,
            tbt_ms=float(tbt_target_ms),
            price_per_hour_usd=price,
        )
        return self._calc_bridge.predict(inputs)

    async def _calibrate(
        self,
        *,
        loop_cfg: LoopConfig,
        workload_factory: Callable[[float, int], WorkloadGenerator],
        workload_cfg: WorkloadConfig,
        ttft_slo_ms: float,
        probe_start_rate: float = DEFAULT_START_RATE,
    ) -> Calibration:
        """Run the probe sweep and emit a stdout summary.

        Each probe builds a fresh workload via the factory (so the
        calibration-phase RNG/conversation state is isolated from the
        measurement window).
        """
        # Warmup phase: /health 200 only proves the HTTP server is up — it
        # does NOT prove CUDA graphs are captured or the KV cache is warm.
        # vLLM captures CUDA graphs lazily per-batch-size: a single warmup
        # request only compiles batch=1, leaving probe(1) at rate=2 to pay
        # the batch=2 compile tax mid-probe (~30s) and false-positive
        # saturation. Dispatch _WARMUP_CONCURRENCY requests all at offset 0
        # so vLLM compiles batch={1..N} up front. No wall-clock cap — we
        # wait for all warmup requests to finish, then discard the records.
        click.echo(
            "calibration: warmup (8 concurrent, compiling CUDA graphs) ...",
            err=True,
        )
        warmup_t0 = time.monotonic()
        await run_loop(
            _warmup_requests(),
            loop_cfg,
            t0=warmup_t0,
            wall_clock_cap_s=None,  # explicit: no cap during warmup
        )

        async def _probe(rate: float, seed: int) -> list[RequestRecord]:
            # Each burst gets its own t0 so its arrival schedule starts
            # at offset 0 (driver_loop sleeps until req.arrival_offset_s).
            # The wall-clock cap bounds the probe at PROBE_S × factor so
            # slow engines (e.g. AWQ-on-T4, residence ~12s) don't burn the
            # full budget on three probes' worth of drain time.
            probe_workload = workload_factory(rate, seed)
            t0_probe = time.monotonic()
            return await run_loop(
                probe_workload.requests(duration_s=PROBE_S),
                loop_cfg,
                t0=t0_probe,
                wall_clock_cap_s=PROBE_S * PROBE_WALL_CLOCK_FACTOR,
            )

        result = await calibrate(
            probe_fn=_probe,
            ttft_slo_ms=ttft_slo_ms,
            start_rate=probe_start_rate,
        )
        # Echo to stdout so paste-and-go users see what rate was picked.
        probe_summary = ", ".join(
            f"{p.rate:g}={'sat' if p.saturated else 'ok'}" for p in result.probes
        )
        click.echo(
            f"rate auto → calibrated to {result.selected_rate:.2f} rps "
            f"(ceiling ~{result.capacity_ceiling:.2f} rps; probes: {probe_summary})",
            err=True,
        )
        return result

    def _await_ready(self, engine: EngineDriver, *, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if engine.healthcheck():
                return
            time.sleep(0.1)
        raise TimeoutError("engine never became healthy")

    def _write_result(self, result: RunResult) -> Path:
        out_dir = self._results_dir / "runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{result.run_id}.json"
        path.write_text(result.model_dump_json(indent=2))
        return path

    # Listing helpers (used by the portal builder and the CLI).
    def list_manifests(self) -> list[RunManifest]:
        return self._manifests.list_all()


def _new_run_id(engine: str, workload: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:6]
    return f"{ts}-{engine}-{workload}-{suffix}"


def load_result(path: Path | str) -> RunResult:
    """Convenience loader; the portal builder uses this."""
    return RunResult.model_validate(json.loads(Path(path).read_text()))
