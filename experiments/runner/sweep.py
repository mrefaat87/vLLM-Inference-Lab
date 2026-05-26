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

from experiments.drivers.base import EngineDriver
from experiments.runner.analysis import analyze
from experiments.runner.calc_bridge import CalcBridge, CalcInputs
from experiments.runner.driver_loop import LoopConfig, run_loop
from experiments.runner.manifest_store import ManifestStore
from experiments.runner.schema import (
    EngineConfig,
    HardwareSpec,
    ModelSpec,
    Prediction,
    RooflineLink,
    RunManifest,
    RunResult,
    RunStatus,
    WorkloadConfig,
)
from experiments.workloads.base import WorkloadGenerator

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
        workload: WorkloadGenerator,
        workload_cfg: WorkloadConfig,
        model: ModelSpec,
        hardware: HardwareSpec,
        roofline_link: RooflineLink,
        run_id: str | None = None,
        notes: str | None = None,
        tbt_target_ms: float = 50.0,
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

            t0 = time.monotonic()
            started_at = datetime.now(timezone.utc)
            loop_cfg = LoopConfig(endpoint=endpoint, model=engine_cfg.model)
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
        )
        return self._calc_bridge.predict(inputs)

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
