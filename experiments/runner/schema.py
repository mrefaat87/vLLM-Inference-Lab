"""Result and manifest schemas.

These pydantic models are the single source of truth for what an experiment
result and a run manifest look like on disk. The portal and any external
consumer should rely only on these shapes.

The ``schema_version`` field on ``RunResult`` is bumped whenever a
breaking change is made to any of the nested fields. Additive changes
(new optional fields) do not bump the version.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Semver-like schema version on the result envelope.
#   MAJOR: breaking change (rename / remove / type-change of an existing field).
#   MINOR: additive change (new optional field) — bump so old consumers can
#          detect the format and so we can write forward-compat tests against
#          a frozen snapshot of each minor.
#   PATCH: docs-only / no field change — do not bump.
#
# 1.0.0 → 1.1.0: added optional Prediction block and Analysis.isl_tokens_p50
# / osl_tokens_p50 percentiles. Old v1.0.0 files load cleanly under this
# schema; new v1.1.0 files MAY load in a v1.0.0 consumer that uses
# extra="ignore" on the envelope (see RunResult below).
# 1.1.0 → 1.2.0: added optional Calibration block recording the rate-probe
# sweep run before measurement. Additive — old files load cleanly (calibration
# is None); old consumers ignore the field via _Envelope.
SCHEMA_VERSION = "1.2.0"


class _Strict(BaseModel):
    """Base model: extra fields rejected, frozen for safety."""

    model_config = ConfigDict(extra="forbid", frozen=False, str_strip_whitespace=True)


class _Envelope(BaseModel):
    """Top-level envelopes that must stay forward-compatible.

    Unknown fields are *ignored* rather than rejected so an old consumer
    can read a future-minor file. Use ``_Strict`` for inner / private
    models where typo-catching matters more than forward compat.
    """

    model_config = ConfigDict(extra="ignore", frozen=False, str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Status enum (shared between manifest and result analysis)
# ---------------------------------------------------------------------------
class RunStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
class EngineConfig(_Strict):
    """Engine-launch configuration.

    Driver implementations consume this; the runner persists it inside
    the result so the run can be reproduced from a single JSON file.
    """

    name: str = Field(..., description="vllm | sglang | trtllm | mock")
    image: str = Field(..., description="container image reference")
    model: str = Field(..., description="HF repo or local path, e.g. meta-llama/Llama-3-70B-Instruct")
    quantization: str = Field(
        default="none",
        description="none | awq-int4 | gptq-int4 | fp8 | int8",
    )
    tensor_parallel: int = Field(default=1, ge=1, le=16)
    max_model_len: int = Field(default=8192, ge=128, le=1_000_000)
    # Cap on concurrent sequences vLLM schedules into a single forward pass.
    # Default 256 mirrors vLLM 0.7.x's built-in default; the calc's
    # recommended_batch overrides this when the snippet is pasted, so the
    # measured run actually probes the calc-recommended operating point
    # rather than running at a silent default. SGLang ignores this field
    # (it uses --max-running-requests) — wiring is engine-specific.
    max_num_seqs: int = Field(
        default=256, ge=1, le=4096,
        description="vLLM --max-num-seqs; defaults to vLLM 0.7.x built-in",
    )
    # Karpenter scheduling hints. The K8s driver translates these to
    # node-affinity selectors so a request for g4dn.xlarge does not get
    # silently rescheduled onto a g4dn.12xlarge by the NodePool.
    instance: str | None = Field(
        default=None,
        description="AWS EC2 instance type, e.g. g5.12xlarge. Drives Karpenter node affinity.",
    )
    gpu: str | None = Field(
        default=None,
        description="GPU model name, e.g. A10G. Used for resource sizing and observability.",
    )
    n_gpu: int | None = Field(
        default=None, ge=1, le=16, description="number of GPUs requested on the node",
    )
    extra_args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_known(cls, v: str) -> str:
        # We are intentionally permissive — new engines should not require
        # a schema edit. But we reject empty / whitespace.
        if not v.strip():
            raise ValueError("engine name must be non-empty")
        return v


class WorkloadConfig(_Strict):
    name: str = Field(..., description="chatbot | agentic_coding | mix | <custom>")
    rate_rps: float = Field(..., gt=0.0, description="target arrival rate, requests/sec")
    duration_s: float = Field(..., gt=0.0)
    warmup_s: float = Field(default=10.0, ge=0.0)
    drain_s: float = Field(default=5.0, ge=0.0)
    seed: int = Field(default=1, ge=0)
    params: dict[str, Any] = Field(default_factory=dict)


class HardwareSpec(_Strict):
    instance: str = Field(..., description="e.g. g5.12xlarge")
    gpu: str = Field(..., description="e.g. A10G, H100")
    n_gpu: int = Field(..., ge=1, le=16)


class ModelSpec(_Strict):
    name: str
    quant: str = Field(default="none")
    tp: int = Field(default=1, ge=1, le=16)


class RooflineLink(_Strict):
    """Join key into the sizing-calculator's data files.

    The portal uses this to look up the predicted curve and overlay
    the empirical point on top.
    """

    model_ref: str = Field(..., description="id from calculators/sizing_calc/src/data/models.json")
    hw_ref: str = Field(..., description="id from calculators/sizing_calc/src/data/hardware.json")


# ---------------------------------------------------------------------------
# Per-request raw record
# ---------------------------------------------------------------------------
class RequestRecord(_Strict):
    request_id: str
    label: str = Field(..., description="workload-defined label, e.g. 'chatbot.turn1'")
    submit_offset_s: float = Field(..., ge=0.0)
    prompt_tokens: int = Field(..., ge=0)
    max_new_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    ttft_s: float | None = Field(default=None, description="time to first token")
    tbt_p50_s: float | None = Field(default=None, description="median inter-token latency")
    end_to_end_s: float | None = None
    error: str | None = None
    conversation_id: str | None = None


# ---------------------------------------------------------------------------
# Analysis (summary stats)
# ---------------------------------------------------------------------------
class Percentiles(_Strict):
    p50: float
    p95: float
    p99: float
    mean: float
    n: int = Field(..., ge=0)


class ThroughputStats(_Strict):
    total_completion_tokens: int = Field(..., ge=0)
    total_prompt_tokens: int = Field(..., ge=0)
    tok_per_sec_avg: float = Field(..., ge=0.0)
    requests_per_sec_avg: float = Field(..., ge=0.0)


class Analysis(_Strict):
    steady_state_requests: int = Field(..., ge=0)
    failed_requests: int = Field(..., ge=0)
    ttft_s: Percentiles | None = None
    tbt_s: Percentiles | None = None
    e2e_s: Percentiles | None = None
    throughput: ThroughputStats
    gpu_util_pct_avg: float | None = Field(default=None, ge=0.0, le=100.0)
    kv_used_frac_p95: float | None = Field(default=None, ge=0.0, le=1.0)
    # Median input / output sequence length in tokens, used by the
    # calculator overlay to default its ISL/OSL sliders so the predicted
    # curve reflects the actual run shape (see Prediction.inputs.isl/osl).
    isl_tokens_p50: int | None = Field(default=None, ge=0)
    osl_tokens_p50: int | None = Field(default=None, ge=0)


# ---------------------------------------------------------------------------
# Prediction block (Flow A: calc → lab)
# ---------------------------------------------------------------------------
class PredictionInputs(_Strict):
    """The inputs the calc's compute() was called with for this run.

    Mirrors `calc.mjs::compute(input)` argument shape — kept loose-typed
    (model/hw are dict-like) so an old lab can ingest a future calc.
    """

    model_key: str
    hw_key: str
    weight_prec: str
    kv_prec: str
    act_prec: str
    isl: int = Field(..., ge=1)
    osl: int = Field(..., ge=1)
    ngpus: int = Field(..., ge=1)
    tbt_ms: float = Field(..., gt=0.0)
    price_per_hour_usd: float | None = None


class PredictionCurvePoint(_Strict):
    batch: int = Field(..., ge=1)
    step_ms: float = Field(..., ge=0.0)
    tps: float = Field(..., ge=0.0)
    cost_per_mtok: float | None = None


class Prediction(_Envelope):
    """Snapshot of the sizing-calculator's compute() output at run time.

    Lives inside RunResult so every measured run carries its own
    falsifiable hypothesis. The portal renders the predicted curve from
    this block directly — no calc.mjs at view time.
    """

    calc_version: str = Field(..., description="git SHA or version string of the calc source")
    data_hash: str = Field(..., description="sha256 of models.json + hardware.json")
    inputs: PredictionInputs
    b_crit: float | None = None
    b_slo: float | None = None
    b_kv: float | None = None
    recommended_batch: float | None = None
    y_max: int | None = None
    curve: list[PredictionCurvePoint] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    # Set when the bridge can't reach the calc — the lab still writes a
    # row so divergence between "we tried" and "no calc available" is
    # observable.
    unavailable_reason: str | None = None


# ---------------------------------------------------------------------------
# Calibration block (rate probe sweep)
# ---------------------------------------------------------------------------
class CalibrationProbe(_Strict):
    rate: float = Field(..., gt=0.0, description="probe arrival rate, requests/sec")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="completed / dispatched")
    ttft_p95_ms: float = Field(..., ge=0.0)
    achieved_rps: float = Field(..., ge=0.0, description="completed / probe duration")
    saturated: bool


class Calibration(_Envelope):
    """Outcome of the auto-rate probe sweep.

    Always written so downstream readers have a uniform shape:
      - method="auto":     probes is non-empty, selected_rate = 0.8 × ceiling.
      - method="explicit": probes is empty, selected_rate == ceiling == --rate.
    """

    method: Literal["auto", "explicit"]
    probes: list[CalibrationProbe] = Field(default_factory=list)
    selected_rate: float = Field(..., gt=0.0)
    capacity_ceiling: float = Field(..., gt=0.0)


# ---------------------------------------------------------------------------
# Top-level run result
# ---------------------------------------------------------------------------
class RunResult(_Envelope):
    schema_version: str = Field(default=SCHEMA_VERSION)
    run_id: str
    started_at: datetime
    finished_at: datetime
    engine: EngineConfig
    model: ModelSpec
    hardware: HardwareSpec
    workload: WorkloadConfig
    analysis: Analysis
    roofline_link: RooflineLink
    # Optional: present iff the calc bridge was reachable at run time.
    # Use _Envelope semantics (extra="ignore") on the envelope so a v1.0.0
    # consumer can still load v1.1.0 files — the `prediction` field is
    # ignored rather than rejected on the older schema.
    prediction: Prediction | None = None
    # Optional: present on runs from v1.2.0+. The auto-rate calibration
    # block records the probe sweep that picked workload.rate_rps; for
    # explicit --rate runs the probes list is empty.
    calibration: Calibration | None = None
    raw_results: list[RequestRecord] = Field(default_factory=list)
    engine_metrics: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None

    @model_validator(mode="after")
    def _check_timeline(self) -> RunResult:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must be >= started_at")
        return self


# ---------------------------------------------------------------------------
# Manifest (what the command center reads)
# ---------------------------------------------------------------------------
class RunManifest(_Strict):
    run_id: str
    status: RunStatus
    engine_name: str
    workload_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result_path: str | None = None
    log_path: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _check_status_consistency(self) -> RunManifest:
        if self.status in (RunStatus.RUNNING, RunStatus.DONE, RunStatus.FAILED):
            if self.started_at is None:
                raise ValueError(f"status={self.status} requires started_at")
        if self.status in (RunStatus.DONE, RunStatus.FAILED):
            if self.finished_at is None:
                raise ValueError(f"status={self.status} requires finished_at")
        if self.status == RunStatus.DONE and self.result_path is None:
            raise ValueError("status=done requires result_path")
        if self.status == RunStatus.FAILED and self.error is None:
            raise ValueError("status=failed requires error")
        return self
