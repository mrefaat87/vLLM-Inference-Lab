"""Bridge from the lab to the sizing calculator's ``compute()``.

Three-tier lookup with graceful degradation. The lab never blocks on the
calc being available — if neither tier resolves, ``predict()`` returns
``None`` and the run still completes (with ``prediction: null`` baked
into the result JSON).

Resolution order:

1. **Grid lookup.** If a matching row exists in the precomputed
   ``predictions/grid.json``, return it. No subprocess.
2. **Live invocation.** Shell out to ``node compute_cli.mjs`` with the
   request on stdin, parse the stdout JSON. Used for off-grid inputs.
3. **Unavailable.** Either Node is missing from PATH or the calc tree
   isn't present (clone-only-the-lab scenario). Returns ``None``; the
   caller writes ``prediction: null`` and a notes line.

The bridge does not raise on calc misconfiguration — that's a
"prediction unavailable" condition, not a lab failure.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from experiments.runner.schema import (
    Prediction,
    PredictionCurvePoint,
    PredictionInputs,
)

log = logging.getLogger(__name__)

# Default location of the calc tree relative to the repo root. Overridable
# via the SIZING_CALC_ROOT env var so the lab can be cloned independently
# (in which case the bridge degrades gracefully).
DEFAULT_CALC_ROOT = Path(__file__).resolve().parents[2] / "calculators" / "sizing_calc"


@dataclass(frozen=True)
class CalcInputs:
    """The bridge's call shape — flat dict that matches the JSON-Schema."""

    model_key: str
    hw_key: str
    weight_prec: str
    kv_prec: str
    act_prec: str
    isl: int
    osl: int
    ngpus: int
    tbt_ms: float
    price_per_hour_usd: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "hw_key": self.hw_key,
            "weight_prec": self.weight_prec,
            "kv_prec": self.kv_prec,
            "act_prec": self.act_prec,
            "isl": int(self.isl),
            "osl": int(self.osl),
            "ngpus": int(self.ngpus),
            "tbt_ms": float(self.tbt_ms),
            "price_per_hour_usd": self.price_per_hour_usd,
        }


class CalcBridge:
    """Stateful (cached) bridge instance.

    Holding a singleton means the grid file is parsed once per process —
    a 5 MB read otherwise dominates the per-run cost.
    """

    def __init__(self, calc_root: Path | None = None) -> None:
        self._calc_root = Path(calc_root) if calc_root else _resolve_calc_root()

    @property
    def grid_path(self) -> Path:
        return self._calc_root / "predictions" / "grid.json"

    @property
    def compute_cli_path(self) -> Path:
        return self._calc_root / "scripts" / "compute_cli.mjs"

    @property
    def hardware_data_path(self) -> Path:
        return self._calc_root / "src" / "data" / "hardware.json"

    @lru_cache(maxsize=1)
    def _hw_price_table(self) -> dict[str, float | None]:
        """Map hw `key` → ``price_per_hour_usd`` from the calc's hardware.json.

        Used by the runner to enrich Prediction.inputs.price_per_hour_usd so
        the lab carries the same default the calc would have applied. Falls
        back to an empty table if the file is missing — the field is optional
        and the run still completes.
        """
        try:
            blob = json.loads(self.hardware_data_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        rows = blob.get("hardware") if isinstance(blob, dict) else None
        if not isinstance(rows, list):
            return {}
        out: dict[str, float | None] = {}
        for row in rows:
            if isinstance(row, dict) and "key" in row:
                v = row.get("price_per_hour_usd")
                out[str(row["key"])] = float(v) if isinstance(v, (int, float)) else None
        return out

    def lookup_hw_price(self, hw_key: str) -> float | None:
        """Return the calc's default ``price_per_hour_usd`` for an hw key.

        Returns ``None`` if the key is unknown or the hardware table can't
        be read (clone-only-the-lab). The lab uses this to populate the
        Prediction.inputs.price_per_hour_usd field at run time.
        """
        return self._hw_price_table().get(hw_key)

    def predict(self, inputs: CalcInputs) -> Prediction | None:
        """Return a Prediction for the given inputs, or None if unavailable."""
        grid = self._load_grid()
        if grid is not None:
            row = _find_row(grid["rows"], inputs)
            if row is not None:
                return _row_to_prediction(row, grid["calc_version"], grid["data_hash"])
        # Fall through to live invocation.
        return self._invoke_live(inputs)

    @lru_cache(maxsize=1)
    def _load_grid(self) -> dict[str, Any] | None:
        if not self.grid_path.exists():
            log.info("calc_bridge: grid not found at %s; will use live calc", self.grid_path)
            return None
        try:
            blob = json.loads(self.grid_path.read_text())
        except json.JSONDecodeError as exc:
            log.warning("calc_bridge: grid at %s is malformed: %s", self.grid_path, exc)
            return None
        if "rows" not in blob or "data_hash" not in blob:
            log.warning("calc_bridge: grid missing required fields")
            return None
        return blob

    def _invoke_live(self, inputs: CalcInputs) -> Prediction | None:
        node = shutil.which("node")
        if node is None:
            log.info("calc_bridge: node not on PATH; prediction unavailable")
            return None
        if not self.compute_cli_path.exists():
            log.info("calc_bridge: %s not found; prediction unavailable", self.compute_cli_path)
            return None
        payload = json.dumps(inputs.as_dict())
        try:
            proc = subprocess.run(
                [node, str(self.compute_cli_path)],
                input=payload,
                text=True,
                capture_output=True,
                timeout=10.0,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.warning("calc_bridge: live invocation failed: %s", exc)
            return None
        if proc.returncode != 0:
            log.warning(
                "calc_bridge: compute_cli exit=%s stderr=%s", proc.returncode, proc.stderr.strip()
            )
            return None
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            log.warning("calc_bridge: compute_cli emitted bad JSON: %s", exc)
            return None
        try:
            return Prediction.model_validate(data)
        except Exception as exc:  # noqa: BLE001 — schema drift is a soft failure
            log.warning("calc_bridge: prediction did not validate: %s", exc)
            return None


def _resolve_calc_root() -> Path:
    override = os.environ.get("SIZING_CALC_ROOT")
    if override:
        return Path(override)
    return DEFAULT_CALC_ROOT


def _input_matches(row_inputs: dict[str, Any], inputs: CalcInputs) -> bool:
    """All scalar fields equal (including precision strings, case-sensitive)."""
    return (
        row_inputs.get("model_key") == inputs.model_key
        and row_inputs.get("hw_key") == inputs.hw_key
        and row_inputs.get("weight_prec") == inputs.weight_prec
        and row_inputs.get("kv_prec") == inputs.kv_prec
        and row_inputs.get("act_prec") == inputs.act_prec
        and int(row_inputs.get("isl", -1)) == inputs.isl
        and int(row_inputs.get("osl", -1)) == inputs.osl
        and int(row_inputs.get("ngpus", -1)) == inputs.ngpus
        and float(row_inputs.get("tbt_ms", -1.0)) == float(inputs.tbt_ms)
    )


def _find_row(rows: list[dict[str, Any]], inputs: CalcInputs) -> dict[str, Any] | None:
    for row in rows:
        if _input_matches(row.get("inputs", {}), inputs):
            return row
    return None


def _row_to_prediction(row: dict[str, Any], calc_version: str, data_hash: str) -> Prediction:
    """Materialize a compact grid row into a fully-formed Prediction."""
    curve = [
        PredictionCurvePoint(
            batch=int(p["batch"]),
            step_ms=float(p["step_ms"]),
            tps=float(p["tps"]),
            cost_per_mtok=p.get("cost_per_mtok"),
        )
        for p in row.get("curve", [])
    ]
    return Prediction(
        calc_version=calc_version,
        data_hash=data_hash,
        inputs=PredictionInputs.model_validate(row["inputs"]),
        b_crit=row.get("b_crit"),
        b_slo=row.get("b_slo"),
        b_kv=row.get("b_kv"),
        recommended_batch=row.get("recommended_batch"),
        y_max=row.get("y_max"),
        curve=curve,
        warnings=list(row.get("warnings", [])),
        unavailable_reason=row.get("unavailable_reason"),
    )


# Convenience module-level singleton + helper so callers don't have to
# construct a CalcBridge themselves in the common case.
_default_bridge: CalcBridge | None = None


def predict(inputs: CalcInputs) -> Prediction | None:
    global _default_bridge
    if _default_bridge is None:
        _default_bridge = CalcBridge()
    return _default_bridge.predict(inputs)
