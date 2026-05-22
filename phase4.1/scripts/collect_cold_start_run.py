#!/usr/bin/env python3
"""Harvest per-stage cold-start timings for one Phase 4.1 baseline run.

Inputs (all optional via flags; main() wires them up from the run config):
  --run-id, --pod, --namespace, --node, --instance-id, --launched-at-epoch-ms

Outputs:
  phase4.1/tests/baseline_runs/<run_id>.json   — structured run record
  phase4.1/tests/baseline_runs/raw/<run_id>.*  — raw event/log artifacts

Design points:
  - Sources chosen for survivability (events, journald, CloudTrail, hostPath jsonl).
    A pod crash mid-run still leaves enough breadcrumbs to extract stage timings.
  - parse_k8s_timestamp returns None on bad input rather than raising — past
    benchmark runs were dropped silently because the parser hit None and crashed
    the aggregate. Here None propagates into validation.notes and the stage gets
    flagged, not silently lost. (See feedback_experiment_design.md.)
  - The collector NEVER infers a missing timestamp from a different source. If
    the canonical signal for a stage is missing, that stage is null and the run
    is marked invalid. Better one bad run than three runs with mixed-source data.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Optional

# All 10 waterfall stages, in the canonical waterfall order.
STAGE_ORDER = [
    "karpenter_scheduling",
    "ec2_spot_fulfillment",
    "node_bootstrap",
    "image_download",
    "image_unpack",
    "model_download_s3",
    "vllm_init_cuda_ctx",
    "weight_load_gpu_mem",
    "cuda_graph_warmup",
    "readiness_probe_pass",
    "first_token_served",
]


_FRAC_RE = re.compile(r"(\.\d+)")


def parse_k8s_timestamp(s: Optional[str]) -> Optional[int]:
    """RFC3339 → epoch ms. None-safe. Handles nanosecond precision.

    Past bug 1: callers passed `None` (event missing a timestamp) and the parser
    raised, dropping the whole run from the aggregate. Now None in → None out.

    Past bug 2: Python 3.9 datetime.fromisoformat rejects fractional seconds
    beyond microseconds (6 digits). Containerd emits 9 digits (nanoseconds).
    We truncate to 6 digits before parsing — millisecond resolution downstream
    means the truncation doesn't lose anything we care about.
    """
    if s is None or not isinstance(s, str) or not s:
        return None
    # Truncate fractional seconds to 6 digits max (microseconds).
    iso = _FRAC_RE.sub(lambda m: m.group(0)[:7], s)
    # Normalize "Z" → "+00:00" for fromisoformat.
    iso = iso.replace("Z", "+00:00") if iso.endswith("Z") else iso
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


@dataclasses.dataclass
class Stage:
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    source: str = ""

    @property
    def duration_ms(self) -> Optional[int]:
        if self.start_ms is None or self.end_ms is None:
            return None
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# K8s / kubectl helpers — thin wrappers, all pluggable for unit tests.
# ---------------------------------------------------------------------------


def kubectl_json(args: list[str]) -> Any:
    out = subprocess.run(
        ["kubectl", *args, "-o", "json"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def get_pod(namespace: str, pod: str) -> dict:
    return kubectl_json(["get", "pod", "-n", namespace, pod])


def get_events_for_pod(namespace: str, pod: str) -> list[dict]:
    data = kubectl_json(
        ["get", "events", "-n", namespace,
         "--field-selector", f"involvedObject.name={pod}"]
    )
    return data.get("items", [])


def get_node(node: str) -> dict:
    return kubectl_json(["get", "node", node])


# ---------------------------------------------------------------------------
# Per-stage extractors.
# Each returns a Stage. Missing data = None fields (not raised).
# ---------------------------------------------------------------------------


def stage_karpenter_scheduling(pod: dict, events: list[dict]) -> Stage:
    """Pod creationTimestamp -> Karpenter Nominated/Launched event.

    Filter events by pod UID. Earlier bug: pods with the same name across runs
    left stale events behind (different UID, same name). Without the UID filter,
    we picked up events from a previous pod and got negative durations.
    """
    s = Stage(source="k8s_events")
    pod_uid = pod.get("metadata", {}).get("uid")
    s.start_ms = parse_k8s_timestamp(pod.get("metadata", {}).get("creationTimestamp"))

    # Pick the EARLIEST matching event with timestamp >= pod creation time.
    # That guards against any stale events that might still slip through.
    earliest: Optional[int] = None
    for e in events:
        if pod_uid and e.get("involvedObject", {}).get("uid") != pod_uid:
            continue
        if e.get("reason", "") not in ("Nominated", "Launched", "DisruptionLaunch"):
            continue
        ts = parse_k8s_timestamp(e.get("eventTime") or e.get("firstTimestamp"))
        if ts is None:
            continue
        if s.start_ms is not None and ts < s.start_ms:
            continue  # stale event, skip
        if earliest is None or ts < earliest:
            earliest = ts
    s.end_ms = earliest
    return s


def stage_node_bootstrap(node: dict, ec2_running_ms: Optional[int]) -> Stage:
    """EC2 running -> Node Ready=True."""
    s = Stage(source="ec2+node_conditions")
    s.start_ms = ec2_running_ms
    for cond in node.get("status", {}).get("conditions", []):
        if cond.get("type") == "Ready" and cond.get("status") == "True":
            s.end_ms = parse_k8s_timestamp(cond.get("lastTransitionTime"))
            break
    return s


def stage_first_token(pod_ready_ms: Optional[int], smoke_ms: Optional[int]) -> Stage:
    s = Stage(source="external_smoke_client")
    s.start_ms = pod_ready_ms
    s.end_ms = smoke_ms
    return s


def stage_readiness_probe(events: list[dict], pod: dict) -> Stage:
    """warmup_done (from tracer jsonl, threaded in by caller via end_ms override)
    -> Pod Ready=True. We only fill the *end* here; start comes from the tracer."""
    s = Stage(source="pod_conditions")
    for cond in pod.get("status", {}).get("conditions", []):
        if cond.get("type") == "Ready" and cond.get("status") == "True":
            s.end_ms = parse_k8s_timestamp(cond.get("lastTransitionTime"))
            break
    return s


def stage_fsx_csi_mount(events: list[dict], pod: dict) -> Optional[Stage]:
    """OPTIONAL stage — Variant H only.

    Returns None when the pod has no `model-fsx` PVC volume (i.e. all variants
    A-G), so it doesn't show up in older runs' JSON. For Variant H, returns a
    Stage from `Scheduled` event time to the `SuccessfulAttachVolume` /
    `MountVolume.SetUp succeeded for volume "model-fsx"` event time. This is
    the FSx Lustre client first-pod-on-node mount cost — not visible in any
    of the standard tracer stages because vLLM hasn't started yet.

    The stage is additive: it does NOT replace any existing stage, and old
    JSONs continue to validate.
    """
    vols = pod.get("spec", {}).get("volumes", []) or []
    has_fsx = any(v.get("name") == "model-fsx" for v in vols)
    if not has_fsx:
        return None

    s = Stage(source="kubelet_events")
    for ev in events:
        reason = ev.get("reason")
        msg = ev.get("message") or ""
        first_seen = parse_k8s_timestamp(ev.get("firstTimestamp") or ev.get("eventTime"))
        if reason == "Scheduled" and s.start_ms is None:
            s.start_ms = first_seen
        # Kubelet emits two distinct mount events; the "SuccessfulMount" reason
        # is the canonical end. The "MountVolume.SetUp succeeded" message also
        # appears as a normal event. We accept either as the end signal.
        if reason in ("SuccessfulMount", "SuccessfulAttachVolume") and "model-fsx" in msg:
            s.end_ms = first_seen
            break
    return s


def stage_model_download(pod: dict) -> Stage:
    """Init-container start/finish.

    Variant E (RunAI Model Streamer) removes the model-download init container
    entirely — vLLM streams weights from S3 directly during weight load. When
    no `model-*` init container is present in the pod spec, this stage
    resolves to a known-zero with source=`absent_init_container` so the
    waterfall renders coherently and validation does NOT flag the run as
    incomplete. The streamer's actual S3-fetch time gets absorbed into
    weight_load_gpu_mem (logged + documented in LEARNINGS.md "Variant E").
    """
    spec_init = pod.get("spec", {}).get("initContainers", []) or []
    has_model_init = any("model" in (c.get("name") or "").lower() for c in spec_init)
    if not has_model_init:
        # Pin start == end at the pod creation time so totals math doesn't
        # accidentally read this as "model download took forever". Both being
        # the same value means duration_ms = 0.
        cts = parse_k8s_timestamp(pod.get("metadata", {}).get("creationTimestamp"))
        return Stage(start_ms=cts, end_ms=cts, source="absent_init_container")

    s = Stage(source="init_container_status")
    for ic in pod.get("status", {}).get("initContainerStatuses", []):
        # Match by name suffix; pod templates may prefix differently per phase.
        if "model" not in ic.get("name", "").lower():
            continue
        term = (ic.get("state") or {}).get("terminated") or {}
        s.start_ms = parse_k8s_timestamp(term.get("startedAt"))
        s.end_ms = parse_k8s_timestamp(term.get("finishedAt"))
        break
    return s


def stages_from_tracer_jsonl(events: list[dict]) -> dict[str, Stage]:
    """Stages 6, 7, 8 come from cold_start_tracer.py JSONL events."""
    by_name: dict[str, int] = {}
    for ev in events:
        name = ev.get("event")
        ms = ev.get("epoch_ms")
        if name and isinstance(ms, int):
            by_name.setdefault(name, ms)  # first occurrence wins (dedupe)

    return {
        "vllm_init_cuda_ctx": Stage(
            source="tracer_jsonl",
            start_ms=by_name.get("process_start"),
            end_ms=by_name.get("cuda_context_ready"),
        ),
        "weight_load_gpu_mem": Stage(
            source="tracer_jsonl",
            start_ms=by_name.get("weight_load_start"),
            end_ms=by_name.get("weight_load_done"),
        ),
        "cuda_graph_warmup": Stage(
            source="tracer_jsonl",
            start_ms=by_name.get("graph_capture_start"),
            end_ms=by_name.get("warmup_done"),
        ),
    }


def parse_containerd_image_pull(
    journal_text: str, target_image: str = "vllm-openai"
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (pull_start_ms, stop_pulling_ms, pull_end_ms) for the matching image.

    Real containerd v1.7 format (journalctl -o short-iso):
      ...time="2026-05-01T02:35:46.589Z" level=info msg="PullImage \"vllm/vllm-openai:v0.19.0\""
      ...time="2026-05-01T02:40:42.925Z" level=info msg="stop pulling image docker.io/vllm/vllm-openai:v0.19.0: ..."
      ...time="2026-05-01T02:40:42.952Z" level=info msg="PullImage \"vllm/vllm-openai:v0.19.0\" returns image reference \"...\""

    Use the high-resolution timestamp inside `time="..."` (not journald's prefix).
    Filter on `target_image` so unrelated system-image pulls don't pollute timing.

    Note: in containerd v1.7+ with overlayfs DiscardUnpackedLayers=true, layer
    fetch and snapshot unpack are concurrent — the gap between stop_pulling and
    PullImage returns is typically <100ms. We still report both so the waterfall
    can show that.
    """
    inner_ts = re.compile(r'time="(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)"')
    pull_start = stop_pulling = pull_end = None
    for line in journal_text.splitlines():
        if target_image not in line:
            continue
        m = inner_ts.search(line)
        if not m:
            continue
        ts = parse_k8s_timestamp(m.group("ts"))
        if ts is None:
            continue
        if "PullImage" in line and "returns image reference" in line and pull_end is None:
            pull_end = ts
        elif "stop pulling image" in line and stop_pulling is None:
            stop_pulling = ts
        elif "PullImage" in line and "returns" not in line and pull_start is None:
            pull_start = ts
    return pull_start, stop_pulling, pull_end


# ---------------------------------------------------------------------------
# Run record assembly.
# ---------------------------------------------------------------------------


def build_run_record(
    run_id: str,
    arm: str,
    pod: dict,
    node: dict,
    events: list[dict],
    tracer_events: list[dict],
    containerd_journal: str,
    instance_id: Optional[str],
    instance_type: Optional[str],
    az: Optional[str],
    ami_id: Optional[str],
    cpu_model: Optional[str],
    kernel: Optional[str],
    ec2_run_request_ms: Optional[int],
    ec2_running_ms: Optional[int],
    pod_ready_ms: Optional[int],
    smoke_first_token_ms: Optional[int],
) -> dict:
    """Assemble the run JSON. Per-stage None fields surface to validation.notes."""
    pull_start, last_fetch, pull_end = parse_containerd_image_pull(containerd_journal)

    stages: dict[str, Stage] = {}
    stages["karpenter_scheduling"] = stage_karpenter_scheduling(pod, events)

    stages["ec2_spot_fulfillment"] = Stage(
        source="cloudtrail+ec2",
        start_ms=ec2_run_request_ms,
        end_ms=ec2_running_ms,
    )

    stages["node_bootstrap"] = stage_node_bootstrap(node, ec2_running_ms)

    stages["image_download"] = Stage(
        source="containerd_journal",
        start_ms=pull_start,
        end_ms=last_fetch,
    )
    stages["image_unpack"] = Stage(
        source="containerd_journal",
        start_ms=last_fetch,
        end_ms=pull_end,
    )

    stages["model_download_s3"] = stage_model_download(pod)
    stages.update(stages_from_tracer_jsonl(tracer_events))

    # OPTIONAL stage — Variant H only. None for variants A-G; populated from
    # kubelet events when the pod has a model-fsx PVC. Kept out of STAGE_ORDER
    # so old runs validate cleanly.
    fsx_mount = stage_fsx_csi_mount(events, pod)

    # Stage 9 (readiness): warmup_done -> Pod Ready
    rdy = stage_readiness_probe(events, pod)
    warmup_done = stages["cuda_graph_warmup"].end_ms
    rdy.start_ms = warmup_done
    stages["readiness_probe_pass"] = rdy

    stages["first_token_served"] = stage_first_token(pod_ready_ms, smoke_first_token_ms)

    # Validation
    missing = [s for s in STAGE_ORDER if stages[s].duration_ms is None]
    negative = [s for s in STAGE_ORDER if (stages[s].duration_ms or 0) < 0]
    notes = []
    if missing:
        notes.append(f"missing_stages: {missing}")
    if negative:
        notes.append(f"negative_durations: {negative}")

    # Probe spec hash (for cross-run consistency check). We hash the readiness
    # probe + startup probe of the first non-init container — that's all the
    # plan cares about for stage 9 comparability.
    import hashlib
    probe_spec = {}
    for c in pod.get("spec", {}).get("containers", []):
        probe_spec[c.get("name")] = {
            "readinessProbe": c.get("readinessProbe"),
            "startupProbe": c.get("startupProbe"),
            "livenessProbe": c.get("livenessProbe"),
        }
    probe_spec_hash = hashlib.sha256(
        json.dumps(probe_spec, sort_keys=True).encode()
    ).hexdigest()[:16]

    return {
        "run_id": run_id,
        "arm": arm,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "node": {
            "name": node.get("metadata", {}).get("name"),
            "instance_id": instance_id,
            "instance_type": instance_type,
            "az": az,
            "ami_id": ami_id,
            "cpu_model": cpu_model,
            "kernel": kernel,
            "spot": True,
        },
        "pod": {
            "name": pod.get("metadata", {}).get("name"),
            "uid": pod.get("metadata", {}).get("uid"),
            "image": (pod.get("spec", {}).get("containers", [{}])[0]).get("image"),
        },
        "stages": {k: stages[k].to_dict() for k in STAGE_ORDER},
        # Optional stages — present only for variants that exercise them.
        # fsx_csi_mount: Variant H. Old variants get an empty dict here.
        "optional_stages": (
            {"fsx_csi_mount_s": fsx_mount.to_dict()} if fsx_mount is not None else {}
        ),
        "totals": {
            "scale_up_to_ready_ms": (
                None if pod_ready_ms is None or stages["karpenter_scheduling"].start_ms is None
                else pod_ready_ms - stages["karpenter_scheduling"].start_ms
            ),
            "scale_up_to_first_token_ms": (
                None if smoke_first_token_ms is None or stages["karpenter_scheduling"].start_ms is None
                else smoke_first_token_ms - stages["karpenter_scheduling"].start_ms
            ),
        },
        "validation": {
            "all_stages_present": not missing,
            "any_negative_durations": bool(negative),
            "probe_spec_hash": probe_spec_hash,
            "notes": notes,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--arm", default="fresh_node")
    p.add_argument("--namespace", default="default")
    p.add_argument("--pod", required=True)
    p.add_argument("--node", required=True)
    p.add_argument("--instance-id", required=True)
    p.add_argument("--instance-type", required=True)
    p.add_argument("--az", required=True)
    p.add_argument("--ami-id")
    p.add_argument("--cpu-model")
    p.add_argument("--kernel")
    p.add_argument("--ec2-run-request-ms", type=int)
    p.add_argument("--ec2-running-ms", type=int)
    p.add_argument("--pod-ready-ms", type=int)
    p.add_argument("--smoke-first-token-ms", type=int)
    p.add_argument("--tracer-jsonl", help="Path to run-<id>.jsonl from cold_start_tracer.py")
    p.add_argument("--containerd-journal", help="Path to containerd journald dump")
    p.add_argument("--out-dir", default="phase4.1/tests/baseline_runs")
    args = p.parse_args()

    pod = get_pod(args.namespace, args.pod)
    node = get_node(args.node)
    events = get_events_for_pod(args.namespace, args.pod)

    tracer_events: list[dict] = []
    if args.tracer_jsonl and pathlib.Path(args.tracer_jsonl).is_file():
        with open(args.tracer_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tracer_events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    journal_text = ""
    if args.containerd_journal and pathlib.Path(args.containerd_journal).is_file():
        journal_text = pathlib.Path(args.containerd_journal).read_text()

    record = build_run_record(
        run_id=args.run_id, arm=args.arm,
        pod=pod, node=node, events=events,
        tracer_events=tracer_events, containerd_journal=journal_text,
        instance_id=args.instance_id, instance_type=args.instance_type,
        az=args.az, ami_id=args.ami_id, cpu_model=args.cpu_model, kernel=args.kernel,
        ec2_run_request_ms=args.ec2_run_request_ms,
        ec2_running_ms=args.ec2_running_ms,
        pod_ready_ms=args.pod_ready_ms,
        smoke_first_token_ms=args.smoke_first_token_ms,
    )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run_id}.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"Wrote {out_path}")
    return 0 if record["validation"]["all_stages_present"] else 1


if __name__ == "__main__":
    sys.exit(main())
