"""Unit tests for collect_cold_start_run.py and cold_start_tracer.py log scanner.

The tracer (formerly "probe") is the in-process instrumentation wrapper that emits
JSONL events at stage boundaries. NOT to be confused with K8s readiness probes.

Run: cd phase4.1 && python3 -m unittest tests.test_collect_cold_start_run -v
"""
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from collect_cold_start_run import (  # type: ignore
    STAGE_ORDER,
    Stage,
    build_run_record,
    parse_containerd_image_pull,
    parse_k8s_timestamp,
    stage_karpenter_scheduling,
    stage_model_download,
    stages_from_tracer_jsonl,
)
import cold_start_tracer  # type: ignore


# ---------------------------------------------------------------------------
# parse_k8s_timestamp — must NOT raise on bad input. Past bug: NoneType.
# ---------------------------------------------------------------------------


class TestParseK8sTimestamp(unittest.TestCase):
    def test_valid_z(self):
        self.assertEqual(
            parse_k8s_timestamp("2026-04-30T15:23:45Z"),
            1777562625000,
        )

    def test_valid_offset(self):
        self.assertEqual(
            parse_k8s_timestamp("2026-04-30T15:23:45+00:00"),
            1777562625000,
        )

    def test_none(self):
        self.assertIsNone(parse_k8s_timestamp(None))

    def test_empty(self):
        self.assertIsNone(parse_k8s_timestamp(""))

    def test_garbage(self):
        self.assertIsNone(parse_k8s_timestamp("not-a-timestamp"))

    def test_int_input(self):
        # int is not a string — function must reject without raising.
        self.assertIsNone(parse_k8s_timestamp(12345))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Probe log scanner — pattern matching.
# ---------------------------------------------------------------------------


class _StubEmitter:
    def __init__(self):
        self.events = []

    def emit(self, name, **extra):
        # First-occurrence dedupe (mirrors real EventEmitter behavior).
        if any(e["event"] == name for e in self.events):
            return
        self.events.append({"event": name, **extra})


class TestTracerLogScanner(unittest.TestCase):
    def test_weight_load_start(self):
        em = _StubEmitter()
        cold_start_tracer.scan_line_for_events(
            "INFO 04-30 15:23:45 Starting to load model Qwen/Qwen2.5-7B-Instruct-AWQ",
            em,
        )
        self.assertEqual(em.events[0]["event"], "weight_load_start")

    def test_weight_load_done(self):
        em = _StubEmitter()
        cold_start_tracer.scan_line_for_events(
            "Loading model weights took 12.4 seconds",
            em,
        )
        self.assertEqual(em.events[0]["event"], "weight_load_done")

    def test_graph_capture_start(self):
        em = _StubEmitter()
        cold_start_tracer.scan_line_for_events(
            "Capturing CUDA graphs for batch sizes [1, 2, 4]",
            em,
        )
        self.assertEqual(em.events[0]["event"], "graph_capture_start")

    def test_warmup_done(self):
        em = _StubEmitter()
        cold_start_tracer.scan_line_for_events(
            "Graph capturing finished in 8.1 seconds",
            em,
        )
        self.assertEqual(em.events[0]["event"], "warmup_done")

    def test_no_match(self):
        em = _StubEmitter()
        cold_start_tracer.scan_line_for_events("INFO Some unrelated log line", em)
        self.assertEqual(em.events, [])


# ---------------------------------------------------------------------------
# Stage extractors — fixture-based.
# ---------------------------------------------------------------------------


class TestStageExtractors(unittest.TestCase):
    def test_karpenter_scheduling(self):
        pod = {"metadata": {"creationTimestamp": "2026-04-30T15:00:00Z"}}
        events = [
            {"reason": "Scheduled", "eventTime": "2026-04-30T15:00:01Z"},
            {"reason": "Launched", "eventTime": "2026-04-30T15:00:08Z"},
        ]
        s = stage_karpenter_scheduling(pod, events)
        self.assertEqual(s.duration_ms, 8000)

    def test_karpenter_missing_event(self):
        # No Launched/Nominated event → end_ms None → duration None, NOT raise.
        pod = {"metadata": {"creationTimestamp": "2026-04-30T15:00:00Z"}}
        s = stage_karpenter_scheduling(pod, [])
        self.assertIsNotNone(s.start_ms)
        self.assertIsNone(s.end_ms)
        self.assertIsNone(s.duration_ms)

    def test_model_download(self):
        pod = {
            "status": {
                "initContainerStatuses": [
                    {
                        "name": "model-download",
                        "state": {
                            "terminated": {
                                "startedAt": "2026-04-30T15:01:00Z",
                                "finishedAt": "2026-04-30T15:01:42Z",
                            }
                        },
                    }
                ]
            }
        }
        s = stage_model_download(pod)
        self.assertEqual(s.duration_ms, 42000)

    def test_probe_jsonl_extract(self):
        events = [
            {"event": "process_start", "epoch_ms": 1000},
            {"event": "cuda_context_ready", "epoch_ms": 5000},
            {"event": "weight_load_start", "epoch_ms": 5500},
            {"event": "weight_load_done", "epoch_ms": 17500},
            {"event": "graph_capture_start", "epoch_ms": 17500},
            {"event": "warmup_done", "epoch_ms": 25000},
        ]
        stages = stages_from_tracer_jsonl(events)
        self.assertEqual(stages["vllm_init_cuda_ctx"].duration_ms, 4000)
        self.assertEqual(stages["weight_load_gpu_mem"].duration_ms, 12000)
        self.assertEqual(stages["cuda_graph_warmup"].duration_ms, 7500)


# ---------------------------------------------------------------------------
# Containerd journal parser.
# ---------------------------------------------------------------------------


class TestContainerdJournal(unittest.TestCase):
    def test_basic_pull(self):
        # Real containerd v1.7 format with `time="..."` inner timestamps.
        journal = (
            '2026-04-30T15:00:00+00:00 host containerd[1]: time="2026-04-30T15:00:00.100Z" '
            'level=info msg="PullImage \\"vllm/vllm-openai:v0.19.0\\""\n'
            '2026-04-30T15:04:30+00:00 host containerd[1]: time="2026-04-30T15:04:30.250Z" '
            'level=info msg="stop pulling image docker.io/vllm/vllm-openai:v0.19.0: '
            'active requests=0, bytes read=9577352921"\n'
            '2026-04-30T15:04:30+00:00 host containerd[1]: time="2026-04-30T15:04:30.300Z" '
            'level=info msg="PullImage \\"vllm/vllm-openai:v0.19.0\\" returns image reference \\"sha256:abc\\""\n'
        )
        start, stop_pulling, end = parse_containerd_image_pull(journal)
        self.assertIsNotNone(start)
        self.assertIsNotNone(stop_pulling)
        self.assertIsNotNone(end)
        self.assertGreater(stop_pulling, start)
        self.assertGreaterEqual(end, stop_pulling)

    def test_filters_other_images(self):
        # System image pull shouldn't pollute the vllm timing.
        journal = (
            '2026-04-30T14:00:00+00:00 host containerd[1]: time="2026-04-30T14:00:00.000Z" '
            'level=info msg="PullImage \\"602401143452.dkr.ecr.us-east-1.amazonaws.com/eks/kube-proxy:v1\\""\n'
            '2026-04-30T15:00:00+00:00 host containerd[1]: time="2026-04-30T15:00:00.000Z" '
            'level=info msg="PullImage \\"vllm/vllm-openai:v0.19.0\\""\n'
        )
        start, _, _ = parse_containerd_image_pull(journal)
        # Should match vllm at 15:00, not kube-proxy at 14:00.
        self.assertEqual(start, parse_k8s_timestamp("2026-04-30T15:00:00.000Z"))


# ---------------------------------------------------------------------------
# Full record assembly with a happy-path fixture.
# ---------------------------------------------------------------------------


class TestBuildRunRecord(unittest.TestCase):
    def _fixture(self):
        pod = {
            "metadata": {
                "name": "vllm-baseline-001",
                "uid": "uid-001",
                "creationTimestamp": "2026-04-30T15:00:00Z",
            },
            "spec": {
                "containers": [
                    {
                        "name": "vllm",
                        "image": "vllm/vllm-openai:v0.19.0",
                        "readinessProbe": {"httpGet": {"path": "/health", "port": 8000},
                                           "initialDelaySeconds": 120, "periodSeconds": 10},
                    }
                ]
            },
            "status": {
                "initContainerStatuses": [
                    {"name": "model-download", "state": {"terminated": {
                        "startedAt": "2026-04-30T15:03:00Z",
                        "finishedAt": "2026-04-30T15:03:42Z",
                    }}}
                ],
                "conditions": [
                    {"type": "Ready", "status": "True",
                     "lastTransitionTime": "2026-04-30T15:08:30Z"},
                ],
            },
        }
        node = {
            "metadata": {"name": "ip-10-5-1-1.ec2.internal"},
            "status": {"conditions": [
                {"type": "Ready", "status": "True",
                 "lastTransitionTime": "2026-04-30T15:01:30Z"},
            ]},
        }
        events = [
            {"reason": "Launched", "eventTime": "2026-04-30T15:00:08Z"},
        ]
        tracer_events = [
            {"event": "process_start", "epoch_ms": 1777735408000},
            {"event": "cuda_context_ready", "epoch_ms": 1777735413000},
            {"event": "weight_load_start", "epoch_ms": 1777735420000},
            {"event": "weight_load_done", "epoch_ms": 1777735480000},
            {"event": "graph_capture_start", "epoch_ms": 1777735480000},
            {"event": "warmup_done", "epoch_ms": 1777735508000},
        ]
        return pod, node, events, tracer_events

    def test_happy_path(self):
        pod, node, events, tracer_events = self._fixture()
        rec = build_run_record(
            run_id="fresh-001", arm="fresh_node",
            pod=pod, node=node, events=events,
            tracer_events=tracer_events, containerd_journal="",
            instance_id="i-abc", instance_type="g4dn.xlarge",
            az="us-east-1a", ami_id="ami-1", cpu_model="Xeon-1", kernel="5.10",
            ec2_run_request_ms=1777735208000,  # 15:00:08
            ec2_running_ms=1777735258000,      # 15:00:58
            pod_ready_ms=1777735710000,        # 15:08:30
            smoke_first_token_ms=1777735715000,
        )
        # All 11 stages present (10 + first_token_served).
        self.assertEqual(set(rec["stages"].keys()), set(STAGE_ORDER))
        # Containerd journal empty → image stages have None durations → run invalid.
        self.assertFalse(rec["validation"]["all_stages_present"])
        self.assertIn("image_download", str(rec["validation"]["notes"]))
        self.assertIn("image_unpack", str(rec["validation"]["notes"]))
        # Probe-jsonl stages should compute correctly.
        self.assertEqual(rec["stages"]["weight_load_gpu_mem"]["duration_ms"], 60000)
        self.assertEqual(rec["stages"]["cuda_graph_warmup"]["duration_ms"], 28000)
        # Hardware fingerprint preserved.
        self.assertEqual(rec["node"]["az"], "us-east-1a")
        # Probe spec hash deterministic (same input → same hash).
        rec2 = build_run_record(
            run_id="fresh-002", arm="fresh_node",
            pod=pod, node=node, events=events,
            tracer_events=tracer_events, containerd_journal="",
            instance_id="i-def", instance_type="g4dn.xlarge",
            az="us-east-1a", ami_id="ami-1", cpu_model="Xeon-1", kernel="5.10",
            ec2_run_request_ms=1777735208000,
            ec2_running_ms=1777735258000,
            pod_ready_ms=1777735710000,
            smoke_first_token_ms=1777735715000,
        )
        self.assertEqual(rec["validation"]["probe_spec_hash"],
                         rec2["validation"]["probe_spec_hash"])


if __name__ == "__main__":
    unittest.main()
