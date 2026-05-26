# Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│ Mac / dev box                                                          │
│                                                                        │
│   exp run --engine vllm --workload chatbot                             │
│     │                                                                  │
│     ▼                                                                  │
│   SweepRunner ──► EngineDriver.start(cfg) ─► kubectl apply             │
│     │                                            │                     │
│     ▼                                            ▼                     │
│   WorkloadGenerator                       ┌──────────────┐             │
│   (async iter of                          │  inference-  │             │
│    Request objects)                       │  lab cluster │             │
│     │                                     │              │             │
│     ▼                                     │  ┌────────┐  │             │
│   driver_loop  ──── aiohttp ──── kubectl ─┼─►│ engine │  │             │
│   (TTFT/TBT)                       PF     │  │  pod   │  │             │
│     │                                     │  └────────┘  │             │
│     ▼                                     └──────────────┘             │
│   analyze ──► RunResult JSON ─► results/runs/<id>.json                 │
│     │                                                                  │
│     ▼                                                                  │
│   ManifestStore ─► results/manifests/<id>.json                         │
│                                                                        │
│   portal.build ─► _site/{command_center,results_explorer}.html         │
└────────────────────────────────────────────────────────────────────────┘
```

## Why this shape

**No portal backend.** Adding a FastAPI gateway would force us to think
about auth, cors, deploy, and websocket reconnect — none of which make
the empirical work better. The portal reads JSON the CLI wrote. Refresh
is the live-update mechanism.

**One driver interface, three engines.** vLLM, SGLang, and TRT-LLM have
very different deploy stories (vLLM is one Deployment; TRT-LLM needs a
pre-build Job; SGLang has its own health endpoint). Hiding all of that
behind `EngineDriver.start/stop/healthcheck/metrics` means the sweep
runner is engine-agnostic and the comparison is apples-to-apples.

**Workloads are pure-Python async generators with a single RNG seed.**
Determinism is critical for comparing engines — if engine A sees a
different traffic mix than engine B, the comparison is noise. The
contract test enforces same-seed reproduction.

## Two-stack non-interference

The lab cluster shares zero mutable state with the sibling `phase*`
clusters in this repository. See [`../eks/README.md`](../eks/README.md#non-interference)
for the table; the guard test
[`tests/build/test_no_phase_collisions.py`](../tests/build/test_no_phase_collisions.py)
catches any accidental cross-contamination at build time.

## Roofline join

The result schema's `roofline_link.{model_ref, hw_ref}` are foreign keys
into the sister sizing calculator's `models.json` / `hardware.json`.
The results explorer's chart layer uses them to plot the empirical
point on top of the predicted curve — a 5x divergence is interesting,
a 50% divergence is noise.
