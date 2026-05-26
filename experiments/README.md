# Empirical Inference Lab

Open-source tooling to run empirical inference experiments across multiple
engines (vLLM, SGLang, TensorRT-LLM) and workload shapes (chatbot, agentic
coding, mixes), and plot the results next to roofline predictions in a static
web portal.

This project is the empirical side of a two-part learning lab. The sister
project (the [sizing calculator](../calculators/sizing_calc/)) predicts batch
critical points, KV budgets, and throughput from hardware and model
specifications. This project measures the same numbers on real GPUs so the
predictions can be validated.

## What's here

| Path | Purpose |
| --- | --- |
| `cli/` | `exp` CLI: `run`, `list`, `status`, `stop` |
| `drivers/` | `EngineDriver` interface and vLLM / SGLang / TRT-LLM implementations |
| `workloads/` | Synthetic workload generators (chatbot, agentic coding, rag, multi-turn, mix) |
| `runner/` | Async load driver, sweep runner, result schema |
| `results/` | JSON results (gitignored except a handful of fixtures) + run manifests |
| `eks/` | Terraform + Kubernetes manifests for the isolated `inference-lab` EKS cluster |
| `portal/` | Static HTML pages that read result JSONs and overlay them on roofline curves |
| `tests/` | Unit, contract, integration, build, and E2E tests |

## Quickstart

> The repo is currently a scaffold. Quickstart instructions will be filled in
> as each engine driver lands. See [`docs/runbook.md`](docs/runbook.md) for
> the latest state.

```bash
# Install (editable, with dev extras)
pip install -e ".[dev]"

# Run the full test pyramid except E2E
pytest

# Lint + type-check
ruff check .
black --check .
mypy --strict .
```

## Design

See [`docs/architecture.md`](docs/architecture.md) for the full picture.
Short version:

- Engines are abstracted behind `EngineDriver` so the engine under test is a
  CLI flag, not a code change.
- Workloads are abstracted behind `WorkloadGenerator` so adding a new traffic
  shape is one file plus tests.
- Runs produce a versioned JSON document; the static portal reads those JSONs
  and joins them against the sizing calculator's `hardware.json` / `models.json`
  to plot empirical points on the predicted roofline curves.
- No backend service. The "command center" is a static HTML page that reads
  the manifest files the CLI writes to disk.

## Non-interference

This stack is fully isolated from sibling `phase*/` stacks in the parent
repository — distinct EKS cluster name (`inference-lab`), VPC, IAM roles, ECR
repositories, S3 buckets, Terraform state, and Karpenter NodePool. A build
test (`tests/build/test_no_phase_collisions.py`) fails the build if any
generated artifact accidentally references the wrong stack.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). All PRs must pass lint, type-check,
unit, contract, integration, and build tests. E2E tests require GPU and AWS
credentials and are gated behind a manual workflow.

## License

[Apache 2.0](LICENSE).
