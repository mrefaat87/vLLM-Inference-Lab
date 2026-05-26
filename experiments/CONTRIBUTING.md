# Contributing

Thanks for your interest. This project is an open-source learning lab for
inference infrastructure, and contributions that deepen the test surface,
add new engine drivers, or sharpen the workload models are very welcome.

## Ground rules

- Be kind. We follow the [Contributor Covenant](CODE_OF_CONDUCT.md).
- Every PR must pass `lint`, `test`, and `build` workflows. CI is the bar.
- Don't widen scope silently. If a PR grows past its title, split it.
- Don't commit results, logs, kubeconfig, Terraform state, or model weights.
  `.gitignore` should already handle this — if it doesn't, fix the
  `.gitignore` in the same PR.

## Development setup

```bash
git clone <repo>
cd experiments
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Running tests

```bash
pytest                              # unit + contract + integration + build
pytest -m unit                      # just unit
pytest --cov=experiments --cov-report=term-missing
RUN_E2E=1 pytest -m e2e             # E2E (requires AWS creds + GPU quota)
```

Coverage target: **≥85%** on `drivers/`, `workloads/`, `runner/`. PRs that
drop coverage below that bar will be asked to add tests.

## Adding a new engine

1. Create `drivers/<engine>_driver.py` implementing `EngineDriver`.
2. Add a Deployment template under `eks/manifests/engines/<engine>.yaml`.
3. Subclass `AbstractEngineDriverContract` in
   `tests/contract/test_<engine>_driver.py` — that's it; the abstract class
   carries the actual assertions.
4. Add an entry to `cli/exp.py`'s engine registry.
5. Document the engine's quirks in `docs/engines/<engine>.md`.

## Built-in workloads

| Name | Shape | Stresses |
| --- | --- | --- |
| `chatbot` | short prompts, Poisson arrivals, log-normal lengths | continuous batching, TTFT under bursty arrivals |
| `agentic_coding` | long shared system prompt, bimodal outputs, bursty | long-shared-prefix caching, decode throughput on long outputs |
| `rag` | 5–15k unique prompts, short answers | prefill throughput, KV budget, cache *misses* |
| `multi_turn` | sessions where prompt grows turn-over-turn | KV reuse policies, eviction strategies |
| `mix` | weighted blend of two children | realism — see how engines do under a non-pure load |

## Adding a new workload

1. Create `workloads/<name>.py` implementing `WorkloadGenerator`.
2. Add a distribution unit test in `tests/unit/workloads/test_<name>.py`
   (KS-test against the declared distributions) + a determinism test
   (same seed → identical request stream).
3. Subclass `AbstractWorkloadContract` in
   `tests/contract/test_<name>_workload.py`.
4. Add to `cli/exp.py`'s workload registry.

## Commit messages and PRs

- Imperative subject under 70 chars.
- Body explains the *why* and any non-obvious trade-offs.
- Reference issues with `Fixes #N` / `Refs #N`.
- Squash before merge.

## Reporting issues

Please include the engine, workload, model, hardware, and the relevant
result JSON (with secrets scrubbed). A reproducible report is a fast report.
