# LLM Inference Sizing Calculator

An in-browser, client-side calculator that sizes an LLM serving deployment
**before** any empirical benchmark run. Pick a GPU, a model, an ISL/OSL pair,
precisions, and SLO targets; get back the recommended parallelism, the three
batch-size bounds (`b_crit`, `b_slo`, `b_kv`), `max_num_batched_tokens`,
estimated throughput, and the empirical sweep grids to feed straight into a
benchmark harness.

> **This is an analytical estimator, not a profiler.** Always verify with an
> empirical sweep before sizing production. The point of this tool is to rule
> out 80% of the search space cheaply.

Companion to [`MODEL_SIZING_SCALING_REFERENCE.md`](../../MODEL_SIZING_SCALING_REFERENCE.md)
— every formula it implements cites a section in that document.

## What it computes

| Metric | Formula | Reference |
|---|---|---|
| `b_crit` | `(peak_FLOPs / HBM_BW) × (bits_per_param / bits_per_activation)` | §1 |
| `b_kv` | `(HBM_total − weights − 10% overhead) / (KV_per_token × (ISL+OSL))` | §3 |
| `b_slo` | largest `B` with `step_time(B) ≤ TBT_SLO` (both mem and compute bounds) | §4 |
| `parallelism` | smallest `TP ∈ {1,2,4,8}` where `weights/TP + headroom ≤ HBM_per_GPU`; else PP | §6 |
| `Y_max` (TP usefulness ceiling) | `d_ff / (B × β)` where `β = HBM_BW / ICI_BW` — sharding beyond this stops cutting decode latency because the ICI all-reduce becomes the floor | §6, Scaling Book |
| `max_num_batched_tokens` | `α × max_batch × ISL`, snapped to ≥ 1024 pow-2, Sarathi band | §10 |
| `throughput` | `B / step_time(B)` at the recommended batch | §4 |
| `P:D ratio` | `(ISL/OSL) × (MFU_prefill / MFU_decode)` for disaggregation sizing | §5.2 |

## Repository layout

```
src/
  calc.mjs                # pure compute (source of truth for formulas)
  chart.mjs               # Chart.js oscilloscope + caliper-marker plugin
  ui.mjs                  # DOM wiring, recompute, patchbay, diagnostics
  styles.css              # engineering-instrument aesthetic
  data/
    hardware.json         # GPU specs (T4, L4, A10G, A100×2, H100, H200)
    models.json           # model architectures (Llama-3 8B/70B, Qwen2.5 7B/14B, Mistral-7B)
tests/
  calc.test.mjs           # node:test unit tests anchored to reference goldens
  data.test.mjs           # data-table schema + plausibility ranges
  test_build.py           # pytest: build produces a valid single-file artifact
  calculator.e2e.spec.mjs # Playwright smoke against the rendered HTML
  fixtures/golden.json    # numerical truth anchors with section citations
```

The published artifact is one file: [`../sizing_calculator.html`](../sizing_calculator.html).
Open it directly in a browser — no server, no build step at view time.

## Build

```bash
python3 ../../tools/build_sizing_calculator_html.py
```

Inlines the `src/` modules and data JSON into a single self-contained HTML
(only external refs are the Chart.js CDN and Google Fonts).

## Tests

```bash
# Unit + data-schema tests (zero deps, Node 20+)
node --test tests/calc.test.mjs tests/data.test.mjs

# Build-script test (requires `pip install pytest`)
python3 -m pytest tests/test_build.py

# End-to-end browser smoke (requires `npm install` + `npx playwright install chromium`)
npx playwright test
```

CI runs all four layers on every PR — see
[`.github/workflows/sizing-calculator.yml`](../../.github/workflows/sizing-calculator.yml).

## Contributing

**Adding a GPU.** Append an entry to `src/data/hardware.json` and add at least
one golden in `tests/fixtures/golden.json` if a reference value exists for the
new hardware. The data-schema test will catch missing fields automatically.

**Adding a model.** Append an entry to `src/data/models.json`. For MLA models,
keep `attn_type: "MLA"` — the calculator surfaces a warning that the KV
estimate is a GQA-equivalent upper bound (the real MLA cache is ~10–20×
smaller; modelling that exactly is a TODO).

**Changing a formula.** Update `src/calc.mjs`, update the corresponding
golden(s) in `tests/fixtures/golden.json` with a fresh `source:` citation,
rerun the unit tests, rebuild, rerun the build + e2e tests.

## Design notes

- **Aesthetic**: engineering instrument panel — hairline rules, oscilloscope
  chart, monospace numerals as the hero. Avoids the SaaS dashboard idiom by
  design (no centered hero, no gradient buttons, no card-with-icon tiles).
- **Self-contained**: one HTML file. Two external CDN deps (Chart.js, Google
  Fonts). Everything else inline.
- **Testable layering**: `calc.mjs` is pure functions and is imported directly
  by the Node test suite, so every formula has a unit test anchored to the
  reference doc. The build script splices `calc.mjs` into the HTML for the
  browser; the build test asserts the two stay in sync.

## License

MIT — see [`LICENSE`](./LICENSE).
