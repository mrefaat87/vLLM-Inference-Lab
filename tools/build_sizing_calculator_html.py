"""Build the self-contained sizing-calculator HTML from the src/ modules.

Splices calc.mjs + chart.mjs + ui.mjs into a single inline <script type="module">,
embeds hardware.json + models.json as <script type="application/json"> blocks,
inlines the CSS, and writes the result to calculators/sizing_calculator.html.

The single-file output is the *artifact*; the src/ modules are the *source*.
This separation is what makes the calc layer testable in node:test (it imports
calc.mjs directly) while the published page stays one file with one CDN ref.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "calculators" / "sizing_calc" / "src"
DST = ROOT / "calculators" / "sizing_calculator.html"


def strip_module_syntax(js: str) -> str:
    """Drop top-level import/export so concatenated modules share one scope.

    We rely on naming consistency: every export becomes a free top-level name in
    the bundled script, and every import resolves to the same name. The unit
    tests still run against the originals via real ESM imports — the bundling
    is only for the browser bundle.
    """
    # Remove every `import ... from "..."` statement (multi-line aware).
    js = re.sub(r"^import\s+[^;]+from\s+['\"][^'\"]+['\"];\s*$", "", js, flags=re.MULTILINE)
    # Strip the `export` keyword from declarations; the names stay defined.
    js = re.sub(r"^export\s+(function|const|let|class)\s+", r"\1 ", js, flags=re.MULTILINE)
    return js


def _escape_for_inline_script(js: str) -> str:
    """Defuse any literal </script> sequences so the HTML parser doesn't see
    them as an end-tag and prematurely close our <script> block. This bit us
    once already — a comment in ui.mjs contained the literal text. The
    backslash form is JS-safe and HTML-safe simultaneously.
    """
    return re.sub(r"</(script)", r"<\\/\1", js, flags=re.IGNORECASE)


def bundle_modules() -> str:
    # Order matters: calc defines compute/DTYPE_BYTES used by chart and ui;
    # chart defines createScope used by ui.
    calc = strip_module_syntax((SRC / "calc.mjs").read_text())
    chart = strip_module_syntax((SRC / "chart.mjs").read_text())
    ui = strip_module_syntax((SRC / "ui.mjs").read_text())
    bundle = "\n// ===== calc.mjs =====\n" + calc + \
             "\n// ===== chart.mjs =====\n" + chart + \
             "\n// ===== ui.mjs =====\n" + ui
    return _escape_for_inline_script(bundle)


def body_html() -> str:
    """The form + readout + chart + sweep + diagnostics markup.

    Field IDs here must match the IDs ui.mjs reads — the build test enforces
    this so the two never drift.
    """
    return """
<header class="strip">
  <div class="strip-left">
    <span class="brand">SIZING CALCULATOR</span>
    <span class="sep"></span>
    <span>vLLM · Inference · Roofline analysis</span>
  </div>
  <div class="strip-right">
    <a href="../reference/MODEL_SIZING_SCALING_REFERENCE.html">↗ MODEL SIZING REFERENCE</a>
  </div>
</header>

<div class="bench">
  <aside class="panel-left">
    <div class="panel-head">
      <span>CONTROL PANEL</span>
      <span class="live-state"><span class="live-dot"></span>LIVE</span>
    </div>
    <form id="sizing-form" autocomplete="off" onsubmit="return false;">
      <div class="field">
        <label>Hardware</label>
        <select id="hw"></select>
        <details id="hw-custom" class="custom">
          <summary>Custom hardware</summary>
          <div class="field-row">
            <div class="field"><label>HBM <span class="unit">GB</span></label><input id="c-hbm-gb" type="number" value="80" min="1"></div>
            <div class="field"><label>HBM BW <span class="unit">GB/s</span></label><input id="c-hbm-bw" type="number" value="3350" min="100"></div>
          </div>
          <div class="field-row">
            <div class="field"><label>FP16 <span class="unit">TFLOPs</span></label><input id="c-fp16" type="number" value="990" min="1"></div>
            <div class="field"><label>FP8 <span class="unit">TFLOPs</span></label><input id="c-fp8" type="number" value="" min="0" placeholder="blank=none"></div>
          </div>
          <div class="field"><label><input id="c-nvlink" type="checkbox" checked> NVLink within node</label></div>
        </details>
      </div>

      <div class="field">
        <label>Model</label>
        <select id="model"></select>
        <details id="model-custom" class="custom">
          <summary>Custom model</summary>
          <div class="field-row">
            <div class="field"><label>Params <span class="unit">B</span></label><input id="c-params" type="number" value="7" min="0.1" step="0.1"></div>
            <div class="field"><label>Layers</label><input id="c-layers" type="number" value="32" min="1"></div>
          </div>
          <div class="field-row">
            <div class="field"><label>d_model</label><input id="c-dmodel" type="number" value="4096" min="1"></div>
            <div class="field"><label>n_heads</label><input id="c-heads" type="number" value="32" min="1"></div>
          </div>
          <div class="field-row">
            <div class="field"><label>n_kv_heads</label><input id="c-kvheads" type="number" value="8" min="1"></div>
            <div class="field"><label>head_dim</label><input id="c-headdim" type="number" value="128" min="1"></div>
          </div>
          <div class="field-row">
            <div class="field"><label>max_context</label><input id="c-maxctx" type="number" value="8192" min="128"></div>
            <div class="field"><label>Attention</label><select id="c-attn"><option>GQA</option><option>MHA</option><option>MQA</option><option>MLA</option></select></div>
          </div>
        </details>
      </div>

      <div class="field-row">
        <div class="field"><label>ISL <span class="unit">tokens</span></label><input id="isl" type="number" value="1024" min="1"></div>
        <div class="field"><label>OSL <span class="unit">tokens</span></label><input id="osl" type="number" value="256" min="1"></div>
      </div>

      <div class="field-row">
        <div class="field"><label>Weight prec</label>
          <select id="weight-prec"><option>FP16</option><option>BF16</option><option>FP8</option><option>INT8</option><option>INT4</option></select></div>
        <div class="field"><label>KV prec</label>
          <select id="kv-prec"><option>FP16</option><option>BF16</option><option>FP8</option><option>INT8</option></select></div>
      </div>

      <div class="field-row">
        <div class="field"><label>Act prec</label>
          <select id="act-prec"><option>FP16</option><option>BF16</option><option>FP8</option></select></div>
        <div class="field"><label>nGPUs</label><input id="ngpus" type="number" value="1" min="1"></div>
      </div>

      <div class="field-row">
        <div class="field"><label>TTFT SLO <span class="unit">ms</span></label><input id="ttft" type="number" value="2000" min="1"></div>
        <div class="field"><label>TBT SLO <span class="unit">ms</span></label><input id="tbt" type="number" value="50" min="1"></div>
      </div>
    </form>
  </aside>

  <main class="panel-right">
    <h1>Sizing Calculator
      <span class="sub">analytical pre-flight · before empirical sweeps</span>
    </h1>
    <p class="lede">An in-browser companion to <em>Model Sizing &amp; Scaling Reference</em>.
    Pick a GPU, model, and SLO; rule out 80% of the empirical search space before burning GPU time.</p>

    <h2>Readouts</h2>
    <section class="readouts" aria-label="metric readouts">
      <div class="readout b-crit">
        <p class="readout-label"><span>B_crit</span><span class="hint">roofline ridge</span></p>
        <span class="readout-value" id="m-bcrit">—</span>
        <p class="readout-sub">memory-bound below · compute-bound above</p>
        <div class="readout-sparkline" id="spark-bcrit"></div>
      </div>
      <div class="readout b-slo">
        <p class="readout-label"><span>B_slo</span><span class="hint">latency cap</span></p>
        <span class="readout-value" id="m-bslo">—</span>
        <p class="readout-sub">largest B keeping step_time ≤ TBT SLO</p>
        <div class="readout-sparkline" id="spark-bslo"></div>
      </div>
      <div class="readout b-kv">
        <p class="readout-label"><span>B_kv</span><span class="hint">HBM cap</span></p>
        <span class="readout-value" id="m-bkv">—</span>
        <p class="readout-sub">KV cache fits in available HBM at ISL+OSL</p>
        <div class="readout-sparkline" id="spark-bkv"></div>
      </div>
      <div class="readout">
        <p class="readout-label"><span>Parallelism</span><span class="hint">TP · PP</span></p>
        <span class="readout-value" id="m-par">—</span>
        <p class="readout-sub" id="m-par-sub">—</p>
      </div>

      <div class="readout">
        <p class="readout-label"><span>max_num_seqs</span><span class="hint">vLLM</span></p>
        <span class="readout-value" id="m-mns">—</span>
        <p class="readout-sub">= min(B_slo, B_kv)</p>
      </div>
      <div class="readout">
        <p class="readout-label"><span>max_num_batched_tokens</span><span class="hint">vLLM</span></p>
        <span class="readout-value" id="m-mnbt">—</span>
        <p class="readout-sub">Sarathi band: <span id="m-mnbt-band">—</span></p>
      </div>
      <div class="readout">
        <p class="readout-label"><span>Throughput</span><span class="hint">tok/s</span></p>
        <span class="readout-value" id="m-tps">—</span>
        <p class="readout-sub" id="m-tps-pg">—</p>
      </div>
      <div class="readout">
        <p class="readout-label"><span>P : D ratio</span><span class="hint">disaggregation hint</span></p>
        <span class="readout-value" id="m-pd">—</span>
        <p class="readout-sub">(ISL/OSL) × MFU<sub>p</sub>/MFU<sub>d</sub></p>
      </div>
    </section>

    <p class="lede" style="margin-top:18px">
      Weights: <span id="m-weights">—</span>  ·  KV: <span id="m-kv">—</span>
    </p>

    <h2>Latency curve</h2>
    <section class="scope" aria-label="step-time vs batch oscilloscope">
      <div class="scope-head">
        <span>STEP TIME · LOG B</span>
        <div class="channels">
          <span class="ch-step"><span class="sw"></span>STEP MS</span>
          <span class="ch-tps"><span class="sw"></span>TOK/S</span>
          <span class="ch-slo"><span class="sw"></span>TBT SLO</span>
        </div>
      </div>
      <div class="scope-canvas-wrap">
        <canvas id="scope-canvas"></canvas>
      </div>
      <div class="scope-foot">
        <span>calipers · B_crit (blue) · B_slo (yellow) · B_kv (green)</span>
        <span>hover for crosshair readout</span>
      </div>
    </section>

    <h2>Sweep grid</h2>
    <p class="lede">Empirical bracket around the analytical bound. Paste into your benchmark runner.</p>
    <div class="patchbay" id="patchbay"></div>

    <div class="snippet">
      <div class="snippet-head">
        <span>↓ vLLM serve · empirical grid</span>
        <button class="copy-btn" id="copy-btn" type="button">[ COPY ]</button>
      </div>
      <pre id="snippet-body">—</pre>
    </div>

    <h2>Diagnostics</h2>
    <div class="diagnostics" id="diagnostics"><div class="empty">[--:--:--] · no issues.</div></div>
  </main>
</div>

<footer class="foot">
  <span>built for hands-on inference work · MIT</span>
  <span>companion to <a href="../reference/MODEL_SIZING_SCALING_REFERENCE.html">MODEL_SIZING_SCALING_REFERENCE</a></span>
</footer>
"""


def wrap_template(css: str, hw_json: str, model_json: str, bundle: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Inference Sizing Calculator</title>
<meta name="description" content="Analytical sizing calculator for LLM inference: parallelism, b_crit/b_slo/b_kv, max_num_batched_tokens, and empirical sweep ranges before any benchmark run.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Martian+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>{css}</style>
</head>
<body>
{body_html()}
<script type="application/json" id="hardware-data">{hw_json}</script>
<script type="application/json" id="models-data">{model_json}</script>
<script type="module">
{bundle}
</script>
</body>
</html>
"""


def main() -> None:
    css = (SRC / "styles.css").read_text()
    hw_json = (SRC / "data" / "hardware.json").read_text()
    model_json = (SRC / "data" / "models.json").read_text()
    bundle = bundle_modules()
    html = wrap_template(css, hw_json, model_json, bundle)
    DST.write_text(html)
    print(f"wrote {DST} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
