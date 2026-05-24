// DOM wiring layer. Reads the embedded data tables, builds the form,
// dispatches recompute on every input change, and renders results into the
// readout tiles, oscilloscope, patchbay, snippet, and diagnostics log.
//
// All numeric formatting funnels through `fmt()` so display rounding stays
// consistent across tiles, chart axes, and the copy-paste snippet.

import { compute, DTYPE_BYTES } from "./calc.mjs";
import { createScope } from "./chart.mjs";

// ---------- data tables ----------
// Embedded as <script type="application/json"> by the build script; the build
// test guarantees these round-trip cleanly.
const HARDWARE = JSON.parse(document.getElementById("hardware-data").textContent).hardware;
const MODELS = JSON.parse(document.getElementById("models-data").textContent).models;

const HW_BY_KEY    = Object.fromEntries(HARDWARE.map((h) => [h.key, h]));
const MODEL_BY_KEY = Object.fromEntries(MODELS.map((m) => [m.key, m]));

// ---------- formatting helpers ----------

// Smart number formatter. Tiny values get a decimal; large values get grouping
// separators; non-finite or zero gets the em-dash placeholder (engineering UI
// convention — "no signal" rather than a misleading "0").
const fmt = (v, opts = {}) => {
  if (v == null || !Number.isFinite(v)) return "—";
  if (opts.percent) return `${(v * 100).toFixed(1)}%`;
  if (opts.gb) return `${v.toFixed(v < 10 ? 2 : 1)}`;
  if (Math.abs(v) >= 10000) return Math.round(v).toLocaleString("en-US");
  if (Math.abs(v) >= 100) return Math.round(v).toString();
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(2);
};

const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
const setHTML = (id, html) => { const el = document.getElementById(id); if (el) el.innerHTML = html; };

// ---------- form setup ----------

function populatePresets() {
  const hwSel = document.getElementById("hw");
  const modelSel = document.getElementById("model");
  for (const h of HARDWARE) hwSel.add(new Option(h.label, h.key));
  hwSel.add(new Option("Custom…", "__custom"));
  for (const m of MODELS) modelSel.add(new Option(m.label, m.key));
  modelSel.add(new Option("Custom…", "__custom"));
  // Sensible defaults: H100 + Llama-3-8B — the most common starting point.
  hwSel.value = "H100-80GB";
  modelSel.value = "llama-3-8b";
}

function readForm() {
  // Custom mode pulls raw fields from the reveal panel; preset mode looks up
  // the table by key. Either way we end up with a plain hw/model object.
  const hwKey = document.getElementById("hw").value;
  const modelKey = document.getElementById("model").value;
  const customHw = hwKey === "__custom";
  const customModel = modelKey === "__custom";
  document.getElementById("hw-custom").open = customHw;
  document.getElementById("model-custom").open = customModel;

  const hw = customHw ? {
    key: "custom", label: "Custom HW",
    hbm_gb: +document.getElementById("c-hbm-gb").value,
    hbm_bw_gbs: +document.getElementById("c-hbm-bw").value,
    fp16_tflops: +document.getElementById("c-fp16").value,
    fp8_tflops: document.getElementById("c-fp8").value ? +document.getElementById("c-fp8").value : null,
    nvlink: document.getElementById("c-nvlink").checked,
  } : HW_BY_KEY[hwKey];

  const model = customModel ? (() => {
    // Custom-mode model: read total + active params. If active is blank/zero,
    // mirror it from total (the dense one-click case). Fallback to legacy
    // single c-params input if it still exists (HTML may not be rebuilt yet).
    const totalEl  = document.getElementById("c-params-total");
    const activeEl = document.getElementById("c-params-active");
    const legacyEl = document.getElementById("c-params");
    const total  = totalEl  ? +totalEl.value  : (legacyEl ? +legacyEl.value : 0);
    const activeRaw = activeEl ? +activeEl.value : 0;
    const active = activeRaw > 0 ? activeRaw : total;
    return ({
    key: "custom", label: "Custom Model",
    params_b_total: total,
    params_b_active: active,
    n_layers: +document.getElementById("c-layers").value,
    d_model: +document.getElementById("c-dmodel").value,
    n_heads: +document.getElementById("c-heads").value,
    n_kv_heads: +document.getElementById("c-kvheads").value,
    head_dim: +document.getElementById("c-headdim").value,
    max_context: +document.getElementById("c-maxctx").value,
    attn_type: document.getElementById("c-attn").value,
  });
  })() : MODEL_BY_KEY[modelKey];

  return {
    hw, model,
    isl: +document.getElementById("isl").value,
    osl: +document.getElementById("osl").value,
    weight_prec: document.getElementById("weight-prec").value,
    kv_prec: document.getElementById("kv-prec").value,
    act_prec: document.getElementById("act-prec").value,
    tbt_ms: +document.getElementById("tbt").value,
    ttft_ms: +document.getElementById("ttft").value,
    ngpus: +document.getElementById("ngpus").value,
  };
}

// ---------- recompute pipeline ----------

let scope = null;

function recompute(skipPulse = false) {
  const input = readForm();
  const out = compute(input);
  paintMetrics(out, input);
  paintChart(out);
  paintSweep(out, input);
  paintDiagnostics(out);
  paintSnippet(out, input);
  if (!skipPulse) pulseTiles();
}

function paintMetrics(out, input) {
  const m = out.metrics;
  setText("m-bcrit", fmt(m.b_crit));
  setText("m-bslo",  fmt(m.b_slo));
  setText("m-bkv",   fmt(m.b_kv));
  setText("m-mns",   fmt(m.max_num_seqs));
  setText("m-mnbt",  fmt(m.max_num_batched_tokens));
  setText("m-mnbt-band", m.max_num_batched_tokens_band);
  setText("m-tps",   fmt(m.throughput_tps));
  setText("m-tps-pg", `${fmt(m.throughput_tps_per_gpu)} per GPU`);
  setText("m-pd",    m.pd_ratio == null ? "—" : `${fmt(m.pd_ratio)} : 1`);
  const p = m.parallelism;
  setText("m-par",   p.fits ? `TP=${p.tp} · PP=${p.pp}` : "—");
  setText("m-par-sub", p.fits ? `${p.replicas} replica${p.replicas === 1 ? "" : "s"} · ${p.reason}` : p.reason);
  setText("m-weights", fmt(m.weights_gb, { gb: true }) + " GB");
  setText("m-kv", fmt(m.kv_per_token_kb) + " kB/token · " + fmt(m.kv_per_seq_gb, { gb: true }) + " GB/seq");

  // Sparkline placement: drop a tiny SVG under each B tile showing curve
  // position. Anchored to the chart curve so the spark and main chart agree.
  drawSparkline("spark-bcrit", out.chart.curve, m.b_crit, "#58a6ff");
  drawSparkline("spark-bslo",  out.chart.curve, m.b_slo,  "#d29922");
  drawSparkline("spark-bkv",   out.chart.curve, m.b_kv,   "#3fb950");
}

function drawSparkline(id, curve, markB, color) {
  const el = document.getElementById(id);
  if (!el || curve.length < 2) return;
  const w = 220, h = 22, pad = 1;
  const maxT = Math.max(...curve.map((p) => p.step_ms));
  const minB = curve[0].B, maxB = curve[curve.length - 1].B;
  const lx = (B) => pad + (Math.log(B) - Math.log(minB)) / (Math.log(maxB) - Math.log(minB)) * (w - 2 * pad);
  const ly = (t) => h - pad - (t / maxT) * (h - 2 * pad);
  const path = curve.map((p, i) => `${i === 0 ? "M" : "L"}${lx(p.B).toFixed(1)},${ly(p.step_ms).toFixed(1)}`).join(" ");
  const mx = markB && Number.isFinite(markB) && markB >= minB && markB <= maxB ? lx(markB) : null;
  el.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" width="100%" height="${h}">
      <path d="${path}" fill="none" stroke="#30363d" stroke-width="1" />
      ${mx != null ? `<line x1="${mx.toFixed(1)}" y1="0" x2="${mx.toFixed(1)}" y2="${h}" stroke="${color}" stroke-width="1"/>` : ""}
    </svg>`;
}

function paintChart(out) {
  if (!scope) {
    scope = createScope(document.getElementById("scope-canvas"), out.chart);
  } else {
    scope.update(out.chart);
  }
}

function paintSweep(out, input) {
  const tracks = [
    { label: "MAX_NUM_SEQS",        values: out.sweep.max_num_seqs,             rec: out.metrics.max_num_seqs },
    { label: "MAX_NUM_BATCHED_TOK", values: out.sweep.max_num_batched_tokens,   rec: out.metrics.max_num_batched_tokens },
    { label: "CONCURRENCY",         values: out.sweep.concurrency,              rec: out.metrics.max_num_seqs },
  ];
  const html = tracks.map(({ label, values, rec }) => `
    <div class="patch-label">${label}</div>
    <div class="patch-track">
      ${values.map((v) => `<span class="chip${v === rec ? " recommended" : ""}">${v}</span>`).join("")}
    </div>
  `).join("");
  setHTML("patchbay", html);
}

function paintDiagnostics(out) {
  const el = document.getElementById("diagnostics");
  if (!out.warnings.length) {
    el.innerHTML = `<div class="empty">[--:--:--] · no issues.</div>`;
    return;
  }
  // Timestamp uses recompute moment, not now() in a loop, so all lines share
  // the same tick — easier to scan than time-jittered logs.
  const t = new Date();
  const hh = String(t.getHours()).padStart(2, "0");
  const mm = String(t.getMinutes()).padStart(2, "0");
  const ss = String(t.getSeconds()).padStart(2, "0");
  const ms = String(t.getMilliseconds()).padStart(3, "0");
  const stamp = `${hh}:${mm}:${ss}.${ms}`;
  el.innerHTML = out.warnings.map((w) => `
    <div class="diag-line">
      <time>[${stamp}]</time>
      <span class="glyph ${w.level === "error" ? "error" : "warn"}">${w.level === "error" ? "■" : "▲"}</span>
      <span class="msg">${escapeHtml(w.msg)}</span>
    </div>
  `).join("");
}

const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function paintSnippet(out, input) {
  const m = out.metrics;
  const lines = [
    `# vLLM serving args (analytical recommendation — verify empirically).`,
    `# Model:    ${input.model.label}   Hardware: ${input.hw.label} × ${input.ngpus}`,
    `# ISL/OSL: ${input.isl}/${input.osl}   TBT SLO: ${input.tbt_ms} ms   Precision: w=${input.weight_prec} kv=${input.kv_prec}`,
    ``,
    `vllm serve <model-id> \\`,
    `  --tensor-parallel-size ${m.parallelism.tp} \\`,
    `  --pipeline-parallel-size ${m.parallelism.pp} \\`,
    `  --max-num-seqs ${m.max_num_seqs} \\`,
    `  --max-num-batched-tokens ${m.max_num_batched_tokens} \\`,
    `  --max-model-len ${input.isl + input.osl} \\`,
    `  --gpu-memory-utilization ${(1 - 0.10).toFixed(2)}`,
    ``,
    `# Empirical sweep grid:`,
    `concurrency_list = [${out.sweep.concurrency.join(", ")}]`,
    `max_num_seqs_grid = [${out.sweep.max_num_seqs.join(", ")}]`,
    `mnbt_grid = [${out.sweep.max_num_batched_tokens.join(", ")}]`,
  ];
  document.getElementById("snippet-body").textContent = lines.join("\n");
}

function pulseTiles() {
  for (const el of document.querySelectorAll(".readout")) {
    el.classList.remove("pulse");
    // force reflow so the animation restarts (CSS @keyframes trick).
    void el.offsetWidth;
    el.classList.add("pulse");
  }
}

// ---------- wiring ----------

function wireCopyButton() {
  const btn = document.getElementById("copy-btn");
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(document.getElementById("snippet-body").textContent);
      btn.classList.add("flash");
      btn.textContent = "[ COPIED ]";
      setTimeout(() => { btn.classList.remove("flash"); btn.textContent = "[ COPY ]"; }, 1200);
    } catch (e) {
      btn.textContent = "[ COPY FAILED ]";
    }
  });
}

export function bootstrap() {
  populatePresets();
  const form = document.getElementById("sizing-form");
  form.addEventListener("input", () => recompute());
  form.addEventListener("change", () => recompute());
  wireCopyButton();
  // Skip the pulse on first paint — pulse should signal change, not arrival.
  recompute(true);
}

// Bootstrap automatically when the script tag executes after DOM ready. The
// build script places <script type="module">…bootstrap()…</script> at the end
// of <body>, so the DOM is already parsed.
bootstrap();
