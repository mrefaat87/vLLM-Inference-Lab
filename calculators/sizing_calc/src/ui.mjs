// DOM wiring layer. Reads the embedded data tables, builds the form,
// dispatches recompute on every input change, and renders results into the
// readout tiles, oscilloscope, patchbay, snippet, and diagnostics log.
//
// All numeric formatting funnels through `fmt()` so display rounding stays
// consistent across tiles, chart axes, and the copy-paste snippet.

import { compute, DTYPE_BYTES } from "./calc.mjs";
import { createScope } from "./chart.mjs";
import { recommendedRate } from "./lab_command.mjs";
import {
  DEFAULT_BRIDGE_URL,
  loadLabRuns,
  filterRuns,
  fetchRun,
  projectRun,
  emptyStateMessage,
} from "./validation.mjs";

// Lab-runs state: fetched once on first recompute, cached for the session.
// `_promise` so concurrent recomputes don't refetch.
const labRunsState = {
  _promise: null,
  _runCache: new Map(),
  async get() {
    if (!this._promise) this._promise = loadLabRuns({ bridgeUrl: DEFAULT_BRIDGE_URL });
    return this._promise;
  },
  fetchRun(href) {
    return fetchRun({ bridgeBase: "./sizing_calc/lab_runs", href, cache: this._runCache });
  },
};
import { mountFormulasDrawer } from "./drawer.mjs";

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
  // Seed price input from the default hardware row.
  syncPriceToHardware();
}

// Sync the price-per-hour input to the currently-selected hardware row's
// default. Called on initial paint and on every hw row swap — within-row
// edits leave the user's override alone. Plan Layer 6 test #12.
function syncPriceToHardware() {
  const hwKey = document.getElementById("hw").value;
  const priceEl = document.getElementById("price-per-hour");
  if (!priceEl) return;
  if (hwKey === "__custom") {
    priceEl.value = ""; // custom row → blank → propagates null cost
    return;
  }
  const hw = HW_BY_KEY[hwKey];
  if (hw && hw.price_per_hour_usd != null) {
    priceEl.value = hw.price_per_hour_usd;
  }
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

  // Price override: blank input → fall back to hw.price_per_hour_usd; for
  // custom hardware with no default this propagates as null and the cost tile
  // renders "—". Anything > 0 wins over the row default.
  const priceRaw = document.getElementById("price-per-hour")?.value;
  const priceParsed = priceRaw === "" || priceRaw == null ? null : +priceRaw;
  const price_per_hour_usd = (priceParsed != null && Number.isFinite(priceParsed) && priceParsed > 0)
    ? priceParsed
    : (hw?.price_per_hour_usd ?? null);

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
    price_per_hour_usd,
  };
}

// ---------- recompute pipeline ----------

// Three independent Chart.js instances, one per channel. Lazy-init on first
// recompute so the page doesn't pay chart construction cost before the first
// computed data is available. Exposed on window.__sizingScopes for E2E tests
// and manual debugging (convention from prior dashboards).
let scopes = null;

function recompute(skipPulse = false) {
  const input = readForm();
  const out = compute(input);
  paintMetrics(out, input);
  paintChart(out);
  paintSweep(out, input);
  paintDiagnostics(out);
  paintSnippet(out, input);
  // Lab runs: fire-and-forget against the bridge. Failures show an
  // empty-state card; they never block the calc render.
  paintLabRuns(input, out).catch((e) => {
    // Last-resort safety net — paintLabRuns already swallows non-throwing
    // failures, so this only fires on a genuine bug worth logging.
    console.warn("paintLabRuns failed", e);
  });
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
  // Cost tile. null (custom HW with blank price) → "—"; Infinity (tps=0) → "—".
  // Two-decimal format reads as currency; vendor list comparisons need cents.
  const costTxt = (m.cost_per_mtok == null || !Number.isFinite(m.cost_per_mtok))
    ? "—"
    : `$${m.cost_per_mtok.toFixed(2)}`;
  setText("m-cost", costTxt);
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
  // Filter out non-positive step_ms (defensive — log() blows up on ≤0).
  const pts = curve.filter((p) => p.step_ms > 0 && p.B > 0);
  if (pts.length < 2) return;
  const minT = Math.min(...pts.map((p) => p.step_ms));
  const maxT = Math.max(...pts.map((p) => p.step_ms));
  const minB = pts[0].B, maxB = pts[pts.length - 1].B;
  const lx = (B) => pad + (B - minB) / (maxB - minB) * (w - 2 * pad);
  // Linear Y. Trade-off: the memory-bound region below B_crit compresses
  // against the bottom edge so curve shape there is barely readable; the
  // compute-bound tail dominates. Kept linear for consistency with the main
  // scopes (also flipped to linear). Falls back to mid-line if degenerate.
  const tSpan = maxT - minT;
  const ly = tSpan > 0
    ? (t) => h - pad - (t - minT) / tSpan * (h - 2 * pad)
    : () => (h - pad - (h - 2 * pad) / 2);
  const path = pts.map((p, i) => `${i === 0 ? "M" : "L"}${lx(p.B).toFixed(1)},${ly(p.step_ms).toFixed(1)}`).join(" ");
  const mx = markB && Number.isFinite(markB) && markB >= minB && markB <= maxB ? lx(markB) : null;
  // Tint the curve with the metric color at low opacity — the old `#30363d`
  // hairline was barely visible against the tile background. Marker stays
  // dominant at 1.5px solid; curve reads as a faint contextual backdrop.
  el.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" width="100%" height="${h}">
      <path d="${path}" fill="none" stroke="${color}" stroke-opacity="0.35" stroke-width="1.2" />
      ${mx != null ? `<line x1="${mx.toFixed(1)}" y1="0" x2="${mx.toFixed(1)}" y2="${h}" stroke="${color}" stroke-width="1.5"/>` : ""}
    </svg>`;
}

function paintChart(out) {
  if (!scopes) {
    // First paint: stagger scope construction by 120ms each so the three
    // panels reveal in top-down reading order. Each scope's own marker
    // stagger (80ms apart) layers on top of this. Skip the stagger entirely
    // under prefers-reduced-motion — construct all three immediately.
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const canvases = {
      latency:    document.getElementById("scope-latency-canvas"),
      throughput: document.getElementById("scope-throughput-canvas"),
      cost:       document.getElementById("scope-cost-canvas"),
    };
    scopes = {};
    const order = ["latency", "throughput", "cost"];
    const mount = (ch) => {
      if (!canvases[ch]) return;
      scopes[ch] = createScope(canvases[ch], ch, out.chart);
    };
    if (reduceMotion) {
      for (const ch of order) mount(ch);
    } else {
      // Mount latency immediately so the first scope is visible at t=0; the
      // other two stagger in. Avoids a 240ms blank-canvas flash on slow GPUs.
      mount("latency");
      setTimeout(() => mount("throughput"), 120);
      setTimeout(() => mount("cost"),       240);
    }
    if (typeof window !== "undefined") {
      window.__sizingScopes = scopes;
      // Back-compat alias for older E2E tests that read window.__sizingChart.
      // Points at the latency scope's chart (the panel that previously *was*
      // "the chart"). Set lazily once latency mounts so the test wait works.
      const setAlias = () => {
        if (scopes.latency) window.__sizingChart = scopes.latency.chart;
        else setTimeout(setAlias, 50);
      };
      setAlias();
    }
  } else {
    for (const s of Object.values(scopes)) s.update(out.chart);
  }
}

async function paintLabRuns(input, _calcOut) {
  const el = document.getElementById("lab-runs");
  if (!el) return;
  const hwKey = input.hw?.key;
  const modelKey = input.model?.key;
  if (!hwKey || !modelKey) {
    el.innerHTML = `<div class="empty">[lab] · custom hw/model — empirical runs join on stable keys only.</div>`;
    return;
  }

  const { index, unavailable } = await labRunsState.get();
  if (unavailable) {
    el.innerHTML = `<div class="empty">[lab] · ${escapeHtml(emptyStateMessage({ unavailable }))}</div>`;
    return;
  }
  const matches = filterRuns(index, modelKey, hwKey);
  if (matches.length === 0) {
    const combos = [...new Map(
      index.map((r) => [`${r.model_ref}|${r.hw_ref}`, { model_ref: r.model_ref, hw_ref: r.hw_ref }])
    ).values()];
    el.innerHTML =
      `<div class="empty">[lab] · ${emptyStateMessage({ unavailable: false, availableCombos: combos })}</div>`;
    return;
  }

  // Fetch each matching run lazily, then render a compact list.
  const fulls = await Promise.all(matches.map((m) => labRunsState.fetchRun(m.result_href)));
  const rows = matches.map((m, i) => {
    const run = fulls[i];
    if (!run) {
      return `<div class="lab-row missing">${escapeHtml(m.run_id)} · <span class="lab-tag">[404]</span></div>`;
    }
    const proj = projectRun(run);
    const div = proj.divergenceRatio;
    const divLabel = div != null ? `${div.toFixed(2)}×` : "—";
    const divClass = div == null ? "neutral"
      : (div < 0.5 || div > 2.0) ? "bad"
      : (div < 0.8 || div > 1.25) ? "warn"
      : "good";
    const tps = proj.tps != null ? proj.tps.toFixed(0) : "—";
    const ttft = proj.ttftMs != null ? proj.ttftMs.toFixed(1) : "—";
    const batch = proj.batchProxy != null ? proj.batchProxy.toFixed(1) : "—";
    return `
      <div class="lab-row">
        <div class="lab-row-head">
          <span class="lab-tag">${escapeHtml(run.engine?.name || "?")}</span>
          <span class="lab-tag">${escapeHtml(run.workload?.name || "?")}</span>
          <code>${escapeHtml(m.run_id)}</code>
        </div>
        <div class="lab-row-metrics">
          <span>batch≈<strong>${batch}</strong></span>
          <span>TTFT p50 <strong>${ttft} ms</strong></span>
          <span>TPS <strong>${tps}</strong> tok/s</span>
          <span class="lab-div lab-div-${divClass}">measured/predicted = <strong>${divLabel}</strong></span>
        </div>
      </div>
    `;
  }).join("");

  const summary = `<div class="lab-summary">[lab] · ${matches.length} run(s) for <code>${escapeHtml(modelKey)}</code> × <code>${escapeHtml(hwKey)}</code></div>`;
  el.innerHTML = summary + rows;
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

// ──────────────────────────────────────────────────────────────────────────
// Calc → lab mapping tables.
//
// The calc thinks in (model.key, weight_prec, hw.key, ngpus). The lab's
// `exp run` CLI takes those plus a few engine-specific strings:
//   --model       full HF repo id (depends on the quantization variant)
//   --quant       lab's --quant token (none/awq/fp8/int8)
//   --tp          tensor-parallel size
//   --instance    AWS instance family.size string (metadata; informational)
//   --gpu        GPU SKU string (metadata)
//   --n-gpu       integer count
//
// These tables centralize the mapping so the snippet emits a runnable
// command, not a half-fledged template.
// ──────────────────────────────────────────────────────────────────────────

// (model.key, weight_prec) → HF repo id. INT4 / INT8 / FP8 variants live
// at separate HF paths because quantized weights are produced offline; the
// engine can't downcast on the fly. Falls back to the unquantized path
// when no special variant is published.
const HF_PATHS = {
  "llama-3-8b": {
    BF16: "meta-llama/Meta-Llama-3-8B-Instruct",
    FP16: "meta-llama/Meta-Llama-3-8B-Instruct",
    INT4: "casperhansen/llama-3-8b-instruct-awq",
    FP8:  "neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
  },
  "llama-3-70b": {
    BF16: "meta-llama/Meta-Llama-3-70B-Instruct",
    FP16: "meta-llama/Meta-Llama-3-70B-Instruct",
    INT4: "casperhansen/llama-3-70b-instruct-awq",
    FP8:  "neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
  },
  "qwen2.5-7b":  { BF16: "Qwen/Qwen2.5-7B-Instruct",  INT4: "Qwen/Qwen2.5-7B-Instruct-AWQ" },
  "qwen2.5-14b": { BF16: "Qwen/Qwen2.5-14B-Instruct", INT4: "Qwen/Qwen2.5-14B-Instruct-AWQ" },
  "mistral-7b":  { BF16: "mistralai/Mistral-7B-Instruct-v0.3" },
  "deepseek-v3": { BF16: "deepseek-ai/DeepSeek-V3", FP8: "deepseek-ai/DeepSeek-V3" },
};
function modelHfPath(modelKey, weightPrec) {
  const table = HF_PATHS[modelKey];
  if (!table) return `<HF-REPO-FOR-${modelKey}>`;
  return table[weightPrec] || table.BF16 || table.FP16 || `<HF-REPO-FOR-${modelKey}>`;
}

// weight_prec → lab --quant token. INT4 defaults to awq because that's the
// most widely shipped int4 path; users running gptq can override on the CLI.
const QUANT_FROM_PREC = { FP16: "none", BF16: "none", FP8: "fp8", INT8: "int8", INT4: "awq" };
function quantArg(weightPrec) { return QUANT_FROM_PREC[weightPrec] || "none"; }

// hw.key → AWS instance family. P-family ships only full 8-GPU nodes;
// G-family ships smaller SKUs that scale 1/4/8 GPUs. Sizes verified
// against the AWS EC2 console (Apr 2025): p4*.24xlarge, p5*.48xlarge.
const INSTANCE_FAMILY = {
  T4: "g4dn", L4: "g6", A10G: "g5",
  "A100-40GB": "p4d", "A100-80GB": "p4de",
  "H100-80GB": "p5",  "H200": "p5en",
};
// Per-family size suffix for an 8-GPU full node.
const FULL_NODE_SUFFIX = {
  p4d: "24xlarge", p4de: "24xlarge",
  p5:  "48xlarge", p5en: "48xlarge",
};
function awsInstance(hwKey, ngpus) {
  const fam = INSTANCE_FAMILY[hwKey];
  if (!fam) return `<INSTANCE-${hwKey}>`;
  if (fam === "g4dn" || fam === "g5" || fam === "g6") {
    // G-family scales: 1 GPU = .xlarge, 4 = .12xlarge, 8 = .24xlarge.
    if (ngpus === 1) return `${fam}.xlarge`;
    if (ngpus === 4) return `${fam}.12xlarge`;
    if (ngpus === 8) return `${fam}.24xlarge`;
    return `${fam}.${ngpus}gpu`;          // odd count → placeholder
  }
  // P-family: only the full 8-GPU SKU exists. If the user picked fewer
  // GPUs the instance type is still the full node; -gpu just tells the
  // engine how many to bind. Emit with a TODO comment so the user knows
  // they're paying for the whole box.
  const suffix = FULL_NODE_SUFFIX[fam] || "48xlarge";
  return `${fam}.${suffix}`;
}

function paintSnippet(out, input) {
  const m = out.metrics;
  // Map the calc's workload-preset value to the lab's --workload name.
  // Most names line up 1:1; the calc's split presets (chatbot_turn1 /
  // chatbot_mid, agentic_step / agentic_edit) collapse onto the lab's
  // single generator. "custom" means no preset is selected.
  const presetEl = document.getElementById("workload-preset");
  const preset = presetEl ? presetEl.value : "custom";
  const LAB_WORKLOAD = {
    chatbot_turn1: "chatbot", chatbot_mid: "chatbot",
    rag: "rag",
    summarize: "rag",                 // closest lab analog (long prompt + short answer)
    extract: "rag",
    copilot_inline: "chatbot",        // small ctx, short output — chatbot generator fits
    tool_call: "agentic_coding",
    agentic_step: "agentic_coding",
    agentic_edit: "agentic_coding",
    reason_easy: "chatbot",           // no dedicated reasoning workload yet
    reason_hard: "chatbot",
    embedding: null,                  // lab doesn't drive /v1/embeddings
    custom: null,
  };
  const labWorkload = LAB_WORKLOAD[preset] ?? null;

  // Which lab driver to spin up — vllm / sglang / trtllm. The
  // analytical model is engine-agnostic (it only knows roofline +
  // KV math), so this knob only affects the exp launch command.
  const engineEl = document.getElementById("lab-engine");
  const labEngine = engineEl ? engineEl.value : "vllm";

  // Recommended arrival rate for the lab. See lab_command.mjs for the
  // derivation — kept in a sibling module so it stays unit-testable.
  const recBatch = (m && (m.recommended_batch ?? m.b_crit)) || 1;
  const rateRps = recommendedRate(recBatch, input.tbt_ms, input.osl, input.ttft_ms);

  const lines = [
    `# vLLM serve args — analytical recommendation; verify empirically below.`,
    `# Model     : ${input.model.label}   Hardware: ${input.hw.label} × ${input.ngpus}`,
    `# Precision : w=${input.weight_prec}  kv=${input.kv_prec}  act=${input.act_prec}`,
    `# Workload  : ISL=${input.isl} tok  OSL=${input.osl} tok  TBT SLO=${input.tbt_ms} ms`,
    ``,
    `vllm serve <model-id> \\`,
    `  --tensor-parallel-size ${m.parallelism.tp} \\`,
    `  --pipeline-parallel-size ${m.parallelism.pp} \\`,
    `  --max-num-seqs ${m.max_num_seqs} \\`,
    `  --max-num-batched-tokens ${m.max_num_batched_tokens} \\`,
    `  --max-model-len ${input.isl + input.osl}   # = ISL ${input.isl} + OSL ${input.osl} \\`,
    `  --gpu-memory-utilization ${(1 - 0.10).toFixed(2)}`,
    ``,
    `# ── Drive measured traffic at this shape (companion empirical lab) ──`,
    labWorkload
      ? `exp launch --engine ${labEngine} --workload ${labWorkload} \\`
      : `# (no lab workload maps to preset "${preset}" — pick a preset above)`,
    // Model + quant + parallelism — derived from the calc inputs so the
    // empirical run and the predicted curve use the same engine config.
    labWorkload ? `  --model ${modelHfPath(input.model.key, input.weight_prec)} \\` : null,
    labWorkload ? `  --quant ${quantArg(input.weight_prec)} \\` : null,
    labWorkload ? `  --tp ${m.parallelism.tp}  --n-gpu ${input.ngpus} \\` : null,
    labWorkload ? `  --instance ${awsInstance(input.hw.key, input.ngpus)}  --gpu ${input.hw.key} \\` : null,
    // Workload knobs.
    labWorkload ? `  --rate ${rateRps}  --duration 300  --warmup 30 \\` : null,
    // Join keys — anchor the result back to this exact calc prediction.
    labWorkload ? `  --model-ref ${input.model.key}  --hw-ref ${input.hw.key} \\` : null,
    labWorkload ? `  --tbt-target-ms ${input.tbt_ms}  --ttft-target-ms ${input.ttft_ms}` : null,
    ``,
    `# Empirical sweep grid (vary one, hold others at recommended):`,
    `concurrency_list      = [${out.sweep.concurrency.join(", ")}]`,
    `max_num_seqs_grid     = [${out.sweep.max_num_seqs.join(", ")}]`,
    `max_num_batched_tok   = [${out.sweep.max_num_batched_tokens.join(", ")}]`,
  ].filter((x) => x !== null);
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

// Extract just the `exp run …` block (the contiguous multi-line command,
// with its trailing line-continuations stripped so the user can paste it
// directly into a terminal).
function extractExpRun(snippetText) {
  const lines = snippetText.split("\n");
  // Accept either verb so older snippets (or someone manually editing
  // back to `exp run`) still work. Match `exp launch ` first since
  // that's what we now emit by default.
  const start = lines.findIndex(
    (l) => l.startsWith("exp launch ") || l.startsWith("exp run "),
  );
  if (start < 0) return null;
  let end = start;
  while (end + 1 < lines.length && lines[end].endsWith(" \\")) end++;
  // Collect, normalize line continuations to single-line, then back into
  // pretty multi-line so the paste reads well in a terminal.
  const block = lines.slice(start, end + 1).join("\n");
  return block;
}

function wireCopyButton(btnId, getText, restoreLabel) {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  const originalLabel = btn.textContent;
  btn.addEventListener("click", async () => {
    const text = getText();
    if (text == null) {
      btn.textContent = "[ NO LAB MAPPING ]";
      setTimeout(() => { btn.textContent = restoreLabel || originalLabel; }, 1200);
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      btn.classList.add("flash");
      btn.textContent = "[ COPIED ]";
      setTimeout(() => {
        btn.classList.remove("flash");
        btn.textContent = restoreLabel || originalLabel;
      }, 1200);
    } catch (e) {
      btn.textContent = "[ COPY FAILED ]";
      setTimeout(() => { btn.textContent = restoreLabel || originalLabel; }, 1500);
    }
  });
}

function wireCopyButtons() {
  const body = () => document.getElementById("snippet-body").textContent;
  wireCopyButton("copy-btn",     () => body(),                  "[ COPY ALL ]");
  wireCopyButton("copy-exp-btn", () => extractExpRun(body()),   "[ COPY EXP RUN ]");
}

// Workload preset → ISL/OSL wiring.
// When the user picks a preset (anything other than "custom"), copy the
// `data-isl` / `data-osl` from the chosen <option> onto the ISL/OSL
// inputs and dispatch input events so the form-level listener
// recomputes. When the user edits either input manually, snap the
// preset select back to "custom" so the visible label doesn't lie.
function wireWorkloadPreset() {
  const sel = document.getElementById("workload-preset");
  const isl = document.getElementById("isl");
  const osl = document.getElementById("osl");
  if (!sel || !isl || !osl) return;
  let suppressRevert = false;     // set while WE are writing to the inputs
  sel.addEventListener("change", () => {
    const opt = sel.selectedOptions[0];
    if (!opt) return;
    const v_isl = opt.dataset.isl;
    const v_osl = opt.dataset.osl;
    if (!v_isl || !v_osl) return; // "Custom" — leave values alone
    suppressRevert = true;
    isl.value = v_isl;
    osl.value = v_osl;
    isl.dispatchEvent(new Event("input", { bubbles: true }));
    osl.dispatchEvent(new Event("input", { bubbles: true }));
    suppressRevert = false;
  });
  const revertToCustom = () => {
    if (suppressRevert) return;
    if (sel.value !== "custom") sel.value = "custom";
  };
  isl.addEventListener("input", revertToCustom);
  osl.addEventListener("input", revertToCustom);
}

// ──────────────────────────────────────────────────────────────────────────
// Query-string prefill (Flow B: lab → calc).
//
// When the calc is opened from the Results Explorer's "open this run in
// calc" link, the URL carries the same input fields the lab's Prediction
// snapshot was built from. We seed each <input>/<select> from the query
// before the first `recompute()` so the first paint already reflects the
// run. Unknown / malformed params are silently ignored — the form keeps
// its built-in defaults, mirroring how unrecognized URLs degrade to a
// "fresh calc" rather than blocking the page.
//
// Field map (query key → DOM id):
//   model       → #model        (select; ignored if value isn't a known option)
//   hw          → #hw
//   weight_prec → #weight-prec
//   kv_prec     → #kv-prec
//   act_prec    → #act-prec
//   isl, osl, ngpus → numeric inputs
//   tbt_ms      → #tbt
//
// Returns a dict of {appliedKey: appliedValue} — useful for unit tests
// asserting which fields actually round-tripped.
// ──────────────────────────────────────────────────────────────────────────
export function prefillFromQueryString(search) {
  // Accept an explicit search string for unit tests; default to the
  // live URL when called without args (the bootstrap path).
  const raw = search ?? (typeof window !== "undefined" ? window.location.search : "");
  if (!raw) return {};
  const params = new URLSearchParams(raw);
  const applied = {};

  // Setter for <select>: only write when the requested value is one of
  // the offered <option>s. Assigning an unknown value would silently set
  // the empty string and mask the form's preset default.
  const setSelect = (id, key) => {
    if (!params.has(key)) return;
    const el = document.getElementById(id);
    if (!el) return;
    const want = params.get(key);
    const ok = Array.from(el.options).some((o) => o.value === want);
    if (!ok) return;
    el.value = want;
    applied[key] = want;
  };
  const setNumber = (id, key) => {
    if (!params.has(key)) return;
    const el = document.getElementById(id);
    if (!el) return;
    const n = Number(params.get(key));
    if (!Number.isFinite(n)) return;          // ignore non-numeric
    el.value = String(n);
    applied[key] = n;
  };

  setSelect("model",       "model");
  setSelect("hw",          "hw");
  setSelect("weight-prec", "weight_prec");
  setSelect("kv-prec",     "kv_prec");
  setSelect("act-prec",    "act_prec");
  setNumber("isl",         "isl");
  setNumber("osl",         "osl");
  setNumber("ngpus",       "ngpus");
  setNumber("tbt",         "tbt_ms");
  return applied;
}

export function bootstrap() {
  populatePresets();
  wireWorkloadPreset();
  // Prefill from ?model=…&hw=… before the first recompute so the initial
  // paint reflects the deep link (e.g. "open this run in calc" from the
  // Results Explorer). Must run AFTER populatePresets so the option lists
  // exist; BEFORE recompute so the first chart uses the prefilled values.
  // After a hw change we also re-sync the $/hr field so the cost tile
  // matches the new row's default price.
  const applied = prefillFromQueryString();
  if (applied.hw) syncPriceToHardware();
  const form = document.getElementById("sizing-form");
  form.addEventListener("input", () => recompute());
  form.addEventListener("change", () => recompute());
  // Hardware row swap → reset price input to the new row's default. Within-row
  // edits leave the user's override alone (the change event only fires on
  // <select> changes, not on price input edits). Fire an `input` event after
  // setting the value so the form-level input listener picks up the new price
  // and recomputes — plan Layer 6 test #12 anchor.
  document.getElementById("hw").addEventListener("change", () => {
    syncPriceToHardware();
    const priceEl = document.getElementById("price-per-hour");
    if (priceEl) priceEl.dispatchEvent(new Event("input", { bubbles: true }));
  });
  wireCopyButtons();
  // Side-drawer for "show all formulas" — populated once from the inlined
  // formulas-data tag and kept hidden until the user opens it. Idempotent;
  // no-op if its DOM scaffold isn't present (test harnesses).
  mountFormulasDrawer();
  // Skip the pulse on first paint — pulse should signal change, not arrival.
  recompute(true);
}

// Bootstrap automatically when the script tag executes after DOM ready. The
// build script places <script type="module">…bootstrap()…</script> at the end
// of <body>, so the DOM is already parsed.
//
// Skip the auto-bootstrap when we're not in a browser (e.g. node --test
// imports for unit tests). The presence of `window` is the canonical
// real-DOM signal; `globalThis.Option` is undefined under bare Node so
// even a stubbed document can't fully fake the form-bootstrap path.
if (typeof window !== "undefined" && typeof Option !== "undefined") {
  bootstrap();
}
