<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>EMPIRICAL LAB · Results Explorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link rel="stylesheet"
  href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Martian+Mono:wght@300;400;500&display=swap" />
<link rel="stylesheet" href="assets/style.css" />
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.js"></script>
</head>
<body>

<div class="strip">
  <div class="strip-left">
    <span class="brand">EMPIRICAL LAB</span>
    <span class="sep"></span>
    <span><span class="live-dot"></span>TELEMETRY</span>
  </div>
  <nav class="bench-switcher" aria-label="bench selector">
    <span class="bs-label">BENCH</span>
    <a href="/calculators/sizing_calculator.html" class="bs-item">
      <span class="bs-num">01</span><span class="bs-name">SIZING CALC</span>
    </a>
    <a href="/experiments/_site/command_center.html" class="bs-item">
      <span class="bs-num">02</span><span class="bs-name">COMMAND</span>
    </a>
    <a href="/experiments/_site/results_explorer.html" class="bs-item active">
      <span class="bs-num">03</span><span class="bs-name">RESULTS</span>
    </a>
    <a id="open-in-calc" href="#" class="bs-item bs-open-in-calc" hidden>
      <span class="bs-num">↗</span><span class="bs-name">OPEN THIS RUN IN CALC</span>
    </a>
  </nav>
  <div class="strip-right">
    <a href="/index.html">← ROOT</a>
  </div>
</div>

<main class="bench">

  <h1>
    <span class="sub">▣ scope · measured vs predicted</span>
    Results Explorer
  </h1>
  <p class="lede">Per-run telemetry: TTFT distribution, inter-token latency, throughput at the operating batch, and the predicted curve from the sizing calculator. The two are joined on <code>(model_ref, hw_ref)</code> and travel together in the result JSON.</p>

  <section class="run-bar">
    <div class="field">
      <label>▶ RUN</label>
      <select id="run-picker"></select>
    </div>
    <div class="field">
      <label>▶ COMPARE WITH</label>
      <select id="compare-picker"><option value="">(none)</option></select>
    </div>
    <div class="stat" id="run-count">— RUNS</div>
  </section>

  <section id="summary"></section>
  <section id="prediction-banner" class="prediction-strip" hidden></section>

  <div class="scopes">
    <section class="scope">
      <div class="scope-head">
        <span class="scope-title">▣ TTFT · DISTRIBUTION</span>
        <span class="scope-channels"><span class="ch-bar"><span class="sw"></span>main</span></span>
      </div>
      <div class="scope-canvas-wrap"><canvas id="ttft-chart"></canvas></div>
      <div class="scope-foot">
        <span>seconds · time-to-first-token</span>
        <span>p50 · p95 · p99 · mean</span>
      </div>
    </section>

    <section class="scope">
      <div class="scope-head">
        <span class="scope-title">▣ TBT · INTER-TOKEN LATENCY</span>
        <span class="scope-channels"><span class="ch-bar"><span class="sw"></span>main</span></span>
      </div>
      <div class="scope-canvas-wrap"><canvas id="tbt-chart"></canvas></div>
      <div class="scope-foot">
        <span>seconds · per-token gap</span>
        <span>streaming throughput proxy</span>
      </div>
    </section>

    <section class="scope scope-tall span-2">
      <div class="scope-head">
        <span class="scope-title">▣ THROUGHPUT · MEASURED VS PREDICTED CURVE</span>
        <span class="scope-channels">
          <span class="ch-pred"><span class="sw"></span>predicted (calc)</span>
          <span class="ch-meas"><span class="sw"></span>measured</span>
        </span>
      </div>
      <div class="scope-canvas-wrap"><canvas id="roofline-chart"></canvas></div>
      <div class="scope-foot">
        <span>batch (log axis) · Little's-Law proxy · tokens/sec</span>
        <span id="roofline-foot-right">—</span>
      </div>
    </section>

    <section class="scope">
      <div class="scope-head">
        <span class="scope-title">▣ COST · $ / M OUTPUT TOKENS</span>
        <span class="scope-channels">
          <span class="ch-pred"><span class="sw"></span>predicted</span>
          <span class="ch-meas"><span class="sw"></span>measured</span>
        </span>
      </div>
      <div class="scope-canvas-wrap"><canvas id="cost-chart"></canvas></div>
      <div class="scope-foot">
        <span>batch (log axis) · $ / 1M completion tokens</span>
        <span id="cost-foot-right">—</span>
      </div>
    </section>

    <section class="scope">
      <div class="scope-head">
        <span class="scope-title">▣ STEP TIME · MS / STEP</span>
        <span class="scope-channels">
          <span class="ch-pred"><span class="sw"></span>predicted</span>
          <span class="ch-meas"><span class="sw"></span>measured</span>
        </span>
      </div>
      <div class="scope-canvas-wrap"><canvas id="steptime-chart"></canvas></div>
      <div class="scope-foot">
        <span>batch (log axis) · ms / decode step</span>
        <span id="steptime-foot-right">—</span>
      </div>
    </section>

    <section class="scope">
      <div class="scope-head">
        <span class="scope-title">▣ THROUGHPUT · STEADY STATE</span>
        <span class="scope-channels"><span class="ch-tps"><span class="sw"></span>tok/s</span></span>
      </div>
      <div class="scope-canvas-wrap"><canvas id="throughput-chart"></canvas></div>
      <div class="scope-foot">
        <span>operating throughput</span>
        <span>steady state, warmup excluded</span>
      </div>
    </section>

    <section class="scope">
      <div class="scope-head">
        <span class="scope-title">▣ TIMELINE · TTFT VS SUBMIT</span>
        <span class="scope-channels"><span class="ch-pred"><span class="sw"></span>TTFT (s)</span></span>
      </div>
      <div class="scope-canvas-wrap"><canvas id="timeline-chart"></canvas></div>
      <div class="scope-foot">
        <span>submit offset (s)</span>
        <span>each dot = one request</span>
      </div>
    </section>
  </div>

  <div id="empty" class="empty" hidden>
    [--:--:--] · no runs to explore.
    <span class="hint">Kick one off with <code>exp run …</code> then rebuild the portal.</span>
  </div>
</main>

<footer class="foot">
  <span>EMPIRICAL LAB · MIT</span>
  <span><a href="/calculators/sizing_calculator.html">↗ SIZING CALCULATOR (sibling bench)</a></span>
</footer>

<script>
(async function () {
  // ────────────────────────────────────────────────────────────────────
  // Theme tokens — keep Chart.js visually consistent with the scope panels.
  // ────────────────────────────────────────────────────────────────────
  const css = getComputedStyle(document.documentElement);
  const T = {
    text:   css.getPropertyValue('--text').trim()   || '#e6edf3',
    muted:  css.getPropertyValue('--muted').trim()  || '#8b949e',
    dim:    css.getPropertyValue('--dim').trim()    || '#6e7681',
    grid:   css.getPropertyValue('--grid').trim()   || '#1a1f29',
    rule:   css.getPropertyValue('--rule').trim()   || '#1f2937',
    accent: css.getPropertyValue('--accent').trim() || '#58a6ff',
    bCrit:  css.getPropertyValue('--b-crit').trim() || '#58a6ff',
    bSlo:   css.getPropertyValue('--b-slo').trim()  || '#d29922',
    bKv:    css.getPropertyValue('--b-kv').trim()   || '#3fb950',
    danger: css.getPropertyValue('--danger').trim() || '#f85149',
    mono:   '"Martian Mono", ui-monospace, SFMono-Regular, Menlo, monospace',
  };
  if (window.Chart) {
    Chart.defaults.font.family = T.mono;
    Chart.defaults.font.size = 10;
    Chart.defaults.color = T.muted;
    Chart.defaults.borderColor = T.rule;
    Chart.defaults.plugins.legend.display = false;
  }

  // ────────────────────────────────────────────────────────────────────
  // Load the runs index from the portal.
  // ────────────────────────────────────────────────────────────────────
  const index = await (await fetch('assets/runs.json')).json();
  const done = index.runs.filter(r => r.result_href);
  document.getElementById('run-count').textContent =
    `${done.length} RUN${done.length === 1 ? '' : 'S'}`;

  if (!done.length) {
    document.getElementById('empty').hidden = false;
    document.querySelector('.scopes').hidden = true;
    document.querySelector('.run-bar').hidden = true;
    return;
  }

  const runPicker = document.getElementById('run-picker');
  const cmpPicker = document.getElementById('compare-picker');
  for (const r of done) {
    const label = `${r.engine} · ${r.workload} · ${r.run_id.slice(-12)}`;
    runPicker.insertAdjacentHTML('beforeend', `<option value="${r.run_id}">${label}</option>`);
    cmpPicker.insertAdjacentHTML('beforeend', `<option value="${r.run_id}">${label}</option>`);
  }
  const hash = decodeURIComponent(location.hash.slice(1));
  if (hash) runPicker.value = hash;

  // ────────────────────────────────────────────────────────────────────
  // Render plumbing.
  // ────────────────────────────────────────────────────────────────────
  const charts = {};

  async function render() {
    const main = done.find(r => r.run_id === runPicker.value) || done[0];
    const cmp = cmpPicker.value ? done.find(r => r.run_id === cmpPicker.value) : null;
    const mainBlob = await (await fetch(main.result_href)).json();
    const cmpBlob = cmp ? await (await fetch(cmp.result_href)).json() : null;

    renderSummary(mainBlob, cmpBlob);
    renderPredictionBanner(mainBlob);
    renderTTFT(mainBlob, cmpBlob);
    renderTBT(mainBlob, cmpBlob);
    renderRooflineOverlay(mainBlob);
    renderCostOverlay(mainBlob);
    renderStepTimeOverlay(mainBlob);
    renderThroughput(mainBlob, cmpBlob);
    renderTimeline(mainBlob);
    updateOpenInCalcLink(mainBlob);
  }

  // ────────────────────────────────────────────────────────────────────
  // Build the "Open this run in the sizing calculator" link from the
  // currently-selected run's prediction.inputs. The calc reads the same
  // field set out of window.location.search (see ui.mjs::prefillFromQueryString).
  // Hidden when no prediction block exists (the link would have no inputs to seed).
  // ────────────────────────────────────────────────────────────────────
  function updateOpenInCalcLink(a) {
    const el = document.getElementById('open-in-calc');
    if (!el) return;
    const inp = a.prediction && a.prediction.inputs;
    if (!inp) { el.hidden = true; return; }
    const qs = new URLSearchParams({
      model:       inp.model_key || '',
      hw:          inp.hw_key || '',
      weight_prec: inp.weight_prec || '',
      kv_prec:     inp.kv_prec || '',
      act_prec:    inp.act_prec || '',
      isl:         String(inp.isl ?? ''),
      osl:         String(inp.osl ?? ''),
      ngpus:       String(inp.ngpus ?? ''),
      tbt_ms:      String(inp.tbt_ms ?? ''),
    });
    el.href = `/calculators/sizing_calculator.html?${qs.toString()}`;
    el.hidden = false;
  }

  // Measured cost per Mtok and measured step time per (batch proxy) point.
  // Surface these to the cost / step-time overlay renderers so they can
  // drop a scatter point alongside the predicted curve. Returns null fields
  // when the underlying analysis numbers aren't available — the panel then
  // degrades to a predicted-only line, matching empty-prediction behavior.
  function measuredPoint(a) {
    const rps  = a.analysis?.throughput?.requests_per_sec_avg;
    const ttft = a.analysis?.ttft_s?.p50;
    const tps  = a.analysis?.throughput?.tok_per_sec_avg;
    const tbt  = a.analysis?.tbt_s?.p50;
    const dur  = a.workload?.duration_s;
    const totTok = a.analysis?.throughput?.total_completion_tokens;
    const price = a.prediction?.inputs?.price_per_hour_usd;
    const ngpus = a.prediction?.inputs?.ngpus ?? a.hardware?.n_gpu;
    let batchProxy = null;
    if (rps != null && ttft != null) batchProxy = Math.max(1, rps * ttft + 1);
    // Measured $/Mtok: fleet $/hr × duration_s / 3600 ÷ (totTok / 1e6).
    let costMtok = null;
    if (price != null && ngpus != null && dur != null && totTok > 0) {
      const fleetUsd = price * ngpus * (dur / 3600.0);
      costMtok = fleetUsd / (totTok / 1e6);
    }
    const stepMs = tbt != null ? tbt * 1000 : null;
    return { batchProxy, tps, costMtok, stepMs };
  }

  function fmt(n, digits = 3) {
    if (n == null || !isFinite(n)) return '—';
    return Number(n).toFixed(digits);
  }
  function renderSummary(a, b) {
    const cells = (r) => `
      <div class="cell"><span class="k">ENGINE</span><span class="v">${r.engine.name}</span></div>
      <div class="cell"><span class="k">WORKLOAD</span><span class="v">${r.workload.name}</span></div>
      <div class="cell"><span class="k">MODEL</span><span class="v">${r.model.name}</span></div>
      <div class="cell"><span class="k">HW</span><span class="v">${r.hardware.instance} · ${r.hardware.n_gpu}× ${r.hardware.gpu}</span></div>
      <div class="cell"><span class="k">TTFT p50 / p95</span><span class="v">${fmt(r.analysis.ttft_s?.p50)} / ${fmt(r.analysis.ttft_s?.p95)} s</span></div>
      <div class="cell"><span class="k">THROUGHPUT</span><span class="v">${fmt(r.analysis.throughput.tok_per_sec_avg, 1)} tok/s</span></div>
      <div class="cell"><span class="k">REQUESTS · FAILED</span><span class="v">${r.analysis.steady_state_requests} · ${r.analysis.failed_requests}</span></div>
    `;
    const main = `<div class="summary">${cells(a)}</div>`;
    const cmp = b ? `<div class="summary compare">${cells(b)}</div>` : '';
    document.getElementById('summary').innerHTML = main + cmp;
  }

  function renderPredictionBanner(a) {
    const banner = document.getElementById('prediction-banner');
    const pred = a.prediction;
    const calib = a.calibration;

    // Calibration chip — shown only for auto-calibrated runs so users can
    // see what rate the lab picked and where the ceiling landed. Older
    // result files without a calibration field skip this silently.
    const calibChip = (calib && calib.method === 'auto') ? `
      <span class="metric-chip calib">
        <span class="k">rate (auto)</span>
        <span class="v">${calib.selected_rate.toFixed(2)} rps</span>
        <span class="sub">ceiling ~${calib.capacity_ceiling.toFixed(2)} rps · ${calib.probes.length} probes</span>
      </span>` : '';

    if (!pred) {
      // No baked prediction. Still surface calibration if present.
      if (calibChip) {
        banner.innerHTML = `<span class="ps-label">CALIBRATION</span>${calibChip}`;
        banner.hidden = false;
      } else {
        banner.hidden = true;
      }
      document.getElementById('roofline-foot-right').textContent =
        'no prediction baked into result';
      return;
    }

    let ratio = null;
    const rps = a.analysis?.throughput?.requests_per_sec_avg;
    const ttft = a.analysis?.ttft_s?.p50;
    const tps = a.analysis?.throughput?.tok_per_sec_avg;
    if (tps != null && rps != null && ttft != null && pred.curve.length > 0) {
      const batchProxy = Math.max(1, rps * ttft + 1);
      const nearest = pred.curve.reduce(
        (best, p) => Math.abs(p.batch - batchProxy) < Math.abs(best.batch - batchProxy) ? p : best,
        pred.curve[0],
      );
      if (nearest.tps > 0) ratio = tps / nearest.tps;
    }
    const ratioClass = ratio == null ? '' :
      (ratio < 0.5 || ratio > 2.0) ? 'bad' :
      (ratio < 0.8 || ratio > 1.25) ? 'warn' : 'good';
    const ratioStr = ratio == null ? '—' : `${ratio.toFixed(2)}×`;
    document.getElementById('roofline-foot-right').textContent =
      ratio == null ? '—' : `measured/predicted = ${ratioStr}`;

    banner.innerHTML = `
      <span class="ps-label">PREDICTION</span>
      <span class="metric-chip"><span class="k">calc</span><code>${pred.calc_version}</code></span>
      <span class="metric-chip"><span class="k">data</span><code>${pred.data_hash.slice(0, 10)}…</code></span>
      <span class="metric-chip bc"><span class="k">B_crit</span><span class="v">${pred.b_crit ?? '—'}</span></span>
      <span class="metric-chip bs"><span class="k">B_slo</span><span class="v">${pred.b_slo ?? '—'}</span></span>
      <span class="metric-chip bk"><span class="k">B_kv</span><span class="v">${pred.b_kv ?? '—'}</span></span>
      <span class="metric-chip ratio ${ratioClass}"><span class="k">measured/predicted</span><span class="v">${ratioStr}</span></span>
      ${calibChip}
    `;
    banner.hidden = false;
  }

  // ── chart helpers ──
  const axisTitle = (text) => ({ display: true, text, color: T.muted, font: { size: 10, family: T.mono } });
  function destroy(id) { if (charts[id]) { charts[id].destroy(); delete charts[id]; } }

  function renderTTFT(a, b)   { drawBar('ttft-chart', a.analysis.ttft_s, b?.analysis.ttft_s, 'seconds'); }
  function renderTBT(a, b)    { drawBar('tbt-chart',  a.analysis.tbt_s,  b?.analysis.tbt_s,  'seconds'); }

  function drawBar(id, mainPct, cmpPct, unit) {
    destroy(id);
    const fields = ['p50', 'p95', 'p99', 'mean'];
    const datasets = [];
    if (mainPct) datasets.push({
      label: 'main',
      data: fields.map(f => mainPct[f]),
      backgroundColor: T.accent,
      borderColor: T.accent,
      borderWidth: 1,
      borderRadius: 0,
      barPercentage: 0.6,
      categoryPercentage: 0.7,
    });
    if (cmpPct) datasets.push({
      label: 'compare',
      data: fields.map(f => cmpPct[f]),
      backgroundColor: T.bSlo,
      borderColor: T.bSlo,
      borderWidth: 1,
      borderRadius: 0,
      barPercentage: 0.6,
      categoryPercentage: 0.7,
    });
    charts[id] = new Chart(document.getElementById(id), {
      type: 'bar',
      data: { labels: fields, datasets },
      options: {
        maintainAspectRatio: false, responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: T.grid, display: false }, ticks: { color: T.muted } },
          y: { grid: { color: T.grid }, ticks: { color: T.muted }, title: axisTitle(unit) },
        },
      },
    });
  }

  function renderRooflineOverlay(a) {
    const id = 'roofline-chart';
    destroy(id);
    const datasets = [];
    const pred = a.prediction;
    if (pred && Array.isArray(pred.curve) && pred.curve.length > 0) {
      datasets.push({
        label: 'predicted',
        type: 'line',
        data: pred.curve.map(p => ({ x: p.batch, y: p.tps })),
        borderColor: T.accent,
        backgroundColor: 'transparent',
        pointBackgroundColor: T.accent,
        pointBorderColor: T.accent,
        pointRadius: 1.5,
        pointHoverRadius: 4,
        borderWidth: 1.5,
        tension: 0.22,
        showLine: true,
      });
    }
    const rps = a.analysis?.throughput?.requests_per_sec_avg;
    const ttft = a.analysis?.ttft_s?.p50;
    const tps = a.analysis?.throughput?.tok_per_sec_avg;
    if (rps != null && ttft != null && tps != null) {
      const batchProxy = Math.max(1, rps * ttft + 1);
      datasets.push({
        label: 'measured',
        type: 'scatter',
        data: [{ x: batchProxy, y: tps }],
        backgroundColor: T.bSlo,
        borderColor: T.bSlo,
        pointRadius: 7,
        pointHoverRadius: 9,
      });
    }
    const canvas = document.getElementById(id);
    if (datasets.length === 0) { canvas.style.opacity = 0.35; return; }
    canvas.style.opacity = 1;
    charts[id] = new Chart(canvas, {
      data: { datasets },
      options: {
        maintainAspectRatio: false, responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { type: 'logarithmic', grid: { color: T.grid }, ticks: { color: T.muted }, title: axisTitle('batch (log)') },
          y: { grid: { color: T.grid }, ticks: { color: T.muted }, title: axisTitle('tokens / sec') },
        },
      },
    });
  }

  // Generic predicted-curve + measured-point overlay used by the cost and
  // step-time panels. Mirrors the throughput overlay (line for predicted,
  // scatter point for measured) but parameterized over which curve field
  // and which measured scalar to plot. Empty / null inputs render an
  // empty canvas at 35% opacity, matching renderRooflineOverlay.
  function drawCurveOverlay({
    id, curve, getY, measured, yTitle, footEl, footFormat,
  }) {
    destroy(id);
    const datasets = [];
    if (Array.isArray(curve) && curve.length > 0) {
      const pts = curve
        .map((p) => ({ x: p.batch, y: getY(p) }))
        .filter((p) => p.y != null && isFinite(p.y));
      if (pts.length > 0) {
        datasets.push({
          label: 'predicted',
          type: 'line',
          data: pts,
          borderColor: T.accent,
          backgroundColor: 'transparent',
          pointBackgroundColor: T.accent,
          pointBorderColor: T.accent,
          pointRadius: 1.5,
          pointHoverRadius: 4,
          borderWidth: 1.5,
          tension: 0.22,
          showLine: true,
        });
      }
    }
    if (measured && measured.x != null && measured.y != null && isFinite(measured.y)) {
      datasets.push({
        label: 'measured',
        type: 'scatter',
        data: [{ x: measured.x, y: measured.y }],
        backgroundColor: T.bSlo,
        borderColor: T.bSlo,
        pointRadius: 7,
        pointHoverRadius: 9,
      });
    }
    if (footEl) {
      const el = document.getElementById(footEl);
      if (el) el.textContent = measured && measured.y != null && isFinite(measured.y)
        ? footFormat(measured.y) : '—';
    }
    const canvas = document.getElementById(id);
    if (datasets.length === 0) { canvas.style.opacity = 0.35; return; }
    canvas.style.opacity = 1;
    charts[id] = new Chart(canvas, {
      data: { datasets },
      options: {
        maintainAspectRatio: false, responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { type: 'logarithmic', grid: { color: T.grid }, ticks: { color: T.muted }, title: axisTitle('batch (log)') },
          y: { grid: { color: T.grid }, ticks: { color: T.muted }, title: axisTitle(yTitle) },
        },
      },
    });
  }

  function renderCostOverlay(a) {
    const meas = measuredPoint(a);
    drawCurveOverlay({
      id: 'cost-chart',
      curve: a.prediction?.curve || [],
      getY: (p) => p.cost_per_mtok,
      measured: { x: meas.batchProxy, y: meas.costMtok },
      yTitle: '$ / M tokens',
      footEl: 'cost-foot-right',
      footFormat: (v) => `measured ≈ $${v.toFixed(2)} / Mtok`,
    });
  }

  function renderStepTimeOverlay(a) {
    const meas = measuredPoint(a);
    drawCurveOverlay({
      id: 'steptime-chart',
      curve: a.prediction?.curve || [],
      getY: (p) => p.step_ms,
      measured: { x: meas.batchProxy, y: meas.stepMs },
      yTitle: 'ms / step',
      footEl: 'steptime-foot-right',
      footFormat: (v) => `measured ≈ ${v.toFixed(1)} ms`,
    });
  }

  function renderThroughput(a, b) {
    const id = 'throughput-chart';
    destroy(id);
    const labels = ['main'];
    const values = [a.analysis.throughput.tok_per_sec_avg];
    if (b) { labels.push('compare'); values.push(b.analysis.throughput.tok_per_sec_avg); }
    charts[id] = new Chart(document.getElementById(id), {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: T.bKv,
          borderColor: T.bKv,
          borderWidth: 1,
          borderRadius: 0,
          barPercentage: 0.4,
          categoryPercentage: 0.5,
        }],
      },
      options: {
        maintainAspectRatio: false, responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: T.grid, display: false }, ticks: { color: T.muted } },
          y: { grid: { color: T.grid }, ticks: { color: T.muted }, title: axisTitle('tok/s') },
        },
      },
    });
  }

  function renderTimeline(a) {
    const id = 'timeline-chart';
    destroy(id);
    const points = (a.raw_results || [])
      .filter(r => r.ttft_s != null)
      .map(r => ({ x: r.submit_offset_s, y: r.ttft_s }));
    charts[id] = new Chart(document.getElementById(id), {
      type: 'scatter',
      data: { datasets: [{
        data: points,
        backgroundColor: T.accent,
        borderColor: T.accent,
        pointRadius: 2.5,
        pointHoverRadius: 5,
      }] },
      options: {
        maintainAspectRatio: false, responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: T.grid }, ticks: { color: T.muted }, title: axisTitle('submit offset (s)') },
          y: { grid: { color: T.grid }, ticks: { color: T.muted }, title: axisTitle('TTFT (s)') },
        },
      },
    });
  }

  runPicker.addEventListener('change', render);
  cmpPicker.addEventListener('change', render);
  render();
})();
</script>
</body>
</html>
