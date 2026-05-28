<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>EMPIRICAL LAB · Command Center</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link rel="stylesheet"
  href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Martian+Mono:wght@300;400;500&display=swap" />
<link rel="stylesheet" href="assets/style.css" />
</head>
<body>

<div class="strip">
  <div class="strip-left">
    <span class="brand">EMPIRICAL LAB</span>
    <span class="sep"></span>
    <span>v0·1·0</span>
  </div>
  <nav class="bench-switcher" aria-label="bench selector">
    <span class="bs-label">BENCH</span>
    <a href="/calculators/sizing_calculator.html" class="bs-item">
      <span class="bs-num">01</span><span class="bs-name">SIZING CALC</span>
    </a>
    <a href="/experiments/_site/command_center.html" class="bs-item active">
      <span class="bs-num">02</span><span class="bs-name">COMMAND</span>
    </a>
    <a href="/experiments/_site/results_explorer.html" class="bs-item">
      <span class="bs-num">03</span><span class="bs-name">RESULTS</span>
    </a>
  </nav>
  <div class="strip-right">
    <a href="/index.html">← ROOT</a>
  </div>
</div>

<main class="bench">

  <h1>
    <span class="sub">▣ telemetry · runs ledger</span>
    Command Center
  </h1>
  <p class="lede">Read-only view of every experiment the lab has acquired. Start, stop, and configure runs from the CLI; refresh this page after a state transition.</p>

  <section class="cli-bench">
    <span class="ps-label">CLI · ENTRY POINT</span>
    <pre><span class="prompt">$</span> exp run --engine vllm --workload chatbot --rate 8 --duration 300</pre>
    <span class="hint">Pre-flight consults the sizing calculator; <code>--preflight strict</code> blocks on errors.</span>
  </section>

  <h2>Runs</h2>

  <div class="runs" id="runs-card">
    <div class="runs-head">
      <span>▶ LEDGER · CHRONOLOGICAL DESCENDING</span>
      <span class="count" id="runs-count">— RUNS</span>
    </div>
    <table class="runs-table" id="runs-table">
      <thead>
        <tr>
          <th>STATUS</th>
          <th>RUN ID</th>
          <th>ENGINE</th>
          <th>WORKLOAD</th>
          <th>MODEL</th>
          <th>HW</th>
          <th>STARTED</th>
          <th>FINISHED</th>
          <th></th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>
  <div id="empty" class="empty" hidden>
    [--:--:--] · no runs.
    <span class="hint">Kick one off with <code>exp run ...</code> and refresh this page.</span>
  </div>
</main>

<footer class="foot">
  <span>EMPIRICAL LAB · MIT</span>
  <span><a href="/calculators/sizing_calculator.html">↗ SIZING CALCULATOR (sibling bench)</a></span>
</footer>

<script>
(async function () {
  const r = await fetch('assets/runs.json');
  const { runs } = await r.json();
  const tbody = document.querySelector('#runs-table tbody');
  const countEl = document.getElementById('runs-count');
  if (!runs.length) {
    document.getElementById('empty').hidden = false;
    document.getElementById('runs-card').hidden = true;
    return;
  }
  runs.sort((a, b) => (b.started_at || b.created_at || '').localeCompare(a.started_at || a.created_at || ''));
  countEl.textContent = `${runs.length} RUN${runs.length === 1 ? '' : 'S'}`;
  function shortTs(iso) {
    if (!iso) return '';
    // Strip seconds-precision and timezone to keep the cell narrow.
    return iso.slice(0, 19).replace('T', ' ');
  }
  function shortId(s) {
    // <YYYYMMDDTHHMMSSZ>-<engine>-<workload>-<6hex>  →  …-<6hex>
    const parts = String(s).split('-');
    return parts.length > 3 ? `…${parts[parts.length - 1]}` : s;
  }
  for (const run of runs) {
    const tr = document.createElement('tr');
    const link = run.result_href
      ? `<a href="/experiments/_site/results_explorer.html#${encodeURIComponent(run.run_id)}">[ VIEW ]</a>`
      : '';
    tr.innerHTML = `
      <td><span class="status status-${run.status}">${run.status}</span></td>
      <td><code title="${run.run_id}">${shortId(run.run_id)}</code></td>
      <td>${run.engine || ''}</td>
      <td>${run.workload || ''}</td>
      <td class="dim">${run.model || '—'}</td>
      <td class="dim">${run.hardware || '—'}</td>
      <td class="dim">${shortTs(run.started_at)}</td>
      <td class="dim">${shortTs(run.finished_at)}</td>
      <td>${link}</td>
    `;
    tbody.appendChild(tr);
  }
})();
</script>
</body>
</html>
