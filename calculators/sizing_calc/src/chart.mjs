// Chart layer: three vertically-stacked oscilloscope panels — Latency,
// Throughput, Cost — each a single Chart.js instance sharing the same log-B
// X axis and caliper markers (b_crit / b_slo / b_kv). The single-canvas dual-
// axis version this replaced visually buried throughput under latency; one
// canvas per channel means each curve gets its full vertical reading.
//
// Exports createScope(canvas, channel, initial) — UI calls it three times
// (one per channel) and stores the returned handles in window.__sizingScopes.
// Assumes window.Chart (Chart.js v4) is loaded via CDN.

const COLOR = {
  bg:        "#0a0e14",
  text:      "#e6edf3",
  muted:     "#8b949e",
  dim:       "#6e7681",
  grid:      "#1a1f29",
  border:    "#30363d",
  // Per-channel line + caliper colors. Calipers always match the metric
  // they mark (b_crit blue, b_slo yellow, b_kv green) regardless of which
  // panel they sit in — keeps the three-panel reading visually consistent.
  latency:   "#58a6ff",
  throughput:"#3fb950",
  cost:      "#d29922",
  slo:       "#8b949e",
  bcrit:     "#58a6ff",
  bslo:      "#d29922",
  bkv:       "#3fb950",
};

// Per-channel config. Picking the right `dataKey` from the curve points and
// the right Y-axis title/formatter keeps createScope() generic — adding a
// fourth channel later would just be a new entry here.
const CHANNELS = {
  latency: {
    dataKey:  "step_ms",
    color:    COLOR.latency,
    glow:     "rgba(88, 166, 255, 0.18)",
    yLabel:   "STEP TIME  (ms)",
    yFormat:  (v) => `${v.toFixed(1)} ms`,
    showSlo:  true,   // dashed TBT line on latency panel only
  },
  throughput: {
    dataKey:  "tokens_per_sec",
    color:    COLOR.throughput,
    glow:     "rgba(63, 185, 80, 0.18)",
    yLabel:   "THROUGHPUT  (tok/s)",
    yFormat:  (v) => `${Math.round(v).toLocaleString("en-US")} tok/s`,
    showSlo:  false,
  },
  cost: {
    dataKey:  "cost_per_mtok",
    color:    COLOR.cost,
    glow:     "rgba(210, 153, 34, 0.18)",
    yLabel:   "$ / Mtok",
    // Cost curve carries nulls when price is unset → Chart.js renders gaps.
    yFormat:  (v) => `$${v.toFixed(2)} / Mtok`,
    showSlo:  false,
  },
};

// Find the chart-area X pixel for a given B value (log scale). Returns null if
// out of the visible range — markers off-screen should be hidden, not clamped.
function xForB(chart, B) {
  const scale = chart.scales.x;
  if (!scale || B == null || !Number.isFinite(B) || B <= 0) return null;
  const x = scale.getPixelForValue(B);
  if (x < scale.left - 1 || x > scale.right + 1) return null;
  return x;
}

// Caliper-marker plugin. Draws three full-height vertical lines (b_crit /
// b_slo / b_kv) with a top-gutter label like an engineering CAD dimension.
// Reusable across all three channels — markers live on per-chart options.
//
// Label placement uses greedy first-fit row assignment: labels are sorted by
// X and placed in the topmost row that has no horizontal collision with an
// already-placed label. Common case (markers well-separated): all three land
// in row 0. Crowded case (e.g., B_crit=296 and B_kv=333 — labels would
// otherwise overlap into mush): the second label drops into row 1. Up to
// 2 rows are supported (chart layout pads 26px above chartArea for them);
// 3-marker pile-ups beyond that fall back to overprint of the topmost row.
const calipersPlugin = {
  id: "calipers",
  afterDatasetsDraw(chart) {
    const markers = chart.options.plugins?.calipers?.markers ?? [];
    const ctx = chart.ctx;
    const { top, bottom, right } = chart.chartArea;
    ctx.save();
    ctx.font = '500 10px "Martian Mono", ui-monospace, monospace';
    ctx.textBaseline = "alphabetic";

    // First pass: project visible markers, measure label width, decide each
    // label's anchor X (clamped to stay on-canvas at the right edge).
    const labelGap = 6;
    const positioned = markers
      .map((m) => {
        const x = xForB(chart, m.B);
        if (x == null) return null;
        const label = `${m.label} · ${Math.round(m.B)}`;
        const w = ctx.measureText(label).width;
        const lx = Math.min(x + 6, right - w - 2);
        return { ...m, x, label, w, lx };
      })
      .filter(Boolean)
      .sort((a, b) => a.lx - b.lx);

    // Greedy first-fit row assignment. rowEdge[i] tracks the right edge of
    // the rightmost label currently in row i; a new label fits in row i iff
    // its lx is past that edge.
    const rowEdge = [];
    for (const m of positioned) {
      let row = 0;
      while (rowEdge[row] != null && m.lx < rowEdge[row]) row++;
      rowEdge[row] = m.lx + m.w + labelGap;
      m.row = row;
    }

    // Second pass: draw drop lines (full height, always at row-0 tick),
    // top tick caps (at chartArea.top), and labels at their assigned row.
    const rowH = 11;
    for (const m of positioned) {
      ctx.strokeStyle = m.color;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.globalAlpha = 0.8;
      ctx.beginPath();
      ctx.moveTo(m.x, top); ctx.lineTo(m.x, bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.moveTo(m.x - 4, top - 1); ctx.lineTo(m.x + 4, top - 1);
      ctx.stroke();
      ctx.fillStyle = m.color;
      ctx.fillText(m.label, m.lx, top - 6 - m.row * rowH);
    }
    ctx.restore();
  },
};

// Crosshair plugin: thin vertical+horizontal line + coordinate badge that
// follows the mouse. Per-chart (each scope gets its own crosshair). The Y
// formatter comes from the channel config so the badge reads "ms" on the
// latency panel, "tok/s" on throughput, "$X.XX/Mtok" on cost.
const crosshairPlugin = {
  id: "crosshair",
  afterDatasetsDraw(chart) {
    const cur = chart._cursor;
    if (!cur) return;
    const { top, bottom, left, right } = chart.chartArea;
    if (cur.x < left || cur.x > right || cur.y < top || cur.y > bottom) return;
    const ctx = chart.ctx;
    const yFormat = chart._yFormat || ((v) => v.toFixed(1));
    const channelColor = chart._channelColor || COLOR.latency;
    ctx.save();
    ctx.strokeStyle = channelColor;
    ctx.lineWidth = 0.5;
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    ctx.moveTo(cur.x, top); ctx.lineTo(cur.x, bottom);
    ctx.moveTo(left, cur.y); ctx.lineTo(right, cur.y);
    ctx.stroke();
    const B = chart.scales.x.getValueForPixel(cur.x);
    const yVal = chart.scales.y.getValueForPixel(cur.y);
    const label = `B=${B < 10 ? B.toFixed(1) : Math.round(B)} · ${yFormat(yVal)}`;
    ctx.font = '500 10px "Martian Mono", ui-monospace, monospace';
    ctx.textBaseline = "top";
    const w = ctx.measureText(label).width;
    const pad = 6;
    const bx = Math.min(cur.x + 10, right - w - pad * 2);
    const by = Math.max(cur.y - 22, top + 2);
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = COLOR.bg;
    ctx.strokeStyle = COLOR.border;
    ctx.lineWidth = 1;
    ctx.fillRect(bx, by, w + pad * 2, 16);
    ctx.strokeRect(bx, by, w + pad * 2, 16);
    ctx.fillStyle = COLOR.text;
    ctx.fillText(label, bx + pad, by + 3);
    ctx.restore();
  },
};

// First-paint marker stagger: hide the caliper labels then fade them in one
// by one after the curve draw finishes. Per-chart — each scope animates
// independently of the others (UI staggers scope starts on top of this).
function staggerMarkerReveal(chart, durations = [0, 80, 160]) {
  const markers = chart.options.plugins?.calipers?.markers ?? [];
  const original = markers.map((m) => ({ ...m }));
  markers.forEach((m) => { m.color = "transparent"; });
  chart.update("none");
  setTimeout(() => {
    markers.forEach((m, i) => {
      setTimeout(() => { m.color = original[i].color; chart.update("none"); }, durations[i] ?? 0);
    });
  }, 600);
}

// Build one oscilloscope scope (canvas + Chart instance + update API).
// `channel` is "latency" | "throughput" | "cost". Returns { chart, update }.
export function createScope(canvas, channel, initial) {
  const cfg = CHANNELS[channel];
  if (!cfg) throw new Error(`createScope: unknown channel "${channel}"`);
  const labels = initial.curve.map((p) => p.B);
  const data = initial.curve.map((p) => p[cfg.dataKey]);

  // Build datasets: glow halo (drawn first, low opacity, fat stroke) + bright
  // line (drawn on top, narrow). Same data; layering creates the CRT effect.
  // Latency panel also gets a TBT-SLO horizontal as a dataset (avoids needing
  // chartjs-plugin-annotation — keeps the page on one CDN script).
  const datasets = [
    { label: `${channel}_glow`, data, yAxisID: "y", borderColor: cfg.glow,
      borderWidth: 6, pointRadius: 0, tension: 0.3, fill: false, order: 3,
      spanGaps: false /* cost curve has nulls — render as gaps */ },
    { label: channel, data, yAxisID: "y", borderColor: cfg.color,
      borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: false, order: 1,
      spanGaps: false },
  ];
  if (cfg.showSlo) {
    datasets.push({
      label: "tbt_slo", data: labels.map(() => initial.tbt_ms), yAxisID: "y",
      borderColor: COLOR.slo, borderWidth: 1, borderDash: [4, 4],
      pointRadius: 0, fill: false, order: 4,
    });
  }

  const chart = new window.Chart(canvas, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600, easing: "easeOutQuart" },
      interaction: { mode: "nearest", intersect: false },
      // Top padding leaves room for the caliper labels (drawn above
      // chartArea.top by calipersPlugin, stacked into up to 2 rows when
      // markers overlap horizontally). Without padding the labels render
      // above the canvas boundary and get clipped. 26px = 2 rows × 11px + 4
      // breathing. Right padding deliberately omitted — the rightmost tick
      // ("1,500") needs the full plot width; the label-position clamp inside
      // calipersPlugin keeps rightmost labels on-canvas already.
      layout: { padding: { top: 26 } },
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
        calipers: { markers: [
          { B: initial.bcrit, color: COLOR.bcrit, label: "B_crit" },
          { B: initial.bslo,  color: COLOR.bslo,  label: "B_slo"  },
          { B: initial.bkv,   color: COLOR.bkv,   label: "B_kv"   },
        ] },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "BATCH SIZE  (B)", color: COLOR.muted,
                   font: { family: '"Martian Mono", monospace', size: 10, weight: "500" } },
          ticks: { color: COLOR.dim, font: { family: '"Martian Mono", monospace', size: 10 } },
          grid:  { color: COLOR.grid, drawBorder: false },
        },
        y: {
          position: "left",
          title: { display: true, text: cfg.yLabel, color: COLOR.muted,
                   font: { family: '"Martian Mono", monospace', size: 10, weight: "500" } },
          ticks: { color: COLOR.dim, font: { family: '"Martian Mono", monospace', size: 10 } },
          grid:  { color: COLOR.grid, drawBorder: false },
          beginAtZero: true,
        },
      },
    },
    plugins: [calipersPlugin, crosshairPlugin],
  });
  // Stash channel-specific formatter + color where the crosshair plugin
  // (which is shared across all scopes) can read them per-chart.
  chart._yFormat = cfg.yFormat;
  chart._channelColor = cfg.color;

  // Crosshair tracking — independent per canvas/chart.
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    chart._cursor = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    chart.draw();
  });
  canvas.addEventListener("mouseleave", () => { chart._cursor = null; chart.draw(); });

  // First-paint stagger per-scope (skipped under reduced-motion). UI staggers
  // scope construction itself ~120ms apart so the three panels reveal in
  // top-down reading order.
  if (typeof window !== "undefined" &&
      !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    staggerMarkerReveal(chart);
  }

  return {
    chart,
    channel,
    update(data) {
      const labels = data.curve.map((p) => p.B);
      const next = data.curve.map((p) => p[cfg.dataKey]);
      chart.data.labels = labels;
      chart.data.datasets[0].data = next; // glow
      chart.data.datasets[1].data = next; // bright
      if (cfg.showSlo) {
        chart.data.datasets[2].data = labels.map(() => data.tbt_ms);
        // Keep the hidden legacy tps dataset in sync so its length matches
        // labels (Chart.js complains about ragged datasets on resize).
        if (chart.data.datasets[3]) {
          chart.data.datasets[3].data = data.curve.map((p) => p.tokens_per_sec);
        }
      }
      const m = chart.options.plugins.calipers.markers;
      m[0].B = data.bcrit; m[1].B = data.bslo; m[2].B = data.bkv;
      chart.update();
    },
  };
}
