// Chart layer: oscilloscope-styled latency/throughput curve with caliper
// markers for the three batch bounds, a dashed TBT SLO line, and a hover
// crosshair. Assumes window.Chart (Chart.js v4) is loaded via CDN.
//
// Exports a single createScope() that owns the Chart instance and exposes
// .update() for the UI's recompute handler. All visual decisions live here;
// ui.mjs only feeds data in.

const COLOR = {
  bg:     "#0a0e14",
  text:   "#e6edf3",
  muted:  "#8b949e",
  dim:    "#6e7681",
  grid:   "#1a1f29",
  border: "#30363d",
  step:   "#58a6ff",
  tps:    "#3fb950",
  slo:    "#8b949e",
  bcrit:  "#58a6ff",
  bslo:   "#d29922",
  bkv:    "#3fb950",
  glow:   "rgba(88, 166, 255, 0.18)",
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
// Markers live on the chart's `options.plugins.calipers.markers` array so
// they're trivially updated from outside.
const calipersPlugin = {
  id: "calipers",
  afterDatasetsDraw(chart) {
    const markers = chart.options.plugins?.calipers?.markers ?? [];
    const ctx = chart.ctx;
    const { top, bottom } = chart.chartArea;
    ctx.save();
    ctx.font = '500 10px "Martian Mono", ui-monospace, monospace';
    ctx.textBaseline = "alphabetic";
    for (const m of markers) {
      const x = xForB(chart, m.B);
      if (x == null) continue;
      // Drop line.
      ctx.strokeStyle = m.color;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.globalAlpha = 0.8;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      // Top tick (6px horizontal cap, like a dimension caliper).
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.moveTo(x - 4, top - 1);
      ctx.lineTo(x + 4, top - 1);
      ctx.stroke();
      // Top-gutter label: "B_crit · 295"
      const label = `${m.label} · ${Math.round(m.B)}`;
      const w = ctx.measureText(label).width;
      // Shift left for the rightmost marker so labels never clip the canvas.
      const lx = Math.min(x + 6, chart.chartArea.right - w - 2);
      ctx.fillStyle = m.color;
      ctx.fillText(label, lx, top - 6);
    }
    ctx.restore();
  },
};

// Crosshair plugin: thin cyan vertical line + coordinate badge that follows
// the mouse. Reads chart._cursor (set by the canvas mousemove handler below).
const crosshairPlugin = {
  id: "crosshair",
  afterDatasetsDraw(chart) {
    const cur = chart._cursor;
    if (!cur) return;
    const { top, bottom, left, right } = chart.chartArea;
    if (cur.x < left || cur.x > right || cur.y < top || cur.y > bottom) return;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = COLOR.step;
    ctx.lineWidth = 0.5;
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    ctx.moveTo(cur.x, top); ctx.lineTo(cur.x, bottom);
    ctx.moveTo(left, cur.y); ctx.lineTo(right, cur.y);
    ctx.stroke();
    // Coordinate badge — reads the scale to invert pixel→value.
    const B = chart.scales.x.getValueForPixel(cur.x);
    const tMs = chart.scales.y.getValueForPixel(cur.y);
    const label = `B=${B < 10 ? B.toFixed(1) : Math.round(B)} · t=${tMs.toFixed(1)} ms`;
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

// First-paint marker stagger: fades the three caliper labels in one by one
// after the curve draw finishes. Handled outside Chart.js's animation engine
// because we're rendering markers in afterDatasetsDraw, not as datasets.
function staggerMarkerReveal(chart, durations = [0, 80, 160]) {
  const markers = chart.options.plugins?.calipers?.markers ?? [];
  const original = markers.map((m) => ({ ...m }));
  markers.forEach((m) => { m.color = "transparent"; });
  chart.update("none");
  setTimeout(() => {
    markers.forEach((m, i) => {
      setTimeout(() => { m.color = original[i].color; chart.update("none"); }, durations[i] ?? 0);
    });
  }, 600); // wait for the curve animation
}

export function createScope(canvas, initial) {
  // Build the two-dataset structure: glow halo (drawn first), bright stroke
  // (drawn on top). Same data; the visual layering is what creates the CRT
  // effect without any pixel-shader trickery.
  const labels = initial.curve.map((p) => p.B);
  const stepData = initial.curve.map((p) => p.step_ms);
  const tpsData = initial.curve.map((p) => p.tokens_per_sec);

  const chart = new window.Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        // Glow halo — wide low-opacity stroke behind the bright line.
        { label: "step_ms_glow", data: stepData, yAxisID: "y", borderColor: COLOR.glow,
          borderWidth: 6, pointRadius: 0, tension: 0.3, fill: false, order: 3 },
        // Bright line — the actual readable curve.
        { label: "step_ms", data: stepData, yAxisID: "y", borderColor: COLOR.step,
          borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: false, order: 1 },
        // Throughput on the secondary axis — same B grid.
        { label: "tokens_per_sec", data: tpsData, yAxisID: "y1", borderColor: COLOR.tps,
          borderWidth: 1.2, borderDash: [2, 3], pointRadius: 0, tension: 0.3, fill: false, order: 2 },
        // TBT SLO horizontal — a constant-y line. Implemented as a dataset
        // (rather than annotation) so we stay on a single CDN script.
        { label: "tbt_slo", data: labels.map(() => initial.tbt_ms), yAxisID: "y",
          borderColor: COLOR.slo, borderWidth: 1, borderDash: [4, 4], pointRadius: 0, fill: false, order: 4 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600, easing: "easeOutQuart" },
      interaction: { mode: "nearest", intersect: false },
      plugins: {
        legend: { display: false },           // legend rendered in HTML scope-head
        tooltip: { enabled: false },          // crosshair plugin replaces it
        calipers: { markers: [
          { B: initial.bcrit, color: COLOR.bcrit, label: "B_crit" },
          { B: initial.bslo,  color: COLOR.bslo,  label: "B_slo"  },
          { B: initial.bkv,   color: COLOR.bkv,   label: "B_kv"   },
        ] },
      },
      scales: {
        x: {
          type: "logarithmic",
          title: { display: true, text: "BATCH SIZE  (B)", color: COLOR.muted,
                   font: { family: '"Martian Mono", monospace', size: 10, weight: "500" } },
          ticks: { color: COLOR.dim, font: { family: '"Martian Mono", monospace', size: 10 } },
          grid:  { color: COLOR.grid, drawBorder: false },
        },
        y: {
          position: "left",
          title: { display: true, text: "STEP TIME  (ms)", color: COLOR.muted,
                   font: { family: '"Martian Mono", monospace', size: 10, weight: "500" } },
          ticks: { color: COLOR.dim, font: { family: '"Martian Mono", monospace', size: 10 } },
          grid:  { color: COLOR.grid, drawBorder: false },
          beginAtZero: true,
        },
        y1: {
          position: "right",
          title: { display: true, text: "THROUGHPUT  (tok/s)", color: COLOR.muted,
                   font: { family: '"Martian Mono", monospace', size: 10, weight: "500" } },
          ticks: { color: COLOR.dim, font: { family: '"Martian Mono", monospace', size: 10 } },
          grid:  { display: false },
          beginAtZero: true,
        },
      },
    },
    plugins: [calipersPlugin, crosshairPlugin],
  });

  // Expose to global for E2E + manual debugging — convention from Phase 3/4
  // dashboards. NOT used by app code.
  if (typeof window !== "undefined") window.__sizingChart = chart;

  // Crosshair tracking — set a cursor field that the plugin reads.
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    chart._cursor = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    chart.draw();
  });
  canvas.addEventListener("mouseleave", () => { chart._cursor = null; chart.draw(); });

  // First-paint stagger (skipped under reduced-motion).
  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    staggerMarkerReveal(chart);
  }

  return {
    chart,
    update(data) {
      const labels = data.curve.map((p) => p.B);
      chart.data.labels = labels;
      chart.data.datasets[0].data = data.curve.map((p) => p.step_ms); // glow
      chart.data.datasets[1].data = data.curve.map((p) => p.step_ms); // bright
      chart.data.datasets[2].data = data.curve.map((p) => p.tokens_per_sec);
      chart.data.datasets[3].data = labels.map(() => data.tbt_ms);
      const m = chart.options.plugins.calipers.markers;
      m[0].B = data.bcrit; m[1].B = data.bslo; m[2].B = data.bkv;
      chart.update();
    },
  };
}
