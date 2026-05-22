"""Build HTML version of MODEL_SIZING_SCALING_REFERENCE.md.

Produces a single self-contained MODEL_SIZING_SCALING_REFERENCE.html with:
- Sticky sidebar TOC with active-section highlighting
- Chart.js charts replacing ASCII plots (roofline, latency curve, KV bars)
- New charts the markdown doesn't have: HBM budget, MFU prefill vs decode
- Mermaid diagrams preserved
- Print-friendly typography

Run with: python3 build_sizing_ref_html.py
Requires: python-markdown (already installed at 3.9)
"""
import html as html_lib
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).parent.parent  # repo root (this script lives in tools/)
SRC = ROOT / "reference" / "MODEL_SIZING_SCALING_REFERENCE.md"
DST = ROOT / "reference" / "MODEL_SIZING_SCALING_REFERENCE.html"


# ---------- markdown preprocessing ----------

def replace_ascii_charts(md_text: str) -> str:
    """Swap the hand-drawn ASCII plots for chart placeholders.

    Each placeholder is a div the post-process step finds, sized, and seeds
    with Chart.js. The ASCII versions still live in the .md (which renders
    on GitHub); the HTML build upgrades them to real charts.
    """
    # § 1 roofline plot — anchored on its distinctive opening line.
    md_text = re.sub(
        r"```\n   Achieved FLOPs/s\n.*?prefill chunks\)\n```",
        '<div class="chart-block" data-chart="roofline"></div>',
        md_text,
        flags=re.DOTALL,
    )
    # § 3 KV variant bar chart.
    md_text = re.sub(
        r"```\nMHA   ████.*?\(log scale\)\n```",
        '<div class="chart-block" data-chart="kv-variants"></div>',
        md_text,
        flags=re.DOTALL,
    )
    # § 4 latency-throughput curve.
    md_text = re.sub(
        r"```\n   Tokens/sec \(per replica\)\n.*?TBT grows linearly\n```",
        '<div class="chart-block" data-chart="latency-throughput"></div>',
        md_text,
        flags=re.DOTALL,
    )
    return md_text


PARALLELISM_SVG = r"""<div class="chart-block">
<svg viewBox="0 0 820 360" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Tensor parallelism and pipeline parallelism cartoons">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8b949e"/>
    </marker>
  </defs>

  <!-- ===== LEFT PANEL: TP ===== -->
  <text x="200" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#e6edf3">Tensor Parallelism (TP)</text>
  <text x="200" y="46" text-anchor="middle" font-size="11" fill="#8b949e">slice each matmul along the hidden dim</text>

  <!-- X (input) -->
  <rect x="30" y="120" width="40" height="80" fill="#1f6feb" rx="3"/>
  <text x="50" y="165" text-anchor="middle" font-size="13" fill="white" font-weight="600">X</text>

  <!-- × -->
  <text x="85" y="167" font-size="20" fill="#8b949e">×</text>

  <!-- W (sliced across 4 GPUs) -->
  <rect x="105" y="100" width="22" height="120" fill="#1f6feb" stroke="#0d1117"/>
  <rect x="127" y="100" width="22" height="120" fill="#388bfd" stroke="#0d1117"/>
  <rect x="149" y="100" width="22" height="120" fill="#58a6ff" stroke="#0d1117"/>
  <rect x="171" y="100" width="22" height="120" fill="#79c0ff" stroke="#0d1117"/>
  <text x="149" y="167" text-anchor="middle" font-size="13" fill="white" font-weight="600">W</text>
  <text x="116" y="240" text-anchor="middle" font-size="10" fill="#8b949e">GPU0</text>
  <text x="138" y="240" text-anchor="middle" font-size="10" fill="#8b949e">GPU1</text>
  <text x="160" y="240" text-anchor="middle" font-size="10" fill="#8b949e">GPU2</text>
  <text x="182" y="240" text-anchor="middle" font-size="10" fill="#8b949e">GPU3</text>

  <!-- = -->
  <text x="210" y="167" font-size="20" fill="#8b949e">=</text>

  <!-- Y (output, gathered) -->
  <rect x="232" y="120" width="80" height="80" fill="#3fb950" rx="3"/>
  <text x="272" y="165" text-anchor="middle" font-size="13" fill="white" font-weight="600">Y</text>
  <text x="272" y="240" text-anchor="middle" font-size="10" fill="#8b949e">all-reduce</text>

  <text x="200" y="290" text-anchor="middle" font-size="11" fill="#c9d1d9">Each GPU holds a slice of W and computes a partial Y.</text>
  <text x="200" y="307" text-anchor="middle" font-size="11" fill="#c9d1d9">All-reduce gathers the full output every layer.</text>
  <text x="200" y="328" text-anchor="middle" font-size="10.5" fill="#8b949e" font-style="italic">Cuts step time; cost is ICI bandwidth.</text>

  <!-- divider -->
  <line x1="410" y1="50" x2="410" y2="335" stroke="#30363d" stroke-dasharray="3,3"/>

  <!-- ===== RIGHT PANEL: PP ===== -->
  <text x="615" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#e6edf3">Pipeline Parallelism (PP)</text>
  <text x="615" y="46" text-anchor="middle" font-size="11" fill="#8b949e">slice the model by layer groups</text>

  <!-- 3 stages -->
  <rect x="450" y="80" width="100" height="42" fill="#1f6feb" rx="5"/>
  <text x="500" y="100" text-anchor="middle" font-size="11" fill="white" font-weight="600">Layers 1-20</text>
  <text x="500" y="116" text-anchor="middle" font-size="10" fill="#a5d2ff">GPU0</text>

  <rect x="565" y="80" width="100" height="42" fill="#388bfd" rx="5"/>
  <text x="615" y="100" text-anchor="middle" font-size="11" fill="white" font-weight="600">Layers 21-40</text>
  <text x="615" y="116" text-anchor="middle" font-size="10" fill="#a5d2ff">GPU1</text>

  <rect x="680" y="80" width="100" height="42" fill="#58a6ff" rx="5"/>
  <text x="730" y="100" text-anchor="middle" font-size="11" fill="white" font-weight="600">Layers 41-60</text>
  <text x="730" y="116" text-anchor="middle" font-size="10" fill="white">GPU2</text>

  <!-- arrows between stages -->
  <path d="M 552 101 L 562 101" stroke="#8b949e" stroke-width="1.5" marker-end="url(#arr)" fill="none"/>
  <path d="M 667 101 L 677 101" stroke="#8b949e" stroke-width="1.5" marker-end="url(#arr)" fill="none"/>

  <!-- Pipeline schedule -->
  <text x="615" y="160" text-anchor="middle" font-size="11" fill="#8b949e" font-weight="600">Pipeline schedule (time →)</text>

  <!-- GPU0 row -->
  <text x="440" y="187" text-anchor="end" font-size="10" fill="#8b949e">GPU0</text>
  <rect x="448" y="177" width="30" height="14" fill="#1f6feb"/>
  <rect x="478" y="177" width="30" height="14" fill="#1f6feb"/>
  <rect x="508" y="177" width="30" height="14" fill="#21262d" stroke="#30363d" stroke-dasharray="2,2"/>
  <rect x="538" y="177" width="30" height="14" fill="#21262d" stroke="#30363d" stroke-dasharray="2,2"/>
  <rect x="568" y="177" width="30" height="14" fill="#1f6feb"/>
  <rect x="598" y="177" width="30" height="14" fill="#1f6feb"/>

  <!-- GPU1 row -->
  <text x="440" y="207" text-anchor="end" font-size="10" fill="#8b949e">GPU1</text>
  <rect x="448" y="197" width="30" height="14" fill="#21262d" stroke="#30363d" stroke-dasharray="2,2"/>
  <rect x="478" y="197" width="30" height="14" fill="#388bfd"/>
  <rect x="508" y="197" width="30" height="14" fill="#388bfd"/>
  <rect x="538" y="197" width="30" height="14" fill="#21262d" stroke="#30363d" stroke-dasharray="2,2"/>
  <rect x="568" y="197" width="30" height="14" fill="#388bfd"/>
  <rect x="598" y="197" width="30" height="14" fill="#388bfd"/>

  <!-- GPU2 row -->
  <text x="440" y="227" text-anchor="end" font-size="10" fill="#8b949e">GPU2</text>
  <rect x="448" y="217" width="30" height="14" fill="#21262d" stroke="#30363d" stroke-dasharray="2,2"/>
  <rect x="478" y="217" width="30" height="14" fill="#21262d" stroke="#30363d" stroke-dasharray="2,2"/>
  <rect x="508" y="217" width="30" height="14" fill="#58a6ff"/>
  <rect x="538" y="217" width="30" height="14" fill="#58a6ff"/>
  <rect x="568" y="217" width="30" height="14" fill="#21262d" stroke="#30363d" stroke-dasharray="2,2"/>
  <rect x="598" y="217" width="30" height="14" fill="#58a6ff"/>

  <!-- legend -->
  <rect x="645" y="180" width="12" height="12" fill="#1f6feb"/>
  <text x="662" y="190" font-size="10" fill="#c9d1d9">computing</text>
  <rect x="645" y="200" width="12" height="12" fill="#21262d" stroke="#30363d" stroke-dasharray="2,2"/>
  <text x="662" y="210" font-size="10" fill="#c9d1d9">bubble (idle)</text>

  <text x="615" y="290" text-anchor="middle" font-size="11" fill="#c9d1d9">Each request crosses all stages in sequence.</text>
  <text x="615" y="307" text-anchor="middle" font-size="11" fill="#c9d1d9">Throughput scales; latency suffers from the bubbles.</text>
  <text x="615" y="328" text-anchor="middle" font-size="10.5" fill="#8b949e" font-style="italic">Only used when one node can't hold the model.</text>
</svg>
<p class="chart-caption">TP slices the matmul (cuts per-step latency, costs all-reduce bandwidth). PP slices the model (raises throughput, costs pipeline bubbles). These are the two operations that justify almost every multi-GPU inference topology.</p>
</div>"""


def inject_new_charts(md_text: str) -> str:
    """Insert new visuals that don't exist in the markdown source."""
    # MFU bar chart after the prefill/decode table in § 2.
    md_text = md_text.replace(
        "| Sharding that helps | TP, SP, FSDP | TP only (FSDP/DP useless) |",
        "| Sharding that helps | TP, SP, FSDP | TP only (FSDP/DP useless) |\n\n"
        '<div class="chart-block" data-chart="mfu-comparison"></div>',
    )
    # HBM budget chart in § 5.2 right after the topology table closes.
    md_text = md_text.replace(
        "The **disagg ratio** drops out of the P:D load math:",
        '<div class="chart-block" data-chart="hbm-budget"></div>\n\n'
        "The **disagg ratio** drops out of the P:D load math:",
    )
    # Parallelism cartoon in § 6, after the strategy table.
    md_text = md_text.replace(
        "| **FSDP** | Weights gathered per layer | Marginal | **Unviable** for decode (param loading dominates) | — |",
        "| **FSDP** | Weights gathered per layer | Marginal | **Unviable** for decode (param loading dominates) | — |\n\n"
        + PARALLELISM_SVG,
    )
    return md_text


# ---------- HTML post-processing ----------

def rewrite_mermaid_blocks(html_text: str) -> str:
    """python-markdown wraps ```mermaid as <pre><code class=...>; mermaid.js
    expects <div class="mermaid">. Convert."""
    pattern = re.compile(
        r'<pre><code class="language-mermaid">(.*?)</code></pre>',
        re.DOTALL,
    )
    return pattern.sub(
        lambda m: '<div class="mermaid">' + html_lib.unescape(m.group(1)) + '</div>',
        html_text,
    )


def build_toc(md_text: str) -> str:
    """Extract h2 headings and produce a flat sidebar TOC.

    h3s under § 0 (the workflow) get nested so the most-used section is
    navigable in one click.
    """
    lines = md_text.splitlines()
    items = []
    in_section_zero = False
    for line in lines:
        m2 = re.match(r"^## (.+)$", line)
        m3 = re.match(r"^### (.+)$", line)
        if m2:
            title = m2.group(1).strip()
            anchor = slugify(title)
            items.append((2, title, anchor))
            in_section_zero = title.startswith("0.")
        elif m3 and in_section_zero:
            title = m3.group(1).strip()
            anchor = slugify(title)
            items.append((3, title, anchor))
    out = ['<nav class="toc"><h2>Contents</h2><ul>']
    for level, title, anchor in items:
        cls = "toc-l3" if level == 3 else "toc-l2"
        out.append(f'<li class="{cls}"><a href="#{anchor}">{html_lib.escape(title)}</a></li>')
    out.append("</ul></nav>")
    return "\n".join(out)


def slugify(text: str) -> str:
    """Match python-markdown's `toc` extension slug rules so anchors line up."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")


# ---------- the HTML template ----------

CSS = r"""
:root {
  --bg: #0d1117;
  --bg-card: #161b22;
  --bg-elev: #1c2128;
  --text: #e6edf3;
  --text-body: #c9d1d9;
  --text-mute: #8b949e;
  --accent: #58a6ff;
  --accent-bright: #79c0ff;
  --em: #d2a8ff;
  --warn: #d29922;
  --warn-bg: rgba(210,153,34,0.08);
  --ok: #3fb950;
  --rose: #f85149;
  --border: #30363d;
  --code-bg: #161b22;
  --code-text: #e6edf3;
  --shadow: 0 1px 3px rgba(0,0,0,0.3);
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 20px; background: var(--bg); }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', "Segoe UI", Inter, system-ui, sans-serif;
  font-size: 14.5px;
  line-height: 1.7;
  color: var(--text-body);
  background: var(--bg);
  margin: 0;
}

.layout { display: grid; grid-template-columns: 280px 1fr; max-width: 1320px; margin: 0 auto; }

.toc {
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
  padding: 24px 16px 24px 24px;
  border-right: 1px solid var(--border);
  background: var(--bg);
  font-size: 13px;
}
.toc h2 {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-mute);
  margin: 0 0 12px 0;
  border: none;
  padding: 0;
}
.toc ul { list-style: none; padding: 0; margin: 0; }
.toc li { margin: 2px 0; }
.toc a {
  display: block;
  padding: 4px 10px;
  border-radius: 4px;
  color: var(--text-body);
  text-decoration: none;
  border-left: 2px solid transparent;
}
.toc a:hover { background: var(--bg-card); color: var(--text); }
.toc a.active { background: var(--bg-card); color: var(--accent); border-left-color: var(--accent); font-weight: 600; }
.toc-l3 a { padding-left: 26px; font-size: 12px; color: var(--text-mute); }

.content {
  padding: 40px 56px 80px 56px;
  max-width: 940px;
}

h1 { font-size: 30px; line-height: 1.2; margin: 0 0 8px 0; letter-spacing: -0.01em; color: var(--text); font-weight: 700; }
h2 {
  font-size: 22px;
  margin: 56px 0 16px 0;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
  letter-spacing: -0.01em;
  color: var(--text);
  font-weight: 700;
}
h3 { font-size: 17px; margin: 32px 0 12px 0; color: var(--accent); font-weight: 600; }
h4 { font-size: 13px; margin: 20px 0 8px 0; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-mute); font-weight: 600; }

p { margin: 12px 0; color: var(--text-body); }
strong { color: var(--text); font-weight: 600; }
em { color: var(--em); font-style: normal; }
a { color: var(--accent); text-decoration: none; border-bottom: 1px solid transparent; }
a:hover { border-bottom-color: var(--accent); }

ul, ol { padding-left: 22px; color: var(--text-body); }
li { margin: 6px 0; }

blockquote {
  margin: 16px 0;
  padding: 12px 16px;
  background: var(--warn-bg);
  border-left: 3px solid var(--warn);
  border-radius: 4px;
  color: var(--text-body);
}
blockquote p { margin: 0; color: var(--text-body); }
blockquote strong { color: var(--warn); }

code {
  font-family: 'SF Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12.5px;
  background: var(--code-bg);
  color: var(--accent-bright);
  padding: 1px 6px;
  border-radius: 3px;
}
pre {
  background: var(--code-bg);
  padding: 14px 16px;
  border-radius: 8px;
  overflow-x: auto;
  font-size: 12px;
  line-height: 1.55;
  border: 1px solid var(--border);
  color: var(--text-body);
}
pre code { background: transparent; padding: 0; font-size: 12px; color: var(--text-body); }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 16px 0;
  font-size: 13px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}
th, td {
  text-align: left;
  padding: 9px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
  color: var(--text-body);
}
th {
  background: var(--bg-elev);
  font-weight: 600;
  color: var(--text-mute);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border-bottom: 2px solid var(--border);
}
tr:hover td { background: var(--bg-card); }
tr:last-child td { border-bottom: none; }

hr { border: none; border-top: 1px solid var(--border); margin: 32px 0; }

.chart-block {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px;
  margin: 24px 0;
  box-shadow: var(--shadow);
}
.chart-block canvas { max-width: 100%; }
.chart-block svg { width: 100%; height: auto; display: block; }
.chart-caption {
  font-size: 12px;
  color: var(--text-mute);
  margin: 12px 0 0 0;
  font-style: italic;
}

.mermaid {
  text-align: center;
  margin: 24px 0;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px;
  box-shadow: var(--shadow);
}

.header-bar {
  background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #1c2128 100%);
  color: var(--text);
  padding: 36px 56px;
  margin: -40px -56px 28px -56px;
  border-bottom: 1px solid var(--border);
}
.header-bar h1 { color: var(--text); margin: 0 0 6px 0; }
.header-bar p { margin: 0; color: var(--text-mute); font-size: 14px; }
.header-bar .meta { margin-top: 14px; font-size: 12px; color: var(--text-mute); opacity: 0.85; }

.reading-guide {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 10px;
  margin: 12px 0 36px 0;
}
.reading-guide .card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  font-size: 12px;
  transition: border-color 0.15s;
}
.reading-guide .card:hover { border-color: var(--accent); }
.reading-guide .card strong { display: block; font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--accent); margin-bottom: 4px; }
.reading-guide .card a { color: var(--text-body); border: none; }
.reading-guide .card a:hover { color: var(--accent); }

@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  .toc { position: static; height: auto; border-right: none; border-bottom: 1px solid var(--border); padding: 16px; }
  .content { padding: 24px 16px; }
  .header-bar { margin: -24px -16px 24px -16px; padding: 24px 16px; }
}

@media print {
  body { background: white; color: black; }
  .toc { display: none; }
  .layout { grid-template-columns: 1fr; }
}
"""


CHART_JS = r"""
const COLORS = {
  bg: '#0d1117', card: '#161b22', text: '#e6edf3', body: '#c9d1d9', mute: '#8b949e',
  accent: '#58a6ff', accentSoft: 'rgba(88,166,255,0.22)',
  bright: '#79c0ff',
  warn: '#d29922', warnSoft: 'rgba(210,153,34,0.22)',
  ok: '#3fb950', okSoft: 'rgba(63,185,80,0.22)',
  rose: '#f85149', roseSoft: 'rgba(248,81,73,0.22)',
  purple: '#d2a8ff', purpleSoft: 'rgba(210,168,255,0.22)',
  border: '#30363d', borderSoft: 'rgba(48,54,61,0.6)',
};

Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'SF Pro', 'Segoe UI', Inter, system-ui, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.color = COLORS.body;
Chart.defaults.borderColor = COLORS.border;

function makeRoofline(canvas) {
  // Hardware: assume peak = 200 (arbitrary unit), HBM_BW = 1; roofline knee at AI = 200.
  // Curve: y = min(AI * BW, peak)
  const labels = [], data = [];
  for (let ai = 1; ai <= 2000; ai = Math.ceil(ai * 1.25)) {
    labels.push(ai);
    data.push(Math.min(ai, 200));
  }
  new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Achieved FLOPs/s',
        data,
        borderColor: COLORS.accent,
        backgroundColor: COLORS.accentSoft,
        fill: true,
        tension: 0,
        pointRadius: 0,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      aspectRatio: 2.4,
      plugins: {
        legend: { display: false },
        title: { display: true, text: 'Hardware roofline — achieved FLOPs/s vs arithmetic intensity', font: { size: 13, weight: '600' } },
        annotation: {},
      },
      scales: {
        x: {
          type: 'logarithmic',
          title: { display: true, text: 'Arithmetic intensity (FLOPs / byte)', color: COLORS.mute },
          grid: { color: COLORS.border },
        },
        y: {
          title: { display: true, text: 'Achieved FLOPs/s (arb. units, peak = 200)', color: COLORS.mute },
          grid: { color: COLORS.border },
          max: 240,
        },
      },
    },
  });
}

function makeLatencyThroughput(canvas) {
  // Memory-bound up to B_crit ≈ 200; flat past that (compute-bound; throughput per replica ceiling).
  const labels = [], data = [];
  for (let b = 1; b <= 1000; b = Math.ceil(b * 1.18)) {
    labels.push(b);
    data.push(Math.min(b * 5, 1000));
  }
  new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Tokens/sec (per replica)',
        data,
        borderColor: COLORS.ok,
        backgroundColor: COLORS.okSoft,
        fill: true,
        tension: 0,
        pointRadius: 0,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      aspectRatio: 2.4,
      plugins: {
        legend: { display: false },
        title: { display: true, text: 'Throughput vs batch — the B_crit knee', font: { size: 13, weight: '600' } },
      },
      scales: {
        x: {
          type: 'logarithmic',
          title: { display: true, text: 'Concurrent batch B (log)', color: COLORS.mute },
          grid: { color: COLORS.border },
        },
        y: {
          title: { display: true, text: 'Per-replica tokens/sec', color: COLORS.mute },
          grid: { color: COLORS.border },
        },
      },
    },
  });
}

function makeKvVariants(canvas) {
  new Chart(canvas, {
    type: 'bar',
    data: {
      labels: ['MHA', 'GQA-8 (Llama-3)', 'MQA', 'MLA (DeepSeek)'],
      datasets: [{
        label: 'KV cache (GB/seq @ 32K)',
        data: [256, 16, 2.0, 2.18],
        backgroundColor: [COLORS.rose, COLORS.warn, COLORS.ok, COLORS.accent],
        borderWidth: 0,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      aspectRatio: 2.6,
      plugins: {
        legend: { display: false },
        title: { display: true, text: 'KV cache per sequence at 32K context, DeepSeek-V3 architecture (log scale)', font: { size: 13, weight: '600' } },
        tooltip: { callbacks: { label: (c) => c.parsed.x + ' GB' } },
      },
      scales: {
        x: {
          type: 'logarithmic',
          title: { display: true, text: 'GB per sequence (log)', color: COLORS.mute },
          grid: { color: COLORS.border },
        },
        y: { grid: { display: false } },
      },
    },
  });
}

function makeMfuComparison(canvas) {
  new Chart(canvas, {
    type: 'bar',
    data: {
      labels: ['Prefill', 'Decode'],
      datasets: [
        { label: 'Typical low', data: [40, 5], backgroundColor: COLORS.accentSoft, borderColor: COLORS.accent, borderWidth: 1 },
        { label: 'Typical high', data: [20, 10], backgroundColor: COLORS.accent, borderWidth: 0 },
      ],
    },
    options: {
      responsive: true,
      aspectRatio: 2.6,
      plugins: {
        title: { display: true, text: 'MFU range — same GPU, different workloads', font: { size: 13, weight: '600' } },
        legend: { display: false },
        tooltip: { callbacks: { label: function(c) {
          // Stacked bar shows low + range; display total for clarity.
          const idx = c.dataIndex;
          const lows = [40, 5], highs = [60, 15];
          return [c.label + ' MFU range: ' + lows[idx] + '% – ' + highs[idx] + '%'];
        }}},
      },
      scales: {
        x: { stacked: true, grid: { display: false } },
        y: {
          stacked: true,
          beginAtZero: true,
          max: 70,
          title: { display: true, text: 'MFU (%)', color: COLORS.mute },
          grid: { color: COLORS.border },
        },
      },
    },
  });
}

function makeHbmBudget(canvas) {
  // Llama-70B int8 on 4×2 TPU v5e (~128 GB HBM total) at BS=32, 8k seq.
  new Chart(canvas, {
    type: 'bar',
    data: {
      labels: ['HBM budget (4×2 v5e, ~128 GB)'],
      datasets: [
        { label: 'Weights (int8)',  data: [70], backgroundColor: COLORS.accent },
        { label: 'KV @ BS=32, 8k',  data: [42], backgroundColor: COLORS.warn },
        { label: 'Activations',      data: [4],  backgroundColor: COLORS.ok },
        { label: 'Headroom',         data: [12], backgroundColor: COLORS.mute },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      aspectRatio: 4,
      plugins: {
        title: { display: true, text: 'Llama-70B int8 on 4×2 TPU v5e — HBM budget at BS=32, 8k', font: { size: 13, weight: '600' } },
        legend: { position: 'bottom' },
        tooltip: { callbacks: { label: (c) => c.dataset.label + ': ' + c.parsed.x + ' GB' } },
      },
      scales: {
        x: { stacked: true, max: 128, title: { display: true, text: 'GB', color: COLORS.mute }, grid: { color: COLORS.border } },
        y: { stacked: true, grid: { display: false } },
      },
    },
  });
}

const CHART_BUILDERS = {
  'roofline': makeRoofline,
  'latency-throughput': makeLatencyThroughput,
  'kv-variants': makeKvVariants,
  'mfu-comparison': makeMfuComparison,
  'hbm-budget': makeHbmBudget,
};

function initCharts() {
  document.querySelectorAll('.chart-block[data-chart]').forEach(block => {
    const key = block.dataset.chart;
    const builder = CHART_BUILDERS[key];
    if (!builder) return;
    const canvas = document.createElement('canvas');
    block.appendChild(canvas);
    builder(canvas);
  });
}

function initTocHighlight() {
  // Active section in TOC follows scroll position via IntersectionObserver.
  const sections = document.querySelectorAll('h2[id], h3[id]');
  const tocLinks = new Map();
  document.querySelectorAll('.toc a').forEach(a => {
    const id = decodeURIComponent(a.getAttribute('href').slice(1));
    tocLinks.set(id, a);
  });
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const link = tocLinks.get(entry.target.id);
      if (!link) return;
      if (entry.isIntersecting) {
        document.querySelectorAll('.toc a.active').forEach(a => a.classList.remove('active'));
        link.classList.add('active');
      }
    });
  }, { rootMargin: '-20% 0px -75% 0px' });
  sections.forEach(s => observer.observe(s));
}

document.addEventListener('DOMContentLoaded', () => {
  mermaid.initialize({
    startOnLoad: true,
    theme: 'dark',
    themeVariables: {
      fontFamily: Chart.defaults.font.family,
      background: COLORS.card,
      primaryColor: '#1c2128',
      primaryTextColor: COLORS.text,
      primaryBorderColor: COLORS.border,
      lineColor: COLORS.mute,
      secondaryColor: '#1f6feb',
      tertiaryColor: '#21262d',
    },
  });
  initCharts();
  initTocHighlight();
});
"""


READING_GUIDE_HTML = """
<div class="reading-guide">
  <div class="card"><strong>First time</strong><a href="#0-the-objective-workflow-start-here">§0 → §1–4 → §12</a></div>
  <div class="card"><strong>Sizing new deploy</strong><a href="#06-9-step-workflow">§0.6 workflow</a></div>
  <div class="card"><strong>Tuning existing</strong><a href="#9-engine-implementation-vllm-v1">§9 vLLM / §10 TRT-LLM</a></div>
  <div class="card"><strong>Sanity check</strong><a href="#11-decision-flowchart">§11 flowchart</a></div>
</div>
"""


def wrap_template(body: str, toc_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model Sizing &amp; Scaling — Reference</title>
<style>{CSS}</style>
</head>
<body>
<div class="layout">
{toc_html}
<main class="content">
<div class="header-bar">
<h1>Model Sizing &amp; Scaling</h1>
<p>A consolidated reference for sizing and scaling decisions in LLM inference</p>
<div class="meta">Distilled from primary sources · last updated May 2026</div>
</div>
{READING_GUIDE_HTML}
{body}
</main>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>{CHART_JS}</script>
</body>
</html>
"""


# ---------- main ----------

def main():
    md_text = SRC.read_text()
    md_text = replace_ascii_charts(md_text)
    md_text = inject_new_charts(md_text)

    # `toc` extension adds slug IDs to headings; we don't render the [TOC] marker.
    converter = markdown.Markdown(
        extensions=["fenced_code", "tables", "attr_list", "sane_lists", "toc"],
        extension_configs={"toc": {"marker": "", "toc_depth": "2-3"}},
    )
    html_body = converter.convert(md_text)
    html_body = rewrite_mermaid_blocks(html_body)

    toc_html = build_toc(md_text)

    DST.write_text(wrap_template(html_body, toc_html))
    print(f"Wrote {DST} ({DST.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
