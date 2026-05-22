"""Render inference_lab_results.ipynb to a dark-themed HTML matching the
Phase 3/4 dashboard style. Strips code cells, keeps all markdown prose and
chart outputs in their original sequencing.
"""
import base64
import json
from pathlib import Path
import markdown as md

ROOT = Path(__file__).parent.parent  # repo root (this script lives in tools/)
NB = ROOT / "lab" / "inference_lab_results.ipynb"
OUT = ROOT / "lab" / "inference_lab_dark.html"

CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif;
  background: #0d1117; color: #e6edf3;
}
.hero {
  background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #1c2128 100%);
  padding: 48px 40px 40px; border-bottom: 3px solid #58a6ff;
}
.hero h1 { font-size: 32px; font-weight: 700; margin-bottom: 6px; letter-spacing: -0.5px; }
.hero .subtitle { color: #8b949e; font-size: 15px; margin-bottom: 20px; }

.content { max-width: 1100px; margin: 0 auto; padding: 32px 40px 60px; }

h1 { font-size: 26px; font-weight: 700; margin: 32px 0 14px; color: #e6edf3; }
h2 { font-size: 22px; font-weight: 700; margin: 36px 0 12px; color: #e6edf3;
     border-bottom: 2px solid #30363d; padding-bottom: 8px; }
h3 { font-size: 17px; font-weight: 600; margin: 24px 0 10px; color: #58a6ff; }
h4 { font-size: 15px; font-weight: 600; margin: 18px 0 8px; color: #79c0ff; }

p, li { font-size: 14.5px; line-height: 1.75; color: #c9d1d9; margin-bottom: 8px; }
ul, ol { padding-left: 22px; margin-bottom: 12px; }
strong { color: #e6edf3; }
em { color: #d2a8ff; }
hr { border: none; border-top: 1px solid #30363d; margin: 32px 0; }

a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }

code {
  background: #161b22; color: #79c0ff;
  padding: 2px 6px; border-radius: 4px;
  font-family: 'SF Mono', Menlo, monospace; font-size: 12.5px;
  border: 1px solid #21262d;
}
pre {
  background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  padding: 14px 18px; overflow-x: auto; margin: 12px 0;
}
pre code {
  background: transparent; border: 0; padding: 0; color: #c9d1d9; font-size: 12.5px;
}

table {
  width: 100%; border-collapse: collapse; margin: 14px 0 22px;
  font-size: 13px; background: #0d1117;
}
th {
  text-align: left; padding: 9px 12px; font-weight: 600; color: #8b949e;
  border-bottom: 2px solid #30363d; background: #161b22;
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px;
}
td {
  padding: 8px 12px; border-bottom: 1px solid #21262d;
  color: #c9d1d9; font-size: 12.5px;
}
tr:hover td { background: #161b22; }

blockquote {
  border-left: 4px solid #58a6ff; background: rgba(88,166,255,0.08);
  padding: 12px 18px; margin: 14px 0; border-radius: 0 8px 8px 0;
  color: #c9d1d9;
}
blockquote p { color: #c9d1d9; margin-bottom: 4px; }

figure.chart {
  background: #161b22; border: 1px solid #30363d; border-radius: 10px;
  padding: 14px; margin: 18px 0; text-align: center;
}
figure.chart img { max-width: 100%; height: auto; border-radius: 6px; }

.stream-out {
  background: #0d1117; border: 1px dashed #30363d; border-radius: 8px;
  padding: 12px 16px; margin: 10px 0 18px;
  font-family: 'SF Mono', Menlo, monospace; font-size: 11.5px;
  color: #8b949e; white-space: pre-wrap; overflow-x: auto;
  max-height: 360px; overflow-y: auto;
}

.footer {
  text-align: center; padding: 32px; color: #484f58; font-size: 12px;
  border-top: 1px solid #21262d; margin-top: 40px;
}
"""

HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LLM Inference Learning Lab — Results & Analysis</title>
<style>%s</style>
</head>
<body>
<div class="hero">
  <h1>LLM Inference Learning Lab &mdash; Results &amp; Analysis</h1>
  <p class="subtitle">
    Stage 1 (Ollama, Apple M4) &bull; Stage 2 (vLLM on g4dn.xlarge / g5.xlarge Spot) &bull;
    Continuous batching, quantization, KV cache, FP8, prefix caching, soak &amp; overload tests
  </p>
</div>
<div class="content">
""" % CSS

FOOTER = """
</div>
<div class="footer">Generated from inference_lab_results.ipynb &mdash; same content &amp; sequencing, code stripped.</div>
</body>
</html>
"""


def render_markdown(src: str) -> str:
    return md.markdown(
        src,
        extensions=["tables", "fenced_code", "attr_list", "md_in_html"],
    )


def render_outputs(cell) -> str:
    """Render a code cell's outputs (images + text) without showing the code."""
    parts = []
    for out in cell.get("outputs", []):
        data = out.get("data", {})
        if "image/png" in data:
            png = data["image/png"]
            if isinstance(png, list):
                png = "".join(png)
            parts.append(
                f'<figure class="chart"><img src="data:image/png;base64,{png}" alt="chart"/></figure>'
            )
            continue
        if "text/html" in data:
            html = data["text/html"]
            if isinstance(html, list):
                html = "".join(html)
            parts.append(html)
            continue
        if "text/plain" in data:
            txt = data["text/plain"]
            if isinstance(txt, list):
                txt = "".join(txt)
            # Skip noisy matplotlib repr lines like "<Figure size ...>" or "[<matplotlib...>]".
            stripped = txt.strip()
            if stripped.startswith("<") and stripped.endswith(">"):
                continue
            if stripped.startswith("[<") and stripped.endswith(">]"):
                continue
            parts.append(f'<div class="stream-out">{escape(txt)}</div>')
            continue
        # Stream output (stdout/stderr).
        if out.get("output_type") == "stream":
            txt = out.get("text", "")
            if isinstance(txt, list):
                txt = "".join(txt)
            if txt.strip():
                parts.append(f'<div class="stream-out">{escape(txt)}</div>')
    return "\n".join(parts)


def escape(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def main():
    nb = json.loads(NB.read_text())
    chunks = [HEADER]
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            src = "".join(cell["source"])
            chunks.append(render_markdown(src))
        elif cell["cell_type"] == "code":
            chunks.append(render_outputs(cell))
    chunks.append(FOOTER)
    OUT.write_text("\n".join(chunks))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
