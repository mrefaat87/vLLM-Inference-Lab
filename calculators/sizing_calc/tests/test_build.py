"""Validates the build-script output. Catches drift between the src/ modules,
the data tables, and the generated HTML — the kind of error that would let the
calculator silently render a stale chart for weeks.

Run from repo root: `python3 -m pytest calculators/sizing_calc/tests/build.test.py -v`
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "tools" / "build_sizing_calculator_html.py"
SRC = ROOT / "calculators" / "sizing_calc" / "src"
DST = ROOT / "calculators" / "sizing_calculator.html"


@pytest.fixture(scope="module")
def html() -> str:
    # Always rebuild — the test must reflect current src/, not a stale artifact.
    subprocess.run([sys.executable, str(BUILD)], check=True, capture_output=True)
    assert DST.exists(), "build did not produce the output file"
    return DST.read_text()


def test_output_is_single_file(html: str) -> None:
    """No local script/style/image references — only the CDN ones are allowed.

    `file://` users get nothing but the single HTML + Chart.js + Google Fonts.
    Anything else means we forgot to inline an asset.
    """
    # Permitted external refs: Chart.js CDN, Google Fonts, KaTeX CDN (chart
    # canvas, the page's typography, and the formulas drawer respectively).
    # KaTeX's CSS chains to webfont URLs on the same jsdelivr origin so the
    # broad cdn.jsdelivr.net allow covers them. Internal anchor (../) for the
    # back-link to the reference doc is allowed (siblings in the repo).
    allowed = re.compile(
        r"^(https://(cdn\.jsdelivr\.net|fonts\.googleapis\.com|fonts\.gstatic\.com)(/|$)|"
        r"\.\./reference/MODEL_SIZING_SCALING_REFERENCE\.html|"
        r"data:|#)",
    )
    for m in re.finditer(r'(?:src|href)\s*=\s*"([^"]+)"', html):
        url = m.group(1)
        # Skip JS template-literal interpolations like `${expr}` that appear
        # inside the inlined module bundle (e.g. drawer.mjs builds citation
        # links via `<a href="${href}">`). These are JS source, not real HTML
        # attributes the parser will resolve.
        if "${" in url:
            continue
        assert allowed.match(url), f"unexpected external/local reference: {url!r}"


def test_embedded_json_round_trips(html: str) -> None:
    """The embedded <script type=application/json> blocks must equal source JSON."""
    for tag, src_path in [
        ("hardware-data",      SRC / "data" / "hardware.json"),
        ("models-data",        SRC / "data" / "models.json"),
        ("formulas-data",      SRC / "data" / "formulas.json"),
        ("compatibility-data", SRC / "data" / "compatibility.json"),
    ]:
        m = re.search(rf'<script type="application/json" id="{tag}">(.*?)</script>',
                      html, re.DOTALL)
        assert m, f"missing embedded json for {tag}"
        embedded = json.loads(m.group(1))
        original = json.loads(src_path.read_text())
        assert embedded == original, f"{tag} drifted from source"


def test_required_form_ids_present(html: str) -> None:
    """Every form ID that ui.mjs reads must exist in the rendered HTML.

    This is the contract between the JS reader and the HTML markup; without it
    a typo in either silently produces NaN metrics.
    """
    required = [
        "hw", "model", "hw-custom", "model-custom",
        "isl", "osl", "weight-prec", "kv-prec", "act-prec",
        "tbt", "ttft", "ngpus", "sizing-form",
        # Readout tiles
        "m-bcrit", "m-bslo", "m-bkv", "m-mns", "m-mnbt", "m-mnbt-band",
        "m-tps", "m-tps-pg", "m-cost", "m-pd", "m-par", "m-par-sub", "m-weights", "m-kv",
        # Cost wiring: price input + scope sections (one per channel) + canvases.
        "price-per-hour",
        "scope-latency", "scope-throughput", "scope-cost",
        "scope-latency-canvas", "scope-throughput-canvas", "scope-cost-canvas",
        # Sparklines
        "spark-bcrit", "spark-bslo", "spark-bkv",
        # Chart + sweep + snippet + diagnostics
        "patchbay", "snippet-body", "snippet-block", "copy-btn", "copy-exp-btn", "diagnostics",
        # Custom-mode fields
        "c-hbm-gb", "c-hbm-bw", "c-fp16", "c-fp8", "c-nvlink",
        "c-params-total", "c-params-active", "c-layers", "c-dmodel", "c-heads", "c-kvheads",
        "c-headdim", "c-maxctx", "c-attn",
    ]
    for ident in required:
        assert f'id="{ident}"' in html, f"missing form ID #{ident}"


def test_calc_module_inlined_intact(html: str) -> None:
    """Source-of-truth invariant: every export from calc.mjs that the unit tests
    rely on must show up in the inlined bundle. Catches accidental drift in the
    bundler's strip-module-syntax pass.
    """
    calc_src = (SRC / "calc.mjs").read_text()
    # Names whose presence in the bundle proves the file made it in.
    must_have = [
        "function weightsBytes", "function kvPerToken", "function bCrit",
        "function stepTime", "function bSlo", "function bKv",
        "function recommendParallelism", "function recommendMaxBatchedTokens",
        "function pdRatio", "function sweepRanges", "function compute",
        "DTYPE_BYTES", "ACT_OVERHEAD",
        # compatibility module must bundle in (ui.mjs calls validate()).
        "function validate", "gateSnippetButtons",
    ]
    for name in must_have:
        assert name in html, f"missing inlined symbol: {name!r}"
    # Sanity: no leftover module syntax that would break the inline script.
    assert "\nimport " not in html, "leftover import statement in bundle"
    assert "\nexport " not in html, "leftover export statement in bundle"


def test_exactly_one_module_script_open_and_close(html: str) -> None:
    """Regression: a literal </script> in any source file would close our
    inline module script early and silently break the page. The bundler
    escapes them; this test makes sure that defence stays in place.
    """
    # Opening <script tags in source-code strings inside the bundle don't fool
    # the HTML parser; closing </script> tags do. Only the close count matters
    # for parser-correctness. 1 chart.js CDN + 1 katex CDN + 4 JSON blocks
    # (hardware/models/formulas/compatibility) + 1 module = 7.
    closes = len(re.findall(r"</script\s*>", html))
    assert closes == 7, f"unexpected </script tag count: {closes} (a literal </script> probably leaked from a source comment)"


def test_three_scope_canvases_no_duplicate_ids(html: str) -> None:
    """Each scope must have its own canvas with a unique id. Chart.js binds to
    a canvas by element reference, but duplicate IDs let two charts attach to
    the same DOM node and silently overwrite each other's draw state — a real
    bug class. This guards against accidental copy-paste duplication."""
    for canvas_id in ("scope-latency-canvas", "scope-throughput-canvas", "scope-cost-canvas"):
        occurrences = html.count(f'id="{canvas_id}"')
        assert occurrences == 1, f"canvas id {canvas_id!r} appears {occurrences} times (must be exactly 1)"
    # And the count of <canvas …id="scope-…-canvas"> markers must equal 3.
    canvases = re.findall(r'<canvas\s+id="scope-(latency|throughput|cost)-canvas"', html)
    assert len(canvases) == 3, f"expected 3 scope canvases, found {len(canvases)}: {canvases}"


def test_no_obvious_secrets_or_pii(html: str) -> None:
    """OSS release safety check — no API keys, no email addresses leaked."""
    assert not re.search(r"sk-[A-Za-z0-9]{20,}", html), "looks like an API key"
    assert "AKIA" not in html, "looks like an AWS access key"


# ──────────────────────────────────────────────────────────────────────────
# Formulas drawer — scaffold, KaTeX wiring, and byte-budget guards.
# ──────────────────────────────────────────────────────────────────────────


def test_formulas_data_length_matches_source(html: str) -> None:
    """Inlined formulas.json must contain the same number of entries as the
    source file. Catches a build that silently truncates the JSON."""
    m = re.search(r'<script type="application/json" id="formulas-data">(.*?)</script>',
                  html, re.DOTALL)
    assert m, "missing embedded formulas-data tag"
    inlined = json.loads(m.group(1))
    source = json.loads((SRC / "data" / "formulas.json").read_text())
    assert len(inlined["formulas"]) == len(source["formulas"]) > 0


def test_katex_externals_present_and_pinned(html: str) -> None:
    """KaTeX powers the formulas drawer. Both CSS and JS must be referenced,
    both pinned to a version (no `@latest` — supply-chain hygiene), and the
    onload event must dispatch `katex-loaded` so the drawer can upgrade
    fallback <code> blocks once the script arrives."""
    assert "katex.min.css" in html, "KaTeX CSS not linked"
    assert "katex.min.js" in html, "KaTeX JS not linked"
    # Pinned version — refuse `@latest` to keep behavior reproducible.
    assert not re.search(r"katex@latest", html), "KaTeX must be pinned, not @latest"
    assert re.search(r"katex@\d+\.\d+\.\d+", html), "KaTeX version must be pinned with x.y.z"
    assert "katex-loaded" in html, "onload event 'katex-loaded' not wired"


def test_formulas_drawer_scaffold_present(html: str) -> None:
    """The drawer's DOM scaffold (trigger button + drawer + backdrop) must be
    in the rendered HTML, with ARIA wiring so screen readers announce it as a
    dialog and the trigger advertises which element it controls."""
    assert re.search(r'id="formulas-trigger"[^>]*aria-controls="formulas-drawer"',
                     html), "trigger button missing or not wired to drawer"
    assert re.search(r'id="formulas-trigger"[^>]*aria-expanded="false"',
                     html), "trigger must start with aria-expanded=false"
    assert re.search(r'id="formulas-drawer"[^>]*role="dialog"',
                     html), "drawer must have role=dialog"
    assert re.search(r'id="formulas-drawer"[^>]*aria-modal="true"',
                     html), "drawer must have aria-modal=true"
    assert re.search(r'id="formulas-drawer"[^>]*\bhidden\b',
                     html), "drawer must start hidden"
    assert re.search(r'id="formulas-backdrop"[^>]*\bhidden\b',
                     html), "backdrop must start hidden"
    assert 'class="formulas-close"' in html, "missing close button"
    assert 'class="formulas-body"' in html, "missing scrollable body container"


def test_drawer_module_bundled(html: str) -> None:
    """The drawer module must be concatenated into the inline <script type=module>
    bundle — guards the 'I added the file but forgot bundle_modules()' mistake."""
    assert "function mountFormulasDrawer" in html, "drawer.mjs not in bundle"
    # ui.mjs must actually call the mount function (not just import the name).
    assert "mountFormulasDrawer()" in html, "ui.mjs bootstrap doesn't call mountFormulasDrawer"
    # And the drawer's category-order constant should make it through.
    assert "CATEGORY_ORDER" in html, "drawer's CATEGORY_ORDER constant missing — module wasn't bundled fully"


def test_html_byte_budget(html: str) -> None:
    """Page weight ceiling. Current build ~162 KB; budget set at 220 KB so a
    careless edit that inlines a fat asset (e.g. dropping the KaTeX CDN ref
    in favor of bundled CSS, ~330 KB) trips this test before shipping."""
    size = len(html.encode("utf-8"))
    assert size <= 220_000, (
        f"HTML grew to {size:,} bytes (budget 220,000). If this is intentional, "
        f"raise the budget. If not, check what just got inlined."
    )
