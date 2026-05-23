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
    # Permitted external refs: Chart.js CDN, Google Fonts. Internal anchor (../)
    # for the back-link to the reference doc is allowed (siblings in the repo).
    allowed = re.compile(
        r"^(https://(cdn\.jsdelivr\.net|fonts\.googleapis\.com|fonts\.gstatic\.com)(/|$)|"
        r"\.\./reference/MODEL_SIZING_SCALING_REFERENCE\.html|"
        r"data:|#)",
    )
    for m in re.finditer(r'(?:src|href)\s*=\s*"([^"]+)"', html):
        url = m.group(1)
        assert allowed.match(url), f"unexpected external/local reference: {url!r}"


def test_embedded_json_round_trips(html: str) -> None:
    """The embedded <script type=application/json> blocks must equal source JSON."""
    for tag, src_path in [
        ("hardware-data", SRC / "data" / "hardware.json"),
        ("models-data",   SRC / "data" / "models.json"),
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
        "m-tps", "m-tps-pg", "m-pd", "m-par", "m-par-sub", "m-weights", "m-kv",
        # Sparklines
        "spark-bcrit", "spark-bslo", "spark-bkv",
        # Chart + sweep + snippet + diagnostics
        "scope-canvas", "patchbay", "snippet-body", "copy-btn", "diagnostics",
        # Custom-mode fields
        "c-hbm-gb", "c-hbm-bw", "c-fp16", "c-fp8", "c-nvlink",
        "c-params", "c-layers", "c-dmodel", "c-heads", "c-kvheads",
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
    # for parser-correctness. 1 chart.js CDN + 2 JSON blocks + 1 module = 4.
    closes = len(re.findall(r"</script\s*>", html))
    assert closes == 4, f"unexpected </script tag count: {closes} (a literal </script> probably leaked from a source comment)"


def test_no_obvious_secrets_or_pii(html: str) -> None:
    """OSS release safety check — no API keys, no email addresses leaked."""
    assert not re.search(r"sk-[A-Za-z0-9]{20,}", html), "looks like an API key"
    assert "AKIA" not in html, "looks like an AWS access key"
