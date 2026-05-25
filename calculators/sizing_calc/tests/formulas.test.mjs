// Validates `src/data/formulas.json` — the content that powers the in-UI
// formulas drawer. These tests are the regression net that keeps the data
// file from drifting silently away from calc.mjs as the code evolves.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const formulasPath = join(root, "src/data/formulas.json");
const calcPath = join(root, "src/calc.mjs");
const refDocPath = join(root, "../../reference/MODEL_SIZING_SCALING_REFERENCE.md");

const formulas = JSON.parse(readFileSync(formulasPath, "utf8")).formulas;
const calcSrc = readFileSync(calcPath, "utf8");
const calcLines = calcSrc.split("\n");
const refSrc = readFileSync(refDocPath, "utf8");

// Build the set of exported names from calc.mjs by source-grep — no eval,
// no dynamic import, so the test works even if calc.mjs has a typo elsewhere.
const exportRegex = /^export\s+(?:function|const|class)\s+([A-Za-z_$][\w$]*)/gm;
const calcExports = new Set();
for (const m of calcSrc.matchAll(exportRegex)) calcExports.add(m[1]);

// Names that intentionally have NO standalone drawer entry. Two cases:
//  - "compute" — top-level orchestrator; its pieces are documented per-card
//  - The seven constants (ACT_OVERHEAD, TP_CAP, ALPHA, PREFILL_MFU,
//    DECODE_MFU, DTYPE_BYTES, SARATHI_BANDS) — bundled into a single
//    "constants" card whose coverage is checked separately below
const NOT_USER_FACING = new Set([
  "compute",
  "ACT_OVERHEAD", "TP_CAP", "ALPHA", "PREFILL_MFU", "DECODE_MFU",
  "DTYPE_BYTES", "SARATHI_BANDS",
  // Strict-mode TBT threshold (ms) and the corresponding mnbt cap. Implementation
  // detail of recommendMaxBatchedTokens — the "strict band" story is told in the
  // SARATHI_BANDS prose of the constants card, no need for separate symbols.
  "SARATHI_STRICT_TBT_MS", "SARATHI_STRICT_CAP",
]);

const ALLOWED_CATEGORIES = new Set([
  "constants", "memory", "regime", "latency", "capacity",
  "parallelism", "engine_knobs", "disagg", "cost",
]);

const REQUIRED_FIELDS = [
  "id", "name", "category", "expr_tex", "where",
  "description", "derivation", "citation", "calc_fn", "calc_path",
];

const TRUSTED_URL_HOSTS = [
  "arxiv.org", "jax-ml.github.io", "developer.nvidia.com",
  "research.character.ai", "www.usenix.org", "usenix.org",
  "github.com", "huggingface.co", "docs.vllm.ai",
  "deepseek.com", "api-docs.deepseek.com",
];

test("formulas.json: has the expected number of entries", () => {
  // 16 entries documented in the plan: 1 constants card + 15 functions
  // (kvPerToken is split into MHA and MLA cards because the closed forms
  // differ enough to warrant separate cards in the UI). Update both here and
  // in the plan when a new formula lands.
  assert.equal(formulas.length, 16, `expected 16 entries, got ${formulas.length}`);
});

test("formulas.json: every entry has all required fields", () => {
  for (const f of formulas) {
    for (const field of REQUIRED_FIELDS) {
      assert.ok(field in f, `entry "${f.id ?? "<no id>"}" missing field "${field}"`);
    }
  }
});

test("formulas.json: ids are unique", () => {
  // The id is the DOM key in the drawer; duplicates would silently overwrite.
  const seen = new Set();
  for (const f of formulas) {
    assert.ok(!seen.has(f.id), `duplicate id: ${f.id}`);
    seen.add(f.id);
  }
});

test("formulas.json: every category is on the enum", () => {
  for (const f of formulas) {
    assert.ok(ALLOWED_CATEGORIES.has(f.category),
      `entry "${f.id}" has invalid category "${f.category}". Allowed: ${[...ALLOWED_CATEGORIES].join(", ")}`);
  }
});

test("formulas.json: calc_fn matches a real calc.mjs export (or the constants sentinel)", () => {
  // calc_fn must be either an actual exported symbol or the literal string
  // "(constants)" used by the single constants-bundle card.
  for (const f of formulas) {
    if (f.calc_fn === "(constants)") continue;
    assert.ok(calcExports.has(f.calc_fn),
      `entry "${f.id}" references calc_fn "${f.calc_fn}" which is not exported by calc.mjs. ` +
      `Either rename the entry or update calc.mjs.`);
  }
});

test("formulas.json: calc_path line numbers resolve to the matching export", () => {
  // Catches drift when a function moves within calc.mjs. The line must
  // either declare the matching export OR (for the constants card) be one of
  // the lines that defines a constant.
  for (const f of formulas) {
    const m = /^calc\.mjs:(\d+)$/.exec(f.calc_path);
    assert.ok(m, `entry "${f.id}" calc_path "${f.calc_path}" is not "calc.mjs:NUMBER"`);
    const line = calcLines[Number(m[1]) - 1] ?? "";
    if (f.calc_fn === "(constants)") {
      // Any "export const" line is acceptable for the constants card.
      assert.match(line, /^export\s+const\s+/,
        `entry "${f.id}" calc_path line ${m[1]} is "${line.trim()}" — expected an "export const" line`);
    } else {
      const pattern = new RegExp(`^export\\s+(?:function|const|class)\\s+${f.calc_fn}\\b`);
      assert.match(line, pattern,
        `entry "${f.id}" calc_path line ${m[1]} is "${line.trim()}" — expected "export … ${f.calc_fn}"`);
    }
  }
});

test("formulas.json: every user-facing calc.mjs export is documented", () => {
  // Inventory completeness: a contributor adds a new export, this test fires.
  // kvPerToken is referenced twice (MHA + MLA cards) so the set-based check
  // works without a duplicate-detection special case.
  const documentedFns = new Set(formulas.map((f) => f.calc_fn));
  for (const sym of calcExports) {
    if (NOT_USER_FACING.has(sym)) continue;
    assert.ok(documentedFns.has(sym),
      `calc.mjs exports "${sym}" but no entry in formulas.json documents it. ` +
      `Either add a card or add the name to NOT_USER_FACING in this test.`);
  }
});

test("formulas.json: every expr_tex parses cleanly through KaTeX", async () => {
  // The browser will load KaTeX from CDN; we use the npm package here so the
  // test fails noisily on bad LaTeX BEFORE shipping it to users.
  const require = createRequire(import.meta.url);
  const katex = require("katex");
  for (const f of formulas) {
    assert.doesNotThrow(
      () => katex.renderToString(f.expr_tex, { throwOnError: true, displayMode: true }),
      `entry "${f.id}" expr_tex failed KaTeX parse`,
    );
  }
});

test("formulas.json: every where[] entry is a [name, meaning] pair of non-empty strings", () => {
  for (const f of formulas) {
    assert.ok(Array.isArray(f.where), `entry "${f.id}" where is not an array`);
    for (const row of f.where) {
      assert.ok(Array.isArray(row) && row.length === 2,
        `entry "${f.id}" where row is not a 2-tuple: ${JSON.stringify(row)}`);
      assert.equal(typeof row[0], "string", `entry "${f.id}" where[0] not string`);
      assert.equal(typeof row[1], "string", `entry "${f.id}" where[1] not string`);
      assert.ok(row[0].length > 0 && row[1].length > 0,
        `entry "${f.id}" where row has empty member`);
    }
  }
});

test("formulas.json: every citation has either anchor or url", () => {
  for (const f of formulas) {
    assert.ok(Array.isArray(f.citation) && f.citation.length > 0,
      `entry "${f.id}" has no citations`);
    for (const c of f.citation) {
      assert.ok(typeof c.label === "string" && c.label.length > 0,
        `entry "${f.id}" citation missing label: ${JSON.stringify(c)}`);
      const hasAnchor = typeof c.anchor === "string" && c.anchor.length > 0;
      const hasUrl = typeof c.url === "string" && c.url.length > 0;
      assert.ok(hasAnchor || hasUrl,
        `entry "${f.id}" citation "${c.label}" has neither anchor nor url`);
    }
  }
});

test("formulas.json: every citation has at least one Reference §N anchor", () => {
  // Forces every formula to trace back to the reference doc, not just to
  // external papers — keeps the calculator self-documenting.
  for (const f of formulas) {
    const hasRefAnchor = f.citation.some((c) =>
      typeof c.label === "string" && /Reference\s+§/.test(c.label));
    assert.ok(hasRefAnchor,
      `entry "${f.id}" must cite at least one "Reference §N" — found: ${f.citation.map((c) => c.label).join(", ")}`);
  }
});

test("formulas.json: external citation URLs match the trusted-host allowlist", () => {
  // Prevents a drive-by edit from pointing citations at arbitrary blogs.
  for (const f of formulas) {
    for (const c of f.citation) {
      if (!c.url) continue;
      const u = new URL(c.url);
      assert.ok(TRUSTED_URL_HOSTS.includes(u.hostname),
        `entry "${f.id}" citation "${c.label}" points at "${u.hostname}" — not on the trusted allowlist. ` +
        `Add the host to TRUSTED_URL_HOSTS in this test if it's a legitimate primary source.`);
    }
  }
});

test("formulas.json: constants card mentions every exported constant", () => {
  // The single constants card is the only entry that bundles multiple
  // symbols — make sure none of the actual exported constants got dropped.
  const constantsEntry = formulas.find((f) => f.id === "constants");
  assert.ok(constantsEntry, `no entry with id "constants"`);
  // For each constant, accept either the literal name (e.g. "ACT_OVERHEAD",
  // possibly with KaTeX-escaped underscore "\_") OR a conventional symbol
  // (e.g. \alpha for ALPHA) — both are legitimate ways to render the card.
  const expected = [
    { name: "ACT_OVERHEAD", symbols: [] },
    { name: "TP_CAP", symbols: [] },
    { name: "ALPHA", symbols: ["\\alpha"] },
    { name: "PREFILL_MFU", symbols: [] },
    { name: "DECODE_MFU", symbols: [] },
    { name: "DTYPE_BYTES", symbols: [] },
    { name: "SARATHI_BANDS", symbols: [] },
  ];
  const blob = constantsEntry.expr_tex + "\n" +
               constantsEntry.description + "\n" +
               constantsEntry.derivation;
  for (const { name, symbols } of expected) {
    const candidates = [name.replace(/_/g, "\\\\?_"), ...symbols.map((s) => s.replace(/\\/g, "\\\\"))];
    const rx = new RegExp(candidates.join("|"));
    assert.match(blob, rx, `constants card does not mention "${name}" (looked for any of: ${candidates.join(", ")})`);
  }
});

test("formulas.json: anchor citations point at headings that exist in the reference doc", () => {
  // Parse the reference doc's headings and confirm every internal anchor
  // citation lands on a real one.
  const headingRegex = /^#{1,3}\s+(.+?)\s*$/gm;
  const slugs = new Set();
  for (const m of refSrc.matchAll(headingRegex)) {
    // GitHub-style slug: lowercase, drop punctuation entirely, runs of
    // whitespace become a single "-", collapse any resulting "--+" to "-"
    // (matches the slugger GitHub uses for in-doc anchor links).
    const slug = m[1].toLowerCase()
      .replace(/[^a-z0-9\s-]/g, "")
      .trim()
      .replace(/\s+/g, "-")
      .replace(/-+/g, "-");
    slugs.add(slug);
  }
  for (const f of formulas) {
    for (const c of f.citation) {
      if (!c.anchor) continue;
      const hashIdx = c.anchor.indexOf("#");
      if (hashIdx < 0) continue;
      const slug = c.anchor.slice(hashIdx + 1);
      assert.ok(slugs.has(slug),
        `entry "${f.id}" citation "${c.label}" anchor "#${slug}" does not match any heading in MODEL_SIZING_SCALING_REFERENCE.md`);
    }
  }
});
