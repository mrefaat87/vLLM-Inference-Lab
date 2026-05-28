// prefillFromQueryString: unit tests against a stubbed DOM.
//
// The calc's deep-link prefill is the calc-side half of "open this run
// in the sizing calc" from the Results Explorer. It must:
//   1. round-trip a complete, valid query string into the form fields,
//   2. silently ignore unknown <select> values (don't blank out defaults),
//   3. silently ignore non-numeric numeric inputs,
//   4. no-op on an empty query string (browser-opened-directly case),
//   5. leave fields whose query keys are absent alone (partial deep-link).
//
// We stub `document.getElementById` with a tiny in-memory DOM so this
// runs under `node --test` without jsdom — same pattern as
// validation.test.mjs.

import { test } from "node:test";
import { strict as assert } from "node:assert";

// Pre-import DOM stubs. ui.mjs runs top-level code at import time:
//   - reads <script id="hardware-data"> / <script id="models-data"> JSON tags
//   - resolves a CSS media query (for prefers-reduced-motion)
//   - calls bootstrap() which expects a populated form + scope canvases
// Provide a baseline document/window/Chart stub before importing so the
// module loads without crashing. Per-test code can swap in tighter fakes
// via globalThis.document = … to drive prefillFromQueryString itself.

const minimalDataTag = (jsonText) => ({ textContent: jsonText });
const baselineNodes = {
  "hardware-data": minimalDataTag(JSON.stringify({ hardware: [] })),
  "models-data":   minimalDataTag(JSON.stringify({ models: [] })),
};
globalThis.document = {
  getElementById(id) {
    if (id in baselineNodes) return baselineNodes[id];
    // Form-bootstrap reads many ids; return a permissive stub so
    // bootstrap()'s setText / addEventListener calls don't throw.
    return {
      value: "", textContent: "", innerHTML: "",
      options: [],
      add: () => {},
      addEventListener: () => {},
      dispatchEvent: () => {},
      open: false,
      selectedOptions: [{}],
      classList: { add: () => {}, remove: () => {} },
      style: {},
      offsetWidth: 0,
    };
  },
  querySelectorAll: () => [],
  documentElement: {},
};
globalThis.window = {
  matchMedia: () => ({ matches: false }),
};
// Chart constructor isn't called because the canvas stub returns null
// from createScope's guards; keep this null to be explicit.
globalThis.Chart = undefined;

const { prefillFromQueryString } = await import("../src/ui.mjs");

// Build a fake input/select node. <select> nodes carry .options (an
// array of {value}), and .value assignments to unknown options are
// rejected by setSelect — mirror that behavior here for fidelity.
function fakeInput(initial = "") {
  return { value: initial, tagName: "INPUT" };
}
function fakeSelect(options, initial) {
  return {
    tagName: "SELECT",
    options: options.map((v) => ({ value: v })),
    value: initial ?? options[0],
  };
}

// Install a minimal global.document that maps known ids to nodes. Any
// other id resolves to null — same as a real DOM where the element is
// missing. Keep the registry per-test so cases don't bleed.
function installDocument(nodes) {
  globalThis.document = {
    getElementById(id) {
      return nodes[id] || null;
    },
  };
  return nodes;
}

function freshNodes() {
  return installDocument({
    model:         fakeSelect(["llama-3-8b", "qwen2.5-7b", "llama-3-70b", "__custom"], "llama-3-8b"),
    hw:            fakeSelect(["T4", "A10G", "H100-80GB", "__custom"], "H100-80GB"),
    "weight-prec": fakeSelect(["FP16", "BF16", "INT8", "INT4", "FP8"], "BF16"),
    "kv-prec":     fakeSelect(["FP16", "BF16", "FP8"], "FP16"),
    "act-prec":    fakeSelect(["FP16", "BF16", "FP8"], "BF16"),
    isl:           fakeInput("200"),
    osl:           fakeInput("150"),
    ngpus:         fakeInput("1"),
    tbt:           fakeInput("50"),
  });
}

test("prefillFromQueryString: full round-trip seeds every field", () => {
  const n = freshNodes();
  const applied = prefillFromQueryString(
    "?model=qwen2.5-7b&hw=T4&weight_prec=INT8&kv_prec=FP16&act_prec=FP16" +
      "&isl=200&osl=150&ngpus=1&tbt_ms=50",
  );
  assert.equal(n.model.value, "qwen2.5-7b");
  assert.equal(n.hw.value, "T4");
  assert.equal(n["weight-prec"].value, "INT8");
  assert.equal(n["kv-prec"].value, "FP16");
  assert.equal(n["act-prec"].value, "FP16");
  assert.equal(n.isl.value, "200");
  assert.equal(n.osl.value, "150");
  assert.equal(n.ngpus.value, "1");
  assert.equal(n.tbt.value, "50");
  // Every key in the URL applied (no silent drops).
  assert.deepEqual(Object.keys(applied).sort(), [
    "act_prec", "hw", "isl", "kv_prec", "model", "ngpus", "osl", "tbt_ms", "weight_prec",
  ].sort());
});

test("prefillFromQueryString: unknown <select> value leaves default intact", () => {
  const n = freshNodes();
  // "no-such-model" isn't in the options list — must NOT blank out the
  // default, otherwise the form would render an empty selection and
  // recompute would crash on lookups.
  const applied = prefillFromQueryString("?model=no-such-model&hw=A10G");
  assert.equal(n.model.value, "llama-3-8b");       // default preserved
  assert.equal(applied.model, undefined);           // not applied
  assert.equal(n.hw.value, "A10G");                 // valid value did apply
  assert.equal(applied.hw, "A10G");
});

test("prefillFromQueryString: non-numeric numeric is ignored", () => {
  const n = freshNodes();
  const applied = prefillFromQueryString("?isl=not-a-number&osl=300");
  assert.equal(n.isl.value, "200");                 // default preserved
  assert.equal(applied.isl, undefined);
  assert.equal(n.osl.value, "300");
  assert.equal(applied.osl, 300);
});

test("prefillFromQueryString: empty / missing search → no-op", () => {
  const n = freshNodes();
  assert.deepEqual(prefillFromQueryString(""), {});
  assert.deepEqual(prefillFromQueryString("?"), {});
  // Defaults preserved.
  assert.equal(n.model.value, "llama-3-8b");
  assert.equal(n.hw.value, "H100-80GB");
});

test("prefillFromQueryString: partial query leaves untouched fields alone", () => {
  const n = freshNodes();
  prefillFromQueryString("?tbt_ms=25");
  assert.equal(n.tbt.value, "25");
  // Other fields keep their defaults.
  assert.equal(n.model.value, "llama-3-8b");
  assert.equal(n.isl.value, "200");
});

test("prefillFromQueryString: missing DOM nodes degrade silently", () => {
  // Install an empty registry — every getElementById returns null. The
  // helper must not throw; it just applies nothing.
  installDocument({});
  const applied = prefillFromQueryString("?model=qwen2.5-7b&isl=300");
  assert.deepEqual(applied, {});
});
