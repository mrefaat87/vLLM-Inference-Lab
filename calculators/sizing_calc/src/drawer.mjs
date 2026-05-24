// Formulas drawer — slide-out reference panel that renders every formula
// the calculator uses with KaTeX, plus a description / derivation / citation
// expander per row. Content lives in src/data/formulas.json (inlined by the
// build into <script id="formulas-data">). KaTeX comes from the CDN script
// the HTML head includes; if it hasn't loaded yet we render the raw expr_tex
// in a <code> block and upgrade in place once KaTeX arrives.

// Category labels — kept in render order so the drawer reads top-down from
// "things you might know" to "things you came to look up."
const CATEGORY_ORDER = [
  ["constants",     "CONSTANTS"],
  ["memory",        "MEMORY"],
  ["regime",        "ROOFLINE / REGIME"],
  ["latency",       "LATENCY MODEL"],
  ["capacity",      "CAPACITY CEILINGS"],
  ["parallelism",   "PARALLELISM"],
  ["engine_knobs",  "ENGINE KNOBS"],
  ["disagg",        "DISAGGREGATION"],
  ["cost",          "COST"],
];

// Escape user-supplied strings before injecting into innerHTML. Same helper
// pattern used by ui.mjs escapeHtml — kept local so this module is
// self-contained (so the build can decide where to place it in bundle order).
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// Render one formula expression. Tries KaTeX if it's loaded; falls back to a
// <code> block tagged for upgrade-in-place once the CDN script arrives.
function renderExpr(expr) {
  if (typeof window !== "undefined" && window.katex && typeof window.katex.renderToString === "function") {
    try {
      return window.katex.renderToString(expr, { throwOnError: false, displayMode: true });
    } catch (_) {
      // fall through to the plaintext fallback if KaTeX itself throws
    }
  }
  // Fallback path. The data-katex-pending attribute is what the upgrade pass
  // looks for once window.katex becomes available.
  return `<code class="formula-expr-fallback" data-katex-pending="1">${esc(expr)}</code>`;
}

// One row per formula. Math is visible always; everything else is collapsed
// behind a <details> expander so the user sees a clean list by default.
function renderRow(f) {
  const where = f.where.map(([name, meaning]) =>
    `<dt><code>${esc(name)}</code></dt><dd>${esc(meaning)}</dd>`).join("");
  const citations = f.citation.map((c) => {
    const href = c.anchor ?? c.url ?? "#";
    const target = c.url ? ` target="_blank" rel="noopener"` : "";
    return `<a class="formula-cite" href="${esc(href)}"${target}>${esc(c.label)}</a>`;
  }).join("");
  return `
    <article class="formula-row" data-formula-id="${esc(f.id)}">
      <header class="formula-head">
        <span class="formula-name">${esc(f.name)}</span>
        <span class="formula-path">${esc(f.calc_path)}</span>
      </header>
      <div class="formula-expr" data-expr="${esc(f.expr_tex)}">${renderExpr(f.expr_tex)}</div>
      <details class="formula-details">
        <summary>WHY / WHERE / SOURCE</summary>
        <div class="formula-body">
          <p class="formula-description">${esc(f.description)}</p>
          ${where ? `<dl class="formula-where">${where}</dl>` : ""}
          <p class="formula-derivation">${esc(f.derivation)}</p>
          <div class="formula-citations">${citations}</div>
        </div>
      </details>
    </article>
  `;
}

// Group rows by category so the drawer has scannable headings.
function renderAll(formulas) {
  const byCat = new Map();
  for (const f of formulas) {
    if (!byCat.has(f.category)) byCat.set(f.category, []);
    byCat.get(f.category).push(f);
  }
  const sections = CATEGORY_ORDER
    .filter(([key]) => byCat.has(key))
    .map(([key, label]) => `
      <section class="formula-cat" data-cat="${esc(key)}">
        <h3 class="formula-cat-head">${esc(label)}</h3>
        ${byCat.get(key).map(renderRow).join("")}
      </section>
    `).join("");
  return sections;
}

// If KaTeX loads AFTER first paint (CDN slow / cold cache), walk every
// pending fallback and upgrade it in place. Bound to the script's onload via
// the build script; also runs once at mount time in case KaTeX raced ahead.
function upgradePendingKatex() {
  if (typeof window === "undefined" || !window.katex) return;
  const pending = document.querySelectorAll(".formula-expr [data-katex-pending]");
  for (const node of pending) {
    const expr = node.parentElement?.dataset.expr;
    if (!expr) continue;
    try {
      node.parentElement.innerHTML = window.katex.renderToString(expr, {
        throwOnError: false, displayMode: true,
      });
    } catch (_) {
      // Leave the fallback in place; KaTeX renderToString({throwOnError:false})
      // already emits a red error span so the user sees what failed.
    }
  }
}

// Focus-trap helper. While the drawer is open Tab must cycle within it so
// keyboard users don't escape into the background page (WAI-ARIA dialog
// pattern). Returns the keydown handler so the caller can remove it on close.
function trapFocus(drawer, closeBtn) {
  return (e) => {
    if (e.key !== "Tab") return;
    const focusables = drawer.querySelectorAll(
      'a[href], button, summary, [tabindex]:not([tabindex="-1"])');
    if (focusables.length === 0) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault(); last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault(); first.focus();
    }
  };
}

// Entry point — wires the trigger button, the drawer, the backdrop, the
// `f` shortcut, Esc-to-close, focus management, and KaTeX upgrade. Idempotent
// (calling twice is a no-op): guards against double-init by checking for the
// `data-formulas-mounted` flag on the trigger.
export function mountFormulasDrawer() {
  const trigger = document.getElementById("formulas-trigger");
  const drawer = document.getElementById("formulas-drawer");
  const backdrop = document.getElementById("formulas-backdrop");
  const dataTag = document.getElementById("formulas-data");
  if (!trigger || !drawer || !backdrop || !dataTag) return;
  if (trigger.dataset.formulasMounted === "1") return;
  trigger.dataset.formulasMounted = "1";

  const formulas = JSON.parse(dataTag.textContent).formulas;
  // Render once; the data is static so we don't recompute on every open.
  const body = drawer.querySelector(".formulas-body");
  body.innerHTML = renderAll(formulas);
  // If KaTeX is already loaded (cache hit), this is a no-op; otherwise the
  // CDN script's onload (wired by the build) will call us again.
  upgradePendingKatex();
  window.addEventListener("katex-loaded", upgradePendingKatex);

  const closeBtn = drawer.querySelector(".formulas-close");
  let lastFocused = null;
  let trapHandler = null;

  const open = () => {
    if (!drawer.hasAttribute("hidden")) return;
    lastFocused = document.activeElement;
    drawer.removeAttribute("hidden");
    backdrop.removeAttribute("hidden");
    trigger.setAttribute("aria-expanded", "true");
    // Focus the close button so Esc/Enter work immediately + screen readers
    // announce the drawer landmark.
    closeBtn.focus();
    trapHandler = trapFocus(drawer, closeBtn);
    document.addEventListener("keydown", trapHandler);
  };

  const close = () => {
    if (drawer.hasAttribute("hidden")) return;
    drawer.setAttribute("hidden", "");
    backdrop.setAttribute("hidden", "");
    trigger.setAttribute("aria-expanded", "false");
    if (trapHandler) {
      document.removeEventListener("keydown", trapHandler);
      trapHandler = null;
    }
    // Return focus to the trigger so keyboard users don't lose their place.
    if (lastFocused && typeof lastFocused.focus === "function") {
      lastFocused.focus();
    } else {
      trigger.focus();
    }
  };

  trigger.addEventListener("click", open);
  closeBtn.addEventListener("click", close);
  backdrop.addEventListener("click", close);
  // Esc-to-close — works regardless of focus location (focus trap keeps focus
  // inside the drawer anyway).
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !drawer.hasAttribute("hidden")) close();
  });
  // `f` shortcut — only when no input/textarea/select has focus, so it
  // doesn't fight with the form controls.
  document.addEventListener("keydown", (e) => {
    if (e.key !== "f" && e.key !== "F") return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const t = document.activeElement;
    const tag = t?.tagName?.toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select" || t?.isContentEditable) return;
    if (drawer.hasAttribute("hidden")) { e.preventDefault(); open(); }
  });
}
