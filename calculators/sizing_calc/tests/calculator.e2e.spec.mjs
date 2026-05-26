// End-to-end smoke for the built calculator HTML. Confirms the user-visible
// contract: form populates, defaults compute sane numbers, FP8 doubles B_crit,
// warnings render. The unit tests cover the math; this covers the wiring.

import { test, expect } from "@playwright/test";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";
import { execSync } from "node:child_process";

const here = dirname(fileURLToPath(import.meta.url));
const REPO = join(here, "..", "..", "..");
const HTML = join(REPO, "calculators", "sizing_calculator.html");

test.beforeAll(() => {
  // Always rebuild — keeps the test honest if a contributor forgot to run the
  // build before pushing.
  execSync(`python3 ${join(REPO, "tools/build_sizing_calculator_html.py")}`,
           { stdio: "inherit" });
});

test("loads, populates presets, renders default metrics + chart", async ({ page }) => {
  await page.goto(pathToFileURL(HTML).href);
  // Wait for chart to mount — ui.mjs sets window.__sizingChart on createScope.
  await page.waitForFunction(() => window.__sizingChart != null, null, { timeout: 10_000 });

  // Default selection should be H100-80GB + Llama-3-8B.
  await expect(page.locator("#hw")).toHaveValue("H100-80GB");
  await expect(page.locator("#model")).toHaveValue("llama-3-8b");

  // B_crit on H100 BF16 default acts (FP16 in our UI) ≈ 295 — see goldens.
  const bcrit = parseInt(await page.locator("#m-bcrit").textContent(), 10);
  expect(bcrit).toBeGreaterThan(280);
  expect(bcrit).toBeLessThan(320);

  // Latency scope datasets: glow + bright + slo = 3 (throughput and cost
  // moved to their own scopes — they used to be overlays on this canvas).
  const dsCount = await page.evaluate(() => window.__sizingChart.data.datasets.length);
  expect(dsCount).toBe(3);

  // Caliper markers wired with 3 entries.
  const markerCount = await page.evaluate(
    () => window.__sizingChart.options.plugins.calipers.markers.length,
  );
  expect(markerCount).toBe(3);
});

test("FP8 weights+acts on H100 roughly doubles B_crit (590 vs 295)", async ({ page }) => {
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingChart != null);

  const before = parseInt(await page.locator("#m-bcrit").textContent(), 10);
  await page.locator("#weight-prec").selectOption("FP8");
  await page.locator("#act-prec").selectOption("FP8");
  // recompute is synchronous; wait one tick.
  await page.waitForTimeout(50);
  const after = parseInt(await page.locator("#m-bcrit").textContent(), 10);

  expect(after).toBeGreaterThan(before * 1.8);
  expect(after).toBeLessThan(before * 2.2);
});

test("Llama-3-70B on a single T4 surfaces a weights-overflow diagnostic", async ({ page }) => {
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingChart != null);

  await page.locator("#hw").selectOption("T4");
  await page.locator("#model").selectOption("llama-3-70b");
  await page.waitForTimeout(50);

  const diag = await page.locator("#diagnostics").textContent();
  expect(diag.toLowerCase()).toContain("overflow");
});

test("DeepSeek-V3 propagates MoE sparsity to b_crit + total-weights UI tiles", async ({ page }) => {
  // Wires together the two halves of the MoE-correctness refactor:
  //   - b_crit must pick up the sparsity factor (≥ 4000 vs dense ≈ 295)
  //   - weights_gb must use params_b_total (≥ 1000 GB at BF16 vs old 74 GB)
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingChart != null);

  await page.locator("#model").selectOption("deepseek-v3");
  // Force BF16 so the weights tile assertion is anchored to a known precision.
  await page.locator("#weight-prec").selectOption("BF16");
  await page.waitForTimeout(50);

  const bcrit = parseInt(await page.locator("#m-bcrit").textContent().then(t => t.replace(/[^\d]/g, "")), 10);
  expect(bcrit).toBeGreaterThanOrEqual(4000);

  // Weights tile reads e.g. "1342 GB" — strip non-digits before parsing.
  const weightsText = await page.locator("#m-weights").textContent();
  const weightsGb = parseInt(weightsText.replace(/[^\d]/g, ""), 10);
  expect(weightsGb).toBeGreaterThanOrEqual(1000);
});

test("three scopes mount and share caliper marker positions", async ({ page }) => {
  // Plan Layer 6 #10: all three scope panels mount, expose handles on
  // window.__sizingScopes, and carry identical marker B values (they all
  // read from the same compute() output, so any drift is a real bug).
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(
    () => window.__sizingScopes
       && window.__sizingScopes.latency
       && window.__sizingScopes.throughput
       && window.__sizingScopes.cost,
    null, { timeout: 10_000 },
  );

  const markersByScope = await page.evaluate(() => {
    const out = {};
    for (const [name, s] of Object.entries(window.__sizingScopes)) {
      out[name] = s.chart.options.plugins.calipers.markers.map((m) => m.B);
    }
    return out;
  });
  // Same 3 markers per scope, same B values across all three.
  expect(markersByScope.latency).toHaveLength(3);
  expect(markersByScope.throughput).toEqual(markersByScope.latency);
  expect(markersByScope.cost).toEqual(markersByScope.latency);

  // Each scope has its own canvas in the DOM.
  for (const cid of ["scope-latency-canvas", "scope-throughput-canvas", "scope-cost-canvas"]) {
    await expect(page.locator(`#${cid}`)).toBeVisible();
  }
});

test("m-cost tile updates linearly when the price-per-hour override doubles", async ({ page }) => {
  // Plan Layer 6 #11: locks the end-to-end wiring price → readForm → compute
  // → paintMetrics. Doubling the price must double the displayed $/Mtok.
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingScopes?.cost != null);

  // Read raw (unrounded) cost_per_mtok from the chart's underlying data so
  // the 2-decimal tile rounding doesn't dominate the linearity check. The
  // cost scope's bright dataset carries cost_per_mtok at each B; we sample
  // the same index (anchored to recommended batch ≈ midpoint) before/after.
  const readRawCost = async () => page.evaluate(() => {
    const ds = window.__sizingScopes.cost.chart.data.datasets[1].data;
    // First finite positive sample — robust to leading nulls/Infs.
    return ds.find((v) => Number.isFinite(v) && v > 0);
  });

  const base = await readRawCost();
  expect(base).toBeGreaterThan(0);
  // Sanity: tile shows a non-dash value too.
  const tileBase = await page.locator("#m-cost").textContent();
  expect(tileBase).toMatch(/\$\d/);

  // Double the price and confirm raw cost doubles exactly (within float drift).
  const priceInput = page.locator("#price-per-hour");
  const currentPrice = parseFloat(await priceInput.inputValue());
  await priceInput.fill(String(currentPrice * 2));
  await page.waitForTimeout(80);

  const doubled = await readRawCost();
  expect(doubled).toBeGreaterThan(base * 1.99);
  expect(doubled).toBeLessThan(base * 2.01);
});

test("hardware row swap resets price-per-hour to the new row's default", async ({ page }) => {
  // Plan Layer 6 #12: within-row edits keep the user's override, but a row
  // swap should reset to the new hardware's default per-GPU price.
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingScopes?.cost != null);

  // Start on H100. The default is whatever hardware.json carries — we just
  // assert it's not zero / not the T4 price.
  await expect(page.locator("#hw")).toHaveValue("H100-80GB");
  const h100Price = parseFloat(await page.locator("#price-per-hour").inputValue());
  expect(h100Price).toBeGreaterThan(1); // H100 is well above $1/hr

  // Swap to T4 — price input should snap to the T4 default ($0.526).
  await page.locator("#hw").selectOption("T4");
  await page.waitForTimeout(80);
  const t4Price = parseFloat(await page.locator("#price-per-hour").inputValue());
  expect(t4Price).toBeCloseTo(0.526, 3);
});

test("copy button copies the snippet to clipboard", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingChart != null);

  await page.locator("#copy-btn").click();
  await expect(page.locator("#copy-btn")).toHaveText("[ COPIED ]");
  const clipboard = await page.evaluate(() => navigator.clipboard.readText());
  expect(clipboard).toContain("vllm serve");
  expect(clipboard).toContain("--tensor-parallel-size");
});

// [ COPY EXP RUN ] must yield a paste-and-go command:
// - verb is `exp launch` (the orchestrator that auto-rebuilds + opens the browser)
// - `--rate` is a positive number, not the `<rps>` placeholder
// - all join keys land in the command so the lab result re-anchors to this prediction
test("copy exp button emits a launch command with numeric rate", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingChart != null);

  // Pick a preset that maps to a lab workload so the snippet's exp block exists.
  await page.locator("#workload-preset").selectOption("chatbot_turn1");
  await page.waitForTimeout(80);

  await page.locator("#copy-exp-btn").click();
  await expect(page.locator("#copy-exp-btn")).toHaveText("[ COPIED ]");
  const clipboard = await page.evaluate(() => navigator.clipboard.readText());

  expect(clipboard.startsWith("exp launch ")).toBe(true);
  expect(clipboard).not.toContain("<rps>");
  // Rate appears as a number on the --rate line.
  const rateMatch = clipboard.match(/--rate\s+([0-9.]+)/);
  expect(rateMatch).not.toBeNull();
  const rate = parseFloat(rateMatch[1]);
  expect(rate).toBeGreaterThan(0);
  expect(Number.isFinite(rate)).toBe(true);
  // Join keys present so the lab result lines up with this prediction.
  expect(clipboard).toContain("--model-ref");
  expect(clipboard).toContain("--hw-ref");
  expect(clipboard).toContain("--tbt-target-ms");
});

// The lab supports vllm/sglang/trtllm; the calc must emit the selected
// engine in the exp launch command (not just vllm).
test("engine selector flows into the exp launch command", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingChart != null);
  await page.locator("#workload-preset").selectOption("chatbot_turn1");
  await page.waitForTimeout(80);

  for (const engine of ["vllm", "sglang", "trtllm"]) {
    await page.locator("#lab-engine").selectOption(engine);
    await page.waitForTimeout(80);
    await page.locator("#copy-exp-btn").click();
    const clipboard = await page.evaluate(() => navigator.clipboard.readText());
    expect(clipboard).toMatch(new RegExp(`exp launch --engine ${engine}\\b`));
  }
});
