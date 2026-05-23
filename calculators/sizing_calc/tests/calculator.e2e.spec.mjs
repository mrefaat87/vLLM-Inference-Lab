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

  // Chart datasets: glow + bright + tps + slo = 4 datasets.
  const dsCount = await page.evaluate(() => window.__sizingChart.data.datasets.length);
  expect(dsCount).toBe(4);

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
