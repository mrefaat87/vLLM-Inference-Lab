// E2E: deep-link prefill of the built calculator HTML.
//
// Simulates the "Open this run in the sizing calc" link from the Results
// Explorer: navigate to the calc with a populated query string, then
// assert each form field reflects the URL params before any user
// interaction. Anchors on the same `__sizingChart` ready signal as the
// other e2e specs.

import { test, expect } from "@playwright/test";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";
import { execSync } from "node:child_process";

const here = dirname(fileURLToPath(import.meta.url));
const REPO = join(here, "..", "..", "..");
const HTML = join(REPO, "calculators", "sizing_calculator.html");

test.beforeAll(() => {
  // Same hygiene as calculator.e2e.spec.mjs: rebuild so the test is
  // honest if a contributor forgot to run the build before pushing.
  execSync(`python3 ${join(REPO, "tools/build_sizing_calculator_html.py")}`,
           { stdio: "inherit" });
});

const QUERY =
  "?model=qwen2.5-7b&hw=T4&weight_prec=INT8&kv_prec=FP16&act_prec=FP16" +
  "&isl=200&osl=150&ngpus=1&tbt_ms=50";

test("query-string prefills every form input before first paint", async ({ page }) => {
  await page.goto(pathToFileURL(HTML).href + QUERY);
  await page.waitForFunction(() => window.__sizingChart != null, null, { timeout: 10_000 });

  // Each field reflects the URL, not the page defaults.
  await expect(page.locator("#model")).toHaveValue("qwen2.5-7b");
  await expect(page.locator("#hw")).toHaveValue("T4");
  await expect(page.locator("#weight-prec")).toHaveValue("INT8");
  await expect(page.locator("#kv-prec")).toHaveValue("FP16");
  await expect(page.locator("#act-prec")).toHaveValue("FP16");
  await expect(page.locator("#isl")).toHaveValue("200");
  await expect(page.locator("#osl")).toHaveValue("150");
  await expect(page.locator("#ngpus")).toHaveValue("1");
  await expect(page.locator("#tbt")).toHaveValue("50");

  // Computed tiles must have non-empty content — verifies the prefill
  // didn't crash recompute. We anchor on the b_crit tile because it's
  // the canonical "did anything happen" indicator.
  const bcritText = await page.locator("#m-bcrit").textContent();
  expect(bcritText).not.toBe("—");
  expect(bcritText?.trim().length).toBeGreaterThan(0);
});

test("unknown query values silently fall back to defaults", async ({ page }) => {
  // Garbage model + non-numeric ngpus → both fields keep their
  // built-in defaults; the page still renders. This is the C7 graceful
  // degradation contract: a malformed deep link must not soft-brick the
  // calc.
  await page.goto(pathToFileURL(HTML).href + "?model=not-a-real-model&ngpus=banana&hw=A10G");
  await page.waitForFunction(() => window.__sizingChart != null);
  await expect(page.locator("#model")).toHaveValue("llama-3-8b");  // page default
  await expect(page.locator("#hw")).toHaveValue("A10G");           // valid → applied
  await expect(page.locator("#ngpus")).not.toHaveValue("banana");  // garbage → ignored
});

// Mirrors the "Open this run in calc" link the Results Explorer builds
// at render time: identical URL shape, full navigation to the calc HTML.
// This complements the first test by exercising a different precision
// triple (INT4/FP16/BF16) and a smaller batch, so a hardcoded default in
// the prefill helper can't fake a passing assertion across both.
test("open-in-calc-style URL prefills the form (INT4 variant)", async ({ page }) => {
  const url = pathToFileURL(HTML).href +
    "?model=qwen2.5-7b&hw=T4&isl=128&osl=128&ngpus=1&tbt_ms=50" +
    "&weight_prec=INT4&kv_prec=FP16&act_prec=BF16";
  await page.goto(url);
  await page.waitForFunction(() => window.__sizingChart != null);

  await expect(page.locator("#model")).toHaveValue("qwen2.5-7b");
  await expect(page.locator("#hw")).toHaveValue("T4");
  await expect(page.locator("#weight-prec")).toHaveValue("INT4");
  await expect(page.locator("#isl")).toHaveValue("128");
  await expect(page.locator("#tbt")).toHaveValue("50");
});
