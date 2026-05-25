// End-to-end coverage for the formulas drawer. Confirms the user-visible
// contract: closed by default, trigger / shortcut / Esc / backdrop wired,
// every formula renders through KaTeX, the per-row expander reveals the
// description + derivation + citations, focus management is sane.
//
// All assertions run against the BUILT single HTML file — the same artifact
// users get — so a build regression that breaks the drawer fails here.

import { test, expect } from "@playwright/test";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";
import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";

const here = dirname(fileURLToPath(import.meta.url));
const REPO = join(here, "..", "..", "..");
const HTML = join(REPO, "calculators", "sizing_calculator.html");
const FORMULAS_PATH = join(here, "..", "src", "data", "formulas.json");

// Source-of-truth count for the data-driven assertions below.
const FORMULAS_COUNT = JSON.parse(readFileSync(FORMULAS_PATH, "utf8")).formulas.length;

test.beforeAll(() => {
  execSync(`python3 ${join(REPO, "tools/build_sizing_calculator_html.py")}`,
           { stdio: "inherit" });
});

// Helper: wait for KaTeX to finish loading (CDN script is deferred). The
// drawer fires "katex-loaded" on the window once the script's onload runs.
async function waitForKatex(page) {
  await page.waitForFunction(() => window.katex != null, null, { timeout: 15_000 });
}

test("drawer is closed by default", async ({ page }) => {
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingChart != null);

  // Drawer + backdrop start with the `hidden` attribute set, trigger reports
  // collapsed via aria-expanded=false.
  await expect(page.locator("#formulas-drawer")).toHaveAttribute("hidden", "");
  await expect(page.locator("#formulas-backdrop")).toHaveAttribute("hidden", "");
  await expect(page.locator("#formulas-trigger")).toHaveAttribute("aria-expanded", "false");
});

test("clicking the trigger opens the drawer; Esc closes it", async ({ page }) => {
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingChart != null);

  await page.locator("#formulas-trigger").click();
  // After open: hidden attribute removed on both drawer and backdrop, trigger
  // aria-expanded=true, focus inside the drawer.
  await expect(page.locator("#formulas-drawer")).not.toHaveAttribute("hidden", "");
  await expect(page.locator("#formulas-backdrop")).not.toHaveAttribute("hidden", "");
  await expect(page.locator("#formulas-trigger")).toHaveAttribute("aria-expanded", "true");

  // Esc closes and returns focus to the trigger.
  await page.keyboard.press("Escape");
  await expect(page.locator("#formulas-drawer")).toHaveAttribute("hidden", "");
  await expect(page.locator("#formulas-trigger")).toBeFocused();
});

test("`f` shortcut opens the drawer when no input is focused", async ({ page }) => {
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingChart != null);

  // Make sure no form field is focused.
  await page.locator("body").click();
  await page.keyboard.press("f");
  await expect(page.locator("#formulas-drawer")).not.toHaveAttribute("hidden", "");
});

test("`f` shortcut is suppressed when a form input is focused", async ({ page }) => {
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingChart != null);

  // Focus the ISL number input — typing "f" there must NOT open the drawer.
  await page.locator("#isl").focus();
  await page.keyboard.press("f");
  await expect(page.locator("#formulas-drawer")).toHaveAttribute("hidden", "");
});

test("backdrop click closes the drawer", async ({ page }) => {
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingChart != null);

  await page.locator("#formulas-trigger").click();
  await expect(page.locator("#formulas-drawer")).not.toHaveAttribute("hidden", "");

  await page.locator("#formulas-backdrop").click();
  await expect(page.locator("#formulas-drawer")).toHaveAttribute("hidden", "");
});

test("every formula renders through KaTeX (no fallback nodes left)", async ({ page }) => {
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingChart != null);
  await waitForKatex(page);

  await page.locator("#formulas-trigger").click();

  // One .formula-row per entry — count must match the source JSON.
  const rowCount = await page.locator(".formula-row").count();
  expect(rowCount).toBe(FORMULAS_COUNT);

  // Every .formula-expr must contain at least one .katex node (KaTeX's wrapper
  // class). Pending fallbacks should have been upgraded by now.
  const exprsWithKatex = await page.locator(".formula-expr .katex").count();
  expect(exprsWithKatex).toBe(FORMULAS_COUNT);

  // No leftover pending fallback markers.
  const stillPending = await page.locator(".formula-expr [data-katex-pending]").count();
  expect(stillPending).toBe(0);
});

test("no KaTeX parse errors are logged to the console", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (msg) => { if (msg.type() === "error") errors.push(msg.text()); });

  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingChart != null);
  await waitForKatex(page);
  await page.locator("#formulas-trigger").click();
  await page.waitForTimeout(200);

  const katexErrors = errors.filter((e) => /katex|ParseError/i.test(e));
  expect(katexErrors).toEqual([]);
});

test("per-row expander reveals description / derivation / citations", async ({ page }) => {
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingChart != null);
  await waitForKatex(page);
  await page.locator("#formulas-trigger").click();

  // Pick the b_crit row — every drawer build should have it.
  const row = page.locator('[data-formula-id="b_crit"]');
  await expect(row).toBeVisible();

  // <details> starts closed; body is hidden.
  const details = row.locator(".formula-details");
  await expect(details).not.toHaveAttribute("open", "");

  await row.locator(".formula-details > summary").click();
  await expect(details).toHaveAttribute("open", "");
  // After open, description, where-glossary, and citations are reachable.
  await expect(row.locator(".formula-description")).toBeVisible();
  await expect(row.locator(".formula-where")).toBeVisible();
  await expect(row.locator(".formula-cite").first()).toBeVisible();
});

test("citation links have valid href (anchor or http URL)", async ({ page }) => {
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingChart != null);
  await page.locator("#formulas-trigger").click();
  // Open every expander so every citation node is in layout.
  const summaries = page.locator(".formula-details > summary");
  const n = await summaries.count();
  for (let i = 0; i < n; i++) await summaries.nth(i).click();

  const hrefs = await page.locator(".formula-cite").evaluateAll(
    (els) => els.map((a) => a.getAttribute("href")));
  expect(hrefs.length).toBeGreaterThan(0);
  for (const h of hrefs) {
    expect(h).toBeTruthy();
    // Must be either an internal reference doc anchor or an external https URL.
    const ok = h.startsWith("https://") || h.startsWith("../reference/");
    expect(ok, `unexpected href shape: ${h}`).toBeTruthy();
  }
});

test("mobile viewport: drawer spans full width and close button is reachable", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto(pathToFileURL(HTML).href);
  await page.waitForFunction(() => window.__sizingChart != null);

  await page.locator("#formulas-trigger").click();
  // Drawer width: min(520, 100vw) → 375 on iPhone-sized viewport.
  const box = await page.locator("#formulas-drawer").boundingBox();
  expect(box.width).toBe(375);

  // Close button reachable + closes the drawer.
  await page.locator(".formulas-close").click();
  await expect(page.locator("#formulas-drawer")).toHaveAttribute("hidden", "");
});
