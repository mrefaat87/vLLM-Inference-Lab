// Minimal Playwright config — single chromium project, no web server (we open
// the file:// URL of the built HTML directly).
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  testMatch: /.*\.e2e\.spec\.mjs$/,
  // Slightly generous default timeout: font loading + Chart.js bootstrap can
  // take ~2s on a cold cache.
  timeout: 30_000,
  expect: { timeout: 5_000 },
  reporter: "list",
  use: {
    ...devices["Desktop Chrome"],
    // file:// loads need this; some defaults disable local file access.
    launchOptions: { args: ["--allow-file-access-from-files"] },
  },
});
