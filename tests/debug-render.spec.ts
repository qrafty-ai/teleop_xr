import { test, expect } from '@playwright/test';

test.use({
  ignoreHTTPSErrors: true,
  headless: false,
  launchOptions: {
    args: [
      '--use-gl=egl',
      '--enable-webgl',
      '--ignore-gpu-blocklist',
    ],
  },
});

test('verify headless WebGL rendering', async ({ page }) => {
  const consoleLogs: Array<{ type: string; text: string }> = [];

  page.on('console', (msg) => {
    consoleLogs.push({ type: msg.type(), text: msg.text() });
  });

  page.on('pageerror', (error) => {
    console.error('Page error:', error.message);
  });

  await page.goto('https://localhost:4443', { waitUntil: 'networkidle' });

  await page.waitForTimeout(5000);

  const screenshot = await page.screenshot({
    fullPage: true,
    path: 'debug-screenshot.png'
  });

  console.log('\n=== Console Logs ===');
  consoleLogs.forEach(log => {
    console.log(`[${log.type}] ${log.text}`);
  });

  const canvasExists = await page.locator('canvas').count();
  console.log(`\nCanvas elements found: ${canvasExists}`);

  const debugOverlay = await page.locator('div:has-text("DEBUG VISIBLE")').count();
  console.log(`Debug overlay visible: ${debugOverlay > 0}`);
});
