import { test, expect, devices } from '@playwright/test';

test.use({
  ...devices['Pixel 5'],
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

test('Quest-like screenshot test', async ({ page }) => {
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
    path: 'quest-screenshot.png'
  });

  console.log('\n=== Console Logs ===');
  consoleLogs.forEach(log => {
    console.log(`[${log.type}] ${log.text}`);
  });

  const canvasExists = await page.locator('canvas').count();
  console.log(`\nCanvas elements found: ${canvasExists}`);

  const debugOverlay = await page.locator('div:has-text("DEBUG VISIBLE")').count();
  console.log(`Debug overlay visible: ${debugOverlay > 0}`);

  const aScene = await page.locator('a-scene').count();
  console.log(`A-Frame scene elements: ${aScene}`);
});
