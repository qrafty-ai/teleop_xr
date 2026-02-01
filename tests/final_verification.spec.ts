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

test('Final verification screenshot', async ({ page }) => {
  // Listen to browser console logs
  page.on('console', msg => console.log('BROWSER LOG:', msg.text()));
  page.on('pageerror', err => console.log('BROWSER ERROR:', err));

  await page.goto('https://localhost:4443', { waitUntil: 'networkidle' });

  // Wait for scene to load
  await page.waitForSelector('a-scene', { timeout: 10000 });

  await page.waitForTimeout(3000);

  // Check if teleop-dashboard component is registered
  const componentCheck = await page.evaluate(() => {
    const AFRAME = (window as any).AFRAME;
    if (!AFRAME) return { error: 'AFRAME not found' };
    const hasDashboard = !!AFRAME.components['teleop-dashboard'];
    const hasRobotModel = !!AFRAME.components['robot-model'];
    const hasVideoStream = !!AFRAME.components['video-stream'];
    return { hasDashboard, hasRobotModel, hasVideoStream };
  });
  console.log('Component check:', componentCheck);

  await page.waitForTimeout(1000);

  await page.screenshot({
    fullPage: true,
    path: 'final_verification.png'
  });

  const canvasExists = await page.locator('canvas').count();
  expect(canvasExists).toBeGreaterThan(0);
});
