import { test, expect } from '@playwright/test';

test.use({
  ignoreHTTPSErrors: true,
});

test('verify dashboard and settings panels', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 1024 });

  console.log('Navigating to https://localhost:4443...');
  await page.goto('https://localhost:4443', { waitUntil: 'networkidle' });

  console.log('Waiting for #dashboard...');
  const dashboard = page.locator('#dashboard');
  await dashboard.waitFor({ state: 'attached', timeout: 15000 });
  console.log('#dashboard is attached.');

  const buttons = [
    '#camera-settings-btn',
    '#robot-settings-btn',
    '#general-settings-btn'
  ];

  console.log('--- Visibility Status ---');
  for (const btn of buttons) {
    const locator = page.locator(btn);
    await locator.waitFor({ state: 'attached', timeout: 10000 }).catch(() => {});
    const count = await locator.count();
    if (count > 0) {
        const visibleAttr = await locator.getAttribute('visible');
        console.log(`${btn}: Attached. 'visible' attribute: ${visibleAttr}`);
    } else {
        console.log(`${btn}: Not Found in DOM`);
    }
  }

  await page.screenshot({
    fullPage: true,
    path: 'final_ui_complete_verification.png'
  });
  console.log('Screenshot saved to final_ui_complete_verification.png');

  const panelMappings = [
    { btn: '#camera-settings-btn', panel: '#camera-settings-panel' },
    { btn: '#robot-settings-btn', panel: '#robot-settings-panel' },
    { btn: '#general-settings-btn', panel: '#general-settings-panel' }
  ];

  for (const mapping of panelMappings) {
    console.log(`Verifying ${mapping.btn} -> ${mapping.panel}...`);
    const btn = page.locator(mapping.btn);
    if (await btn.count() > 0) {
        const panel = page.locator(mapping.panel);
        const initialVisible = await panel.getAttribute('visible');
        console.log(`Initial ${mapping.panel} 'visible' attribute: ${initialVisible}`);

        console.log(`Clicking ${mapping.btn}...`);
        await btn.evaluate(node => {
          node.dispatchEvent(new CustomEvent('click'));
        });

        await page.waitForTimeout(1000);

        const finalVisible = await panel.getAttribute('visible');
        console.log(`Final ${mapping.panel} 'visible' attribute: ${finalVisible}`);
    } else {
        console.log(`${mapping.btn} not found, skipping click.`);
    }
  }
});
