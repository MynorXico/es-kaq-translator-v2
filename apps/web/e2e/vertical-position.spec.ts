import { expect, test } from "@playwright/test";

// Gated on width AND height (#135) -- a genuine height problem, not a width
// one, confirmed against the real page: a 1366x768 laptop already fits
// .app-shell comfortably (below the height gate, no margin added), while an
// 800x800 window and the maintainer's original 1920x933 screenshot both need
// it (above the gate on both axes).
const BELOW_HEIGHT_GATE = { width: 1440, height: 700 };
const ABOVE_BOTH_GATES = { width: 1920, height: 933 };

test.describe("vertical positioning/balance on tall viewports (#135)", () => {
  test("below the height gate, .app-shell stays flush at the top with no background glow", async ({
    page,
  }) => {
    await page.setViewportSize(BELOW_HEIGHT_GATE);
    await page.goto("/");

    const marginTop = await page
      .locator(".app-shell")
      .evaluate((el) => getComputedStyle(el).marginTop);
    expect(marginTop).toBe("0px");

    const bodyBackground = await page.evaluate(() => getComputedStyle(document.body).backgroundImage);
    expect(bodyBackground).not.toContain("radial-gradient");
  });

  test("above both gates, .app-shell gets a height-scaled top/bottom offset and the page gets a background glow", async ({
    page,
  }) => {
    await page.setViewportSize(ABOVE_BOTH_GATES);
    await page.goto("/");

    const shell = page.locator(".app-shell");
    const marginTop = await shell.evaluate((el) => getComputedStyle(el).marginTop);
    const marginBottom = await shell.evaluate((el) => getComputedStyle(el).marginBottom);
    // clamp(64px, 8vh, 140px) at 933px viewport height -> 8vh = 74.64px,
    // within the clamp's bounds.
    expect(parseFloat(marginTop)).toBeCloseTo(74.64, 0);
    expect(marginTop).toBe(marginBottom);

    const bodyBackground = await page.evaluate(() => getComputedStyle(document.body).backgroundImage);
    expect(bodyBackground).toContain("radial-gradient");
  });
});
