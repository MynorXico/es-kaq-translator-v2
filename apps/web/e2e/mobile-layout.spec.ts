import { expect, test } from "@playwright/test";

// A common small-phone width (iPhone SE/mini class) -- if anything overflows
// here, it'll overflow everywhere.
const MOBILE_VIEWPORT = { width: 320, height: 640 };
const DESKTOP_VIEWPORT = { width: 900, height: 800 };

test.describe("mobile layout (#41)", () => {
  test("the viewport meta tag is present and correct", async ({ page }) => {
    await page.goto("/");
    const viewportContent = await page
      .locator('meta[name="viewport"]')
      .getAttribute("content");
    expect(viewportContent).toBe("width=device-width, initial-scale=1.0");
  });

  test("no horizontal scrolling at a narrow mobile viewport", async ({ page }) => {
    await page.setViewportSize(MOBILE_VIEWPORT);
    await page.goto("/");

    const { scrollWidth, innerWidth } = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth,
    }));

    expect(scrollWidth).toBeLessThanOrEqual(innerWidth);
  });

  test("input and output font-size is at least 16px (avoids iOS Safari auto-zoom)", async ({
    page,
  }) => {
    await page.setViewportSize(MOBILE_VIEWPORT);
    await page.route("**/v1/translate", (route) =>
      route.fulfill({ json: { translation: "Utz" } }),
    );
    await page.goto("/");

    const input = page.getByLabel(/^Texto en /);
    const inputFontSize = await input.evaluate((el) =>
      parseFloat(getComputedStyle(el).fontSize),
    );
    expect(inputFontSize).toBeGreaterThanOrEqual(16);

    await input.fill("Hola");
    await page.getByRole("button", { name: "Traducir" }).click();
    const output = page.getByLabel(/^Traducción en /);
    await expect(output).toHaveValue("Utz");
    const outputFontSize = await output.evaluate((el) =>
      parseFloat(getComputedStyle(el).fontSize),
    );
    expect(outputFontSize).toBeGreaterThanOrEqual(16);
  });

  test("every visible tap target is at least 44x44px", async ({ page }) => {
    await page.route("**/v1/translate", (route) =>
      route.fulfill({ json: { translation: "Utz" } }),
    );
    await page.setViewportSize(MOBILE_VIEWPORT);
    await page.goto("/");
    await page.getByLabel(/^Texto en /).fill("Hola");
    await page.getByRole("button", { name: "Traducir" }).click();
    await expect(page.getByRole("button", { name: "Copiar" })).toBeVisible();

    const targets = [
      page.getByRole("button", { name: "Acerca de" }),
      page.getByRole("button", { name: "Cambiar dirección" }),
      page.getByRole("button", { name: "Borrar el texto de entrada" }),
      page.getByRole("button", { name: "Traducir" }),
      page.getByRole("button", { name: "Copiar" }),
    ];

    for (const target of targets) {
      const box = await target.boundingBox();
      expect(box, `${await target.textContent()} should be visible`).not.toBeNull();
      expect(box!.width).toBeGreaterThanOrEqual(44);
      expect(box!.height).toBeGreaterThanOrEqual(44);
    }
  });

  test("Traducir is full-width (matches the surrounding field-card) and at least 48px tall, at mobile and wider viewports", async ({
    page,
  }) => {
    await page.goto("/");
    // Since #111, the input textarea sits inside a .field-card (its own
    // border/padding), so it's no longer the same width as the full-bleed
    // cta-button -- compare against the field-card itself, which (like the
    // button) spans the full width of .app-body.
    const inputFieldCard = page.locator(".field-card").first();
    const translateButton = page.getByRole("button", { name: "Traducir" });

    for (const viewport of [MOBILE_VIEWPORT, DESKTOP_VIEWPORT]) {
      await page.setViewportSize(viewport);
      const buttonBox = await translateButton.boundingBox();
      const fieldCardBox = await inputFieldCard.boundingBox();

      expect(buttonBox!.height).toBeGreaterThanOrEqual(48);
      expect(Math.abs(buttonBox!.width - fieldCardBox!.width)).toBeLessThan(1);
    }
  });

  test("the translator content column caps at 480px, per #52's mobile spec", async ({ page }) => {
    await page.goto("/");
    const maxWidth = await page.locator("main").evaluate((el) => getComputedStyle(el).maxWidth);
    expect(maxWidth).toBe("480px");
  });

  test("the output textarea's min-height increases at the 600px breakpoint", async ({ page }) => {
    await page.goto("/");
    const output = page.getByLabel(/^Traducción en /);

    await page.setViewportSize(MOBILE_VIEWPORT);
    const mobileMinHeight = await output.evaluate((el) => getComputedStyle(el).minHeight);
    expect(mobileMinHeight).toBe("120px");

    await page.setViewportSize(DESKTOP_VIEWPORT);
    const desktopMinHeight = await output.evaluate((el) => getComputedStyle(el).minHeight);
    expect(desktopMinHeight).toBe("140px");
  });
});
