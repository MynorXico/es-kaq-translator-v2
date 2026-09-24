import { expect, test } from "@playwright/test";

// Below the 800px breakpoint, the layout is unchanged (#52's mobile spec).
const MOBILE_VIEWPORT = { width: 400, height: 800 };
// Between 800px and 1279px, .app-shell widens to 720px -- .app-body must
// actually fill it (this was a confirmed bug: .app-body never got its own
// max-width, so it stayed stuck inheriting the bare `main` selector's 480px).
const TABLET_VIEWPORT = { width: 960, height: 800 };
// At 1280px and up, .app-shell reaches 1120px and the two field-cards sit
// side by side with the Traducir button centered in the gutter between them.
const WIDE_VIEWPORT = { width: 1440, height: 900 };

test.describe("wide desktop layout (#120)", () => {
  test("below 800px, .app-shell and .app-body stay at the 480px mobile cap", async ({
    page,
  }) => {
    await page.setViewportSize(MOBILE_VIEWPORT);
    await page.goto("/");

    const shellWidth = await page
      .locator(".app-shell")
      .evaluate((el) => getComputedStyle(el).maxWidth);
    expect(shellWidth).toBe("480px");

    const inputCard = page.locator(".field-card").first();
    const outputCard = page.locator(".field-card").last();
    const inputBox = await inputCard.boundingBox();
    const outputBox = await outputCard.boundingBox();
    // Single column: the output card sits below the input card, not beside it.
    expect(outputBox!.y).toBeGreaterThan(inputBox!.y + inputBox!.height - 1);
  });

  test("between 800px and 1279px, .app-body fills the widened .app-shell (bug fix)", async ({
    page,
  }) => {
    await page.setViewportSize(TABLET_VIEWPORT);
    await page.goto("/");

    const shell = page.locator(".app-shell");
    const body = page.locator(".app-body");
    const shellBox = await shell.boundingBox();
    const bodyBox = await body.boundingBox();

    expect(await shell.evaluate((el) => getComputedStyle(el).maxWidth)).toBe("720px");
    // .app-body should span (nearly) the full shell width, not be stuck at 480px.
    // (A few px of slack accounts for .app-shell's 1px border on each side.)
    expect(bodyBox!.width).toBeGreaterThan(600);
    expect(Math.abs(bodyBox!.width - shellBox!.width)).toBeLessThan(4);
  });

  test("at 1280px and up, the field-cards sit side by side around a centered Traducir button", async ({
    page,
  }) => {
    await page.setViewportSize(WIDE_VIEWPORT);
    await page.goto("/");

    const shell = page.locator(".app-shell");
    expect(await shell.evaluate((el) => getComputedStyle(el).maxWidth)).toBe("1120px");

    const inputCard = page.locator(".field-card").first();
    const outputCard = page.locator(".field-card").last();
    const translateButton = page.getByRole("button", { name: "Traducir" });

    const inputBox = await inputCard.boundingBox();
    const outputBox = await outputCard.boundingBox();
    const buttonBox = await translateButton.boundingBox();

    // Side by side, not stacked: same row, input to the left of output.
    expect(Math.abs(inputBox!.y - outputBox!.y)).toBeLessThan(2);
    expect(inputBox!.x).toBeLessThan(outputBox!.x);

    // The button sits in the gutter between the two cards.
    expect(buttonBox!.x).toBeGreaterThan(inputBox!.x + inputBox!.width - 1);
    expect(buttonBox!.x + buttonBox!.width).toBeLessThan(outputBox!.x + 1);

    // direction-switch stays a compact centered pill, not stretched full width.
    const directionSwitch = page.locator(".direction-switch");
    const switchWidth = await directionSwitch.evaluate((el) => el.getBoundingClientRect().width);
    expect(switchWidth).toBeLessThanOrEqual(480);

    // Textareas grow taller at this breakpoint.
    const input = page.getByLabel(/^Texto en /);
    const inputMinHeight = await input.evaluate((el) =>
      parseFloat(getComputedStyle(el).minHeight),
    );
    expect(inputMinHeight).toBe(160);
  });

  test("DOM order still matches visual order (input, then Traducir, then output), so tab order is unaffected", async ({
    page,
  }) => {
    await page.setViewportSize(WIDE_VIEWPORT);
    await page.goto("/");

    const order = await page.evaluate(() => {
      const input = document.querySelector('textarea[aria-label^="Texto en"]')!;
      const button = Array.from(document.querySelectorAll("button")).find(
        (el) => el.textContent?.trim() === "Traducir",
      )!;
      const output = document.querySelector('textarea[aria-label^="Traducción en"]')!;

      const inputBeforeButton =
        input.compareDocumentPosition(button) & Node.DOCUMENT_POSITION_FOLLOWING;
      const buttonBeforeOutput =
        button.compareDocumentPosition(output) & Node.DOCUMENT_POSITION_FOLLOWING;
      return { inputBeforeButton: !!inputBeforeButton, buttonBeforeOutput: !!buttonBeforeOutput };
    });

    expect(order.inputBeforeButton).toBe(true);
    expect(order.buttonBeforeOutput).toBe(true);
  });
});
