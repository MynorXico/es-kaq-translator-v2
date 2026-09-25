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
// A much wider row than the direction-switch's 480px cap -- exercises
// whether the pill actually stays capped/centered rather than stretching,
// which a merely-somewhat-wider viewport (like WIDE_VIEWPORT) might not
// reveal (see #122).
const ULTRA_WIDE_VIEWPORT = { width: 1920, height: 1000 };

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

// Non-happy-path states at the >=1280px breakpoint, following up on #120/
// #121's ad-hoc (uncommitted) manual verification -- see #122.
test.describe("wide desktop layout non-happy-path states (#122)", () => {
  test("mismatched input/output content length doesn't stretch the shorter field-card to match the taller one", async ({
    page,
  }) => {
    await page.setViewportSize(WIDE_VIEWPORT);
    await page.goto("/");

    // Over the 2000-char limit: the input field-card grows an extra
    // over-limit-message line, while the output field-card stays at its
    // idle (shorter) height.
    await page.getByLabel(/^Texto en /).fill("a".repeat(2001));
    await expect(page.getByText("El texto supera el límite de 2000 caracteres.")).toBeVisible();

    const inputCard = page.locator(".field-card").first();
    const outputCard = page.locator(".field-card").last();
    const inputBox = await inputCard.boundingBox();
    const outputBox = await outputCard.boundingBox();

    // Same row -- both cards still start at the same y (align-items: start
    // keeps them aligned to the row's start edge, it just doesn't stretch
    // them to match each other's height).
    expect(Math.abs(inputBox!.y - outputBox!.y)).toBeLessThanOrEqual(2);
    // The taller (input) card must not force the shorter (output) card to
    // stretch up to match it.
    expect(inputBox!.height).toBeGreaterThan(outputBox!.height + 10);

    expect(
      await page.locator(".app-body").evaluate((el) => getComputedStyle(el).alignItems),
    ).toBe("start");
  });

  test("the Traducir button stays vertically centered against the taller card's height, not the shorter one", async ({
    page,
  }) => {
    await page.setViewportSize(WIDE_VIEWPORT);
    await page.goto("/");

    await page.getByLabel(/^Texto en /).fill("a".repeat(2001));
    await expect(page.getByText("El texto supera el límite de 2000 caracteres.")).toBeVisible();

    const inputCard = page.locator(".field-card").first();
    const outputCard = page.locator(".field-card").last();
    const translateButton = page.getByRole("button", { name: "Traducir" });

    const inputBox = await inputCard.boundingBox();
    const outputBox = await outputCard.boundingBox();
    const buttonBox = await translateButton.boundingBox();

    // Sanity check: the cards actually differ in height here.
    expect(inputBox!.height).toBeGreaterThan(outputBox!.height + 10);

    const inputCenterY = inputBox!.y + inputBox!.height / 2;
    const outputCenterY = outputBox!.y + outputBox!.height / 2;
    const buttonCenterY = buttonBox!.y + buttonBox!.height / 2;

    // Centered against the taller (input) card...
    expect(Math.abs(buttonCenterY - inputCenterY)).toBeLessThan(4);
    // ...and measurably off-center from the shorter (output) card.
    expect(Math.abs(buttonCenterY - outputCenterY)).toBeGreaterThan(8);
  });

  test("the error state (mocked API failure) renders in the output column without breaking the grid layout", async ({
    page,
  }) => {
    await page.setViewportSize(WIDE_VIEWPORT);
    await page.route("**/v1/translate", (route) => route.abort("failed"));
    await page.goto("/");

    await page.getByLabel(/^Texto en /).fill("Hola");
    await page.getByRole("button", { name: "Traducir" }).click();

    await expect(
      page.getByText("No se pudo conectar. Revisa tu conexión a internet e inténtalo de nuevo."),
    ).toBeVisible();
    const retryButton = page.getByRole("button", { name: "Reintentar" });
    await expect(retryButton).toBeVisible();

    const inputCard = page.locator(".field-card").first();
    const outputCard = page.locator(".field-card").last();
    const translateButton = page.getByRole("button", { name: "Traducir" });

    const inputBox = await inputCard.boundingBox();
    const outputBox = await outputCard.boundingBox();
    const retryBox = await retryButton.boundingBox();

    // Still side by side in the same row -- the error state doesn't push
    // the output column onto its own row or collapse the grid.
    expect(Math.abs(inputBox!.y - outputBox!.y)).toBeLessThan(2);
    expect(inputBox!.x).toBeLessThan(outputBox!.x);

    // The error message + Reintentar button stay within the output card's
    // own horizontal bounds, not overflowing into the gutter or input
    // column.
    expect(retryBox!.x).toBeGreaterThanOrEqual(outputBox!.x - 1);
    expect(retryBox!.x + retryBox!.width).toBeLessThanOrEqual(outputBox!.x + outputBox!.width + 1);

    // The error box (min-height 120px) is naturally shorter than the input
    // card (whose textarea alone has a 160px min-height at this
    // breakpoint) -- it must not get stretched up to match the input card
    // (same align-items: start behavior as the over-limit-message case
    // above, exercised here with genuinely different content).
    expect(inputBox!.height).toBeGreaterThan(outputBox!.height + 10);

    // The Traducir button still centers against the taller (input) card.
    const buttonBox = await translateButton.boundingBox();
    const inputCenterY = inputBox!.y + inputBox!.height / 2;
    const buttonCenterY = buttonBox!.y + buttonBox!.height / 2;
    expect(Math.abs(buttonCenterY - inputCenterY)).toBeLessThan(4);
  });

  test("the direction-switch pill stays capped at 480px and centered, not stretched, when its row is much wider", async ({
    page,
  }) => {
    await page.setViewportSize(ULTRA_WIDE_VIEWPORT);
    await page.goto("/");

    const directionSwitch = page.locator(".direction-switch");
    const appBody = page.locator(".app-body");

    const switchBox = await directionSwitch.boundingBox();
    const bodyBox = await appBody.boundingBox();

    // With the current short "Español"/"Kaqchikel" labels, the pill's own
    // shrink-to-fit width is already well under 480px regardless of the
    // cap, so the bounding-box width check alone wouldn't catch a
    // regression to the cap itself -- assert the actual CSS property too.
    expect(
      await directionSwitch.evaluate((el) => getComputedStyle(el).maxWidth),
    ).toBe("480px");
    expect(switchBox!.width).toBeLessThanOrEqual(480);
    // Centered under the (much wider) row, not stretched or anchored to
    // one side.
    const switchCenterX = switchBox!.x + switchBox!.width / 2;
    const bodyCenterX = bodyBox!.x + bodyBox!.width / 2;
    expect(Math.abs(switchCenterX - bodyCenterX)).toBeLessThan(2);

    // Legibility: the swap button and labels keep their normal, non-
    // compressed size rather than being squeezed to fit some unintended
    // narrower box.
    const swapButton = page.getByRole("button", { name: "Cambiar dirección" });
    const swapBox = await swapButton.boundingBox();
    expect(Math.abs(swapBox!.width - 44)).toBeLessThan(2);
    expect(Math.abs(swapBox!.height - 44)).toBeLessThan(2);

    const dirLabel = page.locator(".dir-label").first();
    expect(await dirLabel.evaluate((el) => getComputedStyle(el).fontSize)).toBe("15px");
  });
});
