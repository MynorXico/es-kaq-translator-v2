import { expect, test } from "@playwright/test";

test("loads the translator page", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Traductor Kaqchikel" })).toBeVisible();
  await expect(page.getByText("Español", { exact: true })).toBeVisible();
  await expect(page.getByText("Kaqchikel", { exact: true })).toBeVisible();
});

test("swaps translation direction", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Español", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Cambiar dirección" }).click();

  await expect(page.getByText("Kaqchikel", { exact: true })).toBeVisible();
  await expect(
    page.getByText("Dirección cambiada: Kaqchikel a Español."),
  ).toBeAttached();
});

test("swapping carries a successful translation into the input", async ({ page }) => {
  await page.route("**/v1/translate", (route) =>
    route.fulfill({ json: { translation: "Utz" } }),
  );

  await page.goto("/");
  const input = page.getByLabel("Texto a traducir");
  await input.fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();
  await expect(page.getByLabel("Traducción")).toHaveValue("Utz");

  await page.getByRole("button", { name: "Cambiar dirección" }).click();

  await expect(input).toHaveValue("Utz");
  await expect(input).toBeFocused();
  await expect(page.getByLabel("Traducción")).toHaveValue("");
});

test("does not force the caret to the end on a later edit when the carried-over text equals the current input", async ({
  page,
}) => {
  await page.route("**/v1/translate", (route) =>
    route.fulfill({ json: { translation: "Hola" } }),
  );

  await page.goto("/");
  const input = page.getByLabel("Texto a traducir");
  await input.fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();
  await expect(page.getByLabel("Traducción")).toHaveValue("Hola");

  // The carried-over translation ("Hola") equals the current input value.
  await page.getByRole("button", { name: "Cambiar dirección" }).click();
  await expect(input).toHaveValue("Hola");
  await expect(input).toBeFocused();

  // Move the caret to right after "Ho" and type a character there.
  await input.press("Home");
  await input.press("ArrowRight");
  await input.press("ArrowRight");
  await input.pressSequentially("X");

  // If the caret-reset behavior were still "armed" from the same-value
  // swap, it would force the caret to the end on this keystroke, landing
  // "X" at the end ("HolaX") instead of where it was actually typed.
  await expect(input).toHaveValue("HoXla");
});

test("translate button is disabled until text is entered", async ({ page }) => {
  await page.goto("/");
  const translateButton = page.getByRole("button", { name: "Traducir" });
  await expect(translateButton).toBeDisabled();

  await page.getByLabel("Texto a traducir").fill("Hola");
  await expect(translateButton).toBeEnabled();
});

test("submitting a translation shows a result", async ({ page }) => {
  await page.route("**/v1/translate", (route) =>
    route.fulfill({ json: { translation: "Utz" } }),
  );

  await page.goto("/");
  await page.getByLabel("Texto a traducir").fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();
  await expect(page.getByLabel("Traducción")).not.toHaveValue("");
});

test("shows a Spanish error message and a working retry button on a network failure", async ({
  page,
}) => {
  let requestCount = 0;
  await page.route("**/v1/translate", (route) => {
    requestCount += 1;
    if (requestCount === 1) {
      return route.abort("failed");
    }
    return route.fulfill({ json: { translation: "Utz" } });
  });

  await page.goto("/");
  await page.getByLabel("Texto a traducir").fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();

  await expect(
    page.getByText("No se pudo conectar. Revisa tu conexión a internet e inténtalo de nuevo."),
  ).toBeVisible();

  await page.getByRole("button", { name: "Reintentar" }).click();

  await expect(page.getByLabel("Traducción")).toHaveValue("Utz");
});

test("clears the input, output, and error state via the clear button, and refocuses the input", async ({
  page,
}) => {
  let requestCount = 0;
  await page.route("**/v1/translate", (route) => {
    requestCount += 1;
    if (requestCount === 1) {
      return route.abort("failed");
    }
    return route.fulfill({ json: { translation: "Utz" } });
  });

  await page.goto("/");
  const input = page.getByLabel("Texto a traducir");
  await input.fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();

  await expect(
    page.getByText("No se pudo conectar. Revisa tu conexión a internet e inténtalo de nuevo."),
  ).toBeVisible();

  await page.getByRole("button", { name: "Borrar el texto de entrada" }).click();

  await expect(input).toHaveValue("");
  await expect(
    page.getByText("No se pudo conectar. Revisa tu conexión a internet e inténtalo de nuevo."),
  ).toBeHidden();
  await expect(input).toBeFocused();
  await expect(page.getByRole("button", { name: "Borrar el texto de entrada" })).toBeHidden();
});

test("ignores a stale in-flight response after Borrar clears the fields mid-request", async ({
  page,
}) => {
  await page.route("**/v1/translate", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 500));
    return route.fulfill({ json: { translation: "Utz" } });
  });

  await page.goto("/");
  const input = page.getByLabel("Texto a traducir");
  await input.fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();

  await expect(page.getByText("Traduciendo…")).toBeVisible();
  await page.getByRole("button", { name: "Borrar el texto de entrada" }).click();

  await expect(input).toHaveValue("");
  await expect(page.getByLabel("Traducción")).toHaveValue("");

  // Give the delayed (now-stale) response time to resolve, then confirm it
  // never landed -- the field should still be idle/empty, not "Utz".
  await page.waitForTimeout(700);
  await expect(page.getByLabel("Traducción")).toHaveValue("");
  await expect(page.getByText("Traduciendo…")).toBeHidden();
});

test("ignores a stale in-flight response after swapping direction mid-request", async ({
  page,
}) => {
  await page.route("**/v1/translate", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 500));
    return route.fulfill({ json: { translation: "Utz" } });
  });

  await page.goto("/");
  const input = page.getByLabel("Texto a traducir");
  await input.fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();

  await expect(page.getByText("Traduciendo…")).toBeVisible();
  await page.getByRole("button", { name: "Cambiar dirección" }).click();

  // Nothing to carry over while a request is in flight, so the direction
  // just flips and the input is left untouched.
  await expect(input).toHaveValue("Hola");
  await expect(page.getByLabel("Traducción")).toHaveValue("");

  await page.waitForTimeout(700);
  await expect(page.getByLabel("Traducción")).toHaveValue("");
  await expect(page.getByText("Traduciendo…")).toBeHidden();
});

test("copies a successful translation to the clipboard with temporary confirmation", async ({
  page,
  context,
}) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.route("**/v1/translate", (route) =>
    route.fulfill({ json: { translation: "Utz" } }),
  );

  await page.goto("/");
  await page.getByLabel("Texto a traducir").fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();

  const copyButton = page.getByRole("button", { name: "Copiar" });
  await expect(copyButton).toBeVisible();
  await copyButton.click();

  await expect(page.getByRole("button", { name: "Copiado" })).toBeVisible();
  const clipboardText = await page.evaluate(() => navigator.clipboard.readText());
  expect(clipboardText).toBe("Utz");

  await expect(page.getByRole("button", { name: "Copiar" })).toBeVisible();
});
