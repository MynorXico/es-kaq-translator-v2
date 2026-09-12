import { expect, test } from "@playwright/test";

test("loads the translator page", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Traductor Kaqchikel" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Español → Kaqchikel" })).toBeVisible();
});

test("toggles translation direction", async ({ page }) => {
  await page.goto("/");
  const toggle = page.getByRole("button", { name: "Español → Kaqchikel" });
  await toggle.click();
  await expect(page.getByRole("button", { name: "Kaqchikel → Español" })).toBeVisible();
});

test("translate button is disabled until text is entered", async ({ page }) => {
  await page.goto("/");
  const translateButton = page.getByRole("button", { name: "Traducir" });
  await expect(translateButton).toBeDisabled();

  await page.getByLabel("Texto a traducir").fill("Hola");
  await expect(translateButton).toBeEnabled();
});

test("submitting a translation shows a result", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Texto a traducir").fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();
  await expect(page.getByLabel("Traducción")).not.toHaveValue("");
});
