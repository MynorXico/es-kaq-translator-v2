import { expect, test } from "@playwright/test";

test("navigates to the About page and back via the header/back links", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("button", { name: "Acerca de" }).click();
  await expect(
    page.getByRole("heading", { name: "Acerca de Traductor Kaqchikel" }),
  ).toBeVisible();

  await page.getByRole("button", { name: "← Volver al traductor" }).click();
  await expect(page.getByRole("heading", { name: "Traductor Kaqchikel" })).toBeVisible();
});
