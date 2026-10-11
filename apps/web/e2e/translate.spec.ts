import { expect, test, type Page } from "@playwright/test";

// ADR 0008: translate() now does POST /v1/translate-jobs (enqueue), then
// polls GET /v1/translate-jobs/{job_id} until a terminal status. The exact
// job_id value never matters to these tests, just that it round-trips from
// the enqueue response into the poll URL.
const JOB_ID = "11111111-1111-1111-1111-111111111111";

// Routes a POST /v1/translate-jobs enqueue to `job_id` (optionally after
// `delayMs`), and a GET /v1/translate-jobs/{job_id} poll straight to a
// `succeeded` response with `translation`. Covers the common "already warm"
// case most of these tests only care about the end-to-end result for.
async function mockSucceedingJob(
  page: Page,
  translation: string,
  { delayMs = 0 }: { delayMs?: number } = {},
): Promise<void> {
  await page.route("**/v1/translate-jobs", async (route) => {
    if (route.request().method() !== "POST") {
      return route.continue();
    }
    if (delayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    return route.fulfill({ status: 202, json: { job_id: JOB_ID } });
  });
  await page.route(`**/v1/translate-jobs/${JOB_ID}`, (route) =>
    route.fulfill({ json: { status: "succeeded", translation } }),
  );
}

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
  await mockSucceedingJob(page, "Utz");

  await page.goto("/");
  const input = page.getByLabel(/^Texto en /);
  await input.fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();
  await expect(page.getByLabel(/^Traducción en /)).toHaveValue("Utz");

  await page.getByRole("button", { name: "Cambiar dirección" }).click();

  await expect(input).toHaveValue("Utz");
  await expect(input).toBeFocused();
  await expect(page.getByLabel(/^Traducción en /)).toHaveValue("");
});

test("does not force the caret to the end on a later edit when the carried-over text equals the current input", async ({
  page,
}) => {
  await mockSucceedingJob(page, "Hola");

  await page.goto("/");
  const input = page.getByLabel(/^Texto en /);
  await input.fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();
  await expect(page.getByLabel(/^Traducción en /)).toHaveValue("Hola");

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

  await page.getByLabel(/^Texto en /).fill("Hola");
  await expect(translateButton).toBeEnabled();
});

test("submitting a translation shows a result", async ({ page }) => {
  await mockSucceedingJob(page, "Utz");

  await page.goto("/");
  await page.getByLabel(/^Texto en /).fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();
  await expect(page.getByLabel(/^Traducción en /)).not.toHaveValue("");
  await expect(
    page.getByText(
      "Esta traducción fue generada automáticamente por un modelo en desarrollo y puede contener errores.",
    ),
  ).toBeVisible();
});

test("polls GET /v1/translate-jobs/{job_id} until the job succeeds", async ({ page }) => {
  let pollCount = 0;
  await page.route("**/v1/translate-jobs", (route) => {
    if (route.request().method() !== "POST") {
      return route.continue();
    }
    return route.fulfill({ status: 202, json: { job_id: JOB_ID } });
  });
  await page.route(`**/v1/translate-jobs/${JOB_ID}`, (route) => {
    pollCount += 1;
    if (pollCount === 1) {
      return route.fulfill({ json: { status: "pending" } });
    }
    return route.fulfill({ json: { status: "succeeded", translation: "Utz" } });
  });

  await page.goto("/");
  await page.getByLabel(/^Texto en /).fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();

  // POLL_INTERVAL_MS (apps/web/src/api.ts) stands between the first
  // "pending" response and the second poll that reports "succeeded".
  await expect(page.getByLabel(/^Traducción en /)).toHaveValue("Utz", { timeout: 10_000 });
});

test("shows a Spanish error message and a working retry button on a network failure", async ({
  page,
}) => {
  let requestCount = 0;
  await page.route("**/v1/translate-jobs", (route) => {
    if (route.request().method() !== "POST") {
      return route.continue();
    }
    requestCount += 1;
    if (requestCount === 1) {
      return route.abort("failed");
    }
    return route.fulfill({ status: 202, json: { job_id: JOB_ID } });
  });
  await page.route(`**/v1/translate-jobs/${JOB_ID}`, (route) =>
    route.fulfill({ json: { status: "succeeded", translation: "Utz" } }),
  );

  await page.goto("/");
  await page.getByLabel(/^Texto en /).fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();

  await expect(
    page.getByText("No se pudo conectar. Revisa tu conexión a internet e inténtalo de nuevo."),
  ).toBeVisible();

  await page.getByRole("button", { name: "Reintentar" }).click();

  await expect(page.getByLabel(/^Traducción en /)).toHaveValue("Utz");
});

test("shows a distinct error message when a poll GET returns 404 (job expired mid-poll)", async ({
  page,
}) => {
  await page.route("**/v1/translate-jobs", (route) => {
    if (route.request().method() !== "POST") {
      return route.continue();
    }
    return route.fulfill({ status: 202, json: { job_id: JOB_ID } });
  });
  await page.route(`**/v1/translate-jobs/${JOB_ID}`, (route) =>
    route.fulfill({ status: 404, json: { error: "No se encontró esa traducción." } }),
  );

  await page.goto("/");
  await page.getByLabel(/^Texto en /).fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();

  await expect(
    page.getByText(
      "No se pudo recuperar esa traducción porque tardó demasiado. Inténtalo de nuevo.",
    ),
  ).toBeVisible();
  await expect(
    page.getByText("No se pudo traducir ese texto. Revisa lo que escribiste e inténtalo de nuevo."),
  ).toBeHidden();
});

test("clears the input, output, and error state via the clear button, and refocuses the input", async ({
  page,
}) => {
  let requestCount = 0;
  await page.route("**/v1/translate-jobs", (route) => {
    if (route.request().method() !== "POST") {
      return route.continue();
    }
    requestCount += 1;
    if (requestCount === 1) {
      return route.abort("failed");
    }
    return route.fulfill({ status: 202, json: { job_id: JOB_ID } });
  });
  await page.route(`**/v1/translate-jobs/${JOB_ID}`, (route) =>
    route.fulfill({ json: { status: "succeeded", translation: "Utz" } }),
  );

  await page.goto("/");
  const input = page.getByLabel(/^Texto en /);
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

test("ignores a stale in-flight response after Borrar clears the fields mid-request, and actually cancels the poll loop", async ({
  page,
}) => {
  let pollCalls = 0;
  await mockSucceedingJob(page, "Utz", { delayMs: 500 });
  await page.route(`**/v1/translate-jobs/${JOB_ID}`, (route) => {
    pollCalls += 1;
    return route.fulfill({ json: { status: "succeeded", translation: "Utz" } });
  });

  await page.goto("/");
  const input = page.getByLabel(/^Texto en /);
  await input.fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();

  await expect(page.getByText("Traduciendo…")).toBeVisible();
  await page.getByRole("button", { name: "Borrar el texto de entrada" }).click();

  await expect(input).toHaveValue("");
  await expect(page.getByLabel(/^Traducción en /)).toHaveValue("");

  // Give the delayed (now-cancelled) enqueue time to resolve, then confirm
  // it never landed, and that the cancelled request never even reached the
  // poll step (ADR 0008: the poll loop is actually cancelled, not just
  // discarded).
  await page.waitForTimeout(700);
  await expect(page.getByLabel(/^Traducción en /)).toHaveValue("");
  await expect(page.getByText("Traduciendo…")).toBeHidden();
  expect(pollCalls).toBe(0);
});

test("ignores a stale in-flight response after swapping direction mid-request, and actually cancels the poll loop", async ({
  page,
}) => {
  let pollCalls = 0;
  await mockSucceedingJob(page, "Utz", { delayMs: 500 });
  await page.route(`**/v1/translate-jobs/${JOB_ID}`, (route) => {
    pollCalls += 1;
    return route.fulfill({ json: { status: "succeeded", translation: "Utz" } });
  });

  await page.goto("/");
  const input = page.getByLabel(/^Texto en /);
  await input.fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();

  await expect(page.getByText("Traduciendo…")).toBeVisible();
  await page.getByRole("button", { name: "Cambiar dirección" }).click();

  // Nothing to carry over while a request is in flight, so the direction
  // just flips and the input is left untouched.
  await expect(input).toHaveValue("Hola");
  await expect(page.getByLabel(/^Traducción en /)).toHaveValue("");

  await page.waitForTimeout(700);
  await expect(page.getByLabel(/^Traducción en /)).toHaveValue("");
  await expect(page.getByText("Traduciendo…")).toBeHidden();
  expect(pollCalls).toBe(0);
});

test("copies a successful translation to the clipboard with temporary confirmation", async ({
  page,
  context,
}) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await mockSucceedingJob(page, "Utz");

  await page.goto("/");
  await page.getByLabel(/^Texto en /).fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();

  const copyButton = page.getByRole("button", { name: "Copiar" });
  await expect(copyButton).toBeVisible();
  await copyButton.click();

  await expect(page.getByRole("button", { name: "Copiado" })).toBeVisible();
  const clipboardText = await page.evaluate(() => navigator.clipboard.readText());
  expect(clipboardText).toBe("Utz");

  await expect(page.getByRole("button", { name: "Copiar" })).toBeVisible();
});

test("gives the input and output direction-matching accessible names and lang attributes", async ({
  page,
}) => {
  await page.goto("/");

  const input = page.getByLabel("Texto en español");
  const output = page.getByLabel("Traducción en kaqchikel");
  await expect(input).toHaveAttribute("lang", "es");
  await expect(output).toHaveAttribute("lang", "cak");

  await page.getByRole("button", { name: "Cambiar dirección" }).click();

  const swappedInput = page.getByLabel("Texto en kaqchikel");
  const swappedOutput = page.getByLabel("Traducción en español");
  await expect(swappedInput).toHaveAttribute("lang", "cak");
  await expect(swappedOutput).toHaveAttribute("lang", "es");
});

test("moves focus to the output once a translation succeeds", async ({ page }) => {
  await mockSucceedingJob(page, "Utz");

  await page.goto("/");
  await page.getByLabel(/^Texto en /).fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();

  const output = page.getByLabel(/^Traducción en /);
  await expect(output).toHaveValue("Utz");
  await expect(output).toBeFocused();
});

test("does not steal focus from the input if the user is still typing when a translation resolves", async ({
  page,
}) => {
  await mockSucceedingJob(page, "Utz", { delayMs: 500 });

  await page.goto("/");
  const input = page.getByLabel(/^Texto en /);
  await input.fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();
  await expect(page.getByText("Traduciendo…")).toBeVisible();

  // The user keeps typing their next query while the request is in flight.
  await input.focus();
  await input.fill("Hola, ¿cómo estás?");
  await expect(input).toBeFocused();

  // Wait for the delayed response to resolve.
  await expect(page.getByLabel(/^Traducción en /)).toHaveValue("Utz");

  // Focus (and what the user typed) must stay on the input.
  await expect(input).toBeFocused();
  await expect(input).toHaveValue("Hola, ¿cómo estás?");
});

test("keyboard tab order follows nav -> swap -> input -> clear -> translate -> copy -> output", async ({
  page,
}) => {
  await mockSucceedingJob(page, "Utz");

  await page.goto("/");
  await page.getByLabel(/^Texto en /).fill("Hola");
  await page.getByRole("button", { name: "Traducir" }).click();
  await expect(page.getByLabel(/^Traducción en /)).toHaveValue("Utz");

  // Start from the nav link (real DOM order should carry it from there).
  await page.getByRole("button", { name: "Acerca de" }).focus();
  await expect(page.getByRole("button", { name: "Acerca de" })).toBeFocused();

  await page.keyboard.press("Tab");
  await expect(page.getByRole("button", { name: "Cambiar dirección" })).toBeFocused();

  await page.keyboard.press("Tab");
  await expect(page.getByLabel(/^Texto en /)).toBeFocused();

  await page.keyboard.press("Tab");
  await expect(page.getByRole("button", { name: "Borrar el texto de entrada" })).toBeFocused();

  await page.keyboard.press("Tab");
  await expect(page.getByRole("button", { name: "Traducir" })).toBeFocused();

  await page.keyboard.press("Tab");
  await expect(page.getByRole("button", { name: "Copiar" })).toBeFocused();

  await page.keyboard.press("Tab");
  await expect(page.getByLabel(/^Traducción en /)).toBeFocused();
});
