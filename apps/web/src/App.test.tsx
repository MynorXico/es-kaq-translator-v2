import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";
import {
  translate,
  TranslateHttpError,
  TranslateNetworkError,
  TranslateTimeoutError,
} from "./api";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, translate: vi.fn() };
});

const mockedTranslate = vi.mocked(translate);

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("App", () => {
  it("renders the translator title and the source/target direction labels", () => {
    render(<App />);
    // Two nodes now carry this text (the app-bar wordmark and the page
    // heading, per #111's app-shell/app-bar structure) -- assert via the
    // heading role so the query stays unambiguous.
    expect(screen.getByRole("heading", { name: "Traductor Kaqchikel" })).toBeInTheDocument();
    expect(screen.getByText("Español")).toBeInTheDocument();
    expect(screen.getByText("Kaqchikel")).toBeInTheDocument();
  });

  it("shows a character counter that updates as the user types", () => {
    render(<App />);
    const input = screen.getByLabelText(/^Texto en /);

    expect(screen.getByText("0 / 2000")).toBeInTheDocument();

    fireEvent.change(input, { target: { value: "Hola" } });

    expect(screen.getByText("4 / 2000")).toBeInTheDocument();
  });

  it("disables translation and explains why once the input exceeds 2000 characters", () => {
    render(<App />);
    const input = screen.getByLabelText(/^Texto en /);
    const translateButton = screen.getByRole("button", { name: "Traducir" });

    fireEvent.change(input, { target: { value: "a".repeat(2001) } });

    expect(screen.getByText("2001 / 2000")).toBeInTheDocument();
    expect(
      screen.getByText("El texto supera el límite de 2000 caracteres."),
    ).toBeInTheDocument();
    expect(translateButton).toBeDisabled();
  });

  it("navigates to the About page via the header nav link, and back via the same slot", () => {
    render(<App />);

    const aboutLink = screen.getByRole("button", { name: "Acerca de" });
    fireEvent.click(aboutLink);

    expect(
      screen.getByRole("heading", { name: "Acerca de Traductor Kaqchikel" }),
    ).toBeInTheDocument();

    const translateLink = screen.getByRole("button", { name: "Traducir" });
    fireEvent.click(translateLink);

    expect(screen.getByText("Español")).toBeInTheDocument();
    expect(screen.getByText("Kaqchikel")).toBeInTheDocument();
  });
});

describe("App translate flow", () => {
  beforeEach(() => {
    mockedTranslate.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows the idle placeholder before any translation is submitted", () => {
    render(<App />);
    expect(screen.getByLabelText(/^Traducción en /)).toHaveAttribute(
      "placeholder",
      "La traducción aparecerá aquí.",
    );
  });

  it("shows a loading indicator while the request is in flight, then the result", async () => {
    const { promise, resolve } = deferred<{ translation: string }>();
    mockedTranslate.mockReturnValue(promise);
    render(<App />);

    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));

    expect(await screen.findByText("Traduciendo…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Traducir" })).toBeDisabled();

    await act(async () => {
      resolve({ translation: "Utz" });
    });

    expect(await screen.findByLabelText(/^Traducción en /)).toHaveValue("Utz");
  });

  it("shows a translation-quality disclaimer once a translation succeeds, but not before", async () => {
    mockedTranslate.mockResolvedValueOnce({ translation: "Utz" });
    render(<App />);

    const disclaimerText =
      "Esta traducción fue generada automáticamente por un modelo en desarrollo y puede contener errores.";
    expect(screen.queryByText(disclaimerText)).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));

    expect(await screen.findByText(disclaimerText)).toBeInTheDocument();
  });

  it("switches to a warming-up message if there's still no response after 4 seconds", async () => {
    vi.useFakeTimers();
    const { promise } = deferred<{ translation: string }>();
    mockedTranslate.mockReturnValue(promise);
    render(<App />);

    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));

    expect(screen.getByText("Traduciendo…")).toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(4000);
    });

    expect(
      screen.getByText("Calentando el modelo. Puede tardar hasta un minuto."),
    ).toBeInTheDocument();
  });

  it("shows a network error message with a retry button on a fetch failure", async () => {
    mockedTranslate.mockRejectedValueOnce(new TranslateNetworkError());
    render(<App />);

    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));

    expect(
      await screen.findByText(
        "No se pudo conectar. Revisa tu conexión a internet e inténtalo de nuevo.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reintentar" })).toBeInTheDocument();
  });

  it("shows a 'text' error message for a 4xx response", async () => {
    mockedTranslate.mockRejectedValueOnce(new TranslateHttpError(400));
    render(<App />);

    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));

    expect(
      await screen.findByText(
        "No se pudo traducir ese texto. Revisa lo que escribiste e inténtalo de nuevo.",
      ),
    ).toBeInTheDocument();
  });

  it("shows a server error message for a 5xx response", async () => {
    mockedTranslate.mockRejectedValueOnce(new TranslateHttpError(500));
    render(<App />);

    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));

    expect(
      await screen.findByText("Hubo un problema en el servidor. Inténtalo de nuevo en unos minutos."),
    ).toBeInTheDocument();
  });

  it("shows a timeout error message when the client-side timeout elapses", async () => {
    mockedTranslate.mockRejectedValueOnce(new TranslateTimeoutError());
    render(<App />);

    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));

    expect(
      await screen.findByText(
        "El modelo está tardando más de lo esperado. Inténtalo de nuevo en unos minutos.",
      ),
    ).toBeInTheDocument();
  });

  it("retries with the current input text when 'Reintentar' is clicked", async () => {
    mockedTranslate.mockRejectedValueOnce(new TranslateNetworkError());
    mockedTranslate.mockResolvedValueOnce({ translation: "Utz" });
    render(<App />);

    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));
    await screen.findByRole("button", { name: "Reintentar" });

    fireEvent.change(screen.getByLabelText(/^Texto en /), {
      target: { value: "Hola, buenos días" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Reintentar" }));

    await waitFor(() => expect(mockedTranslate).toHaveBeenCalledTimes(2));
    expect(mockedTranslate).toHaveBeenLastCalledWith("Hola, buenos días", "es-to-cak");
    expect(await screen.findByLabelText(/^Traducción en /)).toHaveValue("Utz");
  });

  it("ignores a stale in-flight response after the direction is swapped mid-request", async () => {
    const { promise, resolve } = deferred<{ translation: string }>();
    mockedTranslate.mockReturnValue(promise);
    render(<App />);

    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));
    await screen.findByText("Traduciendo…");

    fireEvent.click(screen.getByRole("button", { name: "Cambiar dirección" }));

    // Nothing to carry over while a request is in flight (no successful
    // translation yet), so the direction just flips and the input is left
    // untouched, per spec.
    expect(screen.getByText("Kaqchikel")).toBeInTheDocument();
    expect(screen.getByLabelText(/^Texto en /)).toHaveValue("Hola");
    expect(screen.getByLabelText(/^Traducción en /)).toHaveValue("");

    await act(async () => {
      resolve({ translation: "Utz" });
    });

    // The stale response belonged to the abandoned request; the UI should
    // still be idle, not showing that translation.
    expect(screen.getByLabelText(/^Traducción en /)).toHaveValue("");
    expect(screen.queryByText("Traduciendo…")).not.toBeInTheDocument();
  });

  it("ignores a stale in-flight response after the direction is swapped mid-warming", async () => {
    vi.useFakeTimers();
    const { promise, resolve } = deferred<{ translation: string }>();
    mockedTranslate.mockReturnValue(promise);
    render(<App />);

    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));

    await act(async () => {
      await vi.advanceTimersByTimeAsync(4000);
    });
    expect(
      screen.getByText("Calentando el modelo. Puede tardar hasta un minuto."),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Cambiar dirección" }));

    // Same as swapping mid-loading: nothing to carry over yet, so the
    // direction just flips and the input is left untouched.
    expect(screen.getByText("Kaqchikel")).toBeInTheDocument();
    expect(screen.getByLabelText(/^Texto en /)).toHaveValue("Hola");
    expect(screen.getByLabelText(/^Traducción en /)).toHaveValue("");

    await act(async () => {
      resolve({ translation: "Utz" });
    });

    // The stale response belonged to the abandoned request; the UI should
    // still be idle, not showing that translation.
    expect(screen.getByLabelText(/^Traducción en /)).toHaveValue("");
    expect(
      screen.queryByText("Calentando el modelo. Puede tardar hasta un minuto."),
    ).not.toBeInTheDocument();
  });
});

describe("App copy and clear affordances", () => {
  beforeEach(() => {
    mockedTranslate.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("does not show the clear button when the input is empty", () => {
    render(<App />);
    expect(
      screen.queryByRole("button", { name: "Borrar el texto de entrada" }),
    ).not.toBeInTheDocument();
  });

  it("shows the clear button once the user types, and clicking it clears input, output, and error state, then refocuses the input", async () => {
    mockedTranslate.mockRejectedValueOnce(new TranslateNetworkError());
    render(<App />);
    const input = screen.getByLabelText(/^Texto en /);

    fireEvent.change(input, { target: { value: "Hola" } });
    expect(
      screen.getByRole("button", { name: "Borrar el texto de entrada" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));
    await screen.findByRole("button", { name: "Reintentar" });

    fireEvent.click(screen.getByRole("button", { name: "Borrar el texto de entrada" }));

    expect(input).toHaveValue("");
    expect(
      screen.queryByText("No se pudo conectar. Revisa tu conexión a internet e inténtalo de nuevo."),
    ).not.toBeInTheDocument();
    expect(screen.getByLabelText(/^Traducción en /)).toHaveValue("");
    expect(input).toHaveFocus();
    expect(
      screen.queryByRole("button", { name: "Borrar el texto de entrada" }),
    ).not.toBeInTheDocument();
  });

  it("ignores a stale in-flight response after Borrar clears the fields mid-request", async () => {
    const { promise, resolve } = deferred<{ translation: string }>();
    mockedTranslate.mockReturnValue(promise);
    render(<App />);

    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));
    await screen.findByText("Traduciendo…");

    fireEvent.click(screen.getByRole("button", { name: "Borrar el texto de entrada" }));

    expect(screen.getByLabelText(/^Texto en /)).toHaveValue("");
    expect(screen.getByLabelText(/^Traducción en /)).toHaveValue("");

    await act(async () => {
      resolve({ translation: "Utz" });
    });

    // The stale response belonged to the abandoned (now-cleared) request;
    // the UI should still be idle, not showing that translation.
    expect(screen.getByLabelText(/^Traducción en /)).toHaveValue("");
    expect(screen.queryByText("Traduciendo…")).not.toBeInTheDocument();
  });

  it("only shows the copy button after a successful translation", async () => {
    render(<App />);
    expect(screen.queryByRole("button", { name: "Copiar" })).not.toBeInTheDocument();

    mockedTranslate.mockResolvedValueOnce({ translation: "Utz" });
    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));

    await screen.findByRole("button", { name: "Copiar" });
  });

  it("copies the translation to the clipboard and shows a temporary confirmation", async () => {
    vi.useFakeTimers();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(window.navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });

    mockedTranslate.mockResolvedValueOnce({ translation: "Utz" });
    render(<App />);
    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Traducir" }));
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copiar" }));
    });

    expect(writeText).toHaveBeenCalledWith("Utz");
    expect(screen.getByRole("button", { name: "Copiado" })).toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });

    expect(screen.getByRole("button", { name: "Copiar" })).toBeInTheDocument();
  });

  it("shows a failure message if the clipboard write fails, then reverts", async () => {
    vi.useFakeTimers();
    const writeText = vi.fn().mockRejectedValue(new Error("denied"));
    Object.defineProperty(window.navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });

    mockedTranslate.mockResolvedValueOnce({ translation: "Utz" });
    render(<App />);
    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Traducir" }));
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copiar" }));
    });

    expect(screen.getByRole("button", { name: "No se pudo copiar" })).toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });

    expect(screen.getByRole("button", { name: "Copiar" })).toBeInTheDocument();
  });
});

describe("App direction swap", () => {
  beforeEach(() => {
    mockedTranslate.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("carries a successful translation into the input, flips direction, clears output, and moves the caret to the end", async () => {
    mockedTranslate.mockResolvedValueOnce({ translation: "Utz" });
    render(<App />);

    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Traducir" }));
    });
    await screen.findByLabelText(/^Traducción en /);
    expect(screen.getByLabelText(/^Traducción en /)).toHaveValue("Utz");

    fireEvent.click(screen.getByRole("button", { name: "Cambiar dirección" }));

    const input = screen.getByLabelText<HTMLTextAreaElement>(/^Texto en /);
    expect(input).toHaveValue("Utz");
    expect(input).toHaveFocus();
    expect(input.selectionStart).toBe(input.value.length);
    expect(input.selectionEnd).toBe(input.value.length);
    expect(screen.getByLabelText(/^Traducción en /)).toHaveValue("");
    expect(screen.getByLabelText(/^Traducción en /)).toHaveAttribute(
      "placeholder",
      "La traducción aparecerá aquí.",
    );
  });

  it("does not re-steal focus on a later, unrelated edit when the carried-over text equals the current input", async () => {
    mockedTranslate.mockResolvedValueOnce({ translation: "Hola" });
    render(<App />);
    const input = screen.getByLabelText<HTMLTextAreaElement>(/^Texto en /);
    const swapButton = screen.getByRole("button", { name: "Cambiar dirección" });

    fireEvent.change(input, { target: { value: "Hola" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Traducir" }));
    });
    expect(screen.getByLabelText(/^Traducción en /)).toHaveValue("Hola");

    // The carried-over translation ("Hola") is identical by value to the
    // current input ("Hola"): setInput() is a no-op React update here.
    fireEvent.click(swapButton);
    expect(input).toHaveFocus();

    // Move focus elsewhere (simulating the user clicking away), then make
    // an unrelated edit to the input.
    swapButton.focus();
    expect(swapButton).toHaveFocus();
    fireEvent.change(input, { target: { value: "Holaa" } });

    // A same-value carry-over must not leave the caret-reset behavior
    // "armed" for the next unrelated edit -- focus should stay wherever it
    // was, not jump back to the input.
    expect(swapButton).toHaveFocus();
  });

  it("leaves the input untouched when swapping from idle (nothing to carry over)", () => {
    render(<App />);
    const input = screen.getByLabelText(/^Texto en /);
    fireEvent.change(input, { target: { value: "Hola" } });

    fireEvent.click(screen.getByRole("button", { name: "Cambiar dirección" }));

    expect(input).toHaveValue("Hola");
    expect(screen.getByText("Kaqchikel")).toBeInTheDocument();
  });

  it("clears an error and leaves the input untouched when swapping", async () => {
    mockedTranslate.mockRejectedValueOnce(new TranslateNetworkError());
    render(<App />);

    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));
    await screen.findByRole("button", { name: "Reintentar" });

    fireEvent.click(screen.getByRole("button", { name: "Cambiar dirección" }));

    expect(screen.getByLabelText(/^Texto en /)).toHaveValue("Hola");
    expect(
      screen.queryByText("No se pudo conectar. Revisa tu conexión a internet e inténtalo de nuevo."),
    ).not.toBeInTheDocument();
  });

  it("always flips direction, even with nothing to carry over", () => {
    render(<App />);
    expect(screen.getByText("Español")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Cambiar dirección" }));
    expect(screen.getByText("Kaqchikel")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Cambiar dirección" }));
    expect(screen.getByText("Español")).toBeInTheDocument();
  });

  it("announces the direction change through the shared status region", () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Cambiar dirección" }));

    expect(screen.getByText("Dirección cambiada: Kaqchikel a Español.")).toBeInTheDocument();
  });
});

describe("App accessibility", () => {
  beforeEach(() => {
    mockedTranslate.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("gives the input and output distinct, direction-matching accessible names and lang attributes", () => {
    render(<App />);

    const input = screen.getByLabelText<HTMLTextAreaElement>("Texto en español");
    const output = screen.getByLabelText<HTMLTextAreaElement>("Traducción en kaqchikel");
    expect(input).toHaveAttribute("lang", "es");
    expect(output).toHaveAttribute("lang", "cak");

    fireEvent.click(screen.getByRole("button", { name: "Cambiar dirección" }));

    const swappedInput = screen.getByLabelText<HTMLTextAreaElement>("Texto en kaqchikel");
    const swappedOutput = screen.getByLabelText<HTMLTextAreaElement>("Traducción en español");
    expect(swappedInput).toHaveAttribute("lang", "cak");
    expect(swappedOutput).toHaveAttribute("lang", "es");
  });

  it("moves focus to the output textarea once a translation succeeds", async () => {
    mockedTranslate.mockResolvedValueOnce({ translation: "Utz" });
    render(<App />);

    fireEvent.change(screen.getByLabelText("Texto en español"), { target: { value: "Hola" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Traducir" }));
    });

    const output = screen.getByLabelText("Traducción en kaqchikel");
    expect(output).toHaveValue("Utz");
    expect(output).toHaveFocus();
  });

  it("does not steal focus from the input if the user is still typing when a translation resolves", async () => {
    const { promise, resolve } = deferred<{ translation: string }>();
    mockedTranslate.mockReturnValue(promise);
    render(<App />);

    const input = screen.getByLabelText<HTMLTextAreaElement>("Texto en español");
    fireEvent.change(input, { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));
    await screen.findByText("Traduciendo…");

    // The user keeps typing their next query while the previous request is
    // still in flight -- the input isn't disabled while translating.
    input.focus();
    fireEvent.change(input, { target: { value: "Hola, ¿cómo estás?" } });
    expect(input).toHaveFocus();

    await act(async () => {
      resolve({ translation: "Utz" });
    });

    // Focus (and whatever the user was typing) must stay on the input --
    // it must not get yanked into the now-populated output textarea.
    expect(input).toHaveFocus();
    expect(input).toHaveValue("Hola, ¿cómo estás?");
  });
});
