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

describe("App", () => {
  it("renders the translator title and direction toggle", () => {
    render(<App />);
    expect(screen.getByText("Traductor Kaqchikel")).toBeInTheDocument();
    expect(screen.getByText("Español → Kaqchikel")).toBeInTheDocument();
  });

  it("shows a character counter that updates as the user types", () => {
    render(<App />);
    const input = screen.getByLabelText("Texto a traducir");

    expect(screen.getByText("0 / 2000")).toBeInTheDocument();

    fireEvent.change(input, { target: { value: "Hola" } });

    expect(screen.getByText("4 / 2000")).toBeInTheDocument();
  });

  it("disables translation and explains why once the input exceeds 2000 characters", () => {
    render(<App />);
    const input = screen.getByLabelText("Texto a traducir");
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

    expect(screen.getByText("Español → Kaqchikel")).toBeInTheDocument();
  });
});

describe("App translate flow", () => {
  beforeEach(() => {
    mockedTranslate.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  function deferred<T>() {
    let resolve!: (value: T) => void;
    let reject!: (reason: unknown) => void;
    const promise = new Promise<T>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    return { promise, resolve, reject };
  }

  it("shows the idle placeholder before any translation is submitted", () => {
    render(<App />);
    expect(screen.getByLabelText("Traducción")).toHaveAttribute(
      "placeholder",
      "La traducción aparecerá aquí.",
    );
  });

  it("shows a loading indicator while the request is in flight, then the result", async () => {
    const { promise, resolve } = deferred<{ translation: string }>();
    mockedTranslate.mockReturnValue(promise);
    render(<App />);

    fireEvent.change(screen.getByLabelText("Texto a traducir"), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));

    expect(await screen.findByText("Traduciendo…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Traducir" })).toBeDisabled();

    await act(async () => {
      resolve({ translation: "Utz" });
    });

    expect(await screen.findByLabelText("Traducción")).toHaveValue("Utz");
  });

  it("switches to a warming-up message if there's still no response after 4 seconds", async () => {
    vi.useFakeTimers();
    const { promise } = deferred<{ translation: string }>();
    mockedTranslate.mockReturnValue(promise);
    render(<App />);

    fireEvent.change(screen.getByLabelText("Texto a traducir"), { target: { value: "Hola" } });
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

    fireEvent.change(screen.getByLabelText("Texto a traducir"), { target: { value: "Hola" } });
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

    fireEvent.change(screen.getByLabelText("Texto a traducir"), { target: { value: "Hola" } });
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

    fireEvent.change(screen.getByLabelText("Texto a traducir"), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));

    expect(
      await screen.findByText("Hubo un problema en el servidor. Inténtalo de nuevo en unos minutos."),
    ).toBeInTheDocument();
  });

  it("shows a timeout error message when the client-side timeout elapses", async () => {
    mockedTranslate.mockRejectedValueOnce(new TranslateTimeoutError());
    render(<App />);

    fireEvent.change(screen.getByLabelText("Texto a traducir"), { target: { value: "Hola" } });
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

    fireEvent.change(screen.getByLabelText("Texto a traducir"), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));
    await screen.findByRole("button", { name: "Reintentar" });

    fireEvent.change(screen.getByLabelText("Texto a traducir"), {
      target: { value: "Hola, buenos días" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Reintentar" }));

    await waitFor(() => expect(mockedTranslate).toHaveBeenCalledTimes(2));
    expect(mockedTranslate).toHaveBeenLastCalledWith("Hola, buenos días", "es-to-cak");
    expect(await screen.findByLabelText("Traducción")).toHaveValue("Utz");
  });
});
