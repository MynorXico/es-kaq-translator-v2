import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  translate,
  TranslateHttpError,
  TranslateNetworkError,
  TranslateTimeoutError,
} from "./api";

describe("translate", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("posts the text and direction to the translate endpoint and returns the translation", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ translation: "Utz" }), { status: 200 }),
    );

    const result = await translate("Hola", "es-to-cak");

    expect(result).toEqual({ translation: "Utz" });
    expect(fetch).toHaveBeenCalledWith(
      expect.stringMatching(/\/v1\/translate$/),
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ "Content-Type": "application/json" }),
        body: JSON.stringify({ text: "Hola", direction: "es-to-cak" }),
      }),
    );
  });

  it("throws TranslateHttpError with the status for a 4xx response", async () => {
    vi.mocked(fetch).mockResolvedValue(new Response("{}", { status: 400 }));

    await expect(translate("Hola", "es-to-cak")).rejects.toMatchObject(
      new TranslateHttpError(400),
    );
  });

  it("throws TranslateHttpError with the status for a 5xx response", async () => {
    vi.mocked(fetch).mockResolvedValue(new Response("{}", { status: 500 }));

    await expect(translate("Hola", "es-to-cak")).rejects.toMatchObject(
      new TranslateHttpError(500),
    );
  });

  it("throws TranslateNetworkError when the fetch call itself fails", async () => {
    vi.mocked(fetch).mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(translate("Hola", "es-to-cak")).rejects.toBeInstanceOf(TranslateNetworkError);
  });

  it("throws TranslateTimeoutError once the 60s client-side timeout elapses", async () => {
    vi.useFakeTimers();
    vi.mocked(fetch).mockImplementation((_url, init) => {
      const signal = (init as RequestInit).signal;
      return new Promise((_resolve, reject) => {
        signal?.addEventListener("abort", () => {
          reject(new DOMException("The operation was aborted.", "AbortError"));
        });
      });
    });

    const pending = translate("Hola", "es-to-cak");
    const assertion = expect(pending).rejects.toBeInstanceOf(TranslateTimeoutError);
    await vi.advanceTimersByTimeAsync(60_000);
    await assertion;
  });
});
