import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cancelTranslate,
  POLL_INTERVAL_MS,
  TIMEOUT_MS,
  translate,
  TranslateCancelledError,
  TranslateHttpError,
  TranslateJobNotFoundError,
  TranslateNetworkError,
  TranslateTimeoutError,
} from "./api";

// Matches apps/api/app/jobs.py's uuid4 job_id shape; the exact value never
// matters to these tests, just that it round-trips into the poll URL.
const JOB_ID = "3fa85f64-5717-4562-b3fc-2c963f66afa6";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status });
}

function mockEnqueueThenPoll(pollResponses: Array<() => Promise<Response>>): void {
  let pollCall = 0;
  vi.mocked(fetch).mockImplementation((url) => {
    if (String(url).endsWith("/v1/translate-jobs")) {
      return Promise.resolve(jsonResponse({ job_id: JOB_ID }, 202));
    }
    const respond = pollResponses[Math.min(pollCall, pollResponses.length - 1)];
    pollCall += 1;
    return respond();
  });
}

describe("translate", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("enqueues a job via POST /v1/translate-jobs and returns the translation once GET reports success", async () => {
    mockEnqueueThenPoll([() => Promise.resolve(jsonResponse({ status: "succeeded", translation: "Utz" }))]);

    const result = await translate("Hola", "es-to-cak");

    expect(result).toEqual({ translation: "Utz" });
    expect(fetch).toHaveBeenNthCalledWith(
      1,
      expect.stringMatching(/\/v1\/translate-jobs$/),
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ "Content-Type": "application/json" }),
        body: JSON.stringify({ text: "Hola", direction: "es-to-cak" }),
      }),
    );
    expect(fetch).toHaveBeenNthCalledWith(
      2,
      expect.stringMatching(new RegExp(`/v1/translate-jobs/${JOB_ID}$`)),
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("throws TranslateHttpError when the enqueue request itself returns a non-2xx status", async () => {
    vi.mocked(fetch).mockResolvedValue(jsonResponse({ error: "El texto no puede estar vacío." }, 422));

    await expect(translate("", "es-to-cak")).rejects.toMatchObject(new TranslateHttpError(422));
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("throws TranslateNetworkError when the enqueue request itself fails", async () => {
    vi.mocked(fetch).mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(translate("Hola", "es-to-cak")).rejects.toBeInstanceOf(TranslateNetworkError);
  });

  it("polls again after POLL_INTERVAL_MS while status is pending, then resolves once succeeded", async () => {
    vi.useFakeTimers();
    mockEnqueueThenPoll([
      () => Promise.resolve(jsonResponse({ status: "pending" })),
      () => Promise.resolve(jsonResponse({ status: "succeeded", translation: "Utz" })),
    ]);

    const pending = translate("Hola", "es-to-cak");
    await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS);

    await expect(pending).resolves.toEqual({ translation: "Utz" });
    // One POST + two GETs (pending, then succeeded).
    expect(fetch).toHaveBeenCalledTimes(3);
  });

  it("rejects with a TranslateHttpError carrying the job's status_code once the job fails", async () => {
    vi.useFakeTimers();
    mockEnqueueThenPoll([
      () =>
        Promise.resolve(
          jsonResponse({
            status: "failed",
            status_code: 502,
            error: "El servicio de traducción no está disponible en este momento.",
          }),
        ),
    ]);

    const pending = translate("Hola", "es-to-cak");
    const assertion = expect(pending).rejects.toMatchObject(new TranslateHttpError(502));
    await vi.advanceTimersByTimeAsync(0);
    await assertion;
  });

  it("rejects with TranslateJobNotFoundError when a poll GET returns 404", async () => {
    vi.useFakeTimers();
    mockEnqueueThenPoll([() => Promise.resolve(jsonResponse({ error: "No se encontró esa traducción." }, 404))]);

    const pending = translate("Hola", "es-to-cak");
    const assertion = expect(pending).rejects.toBeInstanceOf(TranslateJobNotFoundError);
    await vi.advanceTimersByTimeAsync(0);
    await assertion;
  });

  it("retries a poll silently after a single transient network failure, then resolves once succeeded", async () => {
    vi.useFakeTimers();
    mockEnqueueThenPoll([
      () => Promise.reject(new TypeError("Failed to fetch")),
      () => Promise.resolve(jsonResponse({ status: "succeeded", translation: "Utz" })),
    ]);

    const pending = translate("Hola", "es-to-cak");
    await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS);

    await expect(pending).resolves.toEqual({ translation: "Utz" });
  });

  it("throws TranslateNetworkError only after several consecutive poll failures, not on the first one", async () => {
    vi.useFakeTimers();
    mockEnqueueThenPoll([() => Promise.reject(new TypeError("Failed to fetch"))]);

    const pending = translate("Hola", "es-to-cak");
    const assertion = expect(pending).rejects.toBeInstanceOf(TranslateNetworkError);

    // Advance well past a single failure's poll interval, but with margin
    // under TIMEOUT_MS, so this is unambiguously the "several consecutive
    // failures" path, not the overall-timeout path.
    await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS * 5);
    await assertion;
    expect(TIMEOUT_MS).toBeGreaterThan(POLL_INTERVAL_MS * 5);
  });

  it("throws TranslateTimeoutError once TIMEOUT_MS elapses with no terminal status", async () => {
    vi.useFakeTimers();
    mockEnqueueThenPoll([() => Promise.resolve(jsonResponse({ status: "pending" }))]);

    const pending = translate("Hola", "es-to-cak");
    const assertion = expect(pending).rejects.toBeInstanceOf(TranslateTimeoutError);
    await vi.advanceTimersByTimeAsync(TIMEOUT_MS);
    await assertion;
  });

  it("gives TIMEOUT_MS real headroom above the Worker Lambda's 90s ceiling (ADR 0008)", () => {
    expect(TIMEOUT_MS).toBeGreaterThanOrEqual(120_000);
  });

  it("cancelTranslate stops the poll loop so no further requests are made", async () => {
    vi.useFakeTimers();
    mockEnqueueThenPoll([() => Promise.resolve(jsonResponse({ status: "pending" }))]);

    const pending = translate("Hola", "es-to-cak");
    pending.catch(() => {
      // Cancellation settles this promise (see the dedicated rejection test
      // below); this suite only cares that polling itself actually stops.
    });
    await vi.advanceTimersByTimeAsync(0);
    const callsBeforeCancel = vi.mocked(fetch).mock.calls.length;

    cancelTranslate();
    await vi.advanceTimersByTimeAsync(TIMEOUT_MS);

    expect(vi.mocked(fetch).mock.calls.length).toBe(callsBeforeCancel);
  });

  it("rejects the pending translate() promise with TranslateCancelledError once cancelled", async () => {
    vi.useFakeTimers();
    vi.mocked(fetch).mockImplementation((url) => {
      if (String(url).endsWith("/v1/translate-jobs")) {
        return Promise.resolve(jsonResponse({ job_id: JOB_ID }, 202));
      }
      // Simulate a poll GET that's still in flight at the moment of
      // cancellation -- cancelTranslate() must settle the outer promise on
      // its own, not wait on this ever resolving.
      return new Promise<Response>(() => {});
    });

    const pending = translate("Hola", "es-to-cak");
    await vi.advanceTimersByTimeAsync(0);

    cancelTranslate();

    await expect(pending).rejects.toBeInstanceOf(TranslateCancelledError);
  });

  it("is a no-op when nothing is in flight", () => {
    expect(() => cancelTranslate()).not.toThrow();
  });
});
