import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";

// Deliberately lives in its own file: App.test.tsx mocks "./api" module-wide
// (translate/cancelTranslate as vi.fn()), which can't exercise the real
// cancelTranslate() -> TranslateCancelledError -> runTranslate() catch path
// (issue #261). Here "./api" is left unmocked; only fetch is stubbed.
//
// React 18 no longer warns about state updates on an unmounted component, so
// there's no console signal to assert on. Instead, every useState setter is
// wrapped to record calls made after the component has unmounted.
const tracker = vi.hoisted(() => ({
  unmounted: false,
  settersCalledAfterUnmount: 0,
}));

vi.mock("react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react")>();
  function useState<S>(initial?: S | (() => S)) {
    const [state, setState] = actual.useState(initial as S | (() => S));
    const trackedSetState = actual.useCallback(
      (value: import("react").SetStateAction<S>) => {
        if (tracker.unmounted) {
          tracker.settersCalledAfterUnmount += 1;
        }
        setState(value);
      },
      [setState],
    );
    return [state, trackedSetState] as const;
  }
  return { ...actual, default: { ...actual, useState }, useState };
});

// Imported after vi.mock (hoisted anyway) so App picks up the wrapped useState.
const { default: App } = await import("./App");

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("App unmount (real api.ts cancellation)", () => {
  beforeEach(() => {
    tracker.unmounted = false;
    tracker.settersCalledAfterUnmount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn((_url: string, init?: RequestInit) => {
        if (init?.method === "POST") {
          return Promise.resolve(jsonResponse({ job_id: "job-1" }, 202));
        }
        return Promise.resolve(jsonResponse({ status: "pending" }));
      }),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("doesn't update state after unmount when the real cancelTranslate() rejects the in-flight translation", async () => {
    const { unmount } = render(<App />);

    fireEvent.change(screen.getByLabelText(/^Texto en /), { target: { value: "Hola" } });
    fireEvent.click(screen.getByRole("button", { name: "Traducir" }));

    // Let the POST resolve so the translation is genuinely mid-poll.
    await act(async () => {
      await Promise.resolve();
    });
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/v1/translate-jobs"),
      expect.objectContaining({ method: "POST" }),
    );

    tracker.unmounted = true;
    unmount();

    // Flush the TranslateCancelledError rejection through runTranslate()'s
    // catch/finally.
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(tracker.settersCalledAfterUnmount).toBe(0);
  });
});
