export type TranslationDirection = "es-to-cak" | "cak-to-es";

export interface TranslateResult {
  translation: string;
}

// uvicorn's default port for `apps/api`'s dev server (see apps/api/README.md
// -- ALLOWED_ORIGINS there defaults to *this* app's dev origin,
// http://localhost:5173, for the reverse direction: allowing apps/web to
// call apps/api cross-origin). Real environments override this via
// VITE_API_BASE_URL, set at build/deploy time once those origins exist
// (see apps/web/README.md).
const DEFAULT_API_BASE_URL = "http://localhost:8000";

// ADR 0008: translate() now does POST /v1/translate-jobs, then polls
// GET /v1/translate-jobs/{job_id} until a terminal status or this overall
// timeout elapses. The Worker Lambda's own ceiling is Duration.seconds(90)
// (infra/cdk/lib/api-stack.ts); add enqueue latency, the final poll's own
// interval wait, response transit time, and clock drift between the client
// and Lambda, and a job that genuinely succeeds at, say, 89s server-side is
// not guaranteed to be observed client-side before a too-tight timeout --
// the exact false-timeout failure mode this ADR exists to prevent. 120s
// (not 100s, per the ADR's own worked example) keeps real headroom above
// the 90s ceiling.
export const TIMEOUT_MS = 120_000;

// ADR 0008: on the order of 1.5-2s -- responsive enough that an
// already-warm translation (the common case) resolves within one or two
// polls, without producing excessive poll volume on a full ~90s cold start
// (up to ~30-60 polls at 2s).
export const POLL_INTERVAL_MS = 2_000;

// ADR 0008: "an occasional transient network blip during polling is
// statistically plausible" over up to ~60 polls -- retry a failed poll GET
// silently rather than failing translate() on the first one, but still give
// up (and surface TranslateNetworkError) well before TIMEOUT_MS if the
// network genuinely appears to be down for the whole poll loop, not just
// one blip.
const MAX_CONSECUTIVE_POLL_FAILURES = 3;

/** Thrown when the request never reaches the server (offline, DNS, CORS, etc.). */
export class TranslateNetworkError extends Error {
  constructor() {
    super("Network error while calling the translate API");
    this.name = "TranslateNetworkError";
  }
}

/** Thrown when the client-side timeout elapses before any response arrives. */
export class TranslateTimeoutError extends Error {
  constructor() {
    super("Translate request timed out");
    this.name = "TranslateTimeoutError";
  }
}

/** Thrown when the server responds with a non-2xx status. */
export class TranslateHttpError extends Error {
  readonly status: number;

  constructor(status: number) {
    super(`Translate API responded with status ${status}`);
    this.name = "TranslateHttpError";
    this.status = status;
  }
}

/**
 * Thrown when a poll GET /v1/translate-jobs/{job_id} returns 404 -- the job
 * genuinely expired mid-poll (ADR 0008's ~1h TTL), or is otherwise unknown.
 * Deliberately distinct from TranslateHttpError: App.tsx's classifyError
 * must not describe this the same way as an actual client-input error.
 */
export class TranslateJobNotFoundError extends Error {
  constructor() {
    super("Translate job not found or expired");
    this.name = "TranslateJobNotFoundError";
  }
}

/**
 * Thrown to settle a still-pending translate() promise when cancelTranslate()
 * is called mid-poll (ADR 0008). App.tsx always bumps its own requestIdRef
 * before calling cancelTranslate(), so this is caught and discarded as a
 * stale response, the same way any other translate() rejection is -- it's
 * exported mainly so cancellation itself is directly testable.
 */
export class TranslateCancelledError extends Error {
  constructor() {
    super("Translate request was cancelled");
    this.name = "TranslateCancelledError";
  }
}

function getApiBaseUrl(): string {
  return import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL;
}

function isAbortError(error: unknown): boolean {
  // Checked by `.name` rather than `instanceof DOMException`/`Error`: the
  // abort signal used for cancellation/TIMEOUT_MS can reject with an error
  // from a different realm than this module's globals (e.g. under jsdom in
  // tests), where cross-realm `instanceof` checks don't hold.
  return (error as { name?: unknown })?.name === "AbortError";
}

interface JobCreatedPayload {
  job_id: string;
}

interface JobStatusPayload {
  status: "pending" | "succeeded" | "failed";
  translation?: string;
  status_code?: number;
  error?: string;
}

// Only one translation is ever in flight at a time (the UI disables
// "Traducir" while isTranslating is true) -- this tracks that single
// session so cancelTranslate() can actually stop its poll loop/timers and
// abort any in-flight request, not just leave its eventual result to be
// discarded by the caller.
interface PollSession {
  settled: boolean;
  controller: AbortController;
  timers: Set<ReturnType<typeof setTimeout>>;
  reject: (error: unknown) => void;
}

let currentSession: PollSession | null = null;

/**
 * Actually cancels the poll loop backing the most recent translate() call,
 * if one is still in flight: clears its pending timers (so no further
 * GET /v1/translate-jobs/{job_id} calls happen), aborts any request that's
 * currently in flight, and settles translate()'s own promise with
 * TranslateCancelledError. A no-op if nothing is in flight.
 *
 * Required (ADR 0008) on "Borrar", a direction swap, or component unmount
 * mid-poll -- without it, an abandoned poll loop would keep re-scheduling
 * itself and polling the server for up to TIMEOUT_MS after the UI has
 * already moved on.
 */
export function cancelTranslate(): void {
  const session = currentSession;
  if (!session || session.settled) {
    return;
  }
  session.settled = true;
  currentSession = null;
  session.controller.abort();
  session.timers.forEach((id) => clearTimeout(id));
  session.timers.clear();
  session.reject(new TranslateCancelledError());
}

export async function translate(
  text: string,
  direction: TranslationDirection,
): Promise<TranslateResult> {
  const controller = new AbortController();
  const session: PollSession = {
    settled: false,
    controller,
    timers: new Set(),
    reject: () => {},
  };
  currentSession = session;

  return new Promise<TranslateResult>((resolve, reject) => {
    session.reject = reject;

    function settle(action: () => void): void {
      if (session.settled) {
        return;
      }
      session.settled = true;
      session.timers.forEach((id) => clearTimeout(id));
      session.timers.clear();
      if (currentSession === session) {
        currentSession = null;
      }
      action();
    }

    const overallTimeoutId = setTimeout(() => {
      controller.abort();
      settle(() => reject(new TranslateTimeoutError()));
    }, TIMEOUT_MS);
    session.timers.add(overallTimeoutId);

    function schedulePoll(jobId: string): void {
      if (session.settled) {
        return;
      }
      const timerId = setTimeout(() => {
        void poll(jobId);
      }, POLL_INTERVAL_MS);
      session.timers.add(timerId);
    }

    let consecutivePollFailures = 0;

    async function poll(jobId: string): Promise<void> {
      if (session.settled) {
        return;
      }

      let response: Response;
      try {
        response = await fetch(`${getApiBaseUrl()}/v1/translate-jobs/${jobId}`, {
          method: "GET",
          signal: controller.signal,
        });
      } catch (error) {
        if (isAbortError(error)) {
          // Either TIMEOUT_MS or cancelTranslate() already aborted this
          // request and has its own settle() call in flight -- nothing more
          // to do here.
          return;
        }
        consecutivePollFailures += 1;
        if (consecutivePollFailures >= MAX_CONSECUTIVE_POLL_FAILURES) {
          settle(() => reject(new TranslateNetworkError()));
          return;
        }
        schedulePoll(jobId);
        return;
      }

      if (response.status === 404) {
        settle(() => reject(new TranslateJobNotFoundError()));
        return;
      }
      if (!response.ok) {
        settle(() => reject(new TranslateHttpError(response.status)));
        return;
      }

      consecutivePollFailures = 0;
      const job = (await response.json()) as JobStatusPayload;

      if (job.status === "succeeded") {
        settle(() => resolve({ translation: job.translation as string }));
        return;
      }
      if (job.status === "failed") {
        settle(() => reject(new TranslateHttpError(job.status_code as number)));
        return;
      }
      schedulePoll(jobId);
    }

    (async () => {
      let response: Response;
      try {
        response = await fetch(`${getApiBaseUrl()}/v1/translate-jobs`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, direction }),
          signal: controller.signal,
        });
      } catch (error) {
        if (isAbortError(error)) {
          return;
        }
        settle(() => reject(new TranslateNetworkError()));
        return;
      }

      if (!response.ok) {
        settle(() => reject(new TranslateHttpError(response.status)));
        return;
      }

      const created = (await response.json()) as JobCreatedPayload;
      void poll(created.job_id);
    })();
  });
}
