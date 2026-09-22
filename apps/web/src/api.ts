export type TranslationDirection = "es-to-cak" | "cak-to-es";

export interface TranslateResult {
  translation: string;
}

// Matches apps/api's ALLOWED_ORIGINS default (see apps/api/app/config.py),
// which assumes a locally-running FastAPI dev server on this port. Real
// environments override this via VITE_API_BASE_URL, set at build/deploy
// time once those origins exist (see apps/web/README.md).
const DEFAULT_API_BASE_URL = "http://localhost:8000";

// apps/api's Serverless Inference-backed endpoint can have real cold-start
// latency (see docs/adr/0001-initial-architecture.md); this is the hard
// client-side cutoff beyond which we give up rather than wait forever.
const TIMEOUT_MS = 60_000;

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

function getApiBaseUrl(): string {
  return import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL;
}

export async function translate(
  text: string,
  direction: TranslationDirection,
): Promise<TranslateResult> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/v1/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, direction }),
      signal: controller.signal,
    });
  } catch (error) {
    // Checked by `.name` rather than `instanceof DOMException`/`Error`: the
    // abort signal used to enforce TIMEOUT_MS can reject with an error from
    // a different realm than this module's globals (e.g. under jsdom in
    // tests), where cross-realm `instanceof` checks don't hold.
    if ((error as { name?: unknown })?.name === "AbortError") {
      throw new TranslateTimeoutError();
    }
    throw new TranslateNetworkError();
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    throw new TranslateHttpError(response.status);
  }

  return (await response.json()) as TranslateResult;
}
