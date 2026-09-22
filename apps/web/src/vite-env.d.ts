/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Base URL of the `apps/api` deployment this frontend build talks to.
   * Set per-environment (dev/qa/prod) at build/deploy time; falls back to
   * the local `apps/api` dev server for `pnpm dev`/`pnpm build`.
   */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
