# apps/web

React + Vite single-page app for the Traductor Kaqchikel UI. Currently a
hello-world scaffold: a direction toggle (Español → Kaqchikel / Kaqchikel →
Español) and a translate box wired to `apps/api`'s `/v1/translate` stub
endpoint via `src/api.ts` (see the "Configuration" section below) — real
model output will replace `apps/api`'s placeholder response once the
fine-tuned model is deployed (ADR 0001).

User-facing copy in this app is written in Spanish (see the root
[`README.md`](../../README.md#language-conventions)); component code,
comments, and tests are written in English.

## Setup

From the repo root (this app is part of the pnpm workspace):

```sh
pnpm install
```

## Commands (run from this directory, or `pnpm --filter web <script>` from the root)

```sh
pnpm dev        # start the Vite dev server
pnpm build      # type-check and build for production
pnpm preview    # serve the production build locally
pnpm test       # run vitest (unit/component)
pnpm test:e2e   # run Playwright e2e tests (builds + serves the app itself)
pnpm lint       # run eslint
```

First time running e2e locally, install the browser once:
`pnpm exec playwright install chromium`. See
[`docs/testing.md`](../../docs/testing.md) for the test pyramid and the
TDD process this project follows.

## Configuration

| Env var | Purpose | Default |
|---|---|---|
| `VITE_API_BASE_URL` | Base URL of the `apps/api` deployment this build calls (`apps/web` and `apps/api` are separate CloudFront/API Gateway origins, per ADR 0001, so this is an absolute URL, not a relative path). | `http://localhost:8000` (the local `apps/api` dev server's default port) |

Each environment's CDK deployment sets `VITE_API_BASE_URL` to that
environment's `apps/api` origin once it's deployed (mirrors how `apps/api`
itself reads `ALLOWED_ORIGINS`, see `apps/api/README.md`).
