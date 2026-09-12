# apps/web

React + Vite single-page app for the Traductor Kaqchikel UI. Currently a
hello-world scaffold: a direction toggle (Español → Kaqchikel / Kaqchikel →
Español) and a translate box wired to a placeholder API client
(`src/api.ts`) that will be swapped for a real call to `apps/api` once the
translation endpoint is live.

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
