# apps/api

FastAPI service exposing the Spanish<->Kaqchikel translation API. Currently
a hello-world scaffold: `GET /health` and a `POST /v1/translate` stub that
returns a placeholder message — real model integration (calling a
SageMaker Serverless Inference endpoint) is tracked separately, see
[`docs/adr/0001-initial-architecture.md`](../../docs/adr/0001-initial-architecture.md).

Code, comments, and tests are written in English; API responses meant for
end users (translations, user-facing error messages) are written in
Spanish.

## Setup

This app manages its own Python environment with [uv](https://docs.astral.sh/uv/):

```sh
cd apps/api
uv sync
```

## Commands

```sh
uv run uvicorn app.main:app --reload   # run the dev server
uv run pytest                          # run tests (unit/ + integration/)
uv run ruff check .                    # lint
```

## Configuration

| Env var | Purpose | Default |
|---|---|---|
| `ALLOWED_ORIGINS` | Comma-separated list of origins allowed to call the API cross-origin (CORS), i.e. the deployed `apps/web` origin for the current environment. | `http://localhost:5173` (Vite's default dev server origin) |

Each environment's CDK deployment sets `ALLOWED_ORIGINS` to its own
`apps/web` origin once that domain exists (see
[`docs/runbooks/domain-and-dns.md`](../../docs/runbooks/domain-and-dns.md)).

Tests are split `tests/unit/` (pure functions, no I/O) and
`tests/integration/` (through the actual FastAPI app via `TestClient`).
See [`docs/testing.md`](../../docs/testing.md) for the full pyramid and
the TDD process this project follows.
