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
uv run pytest                          # run tests
uv run ruff check .                    # lint
```
