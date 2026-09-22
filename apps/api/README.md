# apps/api

FastAPI service exposing the Spanish<->Kaqchikel translation API. Currently
a hello-world scaffold: `GET /health` and a `POST /v1/translate` stub that
returns a placeholder message — real model integration (calling a
SageMaker Serverless Inference endpoint) is tracked separately, see
[`docs/adr/0001-initial-architecture.md`](../../docs/adr/0001-initial-architecture.md).

Deployed as a Lambda container image behind an API Gateway HTTP API (see
[`infra/cdk/lib/api-stack.ts`](../../infra/cdk/lib/api-stack.ts), issue
#96). `app/lambda_handler.py` wraps the same `app.main.app` FastAPI
instance with [Mangum](https://mangum.io/) so it runs unchanged locally
(`uvicorn`) and in Lambda (API Gateway HTTP API, payload format 2.0). The
`Dockerfile` builds the deployed image: it resolves this app's runtime
dependencies from `uv.lock` into a plain `requirements.txt` (uv itself
isn't shipped in the final image), then layers the app's own source on
top of AWS's official Lambda Python base image.

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
| `SAGEMAKER_ENDPOINT_NAME` | Name (not ARN) of the SageMaker Serverless Inference endpoint to invoke for real translations (issues #8/#9). Set by `ApiStack` to `traductor-kaqchikel-translate-<environment>` by default. The Lambda execution role is granted `sagemaker:InvokeEndpoint` scoped to exactly this endpoint's ARN. Not yet read by any code -- `/v1/translate` is still the placeholder stub; #9 should read it via `boto3`'s `sagemaker-runtime` client. | unset locally |

Each environment's CDK deployment sets `ALLOWED_ORIGINS` to its own
`apps/web` origin once that domain exists (see
[`docs/runbooks/domain-and-dns.md`](../../docs/runbooks/domain-and-dns.md)).
As of issue #96, `ApiStack` does not yet set `ALLOWED_ORIGINS` to a real
value (it stays on the localhost default) since `apps/web`'s custom domain
(issue #84) hasn't landed -- wiring that up is a natural follow-up once
both stacks' domains exist.

Tests are split `tests/unit/` (pure functions, no I/O) and
`tests/integration/` (through the actual FastAPI app via `TestClient`).
See [`docs/testing.md`](../../docs/testing.md) for the full pyramid and
the TDD process this project follows.
