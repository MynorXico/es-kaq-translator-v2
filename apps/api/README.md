# apps/api

FastAPI service exposing the Spanish<->Kaqchikel translation API:
`GET /health` and `POST /v1/translate`, which calls the deployed SageMaker
Serverless Inference endpoint (`app/translation.py`, issue #9) — see
[`docs/adr/0001-initial-architecture.md`](../../docs/adr/0001-initial-architecture.md).

Deployed as a Lambda container image behind a regional API Gateway REST
API (v1), fronted by an AWS WAF `WebACL` rate-limiting rule (see
[`infra/cdk/lib/api-stack.ts`](../../infra/cdk/lib/api-stack.ts), issue
#151/ADR 0005 -- migrated off an API Gateway HTTP API, v2, from issue #96,
since AWS WAF cannot attach to that at all). `app/lambda_handler.py` wraps
the same `app.main.app` FastAPI instance with [Mangum](https://mangum.io/)
so it runs unchanged locally (`uvicorn`) and in Lambda (API Gateway REST
API proxy integration). The `Dockerfile` builds the deployed image: it
resolves this app's runtime
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
| `SAGEMAKER_ENDPOINT_NAME` | Name (not ARN) of the SageMaker Serverless Inference endpoint `app/translation.py` invokes via `boto3`'s `sagemaker-runtime` client for real translations (issues #8/#9). Set by `ApiStack`/`app-stage.ts`, which pass through `MlHostingStack`'s real endpoint name for environments that have one (dev, for now). The Lambda execution role is granted `sagemaker:InvokeEndpoint` scoped to exactly this endpoint's ARN. | unset locally (required to call `/v1/translate`) |

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

## Observability

Every `/v1/translate` request logs one structured JSON line via
`app/observability.py` (`app.request` logger): `direction`,
`input_length`, `latency_ms`, `status_code`, and `error_type` (`null` on
success). The raw request/response translation text is never logged --
metadata only (issue #47). Successful requests log at INFO, failed ones
at WARNING, so the two are easy to filter on separately.

`infra/cdk/lib/api-stack.ts` alarms on the API Gateway REST API's own
built-in CloudWatch metrics (5XX count, p90 latency) rather than a
log-based metric filter over these lines -- see that file's comments for
why. It also alarms on the AWS WAF `WebACL`'s `BlockedRequests` metric
(issue #151/ADR 0005), so a rate-limiting spike -- real abuse or an
over-aggressive threshold -- isn't invisible to those two alarms, which
only ever see traffic WAF already let through.
