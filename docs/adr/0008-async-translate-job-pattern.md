# ADR 0008: Asynchronous job pattern for `/v1/translate` (cold-start handling)

- Status: Proposed
- Date: 2026-09-24

## Context

Issue #48 was originally scoped as a response-*contract* design task: once
#9 wired `apps/api` to the real SageMaker Serverless Inference endpoint,
give the frontend a distinguishable signal ("this is slow because of a
cold start", not a generic error) via a status code, header, or response
field once latency crosses a threshold.

Confirmed live against the deployed infrastructure (see issue #48's
pinned comment), the real constraint is deeper than that framing assumed:

- `apps/api`'s Lambda sits behind API Gateway `HttpApi`
  (`infra/cdk/lib/api-stack.ts`), calling a SageMaker Serverless Inference
  endpoint (`infra/cdk/lib/ml-hosting-stack.ts`) that scales to zero.
- **API Gateway has a hard, non-configurable 29-second timeout for Lambda
  proxy integrations** — true for both `HttpApi` and `RestApi`, an AWS
  platform limit, not a CDK/config knob. `apps/api`'s Lambda is already
  deliberately configured at `Duration.seconds(29)` (see the comment on
  that line in `api-stack.ts`) specifically to stay just under it.
- `apps/web` already has UI/UX built around the premise that a cold start
  can take up to roughly a minute: `App.tsx`'s `WARMING_UP_DELAY_MS =
  4000` plus "Calentando el modelo. Puede tardar hasta un minuto.", and
  `api.ts`'s `TIMEOUT_MS = 60_000`.
- Confirmed live: a genuine cold-start invocation causes the Lambda to
  hit its own 29s timeout, and the client receives a raw 500 well before
  either the 60-second client timeout or the "up to a minute" promise is
  ever reached — API Gateway kills the connection itself, regardless of
  what the Lambda/SageMaker side is doing.

This means the original proposal — a same-request signal once latency
crosses a threshold — **cannot work at all** for a cold start that
genuinely exceeds ~29 seconds: there is no way to keep one HTTP request
alive past that point to deliver *any* signal on it, distinguishable or
not. This needs an actual request-pattern change, not a response-shape
tweak.

### Verified: SageMaker has no async invoke for Serverless Inference

Checked rather than assumed (per the issue's explicit ask): SageMaker's
`InvokeEndpointAsync` API is not "an async version of Serverless
Inference" — it belongs to a structurally different SageMaker inference
option, **Asynchronous Inference** (its own `AsyncInferenceConfig` on the
endpoint config), which:

- Requires the request payload to be placed in S3 first (no inline
  payload like `invoke_endpoint`'s `Body=...`) and writes its output back
  to S3 (optionally with an SNS completion notification).
- Runs on provisioned instances, not Serverless Inference's per-ms
  billing — it *can* be configured to autoscale down to zero instances
  (GA since November 2024), but that's a different cost/latency model
  than the currently-deployed Serverless endpoint, not a mode switch on
  the same endpoint.
- Cannot be combined with Serverless Inference on one endpoint — they are
  mutually exclusive endpoint-config production-variant types.

See "Alternatives considered" for why this ADR doesn't adopt it.

## Decision

**Replace `/v1/translate`'s single synchronous request/response with an
enqueue-and-poll job pattern, entirely inside `apps/api`/`infra/cdk`.**
`MlHostingStack`'s SageMaker Serverless Inference endpoint is unchanged —
still invoked synchronously via `boto3`'s `invoke_endpoint`
(`apps/api/app/translation.py`), just from a Lambda that is no longer
bound by API Gateway's 29-second ceiling.

### 1. New resource shape

- `POST /v1/translate-jobs` — same request body as today
  (`{"text": ..., "direction": ...}`). Synchronously creates a job record
  and returns fast — well under 29s, since this path never calls
  SageMaker: `202 Accepted`, `{"job_id": "<uuid>"}`.
- `GET /v1/translate-jobs/{job_id}` — polls the job record:
  - `200 {"status": "pending"}`
  - `200 {"status": "succeeded", "translation": "<text>"}`
  - `200 {"status": "failed", "error": "<user-facing Spanish message>"}`
    (same `{"error": ...}` shape `TranslationServiceError` already uses)
  - `404` if `job_id` is unknown or has expired.
- The old `POST /v1/translate` synchronous contract (200 with
  `{"translation": ...}` inline) is retired. This is a breaking API
  change, judged acceptable now: the project is pre-1.0, `apps/web` is
  the only real consumer, and it ships in the same change as this ADR's
  implementation. OpenAPI docs must be updated to match (closing this
  issue's original acceptance criterion, in the new shape).

### 2. Two Lambda roles, one Docker image

Both share `apps/api`'s existing container image/codebase, distinguished
by handler entry point (e.g. the existing `app/lambda_handler.py`
Mangum/FastAPI entry for the Request Lambda, a new plain-Python
`app/worker_handler.py` for the Worker Lambda, which is invoked directly
with a job payload, not through FastAPI/Mangum):

- **Request Lambda** — fronts API Gateway, keeps today's
  `Duration.seconds(29)` (its work is trivial: a DynamoDB `PutItem`/
  `GetItem` and, on `POST`, one async Lambda invoke — no SageMaker call
  on this path at all). Handles both new routes.
- **Worker Lambda** — never fronted by API Gateway; invoked only via
  `lambda:InvokeFunction` with `InvocationType=Event` from the Request
  Lambda. Runs today's `translate_via_sagemaker` against the unchanged
  SageMaker endpoint, then writes `succeeded`/`failed` (+ result) back to
  the job's DynamoDB item. Configured with its own longer `Duration` —
  long enough to comfortably cover a real cold start with margin (e.g.
  `Duration.seconds(90)`, still far under Lambda's 15-minute hard
  ceiling) — since it is not constrained by API Gateway at all.
- The Worker Lambda's async event-invoke config sets
  **`retryAttempts: 0`** (CDK: `EventInvokeConfig`/
  `maxEventAge`/`retryAttempts` on the function). Lambda's async-invoke
  default is up to 2 automatic retries with 1- and 2-minute backoff
  windows between attempts — appropriate for idempotent background work,
  but wrong here: it would mean a job that failed/timed out sits
  `pending` for several extra minutes before Lambda gives up and fires
  the failure destination, well past the point the client has already
  given up. A single attempt, fail fast, is the right behavior for a
  request a human is actively waiting on (the existing "Reintentar"
  button already gives the user an explicit, cheap way to try again).
- An **`onFailure` Lambda Destination** on the Worker Lambda's async
  invoke config targets a small failure handler (can be a third minimal
  Lambda, or a distinct code path off the same image) that marks the job
  `failed` in DynamoDB. This covers the case where the Worker Lambda is
  killed by its own timeout or crashes before its own code gets a chance
  to write a terminal status — Lambda's async-invocation failure
  destinations *do* fire on timeout (confirmed: a timeout counts as a
  function error for this purpose), so a client can never be left
  polling a `pending` job forever.

### 3. New DynamoDB table (`TranslateJobsTable`)

On-demand billing, partition key `job_id`, a `ttl` attribute (e.g. 1 hour
after creation) so jobs self-expire with no manual cleanup. This is the
project's first use of DynamoDB — flagged explicitly since it's a new
AWS service beyond ADR 0001's original list — but it's a fully-managed,
pay-per-request store, the smallest stateful primitive that fits "one row
per job, read by key," and consistent with the project's serverless-first
bias.

Submitted translation text is now held, transiently, in this table until
its TTL expires. This is new: today the text is never persisted anywhere
(Lambda → SageMaker → response, nothing written to disk/DB). It doesn't
touch ADR 0002/`docs/data-governance.md` (that's about the training
corpus and model weights, not live user requests) and must stay that
way — this job table is never read by, or fed into, the training
pipeline. Default DynamoDB encryption-at-rest applies; the short TTL
bounds how long any given submission is retained.

### 4. `apps/web` changes stay internal to `api.ts`

`translate()`'s external contract — `translate(text, direction):
Promise<TranslateResult>`, and the existing `TranslateNetworkError`/
`TranslateTimeoutError`/`TranslateHttpError` types — is unchanged.
Internally, it now does `POST /v1/translate-jobs`, then polls
`GET /v1/translate-jobs/{job_id}` on a short interval (e.g. every
1.5–2s) until a terminal status or its own overall timeout elapses.

This means `App.tsx`'s loading/`warming`/error state machine, its
`WARMING_UP_DELAY_MS` threshold, the "Calentando el modelo…" copy, and
the "Reintentar" button all keep working **unmodified** — they're driven
by the `Promise` `api.ts` returns, and that shape doesn't change, only
what happens underneath it. This is deliberate: the existing
warming-state UX is a good pattern already built and tested, and this
ADR's job is to give it a mechanism that can actually deliver on its own
promise, not to replace it.

`TIMEOUT_MS` should move from 60s to a value with margin above the
Worker Lambda's new 90s ceiling (e.g. 100s), so the client doesn't give
up on a job that could still legitimately succeed server-side. Whether
"Puede tardar hasta un minuto" still holds, once real cold-start
durations are measured against the new ceiling, is a UX follow-up for
`ux`/`product-owner` — not decided by this ADR.

### 5. `MlHostingStack` is unchanged

The SageMaker endpoint stays Serverless Inference (ADR 0001), invoked the
same way as today. No dependency on SageMaker's separate Asynchronous
Inference endpoint type is introduced by this ADR.

## Consequences

- **Breaking API contract change**: `POST /v1/translate`'s synchronous
  shape is retired in favor of `POST` + `GET /v1/translate-jobs[/{id}]`.
  Acceptable now (pre-1.0, `apps/web` is the only real consumer, ships
  together), but must be reflected in the OpenAPI docs as part of the
  same change — this was this issue's own original acceptance criterion.
- **New AWS service**: DynamoDB, not previously used by this project.
  Small, fully-managed, on-demand billing; no new operational surface
  beyond what CDK already manages for other resources.
- **More moving parts**: two Lambda functions instead of one (plus a
  small failure-handler path), still fully serverless/pay-per-invocation.
  `infra/cdk` gains IAM wiring for Request→Worker async invoke and both
  Lambdas'→DynamoDB access, scoped least-privilege as this project
  already does elsewhere (e.g. `ApiStack`'s existing
  `sagemaker:InvokeEndpoint` scoping).
- **Extra latency floor even on a warm endpoint**: the client now does at
  least two round trips (enqueue + one poll) instead of one. Each is
  small; a warm translation should still resolve within one or two poll
  intervals (a few seconds), not materially changing the experience for
  the common case.
- **Cost**: DynamoDB on-demand and the extra Lambda invocations are
  negligible, low-single-digit-cents cost even at the high end, at this
  project's current low, bursty traffic (ADR 0001) — still strictly
  pay-per-use, preserving ADR 0001's zero-traffic-costs-nothing property
  (unlike the pre-warming alternative below). (DynamoDB's advertised free
  tier is specified in provisioned-capacity terms, so it's not a clean
  fit for the on-demand billing mode this ADR uses — the underlying
  "negligible" claim doesn't depend on it either way.) Poll volume
  depends heavily on scenario: 1-2 polls for an already-warm request, but
  up to roughly 30-60 polls for a full ~60-90s cold start at a 1.5-2s
  poll interval — still cheap in absolute dollar terms, but worth stating
  both cases explicitly since the cold-start case is this ADR's actual
  reason for existing.
- **ADR 0005 (rate limiting, Proposed)** will need to account for the new
  `GET` polling route when it's implemented — polls don't call SageMaker
  and are cheap, but the WAF rule's scope should cover both routes, not
  just `POST`. Left to that ADR's own follow-up implementation ticket.
- **Not addressed by this ADR**: reducing *how often* a cold start
  happens at all. This ADR only makes cold starts survivable by the
  request contract; it doesn't reduce their frequency. Pre-warming
  (considered below) could be layered on top later without conflicting
  with this decision.
- **Implementation follow-up** (not done by this ADR): a new/extended
  `infra/cdk` construct for the DynamoDB table, Worker Lambda, and IAM
  wiring (extending `ApiStack` or a new small stack, `dev`'s call);
  `apps/api/app` gains `worker_handler.py` and a `jobs.py`-style module
  for the DynamoDB read/write + job-id generation; `apps/web/src/api.ts`
  gains the poll loop. Exact module split is left to `dev`. `apps/api`'s
  existing test coverage for the synchronous `/v1/translate` contract
  must be replaced with test-first (per `docs/testing.md`) coverage for
  the new job-creation/poll endpoints and the worker handler — this is a
  named requirement of the implementation follow-up, not an incidental
  side effect of updating the OpenAPI docs. `TranslateJobsTable` and both
  Lambdas are per-environment resources, deployed fresh into each
  environment's own account by the CDK Pipelines promotion (dev/qa/prod
  each get their own table and functions), the same pattern `ApiStack`'s
  existing resources already follow — not a shared/singleton table across
  environments.
- **Observability trade-off**: splitting one logical translate request
  across two Lambdas means its two halves (request handling, SageMaker
  inference) land in separate CloudWatch log groups, correlated only by
  `job_id`. This is also an opportunity (issue #47's planned structured
  logging can cleanly separate request-handling latency from inference
  latency) but only if `job_id`-keyed structured logging is built in from
  the start of the implementation follow-up, rather than retrofitted once
  #47 lands.
- **Duplicate jobs from client retries**: a client retry on
  `POST /v1/translate-jobs` after a network blip (not a request the
  client itself made twice deliberately, but e.g. a dropped response to
  an otherwise-successful enqueue) creates a second, unrelated job with
  no dedup between them. Accepted as out of scope for now, given this
  project's current traffic and the low cost of an occasional duplicate
  translation — a documented decision, not an oversight, revisit if it
  becomes a real problem.

## Alternatives considered

- **Same-request response-contract signal (the issue's original
  proposal)**: rejected — confirmed not physically possible. No response
  can be delivered on a request whose connection API Gateway has already
  closed at 29 seconds. A distinguishable "still warming" signal still
  makes sense conceptually; it just has to arrive on a *different*
  request (the poll) than the one that started the job, which is exactly
  this ADR's mechanism, not a rejection of the original goal.

- **Switch the SageMaker endpoint to Asynchronous Inference** (SageMaker's
  own async endpoint type, its `AsyncInferenceConfig`, verified as
  distinct from and incompatible with Serverless Inference): rejected as
  the primary mechanism. It has real appeal as SageMaker's native
  answer to this exact shape (submit, get notified/poll), but: (a) its
  scale-to-zero autoscaling uses provisioned-instance billing semantics,
  a different cost/latency model than Serverless's per-ms billing, with
  no clear evidence its cold start is any faster; (b) it requires
  writing the request payload to S3 and reading the result back from S3
  (or subscribing to SNS), rather than an inline payload — `apps/api`
  would still need to build essentially the same "enqueue + poll" shape
  this ADR adopts, just with SageMaker's own S3/SNS plumbing standing in
  for DynamoDB + async Lambda invoke. Given equivalent complexity,
  keeping the already-deployed, already-tuned Serverless endpoint (its
  `maxConcurrency`/memory settings, ADR 0001's low/bursty-traffic
  rationale) and building the queueing layer in a component this project
  already owns and tests (`apps/api`) was judged simpler than migrating
  `MlHostingStack` to a different SageMaker inference option. Revisit if
  real usage data later shows Async Inference's cost/latency profile is
  clearly better.

- **SQS queue between the Request and Worker Lambdas**, instead of direct
  async (`Event`) Lambda invocation: rejected for now. Direct async
  invocation already gives one attempt (this ADR sets `retryAttempts: 0`)
  plus an `onFailure` destination, which is enough resilience at this
  project's current low, bursty traffic and its existing
  `maxConcurrency: 2` cap on the SageMaker endpoint. An SQS queue would
  add real value once true backpressure/ordering/queue-depth
  observability matters (e.g. many concurrent cold-start requests
  competing for `maxConcurrency: 2`), but that's speculative at today's
  traffic — revisit if Worker Lambda throttling becomes a real signal.

- **Step Functions to orchestrate enqueue → invoke → poll**: rejected —
  this is a two-step workflow (invoke SageMaker, write a result) with no
  branching, parallelism, or human-in-the-loop step to justify a state
  machine. It would add per-state-transition billing and a new
  orchestration primitive for a shape two Lambdas + one DynamoDB table
  already cover plainly.

- **Scheduled pre-warming ("ping") of the SageMaker endpoint during
  expected usage hours**: considered as a complementary mitigation, not
  adopted as part of this ADR's core mechanism, and not a substitute for
  it — even with pre-warming, whatever cold starts still get through
  still need a request pattern that can survive them, which is what this
  ADR already provides. Rejected as a *primary* fix, for now, because:
  - It reduces cold-start *frequency*, it doesn't eliminate them — the
    survivability mechanism above is still required regardless.
  - This is an early-stage OSS project with near-zero, unpredictable real
    traffic; there's no established "usage hours" pattern yet to
    schedule a ping against, and a guessed daily window could easily
    miss real (nighttime/international-timezone) visitors — exactly the
    audience a public project shouldn't leave cold-start-exposed anyway.
  - It reintroduces real recurring cost specifically at zero real usage,
    working against the entire reason ADR 0001 chose SageMaker Serverless
    (pay nothing when idle) — for a benefit ("fewer cold starts," not
    "none") that's hard to size without real traffic data.
  - Not rejected forever: worth revisiting once real usage data exists
    (clear daily/weekly patterns) to decide whether a cheap `EventBridge`
    schedule + a tiny ping Lambda is worth it at that point — a small,
    additive follow-up on top of, not a replacement for, this ADR.
