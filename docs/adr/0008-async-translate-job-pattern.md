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
- **API Gateway `HttpApi` (what this project uses) has a hard,
  non-configurable 29-second timeout for Lambda proxy integrations** — an
  AWS platform limit, not a CDK/config knob. (`RestApi`, API Gateway's
  older v1 surface, actually allows configuring this timeout anywhere from
  50ms up to the same 29s ceiling — only the maximum is fixed there. This
  project uses `HttpApi`, where 29s is fixed with no lower-configurable
  range at all, so the conclusion below is unaffected either way.)
  `apps/api`'s Lambda is already
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
  SageMaker: `202 Accepted`, `{"job_id": "<uuid>"}`. `job_id` must be
  generated as an unguessable random UUIDv4 (Python's `uuid.uuid4()`),
  not a sequential or otherwise predictable value: since `GET
  /v1/translate-jobs/{job_id}` is unauthenticated, `job_id` is a de facto
  bearer credential for reading back that job's submitted/translated
  text — the first time this project persists and makes real
  user-submitted text network-retrievable at all (see the DynamoDB
  section below). A malformed (non-UUID) `job_id` on `GET` returns `400`,
  distinct from the `404` used for a well-formed but unknown/expired one.
- `GET /v1/translate-jobs/{job_id}` — polls the job record:
  - `200 {"status": "pending"}`
  - `200 {"status": "succeeded", "translation": "<text>"}`
  - `200 {"status": "failed", "status_code": <int>, "error": "<user-facing
    Spanish message>"}` — `status_code` is the numeric HTTP status
    `apps/api` would have returned synchronously for this failure today
    (e.g. 502 for a SageMaker error), carried explicitly in the job body
    since the outer HTTP response is always a `200` here. This is
    required, not optional: `apps/web/src/api.ts` needs it to reconstruct
    a faithful `TranslateHttpError(status_code, error)`, the same way
    `App.tsx`'s `classifyError` already distinguishes client- from
    server-caused failures today. Without it, every job failure would
    look identical to the client regardless of cause.
  - `404` if `job_id` is well-formed but unknown or has expired. Because
    DynamoDB's TTL attribute is a **best-effort background sweep**, not
    synchronous deletion, an item can still physically exist in the table
    past its nominal TTL for up to 48 hours (per DynamoDB's documented
    behavior). The `GET` handler must therefore compare the item's `ttl`
    attribute against the current time itself and return `404` for an
    expired-but-not-yet-swept item, rather than relying on the sweep
    alone to make expired items disappear.
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
  invoke config targets a **third handler entry point on the same shared
  Docker image** (consistent with the "one image, distinguished by
  handler" approach above — not a separate minimal Lambda/image) that
  marks the job `failed` in DynamoDB. This covers the case where the
  Worker Lambda is killed by its own timeout or crashes before its own
  code gets a chance to write a terminal status — Lambda's
  async-invocation failure destinations *do* fire on timeout (confirmed:
  a timeout counts as a function error for this purpose), so a client can
  never be left polling a `pending` job forever. The write from this
  handler and a (much less likely, but possible) near-simultaneous write
  from the Worker Lambda itself finishing right at its timeout boundary
  must not clobber each other — the write logic on both paths uses a
  DynamoDB conditional write (`ConditionExpression` guarding against
  overwriting an already-terminal `status`) so whichever write lands
  first wins and the second is a no-op, not a race.

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
`GET /v1/translate-jobs/{job_id}` until a terminal status or its own
overall timeout elapses.

This means `App.tsx`'s loading/`warming`/error state machine, its
`WARMING_UP_DELAY_MS` threshold, the "Calentando el modelo…" copy, and
the "Reintentar" button all keep working **unmodified** — they're driven
by the `Promise` `api.ts` returns, and that shape doesn't change, only
what happens underneath it. This is deliberate: the existing
warming-state UX is a good pattern already built and tested, and this
ADR's job is to give it a mechanism that can actually deliver on its own
promise, not to replace it.

`POLL_INTERVAL_MS` and `TIMEOUT_MS` must be named, exported constants
(matching today's `TIMEOUT_MS` pattern), not just prose ranges — both so
tests can advance fake timers deterministically instead of fitting
assertions to an undocumented implementation detail, and so the margin
between them is an explicit, checkable number rather than an assumption.
Worst-case timeline, stated explicitly rather than rounded away: the
Worker Lambda's own ceiling is `Duration.seconds(90)`; add enqueue
latency, the final poll's interval wait, response transit time, and
clock drift between the client and Lambda, and a real run finishing at,
say, 89 seconds server-side is not guaranteed to be observed
client-side before a 100s client timeout — producing a false "El modelo
está tardando más de lo esperado" for a job that in fact succeeded
seconds later, exactly the failure mode this ADR exists to prevent,
reintroduced at the margin. `TIMEOUT_MS` should therefore be set with
real headroom above the Worker Lambda's ceiling (e.g. 120s, not 100s),
and `POLL_INTERVAL_MS` on the order of 1.5–2s. Whether "Puede tardar
hasta un minuto" still holds, once real cold-start durations are
measured against the new ceiling, is a UX follow-up for
`ux`/`product-owner` — not decided by this ADR.

The following client behaviors must be decided and implemented as part
of this change, not left implicit:

- **A single failed poll `GET` must not fail the whole `translate()`
  promise immediately.** With up to ~30-60 polls on a real cold start, an
  occasional transient network blip during polling is statistically
  plausible — the client should retry the poll silently and only
  surface a `TranslateNetworkError` after several consecutive poll
  failures, or once its own overall timeout elapses, whichever comes
  first.
- **A `404` from `GET` maps to its own error, not a generic
  `TranslateHttpError` client-error message.** The job may be unknown
  because it genuinely expired mid-poll (a legitimate, if rare, case
  given the ~1 hour TTL) — `apps/web` should surface a message specific
  to "this translation took too long to retrieve," not the misleading
  "no se pudo traducir ese texto" wording used for actual client-input
  errors today.
- **A page reload while a job is in flight** currently means nobody ever
  reads that job's result once it's ready, while the Worker Lambda still
  runs the full (up to 90s) SageMaker call for nothing. This is accepted
  as an out-of-scope trade-off for now, the same way duplicate jobs from
  client retries are accepted below — `apps/web` does not persist
  `job_id` across a reload in this ADR's scope. Revisit (e.g. via
  `sessionStorage`) if this proves costly in practice.
- **The poll loop must be actually cancelled**, not just have its
  eventual result discarded, whenever the client abandons it — "Borrar",
  a direction swap, or a component unmount mid-poll. This is the
  poll-loop equivalent of today's single-`fetch` `AbortController`
  cancellation, and is required to preserve the existing e2e guarantees
  that a stale response never overwrites a newer one (already covered by
  `apps/web/e2e/translate.spec.ts`'s "ignores a stale in-flight response
  after Borrar/after swapping direction" tests) — without it, an
  abandoned poll loop also leaks an ongoing timer/interval, not just a
  discarded value.

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
  - `main.py`'s `CORSMiddleware` currently hardcodes `allow_methods=
    ["POST"]`. A plain cross-origin `GET` with no custom headers is a
    CORS "simple request" and wouldn't actually be blocked by this on its
    own (no preflight is triggered, and `allow_methods` only governs the
    preflight response) — but adding `"GET"` explicitly is still correct
    and should be done defensively rather than relying on that nuance.
  - `main.py`'s existing `@app.exception_handler(TranslationServiceError)`
    becomes dead code once `/v1/translate` retires (no remaining route
    calls `translate_via_sagemaker` directly) and must be removed, not
    left orphaned. `worker_handler.py` is not a FastAPI/Mangum app, so it
    needs a plain `try`/`except` around its `translate_via_sagemaker`
    call, writing `.message`/`status_code` into the job's `failed` record
    instead.
  - Given its size — new AWS service + two Lambdas + IAM/async-invoke/
    destination wiring + two new/one retired FastAPI route + a new worker
    handler + OpenAPI updates + full TDD coverage in `apps/api`, plus a
    poll-loop rewrite and tests in `apps/web` — this should be split into
    roughly three tickets rather than one: (1) `infra/cdk` construct work
    together with `apps/api`'s new routes/worker handler (its tests mock
    `boto3`, so no live infra dependency), and (2) `apps/web/src/api.ts`'s
    poll-loop rewrite as a separate follow-up once the real response
    shape exists to build against.
  - New test scenarios this pattern introduces, to be covered
    test-first alongside the above (not exhaustive, but each is a real,
    previously-nonexistent case): `POST /v1/translate-jobs` must be
    asserted to *never* call the SageMaker mock (that's the entire point
    of the split — a silent regression here would be easy to miss); a
    `GET` that stays `pending` past the client's own timeout; the `404`
    paths (unknown, expired-but-not-yet-swept, and malformed `job_id` as
    a distinct `400`); the `onFailure`-handler's own logic (invoked
    directly with a synthetic Lambda-destination failure-envelope
    fixture — AWS documents the shape — no real AWS needed); the
    DynamoDB conditional-write guard against a late Worker write
    clobbering an already-`onFailure`-written terminal status (and vice
    versa), using pre-seeded conflicting state rather than trying to
    force the actual race; and poll-loop cancellation when the client
    abandons a translation (Borrar/direction swap/unmount), extending
    the existing stale-response e2e tests in
    `apps/web/e2e/translate.spec.ts` to the new poll shape.
  - **The Worker-timeout → `onFailure` → job-marked-`failed` path, and the
    real behavior of Lambda's async-invoke machinery on a genuine
    timeout, cannot be proven in a fast, credential-free CI run** (`moto`'s
    async-invoke/destination support doesn't cover this, and it's a
    platform behavior, not application code). This must be verified with
    a real, scripted check in the `dev` AWS environment — e.g. forcing a
    Worker Lambda timeout deliberately (a temporarily short `Duration` on
    a test job, or a debug flag) and confirming the job record flips to
    `failed` within the expected window — as an explicit gate before
    promoting this feature to qa/prod, not something the CI suite is
    expected to prove on its own.
  - Post-deploy smoke verification for dev→qa→prod promotion needs a
    concrete enqueue-then-poll check (submit a job, poll until terminal,
    assert success) named as part of this follow-up, rather than being
    improvised per environment the way today's single-request smoke
    check might otherwise be copied forward unchanged.
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
