# ADR 0005: Rate limiting / abuse protection for the public API

- Status: Proposed
- Date: 2026-09-18

## Context

`apps/api`'s `POST /v1/translate` endpoint (see `apps/api/app/main.py`)
currently returns a stub response and has no rate limiting. Per ADR 0001,
it runs as a FastAPI Lambda container behind API Gateway; per ADR 0001's
inference decision, once #9 wires it to a real SageMaker Serverless
Inference endpoint, every request will incur a per-invocation SageMaker
cost on top of the Lambda/API Gateway cost. No `infra/cdk` stack for the
API exists yet (only `WebStack`, the pipeline, and `AppStage`), so this
ADR decides the mechanism before that stack is built, not after.

The product has no user accounts or authentication — anyone can call the
public API anonymously, by design (it is a public translator, not a
gated product). That constraint rules out any mechanism that assumes an
identified caller. The project is also serverless-first and cost-
conscious (ADR 0001), with a single solo maintainer who has to operate
whatever is chosen.

Four mechanisms were evaluated:

1. **API Gateway usage plans + API keys** — built-in to API Gateway, free
   beyond the API itself. Usage plans throttle/quota *per API key*, which
   assumes each caller can be handed a distinct key. This project has no
   registration/login flow, and building one just to hand out API keys
   would be a much bigger feature than "rate limit an endpoint" — out of
   scope and contrary to "no user accounts." The only alternative,
   issuing one shared key embedded in the public SPA, provides no real
   per-abuser isolation (it's visible in browser network calls, and
   throttles the entire public user base as one bucket the moment any
   single abuser exhausts it) — worse than no isolation at all, since it
   would also rate-limit legitimate users during an attack. Rejected: the
   entire feature is built around identifying callers, which this
   product deliberately does not do.

2. **AWS WAF rate-based rule in front of API Gateway** — a `WebACL` with
   a rate-based rule, associated directly with the API Gateway REST/HTTP
   API (regional APIs support WAF association natively, no CloudFront
   required). Aggregates and throttles by source IP (or a composite key)
   with no identity/API key needed, which matches anonymous public
   traffic exactly. Fully managed by AWS — no state store, queue, or
   cache to operate. Can later be extended with AWS Managed Rule Groups
   (e.g. bot control, IP reputation) using the same WebACL if abuse
   patterns beyond raw rate turn out to matter, without a redesign.

3. **CloudFront rate limiting** — WAF rate-based rules can also attach to
   a CloudFront distribution, but per ADR 0001 the API is served from its
   own API Gateway custom domain (`api.traductorkaqchikel.com`), not
   behind CloudFront (CloudFront in this project fronts only the static
   `apps/web` SPA via S3). Putting the API behind CloudFront just to gain
   WAF rate limiting would mean standing up a new CDN layer, a new
   distribution, and reasoning about caching semantics for a mutating
   `POST` endpoint (translation requests are not cacheable) — a bigger
   architectural change than the problem calls for, when WAF already
   attaches directly to API Gateway. Rejected: solves nothing that
   option 2 doesn't already solve, at the cost of a new CDN layer.

4. **Lambda-level token bucket** — implementing the rate limiter in the
   FastAPI application code. Individual Lambda invocations don't share
   memory, so a correct token bucket across concurrent invocations needs
   external shared state (DynamoDB or ElastiCache), read and written on
   every request. That means: a new stateful data store to provision,
   monitor, and pay for (DynamoDB's per-request cost is small, but it's
   a new failure mode and a new piece of infrastructure the solo
   maintainer now owns); custom cleanup/TTL logic for stale IP buckets;
   and reimplementing, in application code, behavior AWS WAF already
   provides as a managed service. Rejected: strictly more operational
   surface than option 2 for an equivalent or worse result, which
   directly conflicts with "solo maintainer who doesn't want to operate
   complex shared state."

## Decision

**Adopt an AWS WAF rate-based rule, associated with the API Gateway REST
API, as the abuse-protection mechanism for the public API**, layered on
top of API Gateway's own default account/stage-level throttling.

- A `WebACL` (scope `REGIONAL`) is created in `infra/cdk` alongside the
  future API stack and associated with the API Gateway stage.
- It contains one rate-based rule aggregating by source IP, with a
  starting threshold sized for "no real traffic yet" (WAF's rate-based
  rules evaluate over a trailing 5-minute window; an initial limit in the
  low hundreds of requests per 5 minutes per IP is a reasonable starting
  point) and `Action: Block` for IPs over the threshold. The exact number
  is a tuning parameter, not an architectural one, and can be adjusted
  without a new ADR.
- API Gateway's built-in stage-level throttling (requests/sec and burst,
  configurable with no extra service and no extra cost) is kept as a
  coarse, global backstop underneath the per-IP WAF rule — it doesn't
  distinguish callers, but it caps total concurrent load on the Lambda
  and, transitively, on the SageMaker Serverless endpoint, in case a
  distributed-enough abuse pattern spreads under the per-IP threshold.
- No API keys, usage plans, CloudFront distribution, or Lambda-managed
  shared-state store are introduced for this purpose.
- This ADR decides the mechanism only. The follow-up implementation
  ticket (filed once this ADR is accepted) covers the actual CDK
  construct, the chosen threshold, and wiring it into whichever stack
  ends up hosting the API Gateway resource. Per `qa`'s review of this
  ADR (issue #50), that follow-up ticket must also cover, as named
  requirements rather than incidental side effects:
  - **Verification, in two tiers** — a rate-based rule's trailing
    5-minute evaluation window isn't something CloudFormation synth can
    prove works, only that the resources are shaped correctly. (1) CDK
    assertion tests (`infra/cdk`'s existing Vitest + `aws-cdk-lib/
    assertions` tier, same as `test/api-stack.test.ts`) confirming the
    `WebACL` (`Scope: REGIONAL`), its rate-based rule (`AggregateKeyType:
    IP`, the configured `Limit`, `Action: Block`), and the
    `WebACLAssociation` targeting the right stage exist with the right
    properties — plus a negative assertion that no API key/usage-plan
    resources exist, guarding against drifting back toward the rejected
    option 1. (2) A one-time, manual **live** check post-deploy: `apps/
    api`'s existing `GET /health` route never calls SageMaker, and the
    WAF association is stage-level (covers every route), so sending
    enough requests from one source to `/health` in `dev` (never `qa`/
    `prod`, and never as a scheduled/automated check — see below) is a
    safe, near-zero-cost way to confirm the rule actually blocks at
    threshold, without either paying per-invocation SageMaker cost or
    risking real users. This is throwaway QA verification, like the
    existing post-deploy check that a real request through the deployed
    API returns a translation (`docs/testing.md`) — not a repeatable
    automated test: running it on a schedule, or against `qa`/`prod`,
    would itself look like (or actually be) the abuse pattern it exists
    to catch.
  - **A CloudWatch alarm on the WebACL's own metrics**: the `WebACL`'s
    `VisibilityConfig` must have `CloudWatchMetricsEnabled: true`, and a
    new alarm on its `BlockedRequests` metric (namespace `AWS/WAFV2`)
    is required, not optional — without it, neither a threshold set too
    low (blocking real users) nor a genuine abuse spike is visible to
    anyone until a complaint or a bill arrives. This is the concrete
    mechanism by which "the threshold is a tuning parameter" stays an
    actionable escape hatch rather than a decision nobody revisits; it's
    also what closes the gap noted below, where a WAF block currently
    has zero visibility in the existing `ServerErrorRateAlarm`/
    `HighLatencyAlarm` pair (#47), since WAF intercepts before the
    request reaches API Gateway's own `AWS/ApiGateway` metrics.
  - **A specified block-response shape, wired into `apps/web`'s error
    handling**: whether the block action keeps WAF's default `403` or
    uses a custom response (e.g. `429` with `Retry-After`), that choice
    must be explicit, and `apps/web/src/App.tsx`'s `classifyError` must
    gain a distinct `"rateLimited"` `TranslateErrorKind` (with its own
    Spanish copy, e.g. "Demasiadas solicitudes, inténtalo de nuevo en
    unos minutos.") for that status, plus a unit test for the new
    branch. Today, any non-5xx `TranslateHttpError` — including a WAF
    block — falls into the generic `"client"` kind and shows "revisa lo
    que escribiste" ("check what you wrote"), which is actively
    misleading for a rate-limited user whose input was never the
    problem.

## Consequences

- **New recurring cost**: AWS WAF bills a flat per-`WebACL` monthly fee
  plus a small per-rule monthly fee plus a per-million-requests
  inspection fee (order of a few US dollars per month total at this
  project's current near-zero traffic, growing slowly with traffic).
  Unlike API Gateway's own throttling or a Lambda-level bucket, this is a
  fixed cost that accrues even at zero traffic, for as long as the
  `WebACL` exists — this is the concrete cost/complexity trade-off this
  ADR surfaces for project-owner sign-off rather than deciding silently,
  consistent with how ADR 0003 was left `Proposed` for its own
  judgment-call trade-off. It is a small, bounded, and predictable cost
  compared to the unbounded downside of an unthrottled endpoint in front
  of a pay-per-invocation SageMaker Serverless Inference endpoint.
- The WAF `WebACL` and its rate-based rule must be defined in
  `infra/cdk` (CDK-native, per ADR 0004's conventions), not configured
  by hand in the console, so it survives redeploys and is reviewable in
  PRs like any other infra change.
- No new AWS account-tied values, secrets, or account IDs are introduced
  by this decision — the rate threshold and WebACL configuration are
  ordinary (non-secret) CDK construct properties, not values that need
  SSM Parameter Store indirection the way cross-account IDs did in
  ADR 0004.
- If usage patterns later reveal abuse that per-IP rate limiting alone
  doesn't catch (e.g. rotating IPs, credential-stuffing-style patterns
  once/if accounts are ever introduced), the same `WebACL` can be
  extended with AWS Managed Rule Groups (bot control, IP reputation)
  without revisiting this ADR's core mechanism choice.
- If/when the product ever grows real user accounts or a paid/metered
  tier, API Gateway usage plans + API keys (or a proper auth-based quota)
  become relevant again for per-customer quotas — that would be a
  separate, future ADR, not a reason to hold off on WAF now.
- **Accepted risk: shared-IP false positives.** Per-IP aggregation can
  collectively block a batch of distinct legitimate users behind one
  shared source IP (a school, library, or corporate NAT) during
  ordinary, non-abusive usage, not just during a real attack. This is
  not mitigated in v1 — it's surfaced here for the same reason the WAF
  cost trade-off above is, rather than being decided silently. If it
  proves to be a real problem in practice, AWS WAF supports a composite
  aggregate key (e.g. source IP + `User-Agent`) on the same rule, which
  narrows false positives without changing this ADR's core mechanism —
  the same "tuning, not architectural" escape hatch the threshold number
  itself already has.

## Alternatives considered

- **API Gateway usage plans + API keys**: rejected — designed around
  identifying individual callers, which conflicts with this being a
  public, account-free product; a single shared key gives no real
  per-abuser isolation and can collectively throttle legitimate users
  during an attack.
- **CloudFront rate limiting**: rejected for now — would require moving
  the API behind a new CloudFront distribution that ADR 0001 explicitly
  does not put it behind, for no capability WAF-on-API-Gateway doesn't
  already provide directly.
- **Lambda-level token bucket with DynamoDB/ElastiCache-backed shared
  state**: rejected — adds a new stateful data store, its own cost and
  failure modes, and custom cleanup logic, to reimplement in application
  code what AWS WAF already provides as a managed service; a poor fit
  for a solo maintainer optimizing for low operational overhead.
- **Do nothing until after #9 ships and real abuse is observed**:
  rejected as the primary plan — the ticket's own premise is that an
  open, unauthenticated endpoint in front of a pay-per-invocation
  SageMaker Serverless endpoint is a cost-exposure risk from the moment
  #9 ships, not something to react to after an unexpected bill. Standing
  up the mechanism (even at a conservative/high initial threshold) before
  or alongside #9 is cheap insurance relative to that downside.
