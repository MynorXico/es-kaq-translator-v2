# infra/cdk

AWS CDK (TypeScript) application defining all infrastructure. The app is a
single self-mutating [CDK Pipelines](https://docs.aws.amazon.com/cdk/v2/guide/cdk_pipeline.html)
pipeline (`lib/pipeline-stack.ts`, deployed once into the `translator-tooling`
account) that promotes every environment stack through `Dev` -> `Qa` -> `Prod`
on every push to `main`, with a manual approval gate before `Prod`. See
[ADR 0004](../../docs/adr/0004-cdk-pipelines-deployment.md) for the full
design and [ADR 0001](../../docs/adr/0001-initial-architecture.md) for the
overall AWS layout.

Per-environment infrastructure is defined once in `lib/app-stage.ts`'s
`TranslatorStage` and instantiated per stage inside the pipeline — adding a
new stack to every environment means adding it there, not writing new
per-environment deploy scripts. Currently:

- `WebStack` — a private S3 bucket serving `apps/web`'s built SPA through a
  CloudFront distribution with SPA-routing fallback (see below for what's
  still missing: the custom domain).
- `DataStack` — private training data bucket + SageMaker execution role.
- `ApiStack` — `apps/api` (FastAPI) deployed as a Lambda container image
  behind a regional API Gateway REST API (v1), fronted by an AWS WAF
  `WebACL` with a per-IP rate-based rule (issue #151, ADR 0005 — a
  migration off the originally-used `HttpApi`, v2, since AWS WAF cannot
  attach to that at all). See its doc comment in `lib/api-stack.ts` for
  the environment-variable contract used to wire in the SageMaker
  Serverless Inference endpoint from issue #8/#9.

`WebStack` doesn't have its custom `app[-<env>].traductorkaqchikel.com`
domain wired up yet (reachable at its CloudFront distribution's default
`*.cloudfront.net` domain in the meantime) — see
[the domain runbook](../../docs/runbooks/domain-and-dns.md) and issue #99
for why (a cross-account DNS-write mechanism that doesn't exist yet, since
the hosted zone and this stack's CloudFront distribution live in different
AWS accounts).

The pipeline stack also includes `PipelineFailureAlerting`
(`lib/pipeline-alerting.ts`, issue #110): an EventBridge rule + SNS topic
that emails a maintainer within minutes if `TraductorKaqchikelPipeline`'s
execution fails. See
[the pipeline-failure-alerting runbook](../../docs/runbooks/pipeline-failure-alerting.md)
for how to configure the notification email and how to verify it's
working.

## Setup

From the repo root (this app is part of the pnpm workspace):

```sh
pnpm install
```

## Commands (run from this directory, or `pnpm --filter infra-cdk <script>` from the root)

```sh
pnpm build   # type-check
pnpm test    # run CDK assertion tests (see below)
pnpm synth   # synthesize CloudFormation templates (cdk.out/)
```

## Config: account IDs and the GitHub connection ARN

`bin/app.ts` needs 4 AWS account IDs (tooling/dev/qa/prod) and a GitHub
CodeStar/CodeConnections ARN before it can build the pipeline. Per
`CLAUDE.md`, none of these are ever literal strings in this repo. There are
two ways `bin/app.ts` resolves them, chosen via CDK context:

- **Real deploy** (default): fetched live from SSM Parameter Store in
  `translator-tooling` (`/traductor-kaqchikel/accounts/{tooling,dev,qa,prod}`,
  `/traductor-kaqchikel/github-connection-arn`) via a direct AWS SDK call.
  Requires AWS credentials for `translator-tooling` (a human's SSO session
  locally, or the pipeline's own CodeBuild synth role when self-mutating).
- **Mock accounts** (`--context useMockAccounts=true`): skips AWS entirely
  and uses hardcoded, obviously-fake placeholder account IDs/ARN. This is
  what CI's PR checks use — they must stay credential-free (see
  `.claude/agents/devops.md`) — and what you should use for any local synth
  check that isn't targeting a real account:

  ```sh
  pnpm synth --context useMockAccounts=true
  ```

## Building and deploying `apps/web`

The pipeline's synth step (`lib/pipeline-stack.ts`) builds `apps/web`
**once per environment** before `cdk synth` runs (issue #263) — `dev`,
`qa`, and `prod` each get their own `vite build`, each with their own
`VITE_API_BASE_URL` baked in and written to their own output directory
(`dist-dev`/`dist-qa`/`dist-prod`, see `lib/config.ts`'s
`webBuildOutputDirName`). This can't be a single shared build: Vite
inlines `VITE_API_BASE_URL` into the built JS bundle at build time, not
read at runtime, so one build can only ever point at one API. (This
replaced a single shared build that silently baked dev's API URL into
every environment — see issue #263 for how that surfaced as qa's web app
calling dev's API.)

Each environment's `VITE_API_BASE_URL` (see `apps/web/README.md`) comes
from `PipelineConfig.webApiBaseUrls.{dev,qa,prod}`, fetched by `bin/app.ts`
from the matching `/traductor-kaqchikel/api-urls/{dev,qa,prod}` SSM
parameter (issue #130) — it defaults to an empty string only when that
environment's parameter is absent, which now means a same-origin relative
request against that environment's own CloudFront distribution, failing
harmlessly and surfacing as a generic client-side error (not an actual
HTTP 404 — see the doc comment on `PipelineConfig.webApiBaseUrls` for
why). An environment without a real `ApiStack` yet (prod, for now) is
expected to have an absent SSM parameter until it gets one — see
[the api-url-ssm-parameters runbook](../../docs/runbooks/api-url-ssm-parameters.md)
for how to populate it once that changes.

A local `cdk synth`/`cdk deploy` needs `apps/web/dist-dev`,
`apps/web/dist-qa`, and `apps/web/dist-prod` to already exist (run the
three `VITE_API_BASE_URL=... pnpm --filter web exec vite build --outDir
dist-<env>` commands from `lib/pipeline-stack.ts`'s synth step by hand)
unless you're using `--context useMockAccounts=true`, which points every
environment at the same tiny fixture directory instead
(`test/fixtures/site`) so it never depends on a real build.

## Tests

CDK assertion tests (`pnpm test`, Vitest + `aws-cdk-lib/assertions`) are this
project's infra "unit" tier — see
[`docs/testing.md`](../../docs/testing.md) for the full pyramid. They call
`buildPipelineApp` directly with fake config values, never touching real AWS
or real account IDs. `test/mock-synth.test.ts` additionally shells out to
`cdk synth --context useMockAccounts=true` with every `AWS_*` env var
stripped, as an end-to-end guardrail that the CI-safe synth path actually
works without credentials.

## One-time bootstrap

Done — see [the bootstrap runbook](../../docs/runbooks/cdk-pipelines-bootstrap.md)
for the exact steps performed (account bootstrapping, SSM parameters, the
GitHub connection handshake, and the first pipeline deploy). The pipeline
is self-mutating from here on; a fresh `cdk deploy` of this stack is only
needed again if rebuilding from scratch.
