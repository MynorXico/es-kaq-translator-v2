# infra/cdk

AWS CDK (TypeScript) application defining all infrastructure. The app is a
single self-mutating [CDK Pipelines](https://docs.aws.amazon.com/cdk/v2/guide/cdk_pipeline.html)
pipeline (`lib/pipeline-stack.ts`, deployed once into the `translator-tooling`
account) that promotes every environment stack through `Dev` -> `Qa` -> `Prod`
on every push to `main`, with a manual approval gate before `Prod`. See
[ADR 0004](../../docs/adr/0004-cdk-pipelines-deployment.md) for the full
design and [ADR 0001](../../docs/adr/0001-initial-architecture.md) for the
overall AWS layout.

Per-environment infrastructure (currently just `WebStack`, an S3 bucket
placeholder for the future SPA hosting) is defined once in
`lib/app-stage.ts`'s `TranslatorStage` and instantiated per stage inside the
pipeline — adding a new stack to every environment means adding it there,
not writing new per-environment deploy scripts.

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
