# ADR 0004: CDK Pipelines cross-account deployment

- Status: Accepted
- Date: 2026-09-15

## Context

Nothing has been deployed to AWS yet — verified zero CloudFormation
stacks and no `cdk bootstrap` in any of the four accounts
(`translator-tooling`/`-dev`/`-qa`/`-prod`). ADR 0001 called for a
self-mutating CDK Pipelines pipeline in `translator-tooling` promoting
through dev → qa → prod with manual approval only before prod, but left
the actual mechanics unspecified.

One real design problem needed solving: a CDK `Stage`/`Stack`'s
`env.account` must be a literal string known at synth time (CDK uses it
to resolve cross-account asset/deploy roles) — it can't be a
CloudFormation deploy-time token the way a normal config value could be.
That collides directly with this project's standing rule (`CLAUDE.md`):
never commit real AWS account IDs to this public repo. Ordinary
"reference a value from SSM at deploy time" (`StringParameter.valueForStringParameter`,
resolved via a CloudFormation dynamic reference) doesn't work here
specifically because the value is needed before/during synth, not after.

## Decision

- **Account IDs and the GitHub CodeStar Connection ARN live in SSM
  Parameter Store** in `translator-tooling`, under
  `/traductor-kaqchikel/accounts/{tooling,dev,qa,prod}` and
  `/traductor-kaqchikel/github-connection-arn`. Never as literal strings
  in git, never cached into a committed `cdk.context.json`.
- **`bin/app.ts` resolves them via a direct async SSM `GetParameter` call**
  (AWS SDK, not a CDK-construct-level lookup) in a top-level `async
  main()`, before constructing the CDK `App`. This works identically
  whether run locally (a human's `translator-tooling` SSO credentials) or
  inside the pipeline's own CodeBuild synth step (the CodeBuild project's
  execution role, granted `ssm:GetParameter` on exactly these parameter
  ARNs via `CodeBuildStep`'s `rolePolicyStatements`).
- **The actual pipeline/stage construction is a separate, synchronous,
  pure function** (`buildPipelineApp(app, config)` in
  `lib/pipeline-stack.ts`, taking already-resolved account
  IDs/region/connection ARN as plain arguments) — `main()`'s only job is
  fetching config and calling it. This is what makes it unit-testable:
  CDK assertion tests call `buildPipelineApp` directly with fake account
  ID strings and assert on the synthesized template, never touching real
  AWS or real account IDs.
- **Pipeline structure**: one self-mutating `CodePipeline` (from
  `aws-cdk-lib/pipelines`) in `translator-tooling`, source via a
  `CodeStarConnectionsSourceAction`/`CodePipelineSource.connection()`
  against `main` on GitHub, three stages each wrapping the existing
  `WebStack` (and future stacks as they're added):
  1. **Dev** — deploys automatically on every pipeline run.
  2. **QA** — deploys automatically immediately after Dev succeeds.
  3. **Prod** — behind a manual approval step; nothing deploys to
     `translator-prod` without an explicit click.
- **Cross-account trust**: each of `translator-dev`/`-qa`/`-prod` must be
  bootstrapped with `--trust <translator-tooling-account-id>` so the
  pipeline can assume deployment roles there; `translator-tooling` itself
  gets a plain self-bootstrap. This is a one-time manual/CLI step (not
  something CDK code itself does), tracked in a runbook.
- **GitHub source connection**: created via `aws codestar-connections
  create-connection` (scriptable), but AWS requires a human to complete
  the OAuth handshake for it in the Console — this one step cannot be
  automated away, by AWS's own design (there's no API for it).

## Consequences

- `infra/cdk/bin/app.ts` changes from "define `WebStack` directly, deploy
  per-environment by hand via `--context environmentName=`" to "define
  the pipeline stack, which internally deploys `WebStack` per stage." The
  old manual-per-environment-context deploy path is superseded — update
  `infra/cdk/README.md` accordingly.
- A new runbook (`docs/runbooks/cdk-pipelines-bootstrap.md`) documents
  the one-time setup: bootstrapping all 4 accounts with the right trust
  relationships, creating the SSM parameters, creating the CodeStar
  connection, completing its handshake, and the first pipeline deploy.
- Adding a new stack to an environment later means adding it inside the
  `Stage` construct in `lib/pipeline-stack.ts`, not writing new
  per-environment deploy scripts.
- The CodeBuild synth role's `ssm:GetParameter` grant must be scoped to
  exactly the parameter ARNs it needs — no wildcard `ssm:*` or
  account-wide resource scope.

## Alternatives considered

- **CDK construct-level SSM lookup** (`StringParameter.valueFromLookup`):
  rejected — this caches the resolved value into `cdk.context.json`,
  which would either commit the real account IDs to git or require
  remembering to gitignore and regenerate that file per environment,
  fragile either way.
- **Deploy-time CloudFormation dynamic reference**
  (`valueForStringParameter`): rejected — doesn't work for `env.account`,
  which CDK needs as a literal at synth time, not a token resolved during
  deployment.
- **Committing a gitignored local `accounts.json`** populated by hand per
  machine: rejected as the primary mechanism — it still needs to exist
  inside the CodeBuild synth container somehow (same problem, one layer
  down), and SSM Parameter Store already solves "share a small
  non-secret config value across a human's laptop and a CodeBuild job in
  the same account" natively.
