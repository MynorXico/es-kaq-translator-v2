# infra/cdk

AWS CDK (TypeScript) application defining all infrastructure. Currently a
minimal scaffold with a single `WebStack` (S3 bucket placeholder for the
future SPA hosting, parameterized by environment name). The CDK Pipelines
deployment pipeline and the remaining per-environment stacks (API, ML
hosting, data) will be added incrementally — see
[`docs/adr/0001-initial-architecture.md`](../../docs/adr/0001-initial-architecture.md).

## Setup

From the repo root (this app is part of the pnpm workspace):

```sh
pnpm install
```

## Commands (run from this directory, or `pnpm --filter infra-cdk <script>` from the root)

```sh
pnpm build   # type-check
pnpm test    # run stack assertion tests
pnpm synth   # synthesize CloudFormation templates (cdk.out/)
```

`cdk synth` defaults to the `dev` environment name; override with
`--context environmentName=qa` (or `prod`).

CDK assertion tests (`pnpm test`) are this project's infra "unit" tier —
see [`docs/testing.md`](../../docs/testing.md) for the full pyramid.
