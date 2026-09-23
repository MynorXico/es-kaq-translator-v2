#!/usr/bin/env node
import path from "node:path";
import { App } from "aws-cdk-lib";
import { GetParameterCommand, SSMClient } from "@aws-sdk/client-ssm";
import { webHostName } from "../lib/config";
import { buildPipelineApp, type PipelineConfig } from "../lib/pipeline-stack";

// Single source of truth for the region everything in this project lives in
// (tooling pipeline + all three deploy stages). See ADR 0004.
const REGION = "us-east-1";

// All of this project's SSM parameters live under this path (ADR 0004).
const SSM_PARAMETER_PATH_PREFIX = "/traductor-kaqchikel";

const MOCK_REPO_STRING = "MynorXico/es-kaq-translator-v2";

/**
 * Obviously-fake placeholder config used only when `useMockAccounts=true`
 * is passed as CDK context. This keeps `cdk synth` working with zero AWS
 * credentials, which PR checks in CI require (see `.claude/agents/devops.md`
 * and `.github/workflows/ci.yml`) — real account IDs are never available
 * there, and never should be.
 */
function mockConfig(): PipelineConfig {
  return {
    toolingAccount: "111111111111",
    devAccount: "222222222222",
    qaAccount: "333333333333",
    prodAccount: "444444444444",
    region: REGION,
    connectionArn: "arn:aws:codeconnections:us-east-1:111111111111:connection/mock-connection-id",
    repoString: MOCK_REPO_STRING,
    // Mock synth (CI's credential-free PR check, test/mock-synth.test.ts)
    // always "already validated" so ADR 0007's new-certificate guard never
    // blocks it -- it exercises WebStack's cert/domain wiring, not the
    // human DNS-validation workflow (see test/web-stack.test.ts and
    // test/app.test.ts for guard-specific coverage).
    previouslyValidatedWebCertDomains: {
      dev: webHostName("dev"),
      qa: webHostName("qa"),
      prod: webHostName("prod"),
    },
  };
}

async function fetchSsmParameter(client: SSMClient, name: string): Promise<string> {
  const response = await client.send(new GetParameterCommand({ Name: name }));
  const value = response.Parameter?.Value;
  if (!value) {
    throw new Error(`SSM parameter "${name}" exists but has no value`);
  }
  return value;
}

/**
 * Like `fetchSsmParameter`, but treats a missing parameter (not yet
 * created -- e.g. the first-ever certificate for an environment) as
 * "no previously-recorded value" rather than an error. Used only for the
 * ADR 0007 previously-validated-domain lookups, where absence is an
 * expected, fail-safe state (`WebStack`'s guard treats `undefined` the
 * same as a mismatch, never an automatic match).
 */
async function fetchOptionalSsmParameter(client: SSMClient, name: string): Promise<string | undefined> {
  try {
    const response = await client.send(new GetParameterCommand({ Name: name }));
    return response.Parameter?.Value;
  } catch (error) {
    if (error instanceof Error && error.name === "ParameterNotFound") {
      return undefined;
    }
    throw error;
  }
}

/**
 * Fetches real config from SSM Parameter Store in `translator-tooling`.
 * Used both for local synth/deploy (a human's tooling-account credentials)
 * and inside the pipeline's own CodeBuild synth step (its execution role,
 * scoped to exactly these parameter ARNs — see `lib/pipeline-stack.ts`).
 */
async function realConfig(): Promise<PipelineConfig> {
  const client = new SSMClient({ region: REGION });

  const [
    toolingAccount,
    devAccount,
    qaAccount,
    prodAccount,
    connectionArn,
    devWebCertDomain,
    qaWebCertDomain,
    prodWebCertDomain,
  ] = await Promise.all([
    fetchSsmParameter(client, `${SSM_PARAMETER_PATH_PREFIX}/accounts/tooling`),
    fetchSsmParameter(client, `${SSM_PARAMETER_PATH_PREFIX}/accounts/dev`),
    fetchSsmParameter(client, `${SSM_PARAMETER_PATH_PREFIX}/accounts/qa`),
    fetchSsmParameter(client, `${SSM_PARAMETER_PATH_PREFIX}/accounts/prod`),
    fetchSsmParameter(client, `${SSM_PARAMETER_PATH_PREFIX}/github-connection-arn`),
    // ADR 0007: absent (not-yet-created) is expected for a
    // never-yet-validated environment, not an error -- see
    // `fetchOptionalSsmParameter`.
    fetchOptionalSsmParameter(client, `${SSM_PARAMETER_PATH_PREFIX}/domains/dev-web-cert-domain`),
    fetchOptionalSsmParameter(client, `${SSM_PARAMETER_PATH_PREFIX}/domains/qa-web-cert-domain`),
    fetchOptionalSsmParameter(client, `${SSM_PARAMETER_PATH_PREFIX}/domains/prod-web-cert-domain`),
  ]);

  return {
    toolingAccount,
    devAccount,
    qaAccount,
    prodAccount,
    region: REGION,
    connectionArn,
    repoString: MOCK_REPO_STRING,
    previouslyValidatedWebCertDomains: {
      dev: devWebCertDomain,
      qa: qaWebCertDomain,
      prod: prodWebCertDomain,
    },
  };
}

/**
 * ADR 0007's explicit human acknowledgement that this synth is expected to
 * introduce a new/changed ACM certificate, e.g. `cdk deploy --context
 * newCertificateAck=true`. Read directly from CDK context regardless of the
 * `useMockAccounts` path, since it's an operator intent, not account
 * config -- see `WebStack`'s `assertCertificateChangeAcknowledged` for
 * where it's actually enforced.
 */
function resolveNewCertificateAck(app: App): boolean {
  return app.node.tryGetContext("newCertificateAck") === "true";
}

async function resolveConfig(app: App): Promise<PipelineConfig> {
  const useMockAccounts = app.node.tryGetContext("useMockAccounts") === "true";
  const newCertificateAck = resolveNewCertificateAck(app);

  if (useMockAccounts) {
    // eslint-disable-next-line no-console
    console.error(
      "[infra-cdk] useMockAccounts=true: synthesizing with placeholder mock account IDs " +
        "and a fake connection ARN. This is only valid for CI/local synth checks — never " +
        "use this path for a real deployment.",
    );
    return { ...mockConfig(), newCertificateAck };
  }

  const config = await realConfig();
  return { ...config, newCertificateAck };
}

/**
 * `WebStack`'s `BucketDeployment` (lib/web-stack.ts) needs a real directory
 * on disk at synth time. A real (`useMockAccounts` unset) synth always
 * builds `apps/web` first in the pipeline's own CodeBuild synth step (see
 * `lib/pipeline-stack.ts`) or must be built locally first for a manual
 * `cdk synth`/`deploy` -- so the real `apps/web/dist` output is expected to
 * exist by the time this runs. The mock-accounts path (CI's credential-free
 * PR check, `test/mock-synth.test.ts`) instead points at a tiny committed
 * fixture, so it never depends on a real Vite build having run.
 */
function resolveWebSiteContentPath(useMockAccounts: boolean): string {
  return useMockAccounts
    ? path.join(__dirname, "../test/fixtures/site")
    : path.join(__dirname, "../../../apps/web/dist");
}

async function main(): Promise<void> {
  const app = new App();
  const useMockAccounts = app.node.tryGetContext("useMockAccounts") === "true";
  const config = await resolveConfig(app);
  const webSiteContentPath = resolveWebSiteContentPath(useMockAccounts);
  buildPipelineApp(app, config, webSiteContentPath);
}

// Only auto-run when executed directly (`cdk synth`/`deploy`, via ts-node --
// see cdk.json), not when `test/app.test.ts` imports the functions above for
// unit testing (require.main !== module in that case).
if (require.main === module) {
  main().catch((error) => {
    // eslint-disable-next-line no-console
    console.error(error);
    process.exitCode = 1;
  });
}

// Exported for test/app.test.ts only -- bin/app.ts is otherwise a CLI
// entrypoint, not a library other stacks import from.
export {
  fetchOptionalSsmParameter,
  fetchSsmParameter,
  mockConfig,
  realConfig,
  resolveConfig,
  resolveNewCertificateAck,
};
