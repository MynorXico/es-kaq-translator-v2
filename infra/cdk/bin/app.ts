#!/usr/bin/env node
import { App } from "aws-cdk-lib";
import { GetParameterCommand, SSMClient } from "@aws-sdk/client-ssm";
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
 * Fetches real config from SSM Parameter Store in `translator-tooling`.
 * Used both for local synth/deploy (a human's tooling-account credentials)
 * and inside the pipeline's own CodeBuild synth step (its execution role,
 * scoped to exactly these parameter ARNs — see `lib/pipeline-stack.ts`).
 */
async function realConfig(): Promise<PipelineConfig> {
  const client = new SSMClient({ region: REGION });

  const [toolingAccount, devAccount, qaAccount, prodAccount, connectionArn] = await Promise.all([
    fetchSsmParameter(client, `${SSM_PARAMETER_PATH_PREFIX}/accounts/tooling`),
    fetchSsmParameter(client, `${SSM_PARAMETER_PATH_PREFIX}/accounts/dev`),
    fetchSsmParameter(client, `${SSM_PARAMETER_PATH_PREFIX}/accounts/qa`),
    fetchSsmParameter(client, `${SSM_PARAMETER_PATH_PREFIX}/accounts/prod`),
    fetchSsmParameter(client, `${SSM_PARAMETER_PATH_PREFIX}/github-connection-arn`),
  ]);

  return {
    toolingAccount,
    devAccount,
    qaAccount,
    prodAccount,
    region: REGION,
    connectionArn,
    repoString: MOCK_REPO_STRING,
  };
}

async function resolveConfig(app: App): Promise<PipelineConfig> {
  const useMockAccounts = app.node.tryGetContext("useMockAccounts") === "true";

  if (useMockAccounts) {
    // eslint-disable-next-line no-console
    console.error(
      "[infra-cdk] useMockAccounts=true: synthesizing with placeholder mock account IDs " +
        "and a fake connection ARN. This is only valid for CI/local synth checks — never " +
        "use this path for a real deployment.",
    );
    return mockConfig();
  }

  return realConfig();
}

async function main(): Promise<void> {
  const app = new App();
  const config = await resolveConfig(app);
  buildPipelineApp(app, config);
}

main().catch((error) => {
  // eslint-disable-next-line no-console
  console.error(error);
  process.exitCode = 1;
});
