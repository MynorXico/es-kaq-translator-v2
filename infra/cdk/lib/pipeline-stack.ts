import { App, Stack } from "aws-cdk-lib";
import { PolicyStatement } from "aws-cdk-lib/aws-iam";
import { CodeBuildStep, CodePipeline, CodePipelineSource, ManualApprovalStep } from "aws-cdk-lib/pipelines";
import { TranslatorStage } from "./app-stage";

/**
 * Config resolved by `bin/app.ts` (from SSM in a real deploy, or hardcoded
 * mock values in CI/local synth) and passed into this pure, synchronous
 * function. See ADR 0004 for why account IDs/connection ARN must be
 * resolved *before* this function runs, rather than looked up from within
 * CDK constructs.
 */
export interface PipelineConfig {
  toolingAccount: string;
  devAccount: string;
  qaAccount: string;
  prodAccount: string;
  region: string;
  connectionArn: string;
  /** e.g. "MynorXico/es-kaq-translator-v2" */
  repoString: string;
  /**
   * Build-time `VITE_API_BASE_URL` baked into every `apps/web` build this
   * pipeline produces (see apps/web/README.md). Deliberately NOT wired to
   * the real `apps/api` deployment yet -- that's issue #96, being built in
   * parallel with no visibility into its exact CloudFormation output name
   * from here (see issue #84). Defaults to an empty string (a same-origin
   * relative request that safely 404s, rather than silently pointing a
   * deployed environment at a local dev server) until #96 lands, at which
   * point wiring the real URL through here is a small follow-up.
   */
  webApiBaseUrl?: string;
}

// Every SSM parameter this project owns lives under this path (ADR 0004).
// The synth step's IAM role is scoped to exactly this prefix, nothing wider.
const SSM_PARAMETER_PATH_PREFIX = "traductor-kaqchikel";

/**
 * Builds the self-mutating CDK Pipelines `Stack` (deployed once, by hand,
 * into `translator-tooling`) that promotes `WebStack` (and future stacks)
 * through Dev -> Qa -> Prod, with a manual approval gate before Prod.
 *
 * Pure and synchronous so it's unit-testable with fake config values —
 * see `test/pipeline-stack.test.ts`.
 *
 * `webSiteContentPath` is the built `apps/web` static site content (see
 * `WebStackProps.siteContentPath`) shared by every stage -- the synth step
 * below builds `apps/web` once, before running `cdk synth`, so all of
 * Dev/Qa/Prod deploy the exact same build output (issue #84; per-stage
 * builds only become meaningful once each environment has its own real
 * `apps/api` URL to bake in, see `webApiBaseUrl` above).
 */
export function buildPipelineApp(app: App, config: PipelineConfig, webSiteContentPath: string): Stack {
  const stack = new Stack(app, "TraductorKaqchikel-Pipeline", {
    env: { account: config.toolingAccount, region: config.region },
  });

  const source = CodePipelineSource.connection(config.repoString, "main", {
    connectionArn: config.connectionArn,
  });

  const synthStep = new CodeBuildStep("Synth", {
    input: source,
    installCommands: ["corepack enable", "pnpm install --frozen-lockfile"],
    // Build apps/web before synth so `WebStack`'s `BucketDeployment` asset
    // (lib/web-stack.ts) has real content to pick up from disk.
    commands: ["pnpm --filter web build", "pnpm --filter infra-cdk synth"],
    env: {
      VITE_API_BASE_URL: config.webApiBaseUrl ?? "",
    },
    primaryOutputDirectory: "infra/cdk/cdk.out",
    rolePolicyStatements: [
      new PolicyStatement({
        actions: ["ssm:GetParameter"],
        resources: [
          `arn:aws:ssm:${config.region}:${config.toolingAccount}:parameter/${SSM_PARAMETER_PATH_PREFIX}/*`,
        ],
      }),
    ],
  });

  const pipeline = new CodePipeline(stack, "Pipeline", {
    pipelineName: "TraductorKaqchikelPipeline",
    synth: synthStep,
    // Dev/Qa/Prod are separate AWS accounts, so the pipeline's artifact
    // bucket needs a KMS key to let those accounts' deploy roles decrypt
    // pipeline artifacts (required for any cross-account CodePipeline).
    crossAccountKeys: true,
  });

  const devStage = new TranslatorStage(stack, "Dev", {
    environmentName: "dev",
    webSiteContentPath,
    env: { account: config.devAccount, region: config.region },
  });
  pipeline.addStage(devStage);

  const qaStage = new TranslatorStage(stack, "Qa", {
    environmentName: "qa",
    webSiteContentPath,
    env: { account: config.qaAccount, region: config.region },
  });
  pipeline.addStage(qaStage);

  const prodStage = new TranslatorStage(stack, "Prod", {
    environmentName: "prod",
    webSiteContentPath,
    env: { account: config.prodAccount, region: config.region },
  });
  pipeline.addStage(prodStage, {
    pre: [new ManualApprovalStep("PromoteToProd")],
  });

  return stack;
}
