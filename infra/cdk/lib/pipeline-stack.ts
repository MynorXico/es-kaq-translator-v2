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
 */
export function buildPipelineApp(app: App, config: PipelineConfig): Stack {
  const stack = new Stack(app, "TraductorKaqchikel-Pipeline", {
    env: { account: config.toolingAccount, region: config.region },
  });

  const source = CodePipelineSource.connection(config.repoString, "main", {
    connectionArn: config.connectionArn,
  });

  const synthStep = new CodeBuildStep("Synth", {
    input: source,
    installCommands: ["corepack enable", "pnpm install --frozen-lockfile"],
    commands: ["pnpm --filter infra-cdk synth"],
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
    env: { account: config.devAccount, region: config.region },
  });
  pipeline.addStage(devStage);

  const qaStage = new TranslatorStage(stack, "Qa", {
    environmentName: "qa",
    env: { account: config.qaAccount, region: config.region },
  });
  pipeline.addStage(qaStage);

  const prodStage = new TranslatorStage(stack, "Prod", {
    environmentName: "prod",
    env: { account: config.prodAccount, region: config.region },
  });
  pipeline.addStage(prodStage, {
    pre: [new ManualApprovalStep("PromoteToProd")],
  });

  return stack;
}
