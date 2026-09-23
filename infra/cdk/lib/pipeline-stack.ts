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
   * relative request against this environment's own CloudFront
   * distribution, rather than silently pointing a deployed environment at
   * a local dev server) until #96 lands, at which point wiring the real
   * URL through here is a small follow-up. Note this doesn't actually
   * fail with a 404: `WebStack`'s SPA-routing fallback (403/404 ->
   * `index.html` with a 200) means the client gets a 200 with the app's
   * own HTML body instead, which then fails harmlessly client-side
   * (`response.json()` throws parsing HTML as JSON, surfaced to the user
   * as a generic error by `apps/web/src/App.tsx`'s existing error
   * handling) -- not a real HTTP 404.
   */
  webApiBaseUrl?: string;
  /**
   * Per-environment domain name that had a successfully DNS-validated ACM
   * certificate as of the last runbook update (ADR 0007), fetched by
   * `bin/app.ts` from `/traductor-kaqchikel/domains/{env}-web-cert-domain`
   * SSM parameters. An absent entry for a given environment means "no
   * certificate has ever been validated for it yet" -- `WebStack`'s
   * synth-time guard (`assertCertificateChangeAcknowledged`) treats that
   * the same as a mismatch, not an automatic match.
   */
  previouslyValidatedWebCertDomains?: Partial<Record<"dev" | "qa" | "prod", string>>;
  /**
   * Mirrors CDK context `newCertificateAck` (ADR 0007): an explicit human
   * acknowledgement that this synth is expected to introduce a new/changed
   * ACM certificate for at least one environment, and that a human will
   * watch the resulting isolated `cdk deploy` for the manual DNS
   * validation step it will block on. Defaults to `false`.
   */
  newCertificateAck?: boolean;
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
    previouslyValidatedWebDomainName: config.previouslyValidatedWebCertDomains?.dev,
    newCertificateAck: config.newCertificateAck,
    env: { account: config.devAccount, region: config.region },
  });
  pipeline.addStage(devStage);

  const qaStage = new TranslatorStage(stack, "Qa", {
    environmentName: "qa",
    webSiteContentPath,
    previouslyValidatedWebDomainName: config.previouslyValidatedWebCertDomains?.qa,
    newCertificateAck: config.newCertificateAck,
    env: { account: config.qaAccount, region: config.region },
  });
  pipeline.addStage(qaStage);

  const prodStage = new TranslatorStage(stack, "Prod", {
    environmentName: "prod",
    webSiteContentPath,
    previouslyValidatedWebDomainName: config.previouslyValidatedWebCertDomains?.prod,
    newCertificateAck: config.newCertificateAck,
    env: { account: config.prodAccount, region: config.region },
  });
  pipeline.addStage(prodStage, {
    pre: [new ManualApprovalStep("PromoteToProd")],
  });

  return stack;
}
