import { App, Stack } from "aws-cdk-lib";
import { PolicyStatement } from "aws-cdk-lib/aws-iam";
import { CodeBuildStep, CodePipeline, CodePipelineSource, ManualApprovalStep } from "aws-cdk-lib/pipelines";
import { TranslatorStage } from "./app-stage";
import { WEB_ENVIRONMENTS, type WebEnvironment, webBuildOutputDirName } from "./config";
import { PipelineFailureAlerting } from "./pipeline-alerting";

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
   * Per-environment `apps/api` HTTP API Gateway URL (issue #96/#123),
   * fetched by `bin/app.ts` from `/traductor-kaqchikel/api-urls/{env}` SSM
   * parameters -- the same async SSM-Parameter-Store-at-synth-time pattern
   * as `previouslyValidatedWebCertDomains` below (ADR 0004/0007), and for
   * the same reason: `ApiStack`'s real URL is a CloudFormation-resolved
   * value that doesn't exist until *after* that stack deploys, so it can't
   * be read from within this synchronous, pre-deploy synth step the way
   * `MlHostingStack`'s plain constructed endpoint name was threaded into
   * `ApiStack` for issue #9. An absent entry means no real `ApiStack` has
   * been deployed for that environment yet (qa/prod, for now).
   *
   * Each environment's value is baked into that environment's *own* build
   * of `apps/web` (issue #263): the synth step below runs `vite build`
   * once per environment in `WEB_ENVIRONMENTS` (lib/config.ts), each with
   * its own `VITE_API_BASE_URL` and its own output directory (see
   * `webSiteContentPaths`'s doc comment above `buildPipelineApp`) -- a
   * single shared build can't work here because Vite inlines this env var
   * into the built JS bundle at build time, not read at runtime.
   *
   * A missing value for a given environment falls back to an empty string
   * (a same-origin relative request against that environment's own
   * CloudFront distribution, rather than silently pointing a deployed
   * environment at a local dev server, or -- the bug this replaced,
   * issue #263 -- at a *different* environment's real API). Note this
   * doesn't actually fail with a 404:
   * `WebStack`'s SPA-routing fallback (403/404 -> `index.html` with a 200)
   * means the client gets a 200 with the app's own HTML body instead,
   * which then fails harmlessly client-side (`response.json()` throws
   * parsing HTML as JSON, surfaced to the user as a generic error by
   * `apps/web/src/App.tsx`'s existing error handling) -- not a real HTTP
   * 404.
   */
  webApiBaseUrls?: Partial<Record<"dev" | "qa" | "prod", string>>;
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
  /**
   * Email address notified when `TraductorKaqchikelPipeline`'s execution
   * fails (issue #110), fetched by `bin/app.ts` from
   * `/traductor-kaqchikel/alerts/pipeline-failure-email` SSM parameter.
   * Never a literal real address in this repo (CLAUDE.md) -- an absent
   * value still creates the failure-notification SNS topic and EventBridge
   * rule (so nothing is silently skipped), just without a subscriber yet.
   */
  pipelineFailureNotificationEmail?: string;
}

// Every SSM parameter this project owns lives under this path (ADR 0004).
// The synth step's IAM role is scoped to exactly this prefix, nothing wider.
const SSM_PARAMETER_PATH_PREFIX = "traductor-kaqchikel";

// The pipeline's own name, used both for `CodePipeline`'s `pipelineName`
// prop and to filter the EventBridge failure-alerting rule (issue #110) --
// kept as one constant so the two can never drift apart.
const PIPELINE_NAME = "TraductorKaqchikelPipeline";

/**
 * Builds the self-mutating CDK Pipelines `Stack` (deployed once, by hand,
 * into `translator-tooling`) that promotes `WebStack` (and future stacks)
 * through Dev -> Qa -> Prod, with a manual approval gate before Prod.
 *
 * Pure and synchronous so it's unit-testable with fake config values —
 * see `test/pipeline-stack.test.ts`.
 *
 * `webSiteContentPaths` is the built `apps/web` static site content *per
 * environment* (see `WebStackProps.siteContentPath`) -- one absolute path
 * per `WebEnvironment`, each pointing at that environment's own `vite
 * build` output (issue #263). `BucketDeployment`'s underlying CDK asset
 * model needs real, already-built content on disk at synth time, which is
 * why these three builds all happen inside the synth step below, *before*
 * `cdk synth` runs, rather than via a per-stage `CodeBuildStep` added with
 * `pipeline.addStage(stage, { pre: [...] })`: a stage-scoped step only
 * runs *after* synth has already produced (and asset-hashed) the
 * CloudFormation template that references this content, so its output
 * could never reach `WebStack`'s asset. Building three times inside one
 * CodeBuildStep sidesteps that, at the cost of a slightly longer synth
 * step -- an acceptable trade for `apps/web`'s size.
 */
export function buildPipelineApp(
  app: App,
  config: PipelineConfig,
  webSiteContentPaths: Record<WebEnvironment, string>,
): Stack {
  const stack = new Stack(app, "TraductorKaqchikel-Pipeline", {
    env: { account: config.toolingAccount, region: config.region },
  });

  const source = CodePipelineSource.connection(config.repoString, "main", {
    connectionArn: config.connectionArn,
  });

  // One `vite build` per environment, each with its own `VITE_API_BASE_URL`
  // and its own output directory (issue #263) -- see `webSiteContentPaths`'
  // doc comment above for why these all run here, inside the single synth
  // step, rather than as per-stage steps. Single-quoted so an absent URL
  // (environment has no real `ApiStack` yet) still sets the env var to an
  // explicit empty string rather than leaving it unset, and so a real
  // URL's own characters (always a plain `https://` origin, never shell
  // metacharacters) never need escaping.
  const webBuildCommands = WEB_ENVIRONMENTS.map((environmentName) => {
    const apiBaseUrl = config.webApiBaseUrls?.[environmentName] ?? "";
    const outDir = webBuildOutputDirName(environmentName);
    return `VITE_API_BASE_URL='${apiBaseUrl}' pnpm --filter web exec vite build --outDir ${outDir}`;
  });

  const synthStep = new CodeBuildStep("Synth", {
    input: source,
    installCommands: ["corepack enable", "pnpm install --frozen-lockfile"],
    commands: [
      // Type-check once -- shared across every environment's build below,
      // since it doesn't depend on VITE_API_BASE_URL.
      "pnpm --filter web exec tsc --noEmit",
      ...webBuildCommands,
      "pnpm --filter infra-cdk synth",
    ],
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
    pipelineName: PIPELINE_NAME,
    synth: synthStep,
    // Dev/Qa/Prod are separate AWS accounts, so the pipeline's artifact
    // bucket needs a KMS key to let those accounts' deploy roles decrypt
    // pipeline artifacts (required for any cross-account CodePipeline).
    crossAccountKeys: true,
  });

  // Issue #110: notify a maintainer within minutes when this pipeline's
  // execution fails -- see `PipelineFailureAlerting`'s doc comment for why
  // this only needs to listen for execution-level (not stage-level) FAILED
  // events, and `docs/runbooks/pipeline-failure-alerting.md` for the
  // operational writeup.
  new PipelineFailureAlerting(stack, "FailureAlerting", {
    pipelineName: PIPELINE_NAME,
    notificationEmail: config.pipelineFailureNotificationEmail,
  });

  const devStage = new TranslatorStage(stack, "Dev", {
    environmentName: "dev",
    webSiteContentPath: webSiteContentPaths.dev,
    previouslyValidatedWebDomainName: config.previouslyValidatedWebCertDomains?.dev,
    newCertificateAck: config.newCertificateAck,
    env: { account: config.devAccount, region: config.region },
  });
  pipeline.addStage(devStage);

  const qaStage = new TranslatorStage(stack, "Qa", {
    environmentName: "qa",
    webSiteContentPath: webSiteContentPaths.qa,
    previouslyValidatedWebDomainName: config.previouslyValidatedWebCertDomains?.qa,
    newCertificateAck: config.newCertificateAck,
    env: { account: config.qaAccount, region: config.region },
  });
  pipeline.addStage(qaStage);

  const prodStage = new TranslatorStage(stack, "Prod", {
    environmentName: "prod",
    webSiteContentPath: webSiteContentPaths.prod,
    previouslyValidatedWebDomainName: config.previouslyValidatedWebCertDomains?.prod,
    newCertificateAck: config.newCertificateAck,
    env: { account: config.prodAccount, region: config.region },
  });
  pipeline.addStage(prodStage, {
    pre: [new ManualApprovalStep("PromoteToProd")],
  });

  return stack;
}
