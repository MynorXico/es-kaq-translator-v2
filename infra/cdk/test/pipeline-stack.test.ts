import path from "node:path";
import { App } from "aws-cdk-lib";
import { Template } from "aws-cdk-lib/assertions";
import { describe, expect, it } from "vitest";
import { WEB_ENVIRONMENTS, type WebEnvironment, webBuildOutputDirName } from "../lib/config";
import { buildPipelineApp, type PipelineConfig } from "../lib/pipeline-stack";

// Fake, obviously-non-real values only — never real account IDs (CLAUDE.md).
const FAKE_CONFIG: PipelineConfig = {
  toolingAccount: "111111111111",
  devAccount: "222222222222",
  qaAccount: "333333333333",
  prodAccount: "444444444444",
  region: "us-east-1",
  connectionArn: "arn:aws:codeconnections:us-east-1:111111111111:connection/fake-connection-id",
  repoString: "example-org/example-repo",
  // This suite is about pipeline/stage structure, not ADR 0007's new-cert
  // guard (see web-stack.test.ts for that) -- ack unconditionally so every
  // stage's WebStack synthesizes regardless of its domain name.
  newCertificateAck: true,
};

// A tiny committed fixture, not a real `apps/web` build -- see
// web-stack.test.ts. Every environment points at the same fixture here --
// this suite is about pipeline/stage structure and the per-environment
// *build commands*, not about three genuinely different build outputs
// (that's exercised for real by `apps/web`'s own build, not a CDK
// assertion test).
const FIXTURE_SITE_CONTENT_PATH = path.join(__dirname, "fixtures/site");
const FIXTURE_SITE_CONTENT_PATHS: Record<WebEnvironment, string> = {
  dev: FIXTURE_SITE_CONTENT_PATH,
  qa: FIXTURE_SITE_CONTENT_PATH,
  prod: FIXTURE_SITE_CONTENT_PATH,
};

function synthPipelineTemplate() {
  const app = new App();
  const stack = buildPipelineApp(app, FAKE_CONFIG, FIXTURE_SITE_CONTENT_PATHS);
  app.synth();
  return Template.fromStack(stack);
}

/**
 * The Synth CodeBuild project's buildspec `phases.build.commands` -- where
 * each environment's `VITE_API_BASE_URL`-scoped `vite build` invocation
 * now lives (issue #263), since they're no longer a single project-level
 * `Environment.EnvironmentVariables` entry.
 */
function synthBuildCommands(template: Template): string[] {
  const projects = template.findResources("AWS::CodeBuild::Project");
  const synthProject = Object.values(projects).find((project) => {
    const buildSpec = project.Properties.Source?.BuildSpec;
    return typeof buildSpec === "string" && buildSpec.includes("pnpm --filter infra-cdk synth");
  });
  expect(synthProject).toBeDefined();

  const buildSpec = JSON.parse(synthProject!.Properties.Source.BuildSpec as string) as {
    phases: { build: { commands: string[] } };
  };
  return buildSpec.phases.build.commands;
}

describe("buildPipelineApp", () => {
  it("creates the pipeline stack in the tooling account/region", () => {
    const app = new App();
    const stack = buildPipelineApp(app, FAKE_CONFIG, FIXTURE_SITE_CONTENT_PATHS);
    app.synth();

    expect(stack.account).toBe(FAKE_CONFIG.toolingAccount);
    expect(stack.region).toBe(FAKE_CONFIG.region);
  });

  it("defines a CodePipeline with Dev, Qa, and Prod stages in order", () => {
    const template = synthPipelineTemplate();

    const pipelines = template.findResources("AWS::CodePipeline::Pipeline");
    const [pipeline] = Object.values(pipelines);
    expect(pipeline).toBeDefined();

    const stageNames: string[] = pipeline.Properties.Stages.map(
      (stage: { Name: string }) => stage.Name,
    );

    const devIndex = stageNames.findIndex((name) => name.startsWith("Dev"));
    const qaIndex = stageNames.findIndex((name) => name.startsWith("Qa"));
    const prodIndex = stageNames.findIndex((name) => name.startsWith("Prod"));

    expect(devIndex).toBeGreaterThanOrEqual(0);
    expect(qaIndex).toBeGreaterThan(devIndex);
    expect(prodIndex).toBeGreaterThan(qaIndex);
  });

  it("gates only the Prod stage behind a manual approval action", () => {
    const template = synthPipelineTemplate();

    const pipelines = template.findResources("AWS::CodePipeline::Pipeline");
    const [pipeline] = Object.values(pipelines);

    type PipelineStage = {
      Name: string;
      Actions: Array<{ ActionTypeId: { Category: string } }>;
    };
    const stages: PipelineStage[] = pipeline.Properties.Stages;

    const stagesWithManualApproval = stages.filter((stage) =>
      stage.Actions.some((action) => action.ActionTypeId.Category === "Approval"),
    );

    expect(stagesWithManualApproval).toHaveLength(1);
    expect(stagesWithManualApproval[0].Name.startsWith("Prod")).toBe(true);
  });

  // Issue #263: qa (and future prod) must get a build of apps/web with
  // *their own* VITE_API_BASE_URL baked in, not dev's -- a single shared
  // build was the root cause of qa silently calling dev's real API.
  it("bakes each configured environment's own API Gateway URL into that environment's own web build command", () => {
    const app = new App();
    const stack = buildPipelineApp(
      app,
      {
        ...FAKE_CONFIG,
        webApiBaseUrls: {
          dev: "https://fake-dev-api.example.com",
          qa: "https://fake-qa-api.example.com",
        },
      },
      FIXTURE_SITE_CONTENT_PATHS,
    );
    app.synth();
    const commands = synthBuildCommands(Template.fromStack(stack));

    const devCommand = commands.find((command) => command.includes(webBuildOutputDirName("dev")));
    const qaCommand = commands.find((command) => command.includes(webBuildOutputDirName("qa")));
    const prodCommand = commands.find((command) => command.includes(webBuildOutputDirName("prod")));

    expect(devCommand).toContain("VITE_API_BASE_URL='https://fake-dev-api.example.com'");
    expect(qaCommand).toContain("VITE_API_BASE_URL='https://fake-qa-api.example.com'");
    // prod has no configured URL above -- falls back to an empty string
    // (same-origin relative request), per PipelineConfig.webApiBaseUrls'
    // doc comment, rather than reusing dev's or qa's URL.
    expect(prodCommand).toContain("VITE_API_BASE_URL=''");
  });

  it("falls back to an empty string VITE_API_BASE_URL for every environment when none are configured", () => {
    const commands = synthBuildCommands(synthPipelineTemplate());

    for (const environmentName of WEB_ENVIRONMENTS) {
      const command = commands.find((c) => c.includes(webBuildOutputDirName(environmentName)));
      expect(command).toContain("VITE_API_BASE_URL=''");
    }
  });

  it("builds apps/web once per environment, each into its own output directory", () => {
    const commands = synthBuildCommands(synthPipelineTemplate());

    for (const environmentName of WEB_ENVIRONMENTS) {
      const outDir = webBuildOutputDirName(environmentName);
      const matchingCommands = commands.filter((command) => command.includes(outDir));
      expect(matchingCommands).toHaveLength(1);
      expect(matchingCommands[0]).toContain("vite build");
    }

    // Three distinct output directories, not the same one reused.
    const outDirs = new Set(WEB_ENVIRONMENTS.map((env) => webBuildOutputDirName(env)));
    expect(outDirs.size).toBe(WEB_ENVIRONMENTS.length);
  });

  it("scopes the synth step's SSM permissions to the traductor-kaqchikel parameter path", () => {
    const template = synthPipelineTemplate();

    const policies = template.findResources("AWS::IAM::Policy");
    const statementsGrantingGetParameter = Object.values(policies).flatMap((policy) => {
      const statements = policy.Properties.PolicyDocument.Statement as Array<{
        Action: string | string[];
        Resource: unknown;
        Effect: string;
      }>;
      return statements.filter((statement) => {
        const actions = Array.isArray(statement.Action) ? statement.Action : [statement.Action];
        return statement.Effect === "Allow" && actions.includes("ssm:GetParameter");
      });
    });

    expect(statementsGrantingGetParameter.length).toBeGreaterThan(0);

    for (const statement of statementsGrantingGetParameter) {
      // Every action in the statement must be ssm:GetParameter only (no broader ssm actions).
      const actions = Array.isArray(statement.Action) ? statement.Action : [statement.Action];
      expect(actions).toEqual(["ssm:GetParameter"]);

      const resources = Array.isArray(statement.Resource) ? statement.Resource : [statement.Resource];
      for (const resource of resources) {
        const resolvedResource =
          typeof resource === "string" ? resource : JSON.stringify(resource);
        // Must reference the traductor-kaqchikel SSM parameter path, and nothing broader
        // (no bare "*", no un-prefixed "parameter/*").
        expect(resolvedResource).toContain("traductor-kaqchikel");
        expect(resolvedResource).not.toBe("*");
      }
    }
  });

  // Issue #110: pipeline execution failure alerting.
  describe("pipeline failure alerting", () => {
    it("creates an EventBridge rule matching this pipeline's FAILED execution state", () => {
      const template = synthPipelineTemplate();

      template.hasResourceProperties("AWS::Events::Rule", {
        EventPattern: {
          source: ["aws.codepipeline"],
          "detail-type": ["CodePipeline Pipeline Execution State Change"],
          detail: {
            pipeline: ["TraductorKaqchikelPipeline"],
            state: ["FAILED"],
          },
        },
      });
    });

    it("subscribes the configured notification email to the failure topic", () => {
      const app = new App();
      const stack = buildPipelineApp(
        app,
        { ...FAKE_CONFIG, pipelineFailureNotificationEmail: "maintainer@example.com" },
        FIXTURE_SITE_CONTENT_PATHS,
      );
      app.synth();
      const template = Template.fromStack(stack);

      template.hasResourceProperties("AWS::SNS::Subscription", {
        Protocol: "email",
        Endpoint: "maintainer@example.com",
      });
    });

    it("still creates the failure topic (with no subscription) when no notification email is configured", () => {
      const template = synthPipelineTemplate();

      template.resourceCountIs("AWS::SNS::Topic", 1);
      template.resourceCountIs("AWS::SNS::Subscription", 0);
    });
  });
});
