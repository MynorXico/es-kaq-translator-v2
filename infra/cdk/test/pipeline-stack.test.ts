import path from "node:path";
import { App } from "aws-cdk-lib";
import { Template } from "aws-cdk-lib/assertions";
import { describe, expect, it } from "vitest";
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

// A tiny committed fixture, not a real `apps/web` build -- see web-stack.test.ts.
const FIXTURE_SITE_CONTENT_PATH = path.join(__dirname, "fixtures/site");

function synthPipelineTemplate() {
  const app = new App();
  const stack = buildPipelineApp(app, FAKE_CONFIG, FIXTURE_SITE_CONTENT_PATH);
  app.synth();
  return Template.fromStack(stack);
}

describe("buildPipelineApp", () => {
  it("creates the pipeline stack in the tooling account/region", () => {
    const app = new App();
    const stack = buildPipelineApp(app, FAKE_CONFIG, FIXTURE_SITE_CONTENT_PATH);
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

  it("bakes the dev API Gateway URL into the synth step's VITE_API_BASE_URL env var", () => {
    const app = new App();
    const stack = buildPipelineApp(
      app,
      { ...FAKE_CONFIG, webApiBaseUrls: { dev: "https://fake-api.example.com" } },
      FIXTURE_SITE_CONTENT_PATH,
    );
    app.synth();
    const template = Template.fromStack(stack);

    const projects = template.findResources("AWS::CodeBuild::Project");
    const synthProject = Object.values(projects).find((project) => {
      const envVars = project.Properties.Environment?.EnvironmentVariables as
        | Array<{ Name: string }>
        | undefined;
      return envVars?.some((envVar) => envVar.Name === "VITE_API_BASE_URL");
    });
    expect(synthProject).toBeDefined();

    const envVars = synthProject!.Properties.Environment.EnvironmentVariables as Array<{
      Name: string;
      Value: string;
    }>;
    const viteApiBaseUrl = envVars.find((envVar) => envVar.Name === "VITE_API_BASE_URL");
    expect(viteApiBaseUrl?.Value).toBe("https://fake-api.example.com");
  });

  it("falls back to an empty string VITE_API_BASE_URL when no dev API URL is configured", () => {
    const template = synthPipelineTemplate();

    const projects = template.findResources("AWS::CodeBuild::Project");
    const synthProject = Object.values(projects).find((project) => {
      const envVars = project.Properties.Environment?.EnvironmentVariables as
        | Array<{ Name: string }>
        | undefined;
      return envVars?.some((envVar) => envVar.Name === "VITE_API_BASE_URL");
    });
    expect(synthProject).toBeDefined();

    const envVars = synthProject!.Properties.Environment.EnvironmentVariables as Array<{
      Name: string;
      Value: string;
    }>;
    const viteApiBaseUrl = envVars.find((envVar) => envVar.Name === "VITE_API_BASE_URL");
    expect(viteApiBaseUrl?.Value).toBe("");
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
        FIXTURE_SITE_CONTENT_PATH,
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
