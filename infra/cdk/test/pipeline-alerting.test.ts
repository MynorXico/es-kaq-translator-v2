import { App, Stack } from "aws-cdk-lib";
import { Match, Template } from "aws-cdk-lib/assertions";
import { describe, expect, it } from "vitest";
import { PipelineFailureAlerting } from "../lib/pipeline-alerting";

const FAKE_ENV = { account: "111111111111", region: "us-east-1" };
const FAKE_PIPELINE_NAME = "TraductorKaqchikelPipeline";

function synthAlerting(notificationEmail?: string) {
  const app = new App();
  const stack = new Stack(app, "TestAlertingStack", { env: FAKE_ENV });
  new PipelineFailureAlerting(stack, "Alerting", {
    pipelineName: FAKE_PIPELINE_NAME,
    notificationEmail,
  });
  return Template.fromStack(stack);
}

describe("PipelineFailureAlerting", () => {
  it("creates an SNS topic", () => {
    const template = synthAlerting("maintainer@example.com");

    template.resourceCountIs("AWS::SNS::Topic", 1);
  });

  it("subscribes the given email address to the topic", () => {
    const template = synthAlerting("maintainer@example.com");

    template.hasResourceProperties("AWS::SNS::Subscription", {
      Protocol: "email",
      Endpoint: "maintainer@example.com",
    });
  });

  it("creates no subscription when no notification email is configured", () => {
    const template = synthAlerting(undefined);

    template.resourceCountIs("AWS::SNS::Subscription", 0);
    // The topic itself should still exist so a subscription can be added later.
    template.resourceCountIs("AWS::SNS::Topic", 1);
  });

  it("creates an EventBridge rule matching this pipeline's FAILED execution state", () => {
    const template = synthAlerting("maintainer@example.com");

    template.hasResourceProperties("AWS::Events::Rule", {
      EventPattern: {
        source: ["aws.codepipeline"],
        "detail-type": ["CodePipeline Pipeline Execution State Change"],
        detail: {
          pipeline: [FAKE_PIPELINE_NAME],
          state: ["FAILED"],
        },
      },
    });
  });

  it("targets the SNS topic from the EventBridge rule", () => {
    const template = synthAlerting("maintainer@example.com");

    const topics = template.findResources("AWS::SNS::Topic");
    const [topicLogicalId] = Object.keys(topics);

    template.hasResourceProperties("AWS::Events::Rule", {
      Targets: Match.arrayWith([
        Match.objectLike({
          Arn: { Ref: topicLogicalId },
        }),
      ]),
    });
  });

  it("includes the execution ID in the notification message so it's actionable without digging", () => {
    const template = synthAlerting("maintainer@example.com");

    const rules = template.findResources("AWS::Events::Rule");
    const [rule] = Object.values(rules);
    const target = rule.Properties.Targets[0];
    const inputTemplate = JSON.stringify(target.InputTransformer ?? target.Input ?? "");

    expect(inputTemplate).toMatch(/execution-id/);
  });
});
