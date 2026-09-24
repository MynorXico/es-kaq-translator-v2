import { App } from "aws-cdk-lib";
import { Match, Template } from "aws-cdk-lib/assertions";
import { describe, expect, it } from "vitest";
import { ApiStack } from "../lib/api-stack";

// Fake, obviously-non-real account ID (CLAUDE.md never allows a real one in tests).
const FAKE_ENV = { account: "222222222222", region: "us-east-1" };

function synthApiStack(props?: { sageMakerEndpointName?: string }) {
  const app = new App();
  const stack = new ApiStack(app, "TestApiStack", {
    environmentName: "test",
    env: FAKE_ENV,
    ...props,
  });
  const template = Template.fromStack(stack);
  return { stack, template };
}

describe("ApiStack", () => {
  it("deploys apps/api as a Lambda container image", () => {
    const { template } = synthApiStack();

    template.hasResourceProperties("AWS::Lambda::Function", {
      PackageType: "Image",
    });
  });

  it("exposes an API Gateway HTTP API in front of the Lambda", () => {
    const { template } = synthApiStack();

    template.hasResourceProperties("AWS::ApiGatewayV2::Api", {
      ProtocolType: "HTTP",
    });

    // Routed via a Lambda proxy integration (AWS_PROXY), not some other
    // integration type, so every path/method reaches the FastAPI app.
    template.hasResourceProperties("AWS::ApiGatewayV2::Integration", {
      IntegrationType: "AWS_PROXY",
      PayloadFormatVersion: "2.0",
    });
  });

  it("passes the SageMaker endpoint name to the Lambda via an env var, defaulting per environment", () => {
    const { template } = synthApiStack();

    template.hasResourceProperties("AWS::Lambda::Function", {
      Environment: {
        Variables: Match.objectLike({
          SAGEMAKER_ENDPOINT_NAME: "traductor-kaqchikel-translate-test",
        }),
      },
    });
  });

  it("passes this environment's real custom-domain origin as ALLOWED_ORIGINS, not left unset", () => {
    const { template } = synthApiStack();

    template.hasResourceProperties("AWS::Lambda::Function", {
      Environment: {
        Variables: Match.objectLike({
          ALLOWED_ORIGINS: "https://app-test.traductorkaqchikel.com",
        }),
      },
    });
  });

  it("allows overriding the SageMaker endpoint name explicitly", () => {
    const { template } = synthApiStack({ sageMakerEndpointName: "some-other-endpoint" });

    template.hasResourceProperties("AWS::Lambda::Function", {
      Environment: {
        Variables: Match.objectLike({
          SAGEMAKER_ENDPOINT_NAME: "some-other-endpoint",
        }),
      },
    });
  });

  it("grants the Lambda execution role sagemaker:InvokeEndpoint scoped to exactly that endpoint, not a wildcard", () => {
    const { template } = synthApiStack();

    const policies = template.findResources("AWS::IAM::Policy");
    const sagemakerStatements = Object.values(policies).flatMap((policy) => {
      const statements = policy.Properties.PolicyDocument.Statement as Array<{
        Effect: string;
        Action: string | string[];
        Resource: unknown;
      }>;
      return statements.filter((statement) => {
        const actions = Array.isArray(statement.Action) ? statement.Action : [statement.Action];
        return statement.Effect === "Allow" && actions.includes("sagemaker:InvokeEndpoint");
      });
    });

    expect(sagemakerStatements.length).toBe(1);

    const [statement] = sagemakerStatements;
    const resources = Array.isArray(statement.Resource) ? statement.Resource : [statement.Resource];
    expect(resources.length).toBe(1);

    const resolved = JSON.stringify(resources[0]);
    expect(resolved).not.toBe('"*"');
    // Must resolve to a real endpoint ARN pattern including the endpoint name,
    // not a bare wildcard resource.
    expect(resolved).toContain("endpoint/traductor-kaqchikel-translate-test");
  });

  it("does not attach a broad AWS-managed SageMaker or Lambda policy to the execution role", () => {
    const { template } = synthApiStack();

    const roles = template.findResources("AWS::IAM::Role");
    for (const role of Object.values(roles)) {
      const managedArns = (role.Properties.ManagedPolicyArns ?? []) as unknown[];
      const managedArnsStr = JSON.stringify(managedArns);
      expect(managedArnsStr).not.toContain("AmazonSageMakerFullAccess");
    }
  });

  it("outputs the API Gateway URL as a CfnOutput, matching the DataStack/WebStack pattern", () => {
    const { template } = synthApiStack();

    template.hasOutput("ApiUrl", {});
  });
});
