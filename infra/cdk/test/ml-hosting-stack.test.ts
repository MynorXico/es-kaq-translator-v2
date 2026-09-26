import { App, Stack } from "aws-cdk-lib";
import { Match, Template } from "aws-cdk-lib/assertions";
import { Bucket } from "aws-cdk-lib/aws-s3";
import { describe, expect, it } from "vitest";
import { MlHostingStack } from "../lib/ml-hosting-stack";

// Fake, obviously-non-real account ID (CLAUDE.md never allows a real one in
// tests, or anywhere in this public repo -- see MlHostingStack's own
// docstring for why the real Model Package ARN is *constructed*, not
// hardcoded, for exactly this reason).
const FAKE_ENV = { account: "222222222222", region: "us-east-1" };

function synthMlHostingStack(modelPackageVersion = 3) {
  const app = new App();
  // A throwaway bucket standing in for DataStack's real bucket -- this
  // stack only needs to grant its execution role read access to it, not
  // create it.
  const bucketStack = new Stack(app, "TestBucketStack", { env: FAKE_ENV });
  const bucket = new Bucket(bucketStack, "FakeTrainingDataBucket");

  const stack = new MlHostingStack(app, "TestMlHostingStack", {
    environmentName: "test",
    modelPackageVersion,
    modelDataBucket: bucket,
    env: FAKE_ENV,
  });
  const template = Template.fromStack(stack);
  return { stack, template };
}

describe("MlHostingStack", () => {
  it("creates a SageMaker Model referencing the approved Model Package version", () => {
    const { template } = synthMlHostingStack(3);

    template.hasResourceProperties("AWS::SageMaker::Model", {
      Containers: Match.arrayWith([
        Match.objectLike({
          ModelPackageName: Match.stringLikeRegexp(
            "arn:aws:sagemaker:us-east-1:222222222222:model-package/traductor-kaqchikel-es-cak/3",
          ),
        }),
      ]),
    });
  });

  it("creates a Serverless Inference endpoint config with a memory/concurrency limit", () => {
    const { template } = synthMlHostingStack();

    template.hasResourceProperties("AWS::SageMaker::EndpointConfig", {
      ProductionVariants: Match.arrayWith([
        Match.objectLike({
          ServerlessConfig: Match.objectLike({
            MemorySizeInMB: Match.anyValue(),
            MaxConcurrency: Match.anyValue(),
          }),
        }),
      ]),
    });
  });

  it("creates an endpoint pointing at the endpoint config", () => {
    const { template } = synthMlHostingStack();

    template.resourceCountIs("AWS::SageMaker::Endpoint", 1);
  });

  it("outputs the endpoint name for apps/api to resolve at deploy time (issue #9)", () => {
    const { template } = synthMlHostingStack();

    template.hasOutput("MlEndpointName", {
      Value: "traductor-kaqchikel-es-cak-test",
    });
  });

  it("grants the inference execution role read-only access to the model data bucket, not broad S3 access", () => {
    const { template } = synthMlHostingStack();

    const policies = template.findResources("AWS::IAM::Policy");
    const allActions = Object.values(policies).flatMap((policy) => {
      const statements = policy.Properties.PolicyDocument.Statement as Array<{
        Action: string | string[];
      }>;
      return statements.flatMap((s) => (Array.isArray(s.Action) ? s.Action : [s.Action]));
    });

    expect(allActions).not.toContain("s3:*");
    expect(allActions).not.toContain("*");
  });

  it("scopes the execution role's CloudWatch Logs permissions to the SageMaker endpoint log namespace, not a wildcard resource", () => {
    const { template } = synthMlHostingStack();

    const policies = template.findResources("AWS::IAM::Policy");
    const logStatements = Object.values(policies).flatMap((policy) => {
      const statements = policy.Properties.PolicyDocument.Statement as Array<{
        Action: string | string[];
        Resource: unknown;
      }>;
      return statements.filter((s) => {
        const actions = Array.isArray(s.Action) ? s.Action : [s.Action];
        return actions.some((a) => a.startsWith("logs:"));
      });
    });

    expect(logStatements.length).toBeGreaterThan(0);
    for (const statement of logStatements) {
      const resourceJson = JSON.stringify(statement.Resource);
      expect(resourceJson).not.toBe('"*"');
    }
  });

  it("uses a SageMaker service-principal trust policy for the execution role", () => {
    const { template } = synthMlHostingStack();

    template.hasResourceProperties("AWS::IAM::Role", {
      AssumeRolePolicyDocument: Match.objectLike({
        Statement: Match.arrayWith([
          Match.objectLike({
            Principal: { Service: "sagemaker.amazonaws.com" },
          }),
        ]),
      }),
    });
  });

  it("gives the Model and EndpointConfig a version-qualified name, so bumping DEV_MODEL_PACKAGE_VERSION is a safe CloudFormation replacement, not a name collision", () => {
    // Real incident: bumping modelPackageVersion previously failed to
    // deploy with "Cannot create already existing model ... (AlreadyExists)"
    // -- CloudFormation's default create-new-before-delete-old replacement
    // strategy can't create the new Model/EndpointConfig if their physical
    // names are the same fixed string every version uses. Model/
    // EndpointConfig names must change whenever the version does; the
    // Endpoint's own name (this stack's public `endpointName` contract,
    // consumed by ApiStack's SAGEMAKER_ENDPOINT_NAME) must NOT change, or
    // every consumer of a stable endpoint name would break instead.
    const { template: templateV3 } = synthMlHostingStack(3);
    const { template: templateV4 } = synthMlHostingStack(4);

    const modelNameV3 = Object.values(templateV3.findResources("AWS::SageMaker::Model"))[0]
      .Properties.ModelName;
    const modelNameV4 = Object.values(templateV4.findResources("AWS::SageMaker::Model"))[0]
      .Properties.ModelName;
    expect(modelNameV3).not.toBe(modelNameV4);

    const configNameV3 = Object.values(
      templateV3.findResources("AWS::SageMaker::EndpointConfig"),
    )[0].Properties.EndpointConfigName;
    const configNameV4 = Object.values(
      templateV4.findResources("AWS::SageMaker::EndpointConfig"),
    )[0].Properties.EndpointConfigName;
    expect(configNameV3).not.toBe(configNameV4);

    const endpointNameV3 = Object.values(templateV3.findResources("AWS::SageMaker::Endpoint"))[0]
      .Properties.EndpointName;
    const endpointNameV4 = Object.values(templateV4.findResources("AWS::SageMaker::Endpoint"))[0]
      .Properties.EndpointName;
    expect(endpointNameV3).toBe(endpointNameV4);
    expect(endpointNameV3).toBe("traductor-kaqchikel-es-cak-test");
  });
});
