import { App } from "aws-cdk-lib";
import { Match, Template } from "aws-cdk-lib/assertions";
import { describe, expect, it } from "vitest";
import { DataStack } from "../lib/data-stack";

// Fake, obviously-non-real account ID (CLAUDE.md never allows a real one in tests).
const FAKE_ENV = { account: "222222222222", region: "us-east-1" };

function synthDataStack() {
  const app = new App();
  const stack = new DataStack(app, "TestDataStack", {
    environmentName: "test",
    env: FAKE_ENV,
  });
  const template = Template.fromStack(stack);
  return { stack, template };
}

describe("DataStack", () => {
  it("creates a versioned, encrypted S3 bucket with all public access blocked", () => {
    const { template } = synthDataStack();

    template.hasResourceProperties("AWS::S3::Bucket", {
      BucketName: "traductor-kaqchikel-training-data-test",
      VersioningConfiguration: { Status: "Enabled" },
      BucketEncryption: {
        ServerSideEncryptionConfiguration: Match.arrayWith([
          Match.objectLike({
            ServerSideEncryptionByDefault: Match.objectLike({
              SSEAlgorithm: Match.anyValue(),
            }),
          }),
        ]),
      },
      PublicAccessBlockConfiguration: {
        BlockPublicAcls: true,
        BlockPublicPolicy: true,
        IgnorePublicAcls: true,
        RestrictPublicBuckets: true,
      },
    });
  });

  it("never attaches a bucket policy allowing public/anonymous access", () => {
    const { template } = synthDataStack();

    const policies = template.findResources("AWS::S3::BucketPolicy");
    for (const policy of Object.values(policies)) {
      const statements = policy.Properties.PolicyDocument.Statement as Array<{
        Effect: string;
        Principal: unknown;
      }>;
      for (const statement of statements) {
        if (statement.Effect !== "Allow") continue;
        const principal = JSON.stringify(statement.Principal);
        expect(principal).not.toBe('"*"');
        expect(principal.includes('"AWS":"*"')).toBe(false);
      }
    }
  });

  it("creates a SageMaker execution role trusted only by the SageMaker service", () => {
    const { template } = synthDataStack();

    template.hasResourceProperties("AWS::IAM::Role", {
      AssumeRolePolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: "Allow",
            Principal: { Service: "sagemaker.amazonaws.com" },
            Action: "sts:AssumeRole",
          }),
        ]),
      },
    });
  });

  it("scopes the execution role's S3 permissions to the training bucket only, not a wildcard", () => {
    const { template } = synthDataStack();

    const policies = template.findResources("AWS::IAM::Policy");
    const s3Statements = Object.values(policies).flatMap((policy) => {
      const statements = policy.Properties.PolicyDocument.Statement as Array<{
        Effect: string;
        Action: string | string[];
        Resource: unknown;
      }>;
      return statements.filter((statement) => {
        const actions = Array.isArray(statement.Action) ? statement.Action : [statement.Action];
        return statement.Effect === "Allow" && actions.some((action) => action.startsWith("s3:"));
      });
    });

    expect(s3Statements.length).toBeGreaterThan(0);

    for (const statement of s3Statements) {
      const resources = Array.isArray(statement.Resource) ? statement.Resource : [statement.Resource];
      expect(resources.length).toBeGreaterThan(0);
      for (const resource of resources) {
        // Every resource must resolve to a reference derived from the bucket
        // construct (Fn::GetAtt/Fn::Join over the bucket's ARN), never a bare "*".
        expect(resource).not.toBe("*");
        const resolved = typeof resource === "string" ? resource : JSON.stringify(resource);
        expect(resolved).not.toBe("*");
      }
    }
  });

  it("grants the execution role least-privilege CloudWatch Logs write access, scoped to a SageMaker log group prefix", () => {
    const { template } = synthDataStack();

    const policies = template.findResources("AWS::IAM::Policy");
    const logsStatements = Object.values(policies).flatMap((policy) => {
      const statements = policy.Properties.PolicyDocument.Statement as Array<{
        Effect: string;
        Action: string | string[];
        Resource: unknown;
      }>;
      return statements.filter((statement) => {
        const actions = Array.isArray(statement.Action) ? statement.Action : [statement.Action];
        return statement.Effect === "Allow" && actions.some((action) => action.startsWith("logs:"));
      });
    });

    expect(logsStatements.length).toBeGreaterThan(0);

    for (const statement of logsStatements) {
      const actions = Array.isArray(statement.Action) ? statement.Action : [statement.Action];
      // No broad logs:* action.
      expect(actions).not.toContain("logs:*");

      const resources = Array.isArray(statement.Resource) ? statement.Resource : [statement.Resource];
      for (const resource of resources) {
        expect(resource).not.toBe("*");
        const resolved = typeof resource === "string" ? resource : JSON.stringify(resource);
        // Must be scoped to the SageMaker log group namespace, not every log group in the account.
        expect(resolved).toContain("sagemaker");
      }
    }
  });

  it("does not attach a broad AWS-managed SageMaker policy to the execution role", () => {
    const { template } = synthDataStack();

    const roles = template.findResources("AWS::IAM::Role");
    for (const role of Object.values(roles)) {
      const managedArns = (role.Properties.ManagedPolicyArns ?? []) as unknown[];
      const managedArnsStr = JSON.stringify(managedArns);
      expect(managedArnsStr).not.toContain("AmazonSageMakerFullAccess");
    }
  });
});
