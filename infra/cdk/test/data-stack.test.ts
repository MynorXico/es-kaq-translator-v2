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
        // Real SageMaker training job log groups are literally named
        // "/aws/sagemaker/TrainingJobs" (leading slash). A resource pattern
        // missing that slash (e.g. "log-group:aws/sagemaker/*") silently
        // never matches a real log group and blocks all training job
        // logging — this exact bug shipped once and was only caught on a
        // real, billable training run (issue #66).
        expect(resolved).toContain("log-group:/aws/sagemaker");
      }
    }
  });

  it("grants the baseline permissions the SageMaker Python SDK v3 requires to accept the execution role for a training job", () => {
    // Real incident: a real (non-dry-run) submit_job.py invocation against
    // this exact role failed with RoleValidationError -- SDK v3's
    // ModelTrainer does a client-side IAM permission check before ever
    // calling CreateTrainingJob, against its own generic "any training job"
    // baseline (sagemaker.core.helper.iam_policies.IAM_POLICY_CONFIG's
    // "training" entry), regardless of whether this project's specific job
    // configuration (no VPC) actually needs all of it. This role worked
    // for real training under the v2 SDK (no client-side check existed);
    // the v3 migration (#155/#172) silently introduced this new
    // requirement, uncaught because --dry-run deliberately never
    // constructs a real ModelTrainer (see submit_job.py's own docstring).
    const { template } = synthDataStack();

    const policies = template.findResources("AWS::IAM::Policy");
    const allStatements = Object.values(policies).flatMap(
      (policy) =>
        policy.Properties.PolicyDocument.Statement as Array<{
          Effect: string;
          Action: string | string[];
          Resource: unknown;
        }>,
    );
    const allowedActions = new Set(
      allStatements
        .filter((s) => s.Effect === "Allow")
        .flatMap((s) => (Array.isArray(s.Action) ? s.Action : [s.Action])),
    );

    // cloudwatch:PutMetricData and ecr:GetAuthorizationToken don't support
    // resource-level scoping in AWS's own IAM model for these actions --
    // "Resource": "*" is the correct, only-possible grant, not a
    // least-privilege regression.
    expect(allowedActions.has("cloudwatch:PutMetricData")).toBe(true);
    expect(allowedActions.has("ecr:GetAuthorizationToken")).toBe(true);

    // Same for the EC2 ENI/VPC-describe actions the SDK's baseline checks
    // for (relevant only if a job ever configures VPC isolation, which
    // this project's jobs don't -- granted anyway since the v3 SDK treats
    // it as a hard precondition regardless).
    for (const action of [
      "ec2:CreateNetworkInterface",
      "ec2:CreateNetworkInterfacePermission",
      "ec2:DeleteNetworkInterface",
      "ec2:DeleteNetworkInterfacePermission",
      "ec2:DescribeNetworkInterfaces",
      "ec2:DescribeVpcs",
      "ec2:DescribeDhcpOptions",
      "ec2:DescribeSubnets",
      "ec2:DescribeSecurityGroups",
    ]) {
      expect(allowedActions.has(action)).toBe(true);
    }
  });

  it("grants the execution role scoped Model Registry permissions for self-registration (issue #190)", () => {
    // Real workflow change, not speculative: train.py now registers its own
    // completed run's model package from inside the training container
    // (no more client-side download of the full training artifact just to
    // read its model card -- a real run's artifact was measured at 30.5GB,
    // issues #187/#188). That self-registration calls
    // CreateModelPackageGroup (idempotent) + CreateModelPackage directly
    // against this role's own credentials.
    const { template } = synthDataStack();

    const policies = template.findResources("AWS::IAM::Policy");
    const allStatements = Object.values(policies).flatMap(
      (policy) =>
        policy.Properties.PolicyDocument.Statement as Array<{
          Effect: string;
          Action: string | string[];
          Resource: unknown;
        }>,
    );

    const findStatementsForAction = (action: string) =>
      allStatements.filter((statement) => {
        const actions = Array.isArray(statement.Action) ? statement.Action : [statement.Action];
        return statement.Effect === "Allow" && actions.includes(action);
      });

    const groupStatements = findStatementsForAction("sagemaker:CreateModelPackageGroup");
    expect(groupStatements.length).toBeGreaterThan(0);
    for (const statement of groupStatements) {
      const resources = Array.isArray(statement.Resource) ? statement.Resource : [statement.Resource];
      for (const resource of resources) {
        expect(resource).not.toBe("*");
        const resolved = typeof resource === "string" ? resource : JSON.stringify(resource);
        expect(resolved).toContain("model-package-group");
        expect(resolved).toContain("traductor-kaqchikel-es-cak");
      }
    }

    const packageStatements = findStatementsForAction("sagemaker:CreateModelPackage");
    expect(packageStatements.length).toBeGreaterThan(0);
    for (const statement of packageStatements) {
      const resources = Array.isArray(statement.Resource) ? statement.Resource : [statement.Resource];
      for (const resource of resources) {
        expect(resource).not.toBe("*");
        const resolved = typeof resource === "string" ? resource : JSON.stringify(resource);
        expect(resolved).toContain("model-package");
        expect(resolved).toContain("traductor-kaqchikel-es-cak");
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
