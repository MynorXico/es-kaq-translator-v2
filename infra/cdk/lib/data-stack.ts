import { ArnFormat, CfnOutput, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import { BlockPublicAccess, Bucket, BucketEncryption } from "aws-cdk-lib/aws-s3";
import { PolicyStatement, Role, ServicePrincipal } from "aws-cdk-lib/aws-iam";
import type { Construct } from "constructs";

export interface DataStackProps extends StackProps {
  environmentName: string;
}

/**
 * Private storage for the ALMG-derived training corpus and trained model
 * artifacts, plus the SageMaker execution role that reads/writes it during
 * training jobs (ADR 0001's "data" stack, ADR 0002/`docs/data-governance.md`
 * privacy requirements).
 *
 * The bucket is never made public and is only ever referenced by its
 * CDK-generated name/ARN (via `CfnOutput`/cross-stack refs) — no real
 * bucket name or account ID is hardcoded anywhere in this repo.
 */
export class DataStack extends Stack {
  public readonly bucket: Bucket;
  public readonly sageMakerExecutionRole: Role;

  constructor(scope: Construct, id: string, props: DataStackProps) {
    super(scope, id, props);

    this.bucket = new Bucket(this, "TrainingDataBucket", {
      bucketName: `traductor-kaqchikel-training-data-${props.environmentName}`,
      versioned: true,
      encryption: BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      blockPublicAccess: BlockPublicAccess.BLOCK_ALL,
      removalPolicy: RemovalPolicy.RETAIN,
    });

    this.sageMakerExecutionRole = new Role(this, "SageMakerExecutionRole", {
      assumedBy: new ServicePrincipal("sagemaker.amazonaws.com"),
      description:
        "Least-privilege execution role for SageMaker Training Jobs: read/write access to " +
        "only this environment's training data bucket, plus CloudWatch Logs write access " +
        "scoped to SageMaker's own log group namespace. Deliberately not the broad AWS-managed " +
        "AmazonSageMakerFullAccess policy.",
    });

    // Scoped to this bucket's ARN (and its objects) only — never a wildcard resource.
    this.bucket.grantReadWrite(this.sageMakerExecutionRole);

    // SageMaker Training Jobs write their logs under this fixed log group
    // namespace; scope write access there instead of granting logs:* on "*".
    // Two easy-to-miss details, both required to actually match a real
    // CloudWatch log group ARN (arn:aws:logs:<region>:<account>:log-group:
    // /aws/sagemaker/TrainingJobs:*) rather than silently matching nothing:
    // (1) resourceName needs its own leading "/" (CloudWatch log group
    // names for SageMaker are literally "/aws/sagemaker/..."), and
    // (2) ArnFormat.COLON_RESOURCE_NAME, since CDK's formatArn defaults to
    // joining resource+resourceName with a "/" for many services, which
    // for logs ARNs (colon-joined) produces a bogus double-slash
    // ("log-group//aws/sagemaker/*") instead of the real
    // "log-group:/aws/sagemaker/*". This exact combination silently
    // blocked all training job logging on the first real run (issue #66).
    this.sageMakerExecutionRole.addToPolicy(
      new PolicyStatement({
        actions: ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
        resources: [
          this.formatArn({
            service: "logs",
            resource: "log-group",
            resourceName: "/aws/sagemaker/*",
            arnFormat: ArnFormat.COLON_RESOURCE_NAME,
          }),
        ],
      }),
    );

    // Real incident, not a preemptive/speculative grant: a real (non-dry-run)
    // `submit_job.py` invocation against this exact role failed with
    // `RoleValidationError` once the project migrated to SageMaker Python
    // SDK v3 (issues #155/#172). v3's `ModelTrainer` does a client-side IAM
    // permission check before ever calling `CreateTrainingJob`, against the
    // SDK's own generic "any training job" baseline
    // (`sagemaker.core.helper.iam_policies.IAM_POLICY_CONFIG["training"]`)
    // -- regardless of whether this project's specific job configuration
    // (no VPC) actually needs all of it. This exact role successfully ran a
    // real training job under the v2 SDK, which had no such client-side
    // check; the v3 migration silently introduced this new precondition,
    // uncaught because `--dry-run` deliberately never constructs a real
    // `ModelTrainer` (see `submit_job.py`'s own docstring on why). All
    // three actions below require `Resource: "*"` in AWS's own IAM model --
    // none support resource-level scoping, so this isn't a least-privilege
    // regression, it's the only grant these specific actions can take.
    this.sageMakerExecutionRole.addToPolicy(
      new PolicyStatement({
        actions: ["cloudwatch:PutMetricData"],
        resources: ["*"],
      }),
    );
    this.sageMakerExecutionRole.addToPolicy(
      new PolicyStatement({
        actions: ["ecr:GetAuthorizationToken"],
        resources: ["*"],
      }),
    );
    // Same reasoning for the EC2 ENI/VPC-describe actions the SDK's
    // baseline checks for -- relevant only if a job ever configures VPC
    // isolation, which this project's jobs deliberately don't (ADR 0001's
    // serverless-first posture), but granted anyway since v3 treats
    // possessing them as a hard precondition regardless of actual usage.
    this.sageMakerExecutionRole.addToPolicy(
      new PolicyStatement({
        actions: [
          "ec2:CreateNetworkInterface",
          "ec2:CreateNetworkInterfacePermission",
          "ec2:DeleteNetworkInterface",
          "ec2:DeleteNetworkInterfacePermission",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DescribeVpcs",
          "ec2:DescribeDhcpOptions",
          "ec2:DescribeSubnets",
          "ec2:DescribeSecurityGroups",
        ],
        resources: ["*"],
      }),
    );

    // Issue #190 gave this role its own scoped Model Registry write grant
    // (CreateModelPackageGroup/CreateModelPackage) so `train.py` could
    // self-register a completed run's model package from inside the
    // training container, avoiding a client-side download of the (30.5GB,
    // issues #187/#188) training artifact just to read its model card.
    // ADR 0009 reverted that self-registration entirely (issue #201:
    // `CreateModelPackage` validates S3 object existence at call time,
    // before the training container's own toolkit has uploaded the
    // artifact -- structurally unable to succeed). Registration goes back
    // to running under the maintainer's own SSO credentials, as it did
    // before #190 and as `submit_job.py --register-existing` already does
    // today -- this role needs no Model Registry write permission at all.

    new CfnOutput(this, "TrainingDataBucketName", { value: this.bucket.bucketName });
    new CfnOutput(this, "SageMakerExecutionRoleArn", { value: this.sageMakerExecutionRole.roleArn });
  }
}
