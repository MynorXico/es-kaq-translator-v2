import { CfnOutput, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
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
    this.sageMakerExecutionRole.addToPolicy(
      new PolicyStatement({
        actions: ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
        resources: [
          this.formatArn({
            service: "logs",
            resource: "log-group",
            resourceName: "aws/sagemaker/*",
          }),
        ],
      }),
    );

    new CfnOutput(this, "TrainingDataBucketName", { value: this.bucket.bucketName });
    new CfnOutput(this, "SageMakerExecutionRoleArn", { value: this.sageMakerExecutionRole.roleArn });
  }
}
