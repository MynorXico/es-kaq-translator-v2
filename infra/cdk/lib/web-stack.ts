import { CfnOutput, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import { BlockPublicAccess, Bucket } from "aws-cdk-lib/aws-s3";
import type { Construct } from "constructs";

export interface WebStackProps extends StackProps {
  environmentName: string;
}

// Placeholder for the static SPA hosting bucket; CloudFront comes later (ADR 0001).
export class WebStack extends Stack {
  constructor(scope: Construct, id: string, props: WebStackProps) {
    super(scope, id, props);

    const siteBucket = new Bucket(this, "SiteBucket", {
      bucketName: `traductor-kaqchikel-web-${props.environmentName}`,
      blockPublicAccess: BlockPublicAccess.BLOCK_ALL,
      removalPolicy: RemovalPolicy.RETAIN,
    });

    new CfnOutput(this, "SiteBucketName", { value: siteBucket.bucketName });
  }
}
