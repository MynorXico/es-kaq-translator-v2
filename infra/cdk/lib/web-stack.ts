import { CfnOutput, Duration, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import {
  AllowedMethods,
  Distribution,
  ViewerProtocolPolicy,
} from "aws-cdk-lib/aws-cloudfront";
import { S3BucketOrigin } from "aws-cdk-lib/aws-cloudfront-origins";
import { BlockPublicAccess, Bucket } from "aws-cdk-lib/aws-s3";
import { BucketDeployment, Source } from "aws-cdk-lib/aws-s3-deployment";
import type { Construct } from "constructs";

export interface WebStackProps extends StackProps {
  environmentName: string;
  /**
   * Absolute path to the built static site content (e.g. `apps/web`'s
   * Vite `dist/` output) to deploy into this environment's bucket. A CDK
   * asset is created from this directory at synth time, so it must exist
   * on disk by then -- see `lib/pipeline-stack.ts` for how a real deploy
   * builds `apps/web` before running `cdk synth`, and
   * `test/fixtures/site` for the small placeholder used by CDK assertion
   * tests and the CI-safe `useMockAccounts=true` synth path instead.
   */
  siteContentPath: string;
}

/**
 * Static SPA hosting for `apps/web`: a private S3 bucket (never public --
 * `BlockPublicAccess.BLOCK_ALL`), served through a CloudFront distribution
 * using origin access control (OAC), with SPA-routing fallback (403/404 ->
 * `index.html`) and cache invalidation wired into every deploy.
 *
 * The custom `app[-<env>].traductorkaqchikel.com` domain (ACM certificate +
 * Route 53 alias record) is deliberately NOT part of this stack yet -- see
 * issue #84's PR description for the full reasoning: the hosted zone lives
 * in the `translator-tooling` account (docs/runbooks/domain-and-dns.md)
 * while this stack deploys into a different account per environment, and
 * neither ACM DNS validation nor a Route 53 alias record can be written
 * cross-account without a new cross-account IAM mechanism (Route 53 has no
 * resource-based/bucket-policy-style cross-account grant, unlike
 * S3/KMS/SNS). That's a real architecture decision, tracked as a follow-up
 * in issue #99 rather than silently bolted on here. Until it's resolved,
 * each environment is reachable at its distribution's default
 * `*.cloudfront.net` domain (HTTPS out of the box, no ACM cert needed).
 */
export class WebStack extends Stack {
  constructor(scope: Construct, id: string, props: WebStackProps) {
    super(scope, id, props);

    const siteBucket = new Bucket(this, "SiteBucket", {
      bucketName: `traductor-kaqchikel-web-${props.environmentName}`,
      blockPublicAccess: BlockPublicAccess.BLOCK_ALL,
      removalPolicy: RemovalPolicy.RETAIN,
    });

    const distribution = new Distribution(this, "Distribution", {
      defaultBehavior: {
        origin: S3BucketOrigin.withOriginAccessControl(siteBucket),
        viewerProtocolPolicy: ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        allowedMethods: AllowedMethods.ALLOW_GET_HEAD,
      },
      defaultRootObject: "index.html",
      // A Vite SPA handles its own client-side routing; any path CloudFront
      // can't find in the bucket (a client route, not a real object -- S3
      // origins via OAC report those as 403, not 404) should still serve
      // `index.html` with a 200 so the SPA's own router can take over.
      errorResponses: [
        { httpStatus: 403, responseHttpStatus: 200, responsePagePath: "/index.html", ttl: Duration.seconds(0) },
        { httpStatus: 404, responseHttpStatus: 200, responsePagePath: "/index.html", ttl: Duration.seconds(0) },
      ],
    });

    new BucketDeployment(this, "SiteDeployment", {
      sources: [Source.asset(props.siteContentPath)],
      destinationBucket: siteBucket,
      // Vite's build output already versions hashed asset filenames, so a
      // short/no explicit cache policy here plus an invalidation on every
      // deploy (below) is enough for a new deploy to be visible immediately,
      // without relying on a manual `aws cloudfront create-invalidation`.
      distribution,
      distributionPaths: ["/*"],
    });

    new CfnOutput(this, "SiteBucketName", { value: siteBucket.bucketName });
    new CfnOutput(this, "DistributionDomainName", { value: distribution.distributionDomainName });
    new CfnOutput(this, "DistributionId", { value: distribution.distributionId });
  }
}
