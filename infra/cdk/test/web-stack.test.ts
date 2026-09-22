import path from "node:path";
import { App } from "aws-cdk-lib";
import { Match, Template } from "aws-cdk-lib/assertions";
import { describe, expect, it } from "vitest";
import { WebStack } from "../lib/web-stack";

// Fake, obviously-non-real account ID (CLAUDE.md never allows a real one in tests).
const FAKE_ENV = { account: "222222222222", region: "us-east-1" };

// A tiny committed fixture directory, not a real `apps/web` Vite build --
// keeps this suite hermetic and fast (no dependency on `apps/web/dist`
// existing). See lib/pipeline-stack.ts for how a real deploy wires the
// actual build output in.
const FIXTURE_SITE_CONTENT_PATH = path.join(__dirname, "fixtures/site");

function synthWebStack() {
  const app = new App();
  const stack = new WebStack(app, "TestWebStack", {
    environmentName: "test",
    siteContentPath: FIXTURE_SITE_CONTENT_PATH,
    env: FAKE_ENV,
  });
  const template = Template.fromStack(stack);
  return { stack, template };
}

describe("WebStack", () => {
  it("synthesizes a private S3 bucket for the given environment", () => {
    const { template } = synthWebStack();

    template.hasResourceProperties("AWS::S3::Bucket", {
      BucketName: "traductor-kaqchikel-web-test",
      PublicAccessBlockConfiguration: {
        BlockPublicAcls: true,
        BlockPublicPolicy: true,
        IgnorePublicAcls: true,
        RestrictPublicBuckets: true,
      },
    });
  });

  it("never attaches a bucket policy allowing public/anonymous access", () => {
    const { template } = synthWebStack();

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

  it("fronts the bucket with a CloudFront distribution using origin access control, not a public bucket", () => {
    const { template } = synthWebStack();

    template.resourceCountIs("AWS::CloudFront::Distribution", 1);
    template.resourceCountIs("AWS::CloudFront::OriginAccessControl", 1);

    template.hasResourceProperties("AWS::CloudFront::Distribution", {
      DistributionConfig: Match.objectLike({
        DefaultRootObject: "index.html",
        Origins: Match.arrayWith([
          Match.objectLike({
            OriginAccessControlId: Match.anyValue(),
            S3OriginConfig: Match.anyValue(),
          }),
        ]),
      }),
    });
  });

  it("redirects HTTP to HTTPS at the viewer", () => {
    const { template } = synthWebStack();

    template.hasResourceProperties("AWS::CloudFront::Distribution", {
      DistributionConfig: Match.objectLike({
        DefaultCacheBehavior: Match.objectLike({
          ViewerProtocolPolicy: "redirect-to-https",
        }),
      }),
    });
  });

  it("rewrites 403/404 (SPA client-side routes) to index.html with a 200 so client routing works", () => {
    const { template } = synthWebStack();

    template.hasResourceProperties("AWS::CloudFront::Distribution", {
      DistributionConfig: Match.objectLike({
        CustomErrorResponses: Match.arrayWith([
          Match.objectLike({
            ErrorCode: 403,
            ResponseCode: 200,
            ResponsePagePath: "/index.html",
          }),
          Match.objectLike({
            ErrorCode: 404,
            ResponseCode: 200,
            ResponsePagePath: "/index.html",
          }),
        ]),
      }),
    });
  });

  it("deploys the given site content into the bucket and invalidates the distribution's cache on deploy", () => {
    const { template } = synthWebStack();

    const deployments = template.findResources("Custom::CDKBucketDeployment");
    const [deployment] = Object.values(deployments);
    expect(deployment).toBeDefined();
    // Asserts the actual invalidation paths, not just that the property is
    // set -- a regression to `[]` (or omitting it) would silently
    // invalidate nothing on deploy while still passing a `toBeDefined()`
    // check.
    expect(deployment.Properties.DistributionPaths).toEqual(["/*"]);
  });

  it("exposes the bucket name and CloudFront domain as stack outputs", () => {
    const { template } = synthWebStack();

    template.hasOutput("SiteBucketName", {});
    template.hasOutput("DistributionDomainName", {});
  });
});
