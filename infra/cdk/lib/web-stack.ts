import { CfnOutput, Duration, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import * as acm from "aws-cdk-lib/aws-certificatemanager";
import {
  AllowedMethods,
  Distribution,
  ViewerProtocolPolicy,
} from "aws-cdk-lib/aws-cloudfront";
import { S3BucketOrigin } from "aws-cdk-lib/aws-cloudfront-origins";
import { BlockPublicAccess, Bucket } from "aws-cdk-lib/aws-s3";
import { BucketDeployment, Source } from "aws-cdk-lib/aws-s3-deployment";
import type { Construct } from "constructs";
import { webHostName } from "./config";

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
  /**
   * The domain name previously recorded (in the
   * `/traductor-kaqchikel/domains/{environmentName}-web-cert-domain` SSM
   * parameter, read by `bin/app.ts`) as having a successfully DNS-validated
   * ACM certificate for this environment. `undefined` means "no certificate
   * has ever been validated for this environment yet" -- either the SSM
   * parameter doesn't exist (first-ever cert), or it hasn't been created.
   * Compared at synth time against `webHostName(environmentName)`; see
   * `assertCertificateChangeAcknowledged` and ADR 0007.
   */
  previouslyValidatedDomainName?: string;
  /**
   * Explicit human acknowledgement -- CDK context `newCertificateAck=true`,
   * threaded down via `bin/app.ts`/`app-stage.ts` -- that this synth is
   * expected to create a new/changed ACM certificate for this environment,
   * and that a human will run the resulting deploy as an isolated,
   * watched `cdk deploy` rather than an unattended pipeline run (ADR 0007:
   * a stack-level `acm.Certificate` with DNS validation blocks on
   * `CREATE_IN_PROGRESS` until the validation CNAME is added by hand).
   * Defaults to `false`.
   */
  newCertificateAck?: boolean;
}

/**
 * Enforces ADR 0007's synth-time guard: a stack that would introduce a
 * *new* (or changed) ACM certificate for a given environment's domain must
 * not synthesize unless a human has explicitly acknowledged it via
 * `newCertificateAck`. This matters because a DNS-validated `acm.Certificate`
 * blocks CloudFormation on `CREATE_IN_PROGRESS` until a human manually adds
 * the validation CNAME into the `translator-tooling` hosted zone (Route 53
 * has no cross-account write mechanism -- see ADR 0007) -- an unattended
 * Dev/Qa pipeline run (ADR 0004) hitting this unacknowledged would simply
 * wedge itself waiting for nobody.
 *
 * A missing/absent `previouslyValidatedDomainName` (first-ever certificate
 * for this environment) is treated the same as a mismatch -- fail-safe,
 * never an automatic pass.
 *
 * Kept as a small pure function, separate from `WebStack`'s construction,
 * so it's directly unit-testable without synthesizing a stack.
 */
export function assertCertificateChangeAcknowledged(params: {
  domainName: string;
  previouslyValidatedDomainName: string | undefined;
  newCertificateAck: boolean;
}): void {
  const { domainName, previouslyValidatedDomainName, newCertificateAck } = params;

  if (previouslyValidatedDomainName === domainName) {
    return;
  }
  if (newCertificateAck) {
    return;
  }

  throw new Error(
    `WebStack would create a new or changed ACM certificate for "${domainName}" ` +
      `(previously validated domain: ${previouslyValidatedDomainName ?? "<none recorded>"}). ` +
      "Per ADR 0007, this deploy will block on a manual DNS validation step and must be run " +
      "as an isolated, human-watched `cdk deploy`, not left to an unattended pipeline run. " +
      "Re-run with `--context newCertificateAck=true` once you are ready to watch it, then " +
      "update the `/traductor-kaqchikel/domains/{environmentName}-web-cert-domain` SSM " +
      "parameter once validation completes -- see docs/runbooks/domain-and-dns.md.",
  );
}

/**
 * Static SPA hosting for `apps/web`: a private S3 bucket (never public --
 * `BlockPublicAccess.BLOCK_ALL`), served through a CloudFront distribution
 * using origin access control (OAC), with SPA-routing fallback (403/404 ->
 * `index.html`) and cache invalidation wired into every deploy.
 *
 * The custom `app[-<env>].traductorkaqchikel.com` domain is wired up per
 * ADR 0007: the ACM certificate is created directly in this stack's own
 * account with `acm.CertificateValidation.fromDns()` and **no** `hostedZone`
 * argument, since the `traductorkaqchikel.com` hosted zone lives in a
 * different account (`translator-tooling`) than this stack -- Route 53 has
 * no resource-based cross-account grant, so CDK must never attempt that
 * write itself. A human completes DNS validation and the final alias
 * record by hand/script against the tooling account; see
 * `docs/runbooks/domain-and-dns.md` and `infra/scripts/upsert-domain-record.sh`.
 * `assertCertificateChangeAcknowledged` (above) guards against an
 * unattended pipeline run introducing a new certificate and silently
 * wedging on the resulting manual validation step.
 */
export class WebStack extends Stack {
  constructor(scope: Construct, id: string, props: WebStackProps) {
    super(scope, id, props);

    const domainName = webHostName(props.environmentName);
    assertCertificateChangeAcknowledged({
      domainName,
      previouslyValidatedDomainName: props.previouslyValidatedDomainName,
      newCertificateAck: props.newCertificateAck ?? false,
    });

    const certificate = new acm.Certificate(this, "Certificate", {
      domainName,
      // No `hostedZone` argument -- see this class's docstring and ADR 0007
      // for why CDK must never attempt to write the validation record
      // itself (the zone lives in a different AWS account).
      validation: acm.CertificateValidation.fromDns(),
    });

    const siteBucket = new Bucket(this, "SiteBucket", {
      bucketName: `traductor-kaqchikel-web-${props.environmentName}`,
      blockPublicAccess: BlockPublicAccess.BLOCK_ALL,
      removalPolicy: RemovalPolicy.RETAIN,
    });

    const distribution = new Distribution(this, "Distribution", {
      domainNames: [domainName],
      certificate,
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
    // Lets a human run `aws acm describe-certificate --certificate-arn
    // <this>` to retrieve the pending validation CNAME (docs/runbooks/
    // domain-and-dns.md) without hunting for it via `list-certificates`.
    new CfnOutput(this, "CertificateArn", { value: certificate.certificateArn });
  }
}
