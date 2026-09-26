import * as path from "node:path";
import { CfnOutput, Duration, Stack, StackProps } from "aws-cdk-lib";
import { EndpointType, LambdaRestApi } from "aws-cdk-lib/aws-apigateway";
import { Alarm, ComparisonOperator, Metric, TreatMissingData } from "aws-cdk-lib/aws-cloudwatch";
import { PolicyStatement } from "aws-cdk-lib/aws-iam";
import { DockerImageCode, DockerImageFunction } from "aws-cdk-lib/aws-lambda";
import { CfnWebACL, CfnWebACLAssociation } from "aws-cdk-lib/aws-wafv2";
import type { Construct } from "constructs";
import { webHostName } from "./config";

/**
 * Rate-based rule threshold (ADR 0005): requests per 5-minute trailing
 * window per source IP before AWS WAF blocks that IP. "Low hundreds" per
 * the ADR -- a tuning parameter, not an architectural one, adjustable
 * without a new ADR. Also used as the `Retry-After` seconds value below,
 * since it matches the rate-based rule's (default) 5-minute evaluation
 * window.
 */
const WAF_RATE_LIMIT_PER_IP = 300;
const WAF_RATE_LIMIT_WINDOW_SECONDS = 300;

export interface ApiStackProps extends StackProps {
  environmentName: string;
  /**
   * Name (not ARN) of the SageMaker Serverless Inference endpoint this
   * environment's API should call (issues #8/#9). Passed to the Lambda as
   * the `SAGEMAKER_ENDPOINT_NAME` env var, and used to scope the Lambda's
   * `sagemaker:InvokeEndpoint` IAM grant to exactly that endpoint's ARN.
   *
   * The endpoint is referenced by name rather than a live CDK cross-stack
   * construct because `ApiStack` is defined independently of the stack
   * that creates it (`MlHostingStack`, #8) -- see this class's doc comment
   * for the full contract. Defaults to a per-environment naming convention
   * (`traductor-kaqchikel-translate-<environmentName>`) that does *not*
   * match any real endpoint; `app-stage.ts` overrides it with
   * `MlHostingStack`'s real `endpointName` for environments that actually
   * have one (dev, for now -- see #9). Environments without a real
   * `MlHostingStack` yet (qa/prod) fall back to this unmatched default,
   * a known, pre-existing gap.
   */
  sageMakerEndpointName?: string;
}

/**
 * Deploys `apps/api` (FastAPI) as a Lambda container image behind a
 * regional API Gateway REST API (v1), fronted by an AWS WAF `WebACL` with a
 * per-IP rate-based rule (ADR 0001, ADR 0005). See `apps/api/Dockerfile`
 * for the image build (uv-resolved runtime deps + app code, AWS's own
 * Lambda Python base image) and `apps/api/app/lambda_handler.py` for the
 * Mangum ASGI adapter that lets the same FastAPI app run locally (uvicorn)
 * and in Lambda.
 *
 * ## Why a REST API (v1), not an HttpApi (v2) -- issue #151, ADR 0005
 *
 * This stack originally used API Gateway's `HttpApi` (v2), which is
 * cheaper per-request but cannot be associated with an AWS WAF `WebACL` at
 * all -- WAF's `AssociateWebACL` only accepts a regional REST API (v1)
 * stage ARN. ADR 0005 adopted AWS WAF as the abuse-protection mechanism
 * for the public, unauthenticated `/v1/translate-jobs` routes fronting a
 * pay-per-invocation SageMaker Serverless endpoint, which made this
 * migration a required (if costlier) consequence, not a free side effect.
 * `EndpointType.REGIONAL` (not the CDK default `EDGE`) is required too:
 * an edge-optimized REST API is fronted by an AWS-managed CloudFront
 * distribution that isn't directly WAF-associable either.
 *
 * ## Environment-variable contract (read from `os.environ` in `apps/api/app`)
 *
 * | Env var                  | Purpose |
 * |---------------------------|---------|
 * | `SAGEMAKER_ENDPOINT_NAME` | Name (not ARN) of the SageMaker Serverless Inference endpoint `apps/api/app/translation.py` invokes via `boto3`'s `sagemaker-runtime` `invoke_endpoint` (#9). The Lambda's execution role is granted `sagemaker:InvokeEndpoint` scoped to exactly this endpoint's ARN in this account/region -- nothing broader (no wildcard resource, no `AmazonSageMakerFullAccess`). |
 * | `ALLOWED_ORIGINS`         | Pre-existing CORS config (see `apps/api/app/config.py`, issue #46). Set to this environment's real `apps/web` custom domain (`webHostName()`, the same SSOT `WebStack` uses) now that it exists for every environment (issues #84/#99) -- confirmed live that leaving this unset broke every real browser request with a CORS preflight failure, even though a direct `curl` against the API looked fine (curl doesn't enforce CORS). |
 *
 * See `sageMakerEndpointName` above for how this stack's default
 * `SAGEMAKER_ENDPOINT_NAME` convention gets overridden with the real
 * endpoint name where one exists.
 */
export class ApiStack extends Stack {
  public readonly apiFunction: DockerImageFunction;
  public readonly restApi: LambdaRestApi;
  public readonly webAcl: CfnWebACL;

  constructor(scope: Construct, id: string, props: ApiStackProps) {
    super(scope, id, props);

    const sageMakerEndpointName =
      props.sageMakerEndpointName ?? `traductor-kaqchikel-translate-${props.environmentName}`;

    this.apiFunction = new DockerImageFunction(this, "ApiFunction", {
      // apps/api is the Docker build context (Dockerfile lives there);
      // resolved relative to this file so it works regardless of cwd.
      code: DockerImageCode.fromImageAsset(path.join(__dirname, "../../../apps/api")),
      memorySize: 512,
      // Must stay below API Gateway's fixed 30s integration timeout, or a
      // slow (e.g. cold-start) invocation would 504 at the gateway instead
      // of returning the Lambda's own response.
      timeout: Duration.seconds(29),
      environment: {
        SAGEMAKER_ENDPOINT_NAME: sageMakerEndpointName,
        ALLOWED_ORIGINS: `https://${webHostName(props.environmentName)}`,
      },
      description: `Traductor Kaqchikel API (${props.environmentName})`,
    });

    // Least-privilege: invoke exactly this environment's SageMaker
    // endpoint, never a wildcard resource or a broad AWS-managed policy
    // like AmazonSageMakerFullAccess.
    this.apiFunction.addToRolePolicy(
      new PolicyStatement({
        actions: ["sagemaker:InvokeEndpoint"],
        resources: [
          this.formatArn({
            service: "sagemaker",
            resource: "endpoint",
            resourceName: sageMakerEndpointName,
          }),
        ],
      }),
    );

    this.restApi = new LambdaRestApi(this, "RestApi", {
      restApiName: `traductor-kaqchikel-api-${props.environmentName}`,
      description: `Traductor Kaqchikel translation API (${props.environmentName})`,
      handler: this.apiFunction,
      // apps/api owns its own routing (FastAPI); proxy every path/method
      // straight through via a single greedy {proxy+} resource + ANY
      // method rather than re-declaring apps/api's routes here too --
      // mirrors the HttpApi's single $default route this replaces.
      proxy: true,
      // Required for AWS WAF association (ADR 0005) -- see this class's
      // doc comment for why EDGE (the CDK default) doesn't work.
      endpointConfiguration: {
        types: [EndpointType.REGIONAL],
      },
      // Named per environment (dev/qa/prod) rather than left at the CDK
      // default ("prod" for every environment, which would confusingly
      // collide with the literal "prod" *environment* name in its own
      // stage path).
      deployOptions: {
        stageName: props.environmentName,
      },
      // No explicit stage-level throttling override: neither this stack's
      // prior HttpApi nor this RestApi sets one, so both rely on the same
      // AWS account/region default throttle limits as the coarse backstop
      // ADR 0005 describes underneath the per-IP WAF rule below -- there
      // was no existing explicit config here to carry over.
    });

    // `LambdaRestApi.url` always has a trailing slash (e.g. ".../dev/"),
    // unlike the old `HttpApi.apiEndpoint`. apps/web builds requests as
    // `${apiBaseUrl}/v1/translate...` (`apps/web/src/api.ts`) -- leaving
    // the trailing slash in would produce a double slash once
    // concatenated, which API Gateway's exact-path resource matching
    // won't route correctly. Stripped here, once, so nobody deploying
    // this has to rediscover it by hand when copying this output's value
    // into the `api-urls/<env>` SSM parameter `bin/app.ts` reads.
    const apiUrl = this.restApi.url.replace(/\/$/, "");
    new CfnOutput(this, "ApiUrl", { value: apiUrl });
    new CfnOutput(this, "ApiFunctionName", { value: this.apiFunction.functionName });

    // Same per-environment origin `apps/api/app/config.py`'s
    // `ALLOWED_ORIGINS` uses for `apps/api/app/main.py`'s CORSMiddleware
    // -- kept in sync by hand across the CDK (TypeScript)/FastAPI (Python)
    // boundary, since there's no shared config source between them.
    const allowedOrigin = `https://${webHostName(props.environmentName)}`;

    // AWS WAF rate limiting (ADR 0005, issue #151): the abuse-protection
    // mechanism for the public, unauthenticated /v1/translate-jobs routes
    // fronting a pay-per-invocation SageMaker Serverless endpoint. A
    // WebACL associated with the REST API's stage covers every route on
    // it by default (stage-scoped, not route-scoped), so both
    // /v1/translate-jobs and /v1/translate-jobs/{job_id} are covered
    // without listing them explicitly.
    const webAclName = `traductor-kaqchikel-api-${props.environmentName}`;
    this.webAcl = new CfnWebACL(this, "ApiWebAcl", {
      name: webAclName,
      // WAFv2's own Description schema disallows parentheses (unlike most
      // other AWS resources' free-text description fields in this stack),
      // so this deliberately doesn't follow the "Name (env)" phrasing used
      // elsewhere here.
      description: `Rate limiting for the Traductor Kaqchikel API, ${props.environmentName} environment`,
      // REGIONAL (not CLOUDFRONT): this WebACL protects a regional REST
      // API stage, created in the same region/account as ApiStack itself.
      scope: "REGIONAL",
      defaultAction: { allow: {} },
      visibilityConfig: {
        cloudWatchMetricsEnabled: true,
        metricName: `traductor-kaqchikel-api-${props.environmentName}`,
        sampledRequestsEnabled: true,
      },
      rules: [
        {
          name: "RateLimitPerSourceIp",
          priority: 0,
          statement: {
            rateBasedStatement: {
              aggregateKeyType: "IP",
              limit: WAF_RATE_LIMIT_PER_IP,
            },
          },
          action: {
            block: {
              // Custom 429 (not WAF's default 403): the semantically
              // correct status for rate limiting, and lets apps/web (a
              // companion ticket, #152) distinguish this from other 4xx
              // client errors by status code alone.
              //
              // CORS headers are required here too, not optional: WAF
              // intercepts and returns this response *before* FastAPI's
              // CORSMiddleware ever runs, so without them a blocked
              // cross-origin request from apps/web has no
              // Access-Control-Allow-Origin header at all -- the browser's
              // `fetch()` then rejects it as an opaque CORS/network error
              // (indistinguishable from a real network failure), never
              // reaching apps/web's response-handling code far enough to
              // see the 429 status and classify it as rate-limited. Method/
              // header values mirror `apps/api/app/main.py`'s
              // `CORSMiddleware` config exactly (`allow_methods=["POST"]`,
              // `allow_headers=["Content-Type"]`).
              customResponse: {
                responseCode: 429,
                responseHeaders: [
                  { name: "Retry-After", value: `${WAF_RATE_LIMIT_WINDOW_SECONDS}` },
                  { name: "Access-Control-Allow-Origin", value: allowedOrigin },
                  { name: "Access-Control-Allow-Methods", value: "POST" },
                  { name: "Access-Control-Allow-Headers", value: "Content-Type" },
                ],
              },
            },
          },
          visibilityConfig: {
            cloudWatchMetricsEnabled: true,
            metricName: `traductor-kaqchikel-api-rate-limit-${props.environmentName}`,
            sampledRequestsEnabled: true,
          },
        },
      ],
    });

    // No higher-level RestApi/Stage construct prop exists for this
    // (unlike CloudFront's `Distribution.webAclId`) -- an explicit,
    // separate association resource is the only way to wire a WebACL to a
    // REST API stage.
    new CfnWebACLAssociation(this, "ApiWebAclAssociation", {
      resourceArn: this.restApi.deploymentStage.stageArn,
      webAclArn: this.webAcl.attrArn,
    });

    // Basic cost/operational-visibility alarms (issue #47). SageMaker
    // Serverless Inference bills per invocation and scales to zero (ADR
    // 0001), so an elevated error rate or latency is both a user-facing
    // problem and a signal something's gone wrong upstream (e.g. endpoint
    // cold starts, throttling, a bad model deployment) worth surfacing.
    //
    // Sourced from the REST API's own built-in CloudWatch metrics (5XX
    // count, p90 latency) rather than a custom metric filter over
    // `apps/api`'s structured request logs (see `app/observability.py`):
    // `TranslationServiceError` is caught and turned into a normal HTTP
    // 4xx/5xx response by `app/main.py`'s exception handler, so it never
    // shows up as a Lambda-level `Errors` metric -- the API Gateway layer
    // is the one place that reliably sees every response's status code.
    //
    // No alarm action (e.g. an SNS topic) is wired up yet -- these alarms
    // are visible in the CloudWatch console/API immediately, but nothing
    // pages/emails on them yet. Wiring actual on-call notification is a
    // separate concern this ticket doesn't scope.
    new Alarm(this, "ServerErrorRateAlarm", {
      alarmDescription: `Elevated 5xx error rate on the Traductor Kaqchikel API (${props.environmentName})`,
      metric: this.restApi.metricServerError({ period: Duration.minutes(5) }),
      threshold: 5,
      evaluationPeriods: 1,
      comparisonOperator: ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
      // Scale-to-zero means long stretches with literally zero requests
      // (and so no metric data points) are expected and not an incident
      // -- don't let CloudWatch's "missing data" default (breaching) fire
      // false alarms overnight.
      treatMissingData: TreatMissingData.NOT_BREACHING,
    });

    new Alarm(this, "HighLatencyAlarm", {
      alarmDescription: `Elevated latency on the Traductor Kaqchikel API (${props.environmentName})`,
      metric: this.restApi.metricLatency({ period: Duration.minutes(5), statistic: "p90" }),
      // Generous relative to typical translation latency: SageMaker
      // Serverless Inference cold-starts (scale-to-zero) can take several
      // seconds on their own, so this should catch a genuinely degraded
      // endpoint, not a single cold start.
      threshold: Duration.seconds(10).toMilliseconds(),
      evaluationPeriods: 3,
      comparisonOperator: ComparisonOperator.GREATER_THAN_THRESHOLD,
      treatMissingData: TreatMissingData.NOT_BREACHING,
    });

    // ADR 0005 requires this alarm, not as an optional nice-to-have: WAF
    // intercepts and blocks requests before they ever reach API Gateway,
    // so a block spike -- whether a real abuse pattern, or a threshold set
    // too low and blocking real users -- is otherwise invisible to the
    // two alarms above, which only ever see what WAF lets through.
    //
    // No CDK-native metric helper exists on `CfnWebACL` (unlike
    // `RestApiBase.metricServerError`/`metricLatency` above), so this is
    // a raw `Metric` against WAFv2's own published CloudWatch contract:
    // namespace `AWS/WAFV2`, dimensions `WebACL` + `Rule` ("ALL" for the
    // web-ACL-wide aggregate across every rule) + `Region`.
    new Alarm(this, "WafBlockedRequestsAlarm", {
      alarmDescription: `Elevated AWS WAF blocked-request rate on the Traductor Kaqchikel API (${props.environmentName})`,
      metric: new Metric({
        namespace: "AWS/WAFV2",
        metricName: "BlockedRequests",
        dimensionsMap: {
          WebACL: webAclName,
          Rule: "ALL",
          Region: this.region,
        },
        statistic: "Sum",
        period: Duration.minutes(5),
      }),
      // Any sustained blocking is worth a human looking at -- either real
      // abuse (working as intended) or a threshold set too aggressively
      // (a false-positive risk this ADR explicitly accepts and asks to be
      // monitored for, e.g. a shared-IP NAT of legitimate users).
      threshold: 1,
      evaluationPeriods: 3,
      comparisonOperator: ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
      treatMissingData: TreatMissingData.NOT_BREACHING,
    });
  }
}
