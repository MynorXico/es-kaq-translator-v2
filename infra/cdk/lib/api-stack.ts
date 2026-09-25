import * as path from "node:path";
import { CfnOutput, Duration, Stack, StackProps } from "aws-cdk-lib";
import { HttpApi } from "aws-cdk-lib/aws-apigatewayv2";
import { HttpLambdaIntegration } from "aws-cdk-lib/aws-apigatewayv2-integrations";
import { Alarm, ComparisonOperator, TreatMissingData } from "aws-cdk-lib/aws-cloudwatch";
import { PolicyStatement } from "aws-cdk-lib/aws-iam";
import { DockerImageCode, DockerImageFunction } from "aws-cdk-lib/aws-lambda";
import type { Construct } from "constructs";
import { webHostName } from "./config";

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
 * Deploys `apps/api` (FastAPI) as a Lambda container image behind an API
 * Gateway HTTP API (ADR 0001). See `apps/api/Dockerfile` for the image
 * build (uv-resolved runtime deps + app code, AWS's own Lambda Python base
 * image) and `apps/api/app/lambda_handler.py` for the Mangum ASGI adapter
 * that lets the same FastAPI app run locally (uvicorn) and in Lambda.
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
  public readonly httpApi: HttpApi;

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

    this.httpApi = new HttpApi(this, "HttpApi", {
      apiName: `traductor-kaqchikel-api-${props.environmentName}`,
      description: `Traductor Kaqchikel translation API (${props.environmentName})`,
      // apps/api owns its own routing (FastAPI); proxy every path/method
      // straight through via a single $default route rather than
      // re-declaring apps/api's routes here too.
      defaultIntegration: new HttpLambdaIntegration("ApiIntegration", this.apiFunction),
    });

    new CfnOutput(this, "ApiUrl", { value: this.httpApi.apiEndpoint });
    new CfnOutput(this, "ApiFunctionName", { value: this.apiFunction.functionName });

    // Basic cost/operational-visibility alarms (issue #47). SageMaker
    // Serverless Inference bills per invocation and scales to zero (ADR
    // 0001), so an elevated error rate or latency is both a user-facing
    // problem and a signal something's gone wrong upstream (e.g. endpoint
    // cold starts, throttling, a bad model deployment) worth surfacing.
    //
    // Sourced from the HTTP API's own built-in CloudWatch metrics (5xx
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
      metric: this.httpApi.metricServerError({ period: Duration.minutes(5) }),
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
      metric: this.httpApi.metricLatency({ period: Duration.minutes(5), statistic: "p90" }),
      // Generous relative to typical translation latency: SageMaker
      // Serverless Inference cold-starts (scale-to-zero) can take several
      // seconds on their own, so this should catch a genuinely degraded
      // endpoint, not a single cold start.
      threshold: Duration.seconds(10).toMilliseconds(),
      evaluationPeriods: 3,
      comparisonOperator: ComparisonOperator.GREATER_THAN_THRESHOLD,
      treatMissingData: TreatMissingData.NOT_BREACHING,
    });
  }
}
