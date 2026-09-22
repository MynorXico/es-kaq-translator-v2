import * as path from "node:path";
import { CfnOutput, Duration, Stack, StackProps } from "aws-cdk-lib";
import { HttpApi } from "aws-cdk-lib/aws-apigatewayv2";
import { HttpLambdaIntegration } from "aws-cdk-lib/aws-apigatewayv2-integrations";
import { PolicyStatement } from "aws-cdk-lib/aws-iam";
import { DockerImageCode, DockerImageFunction } from "aws-cdk-lib/aws-lambda";
import type { Construct } from "constructs";

export interface ApiStackProps extends StackProps {
  environmentName: string;
  /**
   * Name (not ARN) of the SageMaker Serverless Inference endpoint this
   * environment's API should call (issues #8/#9). Passed to the Lambda as
   * the `SAGEMAKER_ENDPOINT_NAME` env var, and used to scope the Lambda's
   * `sagemaker:InvokeEndpoint` IAM grant to exactly that endpoint's ARN.
   *
   * The endpoint is referenced by name rather than a live CDK cross-stack
   * construct because the stack that creates it (#8) doesn't exist yet at
   * the time `ApiStack` is defined -- see this class's doc comment for the
   * full contract. Defaults to a documented per-environment naming
   * convention (`traductor-kaqchikel-translate-<environmentName>`);
   * override this prop from `app-stage.ts` if #8 ends up naming the real
   * endpoint differently.
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
 * ## Environment-variable contract for #9
 *
 * `#9` (wiring `/v1/translate` to the real SageMaker endpoint) should read
 * these from `os.environ` in `apps/api/app`:
 *
 * | Env var                  | Purpose |
 * |---------------------------|---------|
 * | `SAGEMAKER_ENDPOINT_NAME` | Name (not ARN) of the SageMaker Serverless Inference endpoint to invoke via `boto3`'s `sagemaker-runtime` `invoke_endpoint`. The Lambda's execution role is granted `sagemaker:InvokeEndpoint` scoped to exactly this endpoint's ARN in this account/region -- nothing broader (no wildcard resource, no `AmazonSageMakerFullAccess`). |
 * | `ALLOWED_ORIGINS`         | Pre-existing CORS config (see `apps/api/app/config.py`, issue #46). Deliberately left unset here (falls back to the local dev origin) since `apps/web`'s custom domain (issue #84) hasn't landed yet -- wire this up once that domain exists, rather than guessing its value now. |
 *
 * If #8 names its real endpoint differently than this stack's default
 * convention, pass `sageMakerEndpointName` explicitly from
 * `app-stage.ts` (or update the default here) -- don't hardcode a
 * different name inside `apps/api` itself.
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
  }
}
