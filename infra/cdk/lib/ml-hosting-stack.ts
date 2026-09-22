import { ArnFormat, CfnOutput, Stack, StackProps } from "aws-cdk-lib";
import { CfnEndpoint, CfnEndpointConfig, CfnModel } from "aws-cdk-lib/aws-sagemaker";
import { PolicyStatement, Role, ServicePrincipal } from "aws-cdk-lib/aws-iam";
import type { IBucket } from "aws-cdk-lib/aws-s3";
import type { Construct } from "constructs";

export interface MlHostingStackProps extends StackProps {
  environmentName: string;
  /**
   * The approved SageMaker Model Package *version* to deploy (issue #8).
   * Bump this by hand (in a reviewed commit/PR) whenever a better model is
   * approved -- the same "versioned constant updated by hand" pattern as
   * `ml/training/submit_job.py`'s `CORPUS_VERSION`. See this file's own
   * docstring for why this is a bare version number, not a full ARN.
   */
  modelPackageVersion: number;
  /**
   * The environment's training-data bucket (`DataStack.bucket`) -- the
   * inference execution role needs read access to it to load the model
   * artifact the Model Package points at. Passed in as a real construct
   * reference (not re-derived from a name string) so the IAM grant is a
   * real least-privilege `grantRead`, not a hand-written ARN pattern.
   */
  modelDataBucket: IBucket;
}

// Matches ml/training/submit_job.py's DEFAULT_MODEL_PACKAGE_GROUP_NAME --
// not sensitive (unlike an account ID), so it's fine to duplicate as a
// documented constant here rather than pass it through as a prop.
const MODEL_PACKAGE_GROUP_NAME = "traductor-kaqchikel-es-cak";

// SageMaker Serverless Inference's memory tiers are 1024MB increments up to
// 6144MB (the max as of this writing). The fine-tuned checkpoint's
// `model.safetensors` alone is ~2.1GB (fp32, ~196k-token extended
// vocabulary -- see ml/README.md's vocabulary extension sections), which
// plus interpreter/framework/runtime overhead pushes close to the limit, so
// this uses the maximum tier rather than guessing lower and hitting
// out-of-memory failures on every cold start.
const SERVERLESS_MEMORY_SIZE_IN_MB = 6144;
// Serverless Inference's own hard cap is 200; this project has low, bursty
// traffic (ADR 0001's rationale for choosing serverless at all), so a
// small concurrency limit is enough while keeping a cost/blast-radius cap.
const SERVERLESS_MAX_CONCURRENCY = 2;

/**
 * SageMaker Serverless Inference endpoint serving the fine-tuned Spanish
 * <->Kaqchikel model (issue #8, ADR 0001), referencing an already-approved
 * Model Package from SageMaker Model Registry (`DEFAULT_MODEL_PACKAGE_GROUP_
 * NAME`, registered by `ml/deployment/deploy.py`).
 *
 * ## Why the Model Package ARN is *constructed*, not hardcoded
 *
 * A full Model Package ARN
 * (`arn:aws:sagemaker:<region>:<account-id>:model-package/<group>/<version>`)
 * necessarily embeds the real AWS account ID -- which CLAUDE.md/`docs/
 * runbooks/aws-account-bootstrap.md` forbid committing to this public repo,
 * even in infra code. Only `modelPackageVersion` (a small integer, not
 * sensitive) is hardcoded (by the caller, in `app-stage.ts`); the full ARN
 * is built at synth time via `this.formatArn(...)`, using the stack's own
 * `account`/`region` (a CDK token in a real deploy, a fake test value in
 * `test/ml-hosting-stack.test.ts` -- never a literal real ID in source).
 *
 * ## Scope: dev only, for now
 *
 * SageMaker Model Registry entries are account-scoped, and only the `dev`
 * account has a trained, registered, approved model today (training only
 * runs against `dev`'s corpus bucket, per `ml/training/submit_job.py`'s
 * `--environment` default). Promoting a model to qa/prod (separate AWS
 * accounts) would need either a duplicated registration there or
 * cross-account Model Registry sharing -- neither is addressed by ADR 0001,
 * so this is deliberately out of scope here; `app-stage.ts` only
 * instantiates this stack for `environmentName === "dev"`.
 */
export class MlHostingStack extends Stack {
  constructor(scope: Construct, id: string, props: MlHostingStackProps) {
    super(scope, id, props);

    const modelPackageArn = this.formatArn({
      // Explicit "aws" partition (this project only ever deploys to the
      // commercial partition, per ADR 0001/ADR 0004's account layout) so
      // this resolves to a plain literal string at synth time instead of
      // an `Fn::Join` built from the `AWS::Partition` pseudo parameter --
      // easier to assert against directly in tests, and there's no real
      // scenario here where the partition would ever differ from "aws".
      partition: "aws",
      service: "sagemaker",
      resource: "model-package",
      resourceName: `${MODEL_PACKAGE_GROUP_NAME}/${props.modelPackageVersion}`,
    });

    const executionRole = new Role(this, "SageMakerInferenceExecutionRole", {
      assumedBy: new ServicePrincipal("sagemaker.amazonaws.com"),
      description:
        "Least-privilege execution role for the SageMaker Serverless Inference endpoint: " +
        "read-only access to this environment's training data bucket (to load the model " +
        "artifact) plus CloudWatch Logs write access scoped to SageMaker's own endpoint log " +
        "group namespace. Deliberately not the broad AWS-managed AmazonSageMakerFullAccess " +
        "policy.",
    });

    // Scoped to this bucket's ARN (and its objects) only, read-only --
    // unlike DataStack's training role, this one never needs to write.
    props.modelDataBucket.grantRead(executionRole);

    // Same ArnFormat/leading-slash reasoning as DataStack's training-job
    // log group grant, but for the endpoint's own log group namespace
    // (`/aws/sagemaker/Endpoints/...`, distinct from `/aws/sagemaker/
    // TrainingJobs/...`).
    executionRole.addToPolicy(
      new PolicyStatement({
        actions: ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
        resources: [
          this.formatArn({
            service: "logs",
            resource: "log-group",
            resourceName: "/aws/sagemaker/Endpoints/*",
            arnFormat: ArnFormat.COLON_RESOURCE_NAME,
          }),
        ],
      }),
    );

    const namePrefix = `${MODEL_PACKAGE_GROUP_NAME}-${props.environmentName}`;

    const model = new CfnModel(this, "Model", {
      modelName: namePrefix,
      executionRoleArn: executionRole.roleArn,
      containers: [{ modelPackageName: modelPackageArn }],
    });

    const endpointConfig = new CfnEndpointConfig(this, "EndpointConfig", {
      endpointConfigName: namePrefix,
      productionVariants: [
        {
          variantName: "AllTraffic",
          modelName: model.attrModelName,
          serverlessConfig: {
            memorySizeInMb: SERVERLESS_MEMORY_SIZE_IN_MB,
            maxConcurrency: SERVERLESS_MAX_CONCURRENCY,
          },
        },
      ],
    });
    endpointConfig.addResourceDependency(model);

    const endpoint = new CfnEndpoint(this, "Endpoint", {
      endpointName: namePrefix,
      endpointConfigName: endpointConfig.attrEndpointConfigName,
    });
    endpoint.addResourceDependency(endpointConfig);

    new CfnOutput(this, "MlEndpointName", { value: namePrefix });
  }
}
