import { Stage, StageProps } from "aws-cdk-lib";
import type { Construct } from "constructs";
import { ApiStack } from "./api-stack";
import { DataStack } from "./data-stack";
import { MlHostingStack } from "./ml-hosting-stack";
import { WebStack } from "./web-stack";

export interface TranslatorStageProps extends StageProps {
  environmentName: string;
  /** See `WebStackProps.siteContentPath`. */
  webSiteContentPath: string;
}

// The approved SageMaker Model Package version to deploy (issue #8). Bump
// this by hand, in a reviewed commit/PR, whenever a better model is
// approved -- see `MlHostingStack`'s own docstring for why this is a bare
// version number (never a full ARN, which would embed a real AWS account
// ID) and `ml/README.md`'s "Serving" section for how version 4 (BLEU 8.9 /
// chrF 31.2, issue #82's subword-vocabulary run, repackaged with the fixed
// custom inference handler) was registered. Version 3 -- the first
// registration attempt -- was rejected: real endpoint testing found two
// real bugs in its inference code (see ml/deployment/inference.py's
// output_fn/_strip_leading_direction_tag docstrings), fixed before this
// version was registered.
const DEV_MODEL_PACKAGE_VERSION = 4;

/**
 * One promotion target (dev/qa/prod) for the CDK Pipelines deployment
 * pipeline (ADR 0004). Wraps every stack that should exist in a given
 * environment/account — `WebStack`, `DataStack`, `ApiStack`, and (dev only
 * for now) `MlHostingStack` (see ADR 0001).
 */
export class TranslatorStage extends Stage {
  constructor(scope: Construct, id: string, props: TranslatorStageProps) {
    super(scope, id, props);

    new WebStack(this, "Web", {
      environmentName: props.environmentName,
      siteContentPath: props.webSiteContentPath,
    });

    const dataStack = new DataStack(this, "Data", {
      environmentName: props.environmentName,
    });

    new ApiStack(this, "Api", {
      environmentName: props.environmentName,
    });

    // Model Registry entries are account-scoped, and only "dev" has a
    // trained, registered, approved model right now -- training only runs
    // against dev's corpus bucket (ml/training/submit_job.py's
    // --environment default). Promoting a model to qa/prod would need
    // either a duplicated registration there or cross-account Model
    // Registry sharing, neither of which ADR 0001 addresses -- deferred
    // rather than guessed at here (see MlHostingStack's own docstring).
    if (props.environmentName === "dev") {
      new MlHostingStack(this, "MlHosting", {
        environmentName: props.environmentName,
        modelPackageVersion: DEV_MODEL_PACKAGE_VERSION,
        modelDataBucket: dataStack.bucket,
      });
    }
  }
}
