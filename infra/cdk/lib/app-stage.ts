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
  /** See `WebStackProps.previouslyValidatedDomainName` (ADR 0007). */
  previouslyValidatedWebDomainName?: string;
  /** See `WebStackProps.newCertificateAck` (ADR 0007). */
  newCertificateAck?: boolean;
}

// The approved SageMaker Model Package version to deploy (issue #8). Bump
// this by hand, in a reviewed commit/PR, whenever a better model is
// approved -- see `MlHostingStack`'s own docstring for why this is a bare
// version number (never a full ARN, which would embed a real AWS account
// ID) and `ml/README.md`'s "Serving" section for the full history:
// - Version 3 (issue #82's subword-vocabulary run) -- rejected: real
//   endpoint testing found two real bugs in its inference code (see
//   ml/deployment/inference.py's output_fn/_strip_leading_direction_tag
//   docstrings), fixed before the next registration.
// - Version 4 (same weights as v3, repackaged with the fixed inference
//   handler) -- approved and deployed. Originally registered at BLEU 8.9/
//   chrF 31.2; corrected in place to BLEU 13.3/chrF 35.4 after issues #106
//   (direction-tag leak) and #116 (word-boundary decode bug) were found to
//   have been silently corrupting every prior evaluation.
// - Version 5 (issue #125's vocab-scoping fix, a fresh 5-epoch run) --
//   rejected: BLEU 6.0/chrF 26.3, but confounded by far less cumulative
//   training exposure than v4's 3+5+5-epoch continuation chain (13 total
//   epochs vs. 5), not evidence the fix itself hurts.
// - Version 6 -- never deployed. Registered by hand directly via
//   `aws sagemaker create-model-package` (skipping `ml/deployment/
//   deploy.py`'s repackaging step) with customer metadata only and no
//   `InferenceSpecification`, which `MlHostingStack`'s `Model` resource
//   requires -- deploying it failed with "Inference specification is not
//   present". Deleted; version numbers are never reused after deletion.
// - Version 7 (same underlying run as the deleted v6: issue #125's fix, a
//   fresh 13-epoch run matching v4's cumulative training exposure for a
//   clean comparison) -- registered correctly via `deployment.deploy`,
//   approved, and deployed here. `deploy.py` auto-parses BLEU/chrF from
//   the training run's own inline eval (12.3/35.1); customer metadata was
//   corrected in place to 14.0/36.5, a dedicated `evaluate_checkpoint.py`
//   re-eval on the same full validation set and weights -- a real,
//   uncounfounded improvement over v4 (13.3/35.4), confirming the
//   vocab-scoping fix helps modestly. The two measurements' gap (14.0 vs.
//   12.3) is not yet reconciled -- see the model package's own
//   `metric_note` and issue #125.
const DEV_MODEL_PACKAGE_VERSION = 7;

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
      previouslyValidatedDomainName: props.previouslyValidatedWebDomainName,
      newCertificateAck: props.newCertificateAck,
    });

    const dataStack = new DataStack(this, "Data", {
      environmentName: props.environmentName,
    });

    // Model Registry entries are account-scoped, and only "dev" has a
    // trained, registered, approved model right now -- training only runs
    // against dev's corpus bucket (ml/training/submit_job.py's
    // --environment default). Promoting a model to qa/prod would need
    // either a duplicated registration there or cross-account Model
    // Registry sharing, neither of which ADR 0001 addresses -- deferred
    // rather than guessed at here (see MlHostingStack's own docstring).
    //
    // Created before ApiStack (rather than after, as originally written) so
    // its real `endpointName` can be passed into ApiStack below -- issue #9
    // found that ApiStack's own default endpoint-name convention
    // (`traductor-kaqchikel-translate-<environmentName>`) never matched the
    // real endpoint this stack creates
    // (`traductor-kaqchikel-es-cak-<environmentName>`), because nothing
    // ever overrode it.
    let mlHostingStack: MlHostingStack | undefined;
    if (props.environmentName === "dev") {
      mlHostingStack = new MlHostingStack(this, "MlHosting", {
        environmentName: props.environmentName,
        modelPackageVersion: DEV_MODEL_PACKAGE_VERSION,
        modelDataBucket: dataStack.bucket,
      });
    }

    new ApiStack(this, "Api", {
      environmentName: props.environmentName,
      // Only pass an explicit override where a real endpoint exists (dev,
      // for now -- see MlHostingStack's own "Scope: dev only" docstring).
      // qa/prod fall back to ApiStack's own default, which names a
      // not-yet-existing endpoint there -- an already-known gap (no
      // MlHostingStack there yet), not something this ticket can fix.
      sageMakerEndpointName: mlHostingStack?.endpointName,
    });
  }
}
