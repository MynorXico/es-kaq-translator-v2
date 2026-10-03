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
// - Versions 8-12 -- all rejected. #182's cheap-tier regularization
//   levers (dropout=0.3, label-smoothing=0.1, bpe-dropout-alpha=0.1; v8,
//   v9, v10) each underperformed v7 at 13 epochs, as did checkpoint
//   averaging (#192) and a cleaned-corpus retrain (#199, v12) -- none
//   beat v7 outside normal run-to-run noise (v11, a plain 13-epoch
//   reproduction of v7's exact config with no other changes, established
//   that noise band at roughly +-0.5 BLEU / +-0.2 chrF).
// - Version 13 (issue #205: does more training help under the current,
//   post-#82 vocab-extended setup, re-testing a "diminishing returns"
//   finding from before that extension existed) -- a fresh 20-epoch run,
//   otherwise identical to v11's config. BLEU 16.1/chrF 39.3, a real,
//   substantial improvement over the v11/v7 13-epoch reference (13.5-14.0
//   BLEU / 36.5-36.6 chrF) -- far outside the established noise band.
//   Approved, but registered via `training.submit_job` (traceability
//   only) -- its `InferenceSpecification` points at the *training* DLC
//   image, not an inference one. Deploying it directly failed:
//   CloudFormation's `Endpoint` update rolled back with a real ECR
//   permission error (the training image's repository doesn't grant
//   `sagemaker.amazonaws.com` pull access) -- the live endpoint was
//   unaffected throughout (stayed on v7). See issue #209: `deploy.py`
//   exists specifically to avoid this trap and was bypassed here by
//   mistake.
// - Version 14 -- the actually-deployable registration of v13's same
//   weights, produced via `ml/deployment/deploy.py --source-model-data-url
//   <v13's model.tar.gz>`: a real inference DLC image
//   (`huggingface-pytorch-inference`, not `-training`), but its repackaged
//   artifact was still ~30GB -- `deployment.package_model.
//   repackage_model_artifact` re-bundled the *entire* source artifact,
//   including the `checkpoints/` subtree (#182's per-epoch checkpointing,
//   added after this module was written). CloudFormation's `Endpoint`
//   update rolled back: "Failed to decompress and extract model contents
//   as their size is greater than available disk space." Live endpoint
//   unaffected throughout (stayed on v7). See issue #209/#211 (#211
//   fixed `package_model.py` to exclude `checkpoints/` from the
//   repackaged inference copy, verified directly against this exact
//   artifact's real tar member names).
// - Version 15 -- v13's weights, correctly repackaged this time (fix
//   #211, inference artifact now ~1.9GB, verified before this deploy).
//   Same BLEU/chrF as v13/v14 (same underlying weights) -- deployed here.
// - Version 16 (issue #208: pushing epochs further after #205's 13->20
//   result) -- a fresh 26-epoch run, otherwise identical config. BLEU
//   17.8/chrF 40.5, another real, substantial improvement over the
//   20-epoch result (16.1/39.3) -- the trend keeps climbing, no plateau
//   yet. Registered via `training.submit_job` (traceability only, same
//   as v13); its own raw artifact still predates issue #187's
//   checkpoint-storage fix, so it's still ~30GB.
// - Version 17 -- v16's weights, correctly repackaged via `deploy.py`
//   (inference artifact ~1.9GB, verified before this deploy). Same
//   BLEU/chrF as v16 -- deployed here.
// - Version 18 (issue #214: pushing epochs further after #208's 20->26
//   result) -- a fresh 35-epoch run, submitted with issue #187's fix
//   already active -- its own raw artifact was already small (~1.9GB)
//   straight out of training, no `deploy.py` repackaging-after-the-fact
//   needed to fix bloat. BLEU 18.3/chrF 40.7 -- only a marginal
//   improvement over the 26-epoch result (17.8/40.5), right at the edge
//   of the established run-to-run noise band (+-0.5 BLEU/+-0.2 chrF).
//   The epoch-count curve (13->20->26->35) is clearly flattening;
//   stopping further epoch-count probing here per diminishing returns.
// - Version 19 -- v18's weights, repackaged via `deploy.py` (fast this
//   time -- source artifact already small, no large-artifact workaround
//   needed). Same BLEU/chrF as v18 -- deployed here.
const DEV_MODEL_PACKAGE_VERSION = 19;

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
