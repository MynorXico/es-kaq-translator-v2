# ADR 0010: Cross-account model promotion via scripted artifact duplication

- Status: Proposed
- Date: 2026-10-05

## Context

Per ADR 0001, dev/qa/prod are separate AWS accounts under the
Organization. SageMaker Model Registry entries (Model Package Groups and
the Model Packages inside them) are account-scoped resources -- there is
no such thing as one Model Package Group spanning multiple accounts.
Today, only translator-dev has a trained, registered, Approved Model
Package: training only ever runs against dev's corpus bucket
(ml/training/submit_job.py's --environment default), and the one
script that produces a deployable (inference-ready) Model Package,
ml/deployment/deploy.py, has likewise only ever been run with dev
credentials.

As a direct, explicitly-documented consequence,
infra/cdk/lib/app-stage.ts only instantiates MlHostingStack when
environmentName === "dev", and MlHostingStack's own docstring states
"Scope: dev only, for now." qa and prod have no SageMaker endpoint at
all; ApiStack in those environments falls back to its own default
endpoint-name convention, which names an endpoint that doesn't exist, so
/v1/translate cannot work end-to-end there today (issue #234).

Retraining separately per environment is not a real option: training is
real, billed GPU time (per app-stage.ts's own inline history, dev alone
has run well over a dozen training jobs while searching for a good
configuration), and the entire point of a promotion pipeline is to
promote one validated artifact through environments, not reproduce it
independently in each one.

### How registration/deployment actually works today (read directly from code)

- ml/training/submit_job.py registers the training run's raw artifact
  under the training container's image right after a job completes
  (traceability only -- that Model Package is not servable: wrong
  container, no inference code).
- ml/deployment/deploy.py is the script that produces something
  deployable: given --source-model-data-url (a raw training artifact)
  and --environment (default dev), it (1) resolves that environment's
  {Environment}-Data CloudFormation stack outputs (resolve_stack_outputs,
  matching TranslatorStage's construct IDs, e.g. Dev-Data/Qa-Data/
  Prod-Data) to find its private training-data bucket, (2) downloads the
  raw artifact, repackages it with the custom inference handler
  (deployment.package_model), and uploads the repackaged copy back into
  that same bucket under inference-artifacts/<run_id>/model.tar.gz, and
  (3) registers it (ModelApprovalStatus=Approved by default) into that
  account's traductor-kaqchikel-es-cak Model Package Group, with
  BLEU/chrF/run metadata parsed from the artifact's own model card
  recorded as CustomerMetadataProperties.
- Critically, --environment here only ever selects which CloudFormation
  stack's outputs to read. All three boto3 clients it constructs
  (cloudformation, s3, sagemaker) use whatever AWS credentials are active
  in the caller's shell -- there has never been a reason for this script
  to span two accounts in one invocation, because it has only ever been
  run against dev (by a human with dev SSO credentials active, per
  docs/runbooks/aws-account-bootstrap.md's per-environment profile
  convention).
- infra/cdk/lib/ml-hosting-stack.ts builds the Model Package ARN it
  references via this.formatArn with resourceName "<group>/<version>",
  using the stack's own account/region tokens -- not a hardcoded dev
  account ID. Nothing inside MlHostingStack itself is dev-specific; the
  dev-only gating lives entirely in app-stage.ts's if statement and its
  single DEV_MODEL_PACKAGE_VERSION constant.

### Relevant precedent (existing ADRs, not to be contradicted)

- ADR 0004 established the project's one existing cross-account IAM
  trust: translator-tooling's CDK Pipelines deployment role assumes into
  dev/qa/prod to run CloudFormation deploys. This is a deploy
  orchestration trust, scoped to CDK's own asset/deploy roles -- it says
  nothing about runtime data-plane access between environment accounts,
  and this ADR does not reuse or extend it for that purpose.
- ADR 0007 faced a structurally similar problem (a resource in one
  account needing to affect state that lives in another account --
  there, DNS records in translator-tooling's hosted zone) and explicitly
  rejected a new persistent cross-account IAM role/custom-resource in
  favor of a one-time, scripted-but-manually-triggered step performed by
  a human with the already-privileged account's own credentials --
  reasoning that a new standing trust was a disproportionate blast-radius
  increase for an infrequent (there: "roughly six times to start with")
  operation, for a small, cost-conscious, solo-maintainer OSS project.
- ADR 0009 faced "should this operation be decoupled into new AWS
  infrastructure (EventBridge + Lambda) or stay a manually-triggered,
  client-side script run under the maintainer's own SSO credentials," for
  Model Registry registration specifically, and chose the latter --
  explicitly favoring operational simplicity over new infrastructure for
  a solo maintainer, given a job cadence ("roughly one every few days")
  that doesn't justify the new infrastructure's cost. This ADR's decision
  extends that same reasoning to model promotion, not just registration.

### Options sketched in issue #234

1. Duplicate registration: after a model is approved in dev, re-register
   the same trained artifact into qa's and prod's own Model Registries as
   separate entries.
2. Cross-account Model Registry sharing: grant qa/prod a resource policy
   on dev's Model Package Group so they can reference dev's Model Package
   directly via a cross-account ARN, without copying anything.

Researching option 2 against this project's actual stack surfaced a
wrinkle issue #234 didn't anticipate: referencing a Model Package
cross-account is not, by itself, sufficient to host it. MlHostingStack
creates a SageMaker Model from the Model Package (a container whose
modelPackageName is the Model Package ARN); at creation (and at every
subsequent cold start) SageMaker reads the artifact at
InferenceSpecification.Containers[0].ModelDataUrl using the consuming
account's own execution role, not the Model Package's account. Option 2
therefore needs two new standing cross-account grants, not one: a Model
Package Group resource policy (DescribeModelPackage et al. for qa/prod
principals) and an S3 bucket policy on dev's private training-data bucket
(DataStack.bucket) granting qa/prod's SageMaker execution roles
s3:GetObject. That second grant means dev's own private bucket -- the one
holding ALMG-derived-corpus-adjacent artifacts under ADR 0002's privacy
posture -- would need a standing policy readable by two other accounts'
execution roles, each invoked on every cold start, indefinitely.

## Decision

Adopt option 1: scripted, manually-triggered artifact duplication, with
no new persistent cross-account IAM trust or resource policy.

- A new script, ml/deployment/promote_model.py, is the mechanism. Its
  shape, reusing existing helpers rather than inventing new ones:
  - CLI: --source-environment (default dev), --source-model-package-
    version (the integer version to promote, matching app-stage.ts's
    existing "bump a version number by hand" convention), --target-
    environment, and --source-profile/--target-profile (named AWS
    CLI/SSO profiles, defaulting to the translator-ENV convention
    already established in docs/runbooks/aws-account-bootstrap.md).
    This is the first script in the project that genuinely needs two
    AWS accounts' credentials in one invocation, so unlike
    submit_job.py/deploy.py (which just use boto3.client(...) against
    whatever the shell's single active profile is), it constructs two
    independent boto3.Session(profile_name=...) objects and builds each
    side's clients from the matching session -- explicit, not ambient,
    about which account each call touches.
  - Reads the source Model Package via DescribeModelPackage (source
    session) to get its Image/ModelDataUrl and CustomerMetadataProperties
    directly -- no need to re-download and re-parse a model card the way
    deploy.py does, because deploy.py already computed and stored those
    metrics as metadata the first time this artifact was registered.
  - Downloads the artifact unchanged from the source bucket (source
    session) and re-uploads it unchanged (it is already inference-ready;
    no repackaging step is needed or correct here) into the target
    environment's own bucket, resolved via the existing
    training.submit_job.resolve_stack_outputs helper (called with the
    target session's own CloudFormation client), under a new
    promoted-artifacts/RUN_ID/model.tar.gz prefix -- distinct from
    deploy.py's own inference-artifacts/ prefix, so it's visible at a
    glance in each bucket which artifacts were produced locally there
    versus promoted in from another environment.
  - Registers a new Model Package in the target account's own
    traductor-kaqchikel-es-cak group (reusing
    training.model_registry.ensure_model_package_group), with
    ModelApprovalStatus Approved unconditionally (it was already
    approved upstream; promotion is not a second quality review) and
    CustomerMetadataProperties carrying over the source's own metadata
    plus two new provenance fields: promoted_from_environment and
    promoted_from_model_package_arn (the full source ARN including
    version). Traceability is maintained through this metadata, not
    through numeric version alignment -- see Consequences.
  - Prints the new target Model Package ARN/version, mirroring
    deploy.py's own main() output.
- infra/cdk/lib/app-stage.ts generalizes its single
  DEV_MODEL_PACKAGE_VERSION constant into a small per-environment map:

  ```ts
  const MODEL_PACKAGE_VERSION_BY_ENVIRONMENT: Partial<Record<string, number>> = {
    dev: 19,
    // qa/prod: added once ml/deployment/promote_model.py has been run for
    // that environment -- see docs/runbooks/model-promotion.md. Absence
    // here means "no MlHostingStack/endpoint in that environment yet,"
    // same semantics as today's dev-only `if`.
  };
  ```

  and replaces the dev-only if statement with a data-driven one:

  ```ts
  const modelPackageVersion = MODEL_PACKAGE_VERSION_BY_ENVIRONMENT[props.environmentName];
  let mlHostingStack: MlHostingStack | undefined;
  if (modelPackageVersion !== undefined) {
    mlHostingStack = new MlHostingStack(this, "MlHosting", {
      environmentName: props.environmentName,
      modelPackageVersion,
      modelDataBucket: dataStack.bucket,
    });
  }
  ```

  Nothing else in app-stage.ts needs to change: ApiStack's existing
  sageMakerEndpointName prop (set to mlHostingStack?.endpointName)
  already generalizes for free once mlHostingStack is defined for
  qa/prod too -- issue #9's fix was never actually dev-specific, it just
  had no non-dev case to exercise until now.

- infra/cdk/lib/ml-hosting-stack.ts needs no functional change: its
  Model Package ARN construction already uses this.account/this.region
  (stack-scoped CDK tokens resolved per-environment at synth/deploy
  time), so it naturally resolves to each environment's own account.
  Only its docstring's "Scope: dev only, for now" section needs updating
  to describe the generalized, per-environment-version design and point
  at this ADR instead.
- A new runbook, docs/runbooks/model-promotion.md, documents the human
  process: confirm both the source and target translator-ENV SSO
  profiles are logged in, run promote_model.py, note the printed target
  version, open a PR adding/bumping that environment's entry in
  MODEL_PACKAGE_VERSION_BY_ENVIRONMENT (with an inline comment recording
  which source version it was promoted from, mirroring the existing
  per-version history comment above DEV_MODEL_PACKAGE_VERSION), get it
  reviewed and merged, and watch the first resulting deploy for that
  environment (a brand-new MlHostingStack/Endpoint resource, not an
  update to an existing one) rather than assuming an unattended pipeline
  run will surface a problem cleanly. qa and prod are promoted as
  separate, independent invocations of this process -- promoting to qa
  does not imply or automate promoting the same version to prod; a human
  decides each time whether the model is good enough for the
  public-facing environment.

## Consequences

- No new persistent cross-account IAM trust or resource policy of any
  kind. The blast radius for "who/what can read dev's private training-
  data bucket or Model Registry" stays at "a human with dev's own SSO
  credentials, who already has AdministratorAccess there" -- not a
  standing grant reachable from qa's or prod's own execution roles. This
  matches ADR 0007's and ADR 0009's established preference for this
  project's size/traffic/maintainer model.
- Storage is duplicated: each promoted model's inference-ready artifact
  (currently ~1.9GB after issue #211's checkpoint-exclusion fix) is
  stored redundantly in up to three buckets instead of one. At S3
  Standard pricing this is on the order of cents per month, plus a
  one-time, same-region cross-account transfer cost that is similarly
  negligible -- immaterial for this project's cost posture (ADR 0001).
- dev, qa, and prod end up with three independent Model Registries whose
  version numbers will not line up (e.g. dev's version 19 might become
  qa's version 2, long before prod ever sees it). Traceability across
  environments is carried entirely by CustomerMetadataProperties
  (promoted_from_environment/promoted_from_model_package_arn), not by
  numeric alignment -- this must be documented clearly (in the new
  runbook and in app-stage.ts's comments) so nobody assumes "qa v2 ==
  dev v2."
- Model promotion stays a deliberate, infrequent, human-triggered action,
  not something the CDK Pipeline automates between stages the way it
  already does for WebStack/DataStack/ApiStack (ADR 0004). This is
  intentional: unlike those stacks, a "newer" model isn't automatically
  "better" for every environment (see app-stage.ts's own version history
  of rejected training runs) -- promoting a specific version to a
  specific environment is a judgment call, not a mechanical sync.
- ml/deployment/promote_model.py is new, non-trivial production code
  that spans two AWS accounts in one process -- it needs unit tests with
  mocked per-side boto3 clients, per docs/testing.md's TDD policy. This
  ADR does not write that code; see issue #235.
- Landing this ADR's CDK change (the generalized map + conditional in
  app-stage.ts, and the docstring update in ml-hosting-stack.ts) does
  not, by itself, turn on qa/prod -- MODEL_PACKAGE_VERSION_BY_ENVIRONMENT
  should still only contain dev at that point, because no Model Package
  exists yet in qa's or prod's own registry. promote_model.py must
  actually be run, once per environment, before that environment's entry
  is added. This sequencing -- generalized mechanism merges first,
  decoupled from the operational promotion event -- mirrors how
  DEV_MODEL_PACKAGE_VERSION itself is already "bumped by hand in a
  reviewed PR" only once a specific version is actually approved and
  reachable.
- If model promotion frequency ever grows enough that the manual,
  per-promotion script invocation becomes a real bottleneck (not the
  case today -- training/approval cadence is already on the order of
  days per app-stage.ts's own version history, and promotion to qa/prod
  will be rarer still), revisit toward either a CDK-Pipelines-driven
  promotion stage or option 2's cross-account sharing -- the same
  "revisit if churn changes" escape hatch ADR 0007 used, not a permanent
  ban on automating this further.

## Alternatives considered

- Option 2 (cross-account Model Registry resource-policy sharing):
  rejected as the primary mechanism. As found during this ADR's research
  (see Context), it actually requires two new standing cross-account
  grants, not one -- a Model Package Group resource policy and an S3
  bucket policy on dev's private bucket granting qa/prod's execution
  roles read access -- since hosting a Model Package still means the
  consuming account's execution role reads ModelDataUrl directly. That
  is a larger, continuously-live blast radius (two other accounts'
  hosting infrastructure with standing read access into dev's bucket,
  evaluated on every cold start, not just at promotion time) than this
  project's actual promotion cadence justifies, and more new
  infrastructure to build and audit than option 1. Consistent with ADR
  0007's rejection of a structurally similar persistent-trust design for
  a comparably infrequent operation. Revisit only if promotion frequency
  changes dramatically (see Consequences).
- Retraining independently per environment: already rejected in issue
  #234 itself (real, billed GPU cost; defeats the purpose of a promotion
  pipeline that promotes one validated artifact). Not re-litigated here.
- A fully automated CDK Pipelines stage that runs promote_model.py
  automatically whenever dev's approved version changes: rejected for
  now. This would require granting the pipeline's own translator-tooling
  CodeBuild role standing cross-account data-plane access (S3 + Model
  Registry read/write) into all three environment accounts -- an even
  larger trust expansion than option 2, since it is not even "a human's
  already-privileged credentials" invoked occasionally but a CI role
  capable of promoting a model unattended. Model promotion cadence
  doesn't come close to justifying this, mirroring ADR 0009's rejection
  of automating a simpler, same-account registration step for similar
  cadence/complexity reasons. Could be reconsidered if the project's
  overall CI/CD philosophy shifts toward fully automated ML delivery,
  but that is a larger, separate decision than this issue's scope.
- A single shared, cross-account-readable artifact bucket (e.g. in
  translator-tooling) instead of duplicating into each environment's own
  bucket: would reduce storage duplication from three copies to one, but
  reintroduces the same category of standing cross-account read trust
  (dev/qa/prod's hosting roles all reading a bucket outside their own
  account) that this ADR avoids, and doesn't fit ADR 0001/ADR 0002's
  existing "one DataStack bucket per environment" model. Rejected for a
  storage-cost saving measured in cents per month -- not worth the
  trade.

## Follow-up work (not done by this ADR)

- Implementation of ml/deployment/promote_model.py (test-first, per
  docs/testing.md), the app-stage.ts/ml-hosting-stack.ts changes
  described above, and docs/runbooks/model-promotion.md: tracked as
  issue #235, so this ADR's acceptance isn't blocked on implementation
  review. The first real run of promote_model.py against qa (and,
  separately, prod) is itself part of that follow-up, not this ADR.
