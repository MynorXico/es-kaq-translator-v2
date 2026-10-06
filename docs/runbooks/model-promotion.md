# Runbook: promoting a Model Package from dev to qa/prod

Issue #235, implementing [ADR 0010](../adr/0010-cross-account-model-promotion.md).
Before this existed, only `translator-dev` ever had a trained, registered,
Approved SageMaker Model Package -- `qa`/`prod` had no `MlHostingStack` and
no working `/v1/translate` at all (issue #234). This runbook is the human
process for promoting a specific, already-approved `dev` model into
`qa`/`prod`'s own Model Registry, and for rolling a bad promotion back.

Promotion is a deliberate, infrequent, manually-triggered action, not
something the CDK Pipeline automates between stages -- a "newer" model
isn't automatically "better" for every environment (see
`infra/cdk/lib/app-stage.ts`'s own version-history comments of rejected
training runs). `qa` and `prod` are promoted as separate, independent
invocations: promoting a version to `qa` does not imply or automate
promoting the same version to `prod`.

## Prerequisites

- Both the source and target environments' named AWS SSO profiles
  (`translator-<env>`, per
  [the account bootstrap runbook](aws-account-bootstrap.md)) are logged in:

  ```sh
  aws sso login --profile translator-dev
  aws sso login --profile translator-qa   # or translator-prod
  ```

- You know the source Model Package version to promote (an integer, e.g.
  `19`) and have confirmed it's the one you actually want -- check
  `infra/cdk/lib/app-stage.ts`'s `MODEL_PACKAGE_VERSION_BY_ENVIRONMENT`
  comment history, or run:

  ```sh
  aws sagemaker describe-model-package \
    --profile translator-dev \
    --model-package-name traductor-kaqchikel-es-cak/19
  ```

  and read its `ModelApprovalStatus` (must be `Approved`) and
  `CustomerMetadataProperties` (BLEU/chrF/corpus version/hyperparameters)
  yourself before promoting it -- `promote_model.py` will refuse a
  non-`Approved` source package (see "What the script enforces" below),
  but it does not re-review quality; that's your job here.

## Running the promotion

From `ml/`:

```sh
uv run python -m deployment.promote_model \
  --source-environment dev \
  --source-model-package-version 19 \
  --target-environment qa
```

`--source-profile`/`--target-profile` default to `translator-<environment>`
and rarely need overriding. `--model-package-group-name` defaults to
`traductor-kaqchikel-es-cak` (the same group name used in every
environment's account -- each account's group is still a wholly separate
resource).

On success, the script prints the new target Model Package's ARN and the
S3 URI of the promoted artifact in the target bucket (under
`promoted-artifacts/<run_id>/model.tar.gz`, distinct from `deploy.py`'s
own `inference-artifacts/` prefix). Note the printed **version number** --
the next step needs it. Do not assume it matches the source version; see
"Why version numbers won't line up" below.

Repeat independently for `prod` when/if that's also warranted -- it is a
separate decision, not a second automatic step.

### What the script enforces

- **Hard-fails on a non-`Approved` source.** Immediately after reading the
  source Model Package (before any download/upload/registration),
  `promote_model.py` checks its `ModelApprovalStatus`. If it isn't
  `Approved` (e.g. a typo'd `--source-model-package-version` pointing at a
  `Rejected`/`PendingManualApproval` package), the script exits non-zero
  with a clear error and does nothing else -- it never silently promotes
  an unapproved package into the target account as `Approved`.
- **No repackaging.** The source artifact is already inference-ready
  (produced by `deploy.py`); it's downloaded and re-uploaded unchanged.
- **No second approval gate.** The new target Model Package is registered
  `ModelApprovalStatus=Approved` unconditionally -- promotion is not a
  second quality review, the source's own approval already was that
  review.
- **No new AWS infrastructure or cross-account trust.** The only
  credentials used are your own two per-environment SSO sessions, held for
  the duration of this one invocation -- see ADR 0010's Decision/
  Consequences sections for why this was chosen over cross-account Model
  Registry/S3 sharing.

## After a successful promotion: wiring it into the target environment

1. Open a PR adding (or bumping) that environment's entry in
   `infra/cdk/lib/app-stage.ts`'s `MODEL_PACKAGE_VERSION_BY_ENVIRONMENT`,
   to the version number the script printed. Add an inline comment
   recording which source version/environment it was promoted from,
   mirroring the existing per-version history comments already above that
   map for `dev`. For example, promoting `dev`'s version 19 to `qa` for
   the first time might add:

   ```ts
   const MODEL_PACKAGE_VERSION_BY_ENVIRONMENT: Partial<Record<string, number>> = {
     dev: 19,
     // qa version 1 -- promote_model.py --source-environment dev
     // --source-model-package-version 19 --target-environment qa
     // (BLEU 18.3/chrF 40.7, same weights as dev v19).
     qa: 1,
   };
   ```

2. Get the PR reviewed and merged, same review/deploy path as any other
   change here (no direct pushes to `main`).
3. Watch the first resulting deploy for that environment closely -- this
   creates a brand-new `MlHostingStack`/`Endpoint` resource there, not an
   update to an existing one, so don't assume an unattended pipeline run
   will surface a problem cleanly. Confirm the endpoint comes up healthy
   and a real `/v1/translate` request against that environment's API
   returns *a* translation (QA's usual post-deploy verification step, not
   a BLEU/chrF re-check).

## Why version numbers won't line up across environments

`dev`, `qa`, and `prod` each have their own, wholly independent Model
Registry. `dev`'s version 19 might become `qa`'s version 1, long before
`prod` ever sees anything at all -- there is no mechanism, and no attempt,
to keep these numbers in sync. **Never assume "qa version N" and "dev
version N" are the same weights.**

Traceability across environments is carried entirely by two
`CustomerMetadataProperties` fields `promote_model.py` adds to the new
target Model Package: `promoted_from_environment` and
`promoted_from_model_package_arn` (the full source ARN, including
version). To find out what a given `qa`/`prod` Model Package actually is,
read those fields -- don't infer it from the version number alone:

```sh
aws sagemaker describe-model-package \
  --profile translator-qa \
  --model-package-name traductor-kaqchikel-es-cak/1 \
  --query 'CustomerMetadataProperties'
```

## Rollback

A target environment's Model Registry keeps **every** prior promoted
Model Package -- promotion never deletes or overwrites one. Rolling back
a bad promotion is therefore not a new procedure or script: it's
reverting `MODEL_PACKAGE_VERSION_BY_ENVIRONMENT`'s entry for that
environment back to the previous version number, in a reviewed PR -- the
exact same review/deploy path already used to promote it in the first
place.

For example, if `qa: 2` (just promoted) turns out to be a mistake and
`qa: 1` was the last known-good value, the rollback PR is just:

```diff
 const MODEL_PACKAGE_VERSION_BY_ENVIRONMENT: Partial<Record<string, number>> = {
   dev: 19,
-  qa: 2,
+  qa: 1,
 };
```

merged and deployed the same way as any other change. The old Model
Package (`qa` version 2) is left registered in `qa`'s Model Registry --
version numbers are never reused or deleted after the fact (matching the
existing convention already documented in `app-stage.ts`'s own version
history for `dev`), so there's no cleanup step blocking the rollback
itself. If the bad version should eventually be deleted outright (e.g. it
was registered from genuinely wrong/corrupt data), that's a separate,
deliberate `aws sagemaker delete-model-package` decision -- not a
required part of rolling back which version is *deployed*.

## Related

- [ADR 0010](../adr/0010-cross-account-model-promotion.md) for the full
  design rationale (why scripted duplication over cross-account Model
  Registry sharing, cost/blast-radius trade-offs, etc.).
- [`ml/deployment/promote_model.py`](../../ml/deployment/promote_model.py)
  -- the script itself; its module docstring covers the same ground as
  this runbook from the implementation side.
- [`ml/deployment/deploy.py`](../../ml/deployment/deploy.py) -- produces
  the inference-ready artifact a promotion later duplicates; not run
  again as part of promotion.
