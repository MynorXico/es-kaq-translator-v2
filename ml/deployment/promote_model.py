"""Promote an already-Approved SageMaker Model Package from one
environment's account into another environment's own Model Registry
(issue #235), per [ADR 0010](../../docs/adr/0010-cross-account-model-promotion.md).

## Why this script exists, and why it's different from every other script here

SageMaker Model Registry entries (Model Package Groups and the Model
Packages inside them) are account-scoped resources. Today only
`translator-dev` has a trained, registered, Approved, deployable Model
Package (`training/submit_job.py` only ever trains against `dev`'s
corpus bucket; `deployment/deploy.py` has only ever been run with `dev`
credentials). Retraining independently per environment is not a real
option (real, billed GPU time) -- the point of this script is to promote
*one validated artifact* into `qa`/`prod`, not reproduce it.

ADR 0010 rejected a new persistent cross-account IAM trust (a Model
Package Group resource policy / S3 bucket policy letting `qa`/`prod`'s
own SageMaker execution roles read into `dev`'s private bucket) in favor
of scripted, manually-triggered artifact duplication: download the
already-inference-ready artifact from the source environment's bucket,
re-upload it unchanged into the target environment's own bucket, and
register it as a brand-new Model Package in the target account's own
Model Registry. No new standing grant of any kind -- the only
credentials this script ever uses are a human's own, already-privileged
per-environment SSO sessions, held for the duration of one invocation.

This is the **first script in this project that needs two AWS accounts'
credentials in one invocation**. Every other script
(`training/submit_job.py`, `deployment/deploy.py`,
`register_existing_job`) just uses `boto3.client(...)` against whatever
the shell's single active profile is -- there was never a reason for any
of them to span two accounts. Here, two independent `boto3.Session`
objects (one per side) are constructed explicitly in `main()`, and every
client used below is built from the matching session -- so which account
each call touches is always explicit, never ambient.

## The hard-fail approval guard (devops review on PR #236)

`main()` calls `DescribeModelPackage` against the source Model Package
and checks `ensure_approved(...)` *immediately after*, before any
download/upload/registration happens. A typo'd
`--source-model-package-version` must not be able to silently promote a
`Rejected`/`PendingManualApproval` package into the target account as
`Approved` -- promotion is not a second quality review (that already
happened when the source was approved); this guard just makes sure the
thing actually being duplicated really was.

## What this does NOT do

- No repackaging. The source Model Package's artifact is already
  inference-ready (produced by `deployment/deploy.py`) -- downloaded and
  re-uploaded byte-for-byte unchanged, under a `promoted-artifacts/
  <run_id>/model.tar.gz` prefix in the target bucket (distinct from
  `deploy.py`'s own `inference-artifacts/` prefix, so it's visible at a
  glance which artifacts were produced locally in that environment versus
  promoted in from another one).
- No model-card re-parsing. Everything needed (image URI, model data URL,
  `CustomerMetadataProperties` -- corpus version, hyperparameters,
  BLEU/chrF) is read directly off the source Model Package via
  `DescribeModelPackage`; it was already computed once, at the artifact's
  original `deploy.py` registration.
- No second approval gate. `ModelApprovalStatus=Approved` unconditionally
  in the target account, once the hard-fail guard above confirms the
  source really is `Approved`.

Traceability across environments is carried by two new
`CustomerMetadataProperties` fields (`promoted_from_environment`,
`promoted_from_model_package_arn`), not by numeric version alignment --
`dev`'s version 19 promoted to `qa` will very likely *not* become `qa`'s
version 19 (see ADR 0010's Consequences).

See `docs/runbooks/model-promotion.md` for the human process (including
rollback) around running this script.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import boto3

from training.model_registry import (
    DEFAULT_MODEL_PACKAGE_GROUP_NAME,
    MODEL_PACKAGE_GROUP_DESCRIPTION,
    ensure_model_package_group,
)
from training.submit_job import resolve_stack_outputs

# Model promotion is always registered Approved -- see module docstring.
# Not a CLI flag: unlike training/deployment registration, there is no
# "pending review" state for a promotion, by design (ADR 0010).
APPROVAL_STATUS = "Approved"

# Matches every other script's single-region convention (submit_job.py/
# deploy.py's own DEFAULT_REGION) -- used only as a fallback if a profile
# somehow has no region configured (docs/runbooks/aws-account-bootstrap.md's
# translator-<env> profiles always set one explicitly).
DEFAULT_REGION = "us-east-1"


class SourceModelPackageNotApprovedError(RuntimeError):
    """Raised by `ensure_approved` when the source Model Package is not
    `Approved` -- a hard stop, before any download/upload/registration
    (devops review on PR #236, see module docstring).
    """


def build_profile_name(environment: str) -> str:
    """Default named AWS SSO profile for a given environment, per
    `docs/runbooks/aws-account-bootstrap.md`'s `translator-<env>`
    convention.
    """
    return f"translator-{environment}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Duplicate an already-Approved SageMaker Model Package from one "
            "environment's account into another environment's own Model "
            "Registry (issue #235, ADR 0010) -- no repackaging, no new "
            "cross-account IAM trust."
        )
    )
    parser.add_argument(
        "--source-environment",
        default="dev",
        help="Environment to promote a Model Package *from* (default: dev).",
    )
    parser.add_argument(
        "--source-model-package-version",
        type=int,
        required=True,
        help=(
            "Version number (within --model-package-group-name) of the "
            "source Model Package to promote. Must already be "
            "ModelApprovalStatus=Approved in --source-environment's own "
            "Model Registry -- this script refuses to promote anything else."
        ),
    )
    parser.add_argument(
        "--target-environment",
        required=True,
        help="Environment to promote the Model Package *into* (e.g. qa, prod).",
    )
    parser.add_argument(
        "--source-profile",
        default=None,
        help=(
            "Named AWS SSO profile for --source-environment's account. "
            "Defaults to translator-<source-environment>."
        ),
    )
    parser.add_argument(
        "--target-profile",
        default=None,
        help=(
            "Named AWS SSO profile for --target-environment's account. "
            "Defaults to translator-<target-environment>."
        ),
    )
    parser.add_argument(
        "--model-package-group-name",
        default=DEFAULT_MODEL_PACKAGE_GROUP_NAME,
        help=(
            "Model Package Group name, in *both* the source and target "
            f"accounts (default: {DEFAULT_MODEL_PACKAGE_GROUP_NAME}). Each "
            "account's group is a wholly separate resource -- this is just "
            "the name used on both sides."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Source: describe + the hard-fail approval guard
# ---------------------------------------------------------------------------


def build_model_package_arn(
    *, region: str, account_id: str, model_package_group_name: str, version: int
) -> str:
    """Full Model Package ARN -- the *only* form `DescribeModelPackage`'s
    `ModelPackageName` parameter accepts for a versioned package (issue
    #239: a real `dev` -> `qa` promotion found that the `<group>/<version>`
    shorthand this module previously built fails AWS's real API with
    `ValidationException: ... Member must satisfy regular expression
    pattern: ^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$` -- that pattern is
    `DescribeModelPackage`'s validation for a *bare, unversioned* name; it
    rejects the embedded `/` outright. A full ARN is validated against a
    different, ARN-shaped pattern and works, confirmed directly against
    the real API). Mirrors `infra/cdk/lib/ml-hosting-stack.ts`'s
    `this.formatArn(...)` construction of this exact ARN shape, in Python/
    boto3 form -- the one difference is the account ID: CDK resolves it
    from the stack's own `account` token at synth time, whereas this
    script looks it up at runtime via STS (`get_account_id`), since the
    source account is never the account this script happens to run in.
    """
    return f"arn:aws:sagemaker:{region}:{account_id}:model-package/{model_package_group_name}/{version}"


def get_account_id(sts_client: Any) -> str:
    """The calling session's own AWS account ID, via STS -- used to build
    the source Model Package's full ARN (`build_model_package_arn`), since
    `boto3`/SSO profiles never expose the account ID directly as a plain
    attribute.
    """
    return sts_client.get_caller_identity()["Account"]


def describe_source_model_package(sm_client: Any, *, model_package_arn: str) -> dict[str, Any]:
    return sm_client.describe_model_package(ModelPackageName=model_package_arn)


def ensure_approved(description: dict[str, Any], *, source_model_package_version: int) -> None:
    """Hard-fail guard (devops review on PR #236): raise loudly if the
    source Model Package isn't `Approved`, before any download/upload/
    registration happens. See module docstring.
    """
    status = description.get("ModelApprovalStatus")
    if status == "Approved":
        return

    identifier = description.get(
        "ModelPackageArn", f"version {source_model_package_version}"
    )
    raise SourceModelPackageNotApprovedError(
        f"Source Model Package {identifier} has ModelApprovalStatus={status!r}, "
        "not 'Approved' -- refusing to promote. Promotion duplicates an "
        "already-approved artifact; it is not a second quality review. "
        "Check --source-model-package-version."
    )


def extract_source_artifact_info(description: dict[str, Any]) -> dict[str, Any]:
    """Pull everything needed to promote this artifact directly off the
    `DescribeModelPackage` response -- no model-card re-parsing (it was
    already computed once, at this artifact's original `deploy.py`
    registration).
    """
    container = description["InferenceSpecification"]["Containers"][0]
    return {
        "image_uri": container["Image"],
        "model_data_url": container["ModelDataUrl"],
        "customer_metadata": dict(description.get("CustomerMetadataProperties", {})),
        "model_package_arn": description["ModelPackageArn"],
    }


# ---------------------------------------------------------------------------
# Artifact duplication (unchanged -- no repackaging)
# ---------------------------------------------------------------------------

PROMOTED_ARTIFACT_PREFIX = "promoted-artifacts"


def build_promoted_artifact_key(run_id: str) -> str:
    """Where the duplicated artifact lands in the target bucket --
    distinct from `deploy.py`'s own `inference-artifacts/` prefix, so
    it's visible at a glance which artifacts were produced locally there
    versus promoted in from another environment (ADR 0010).
    """
    return f"{PROMOTED_ARTIFACT_PREFIX}/{run_id}/model.tar.gz"


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3:// URI: {uri!r}")
    bucket, _, key = uri[len("s3://") :].partition("/")
    if not key:
        raise ValueError(f"Not an s3:// URI: {uri!r}")
    return bucket, key


def download_source_artifact(s3_client: Any, source_model_data_url: str, destination: Path) -> None:
    bucket, key = _parse_s3_uri(source_model_data_url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, str(destination))


def upload_promoted_artifact(s3_client: Any, local_path: Path, bucket: str, key: str) -> str:
    s3_client.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


# ---------------------------------------------------------------------------
# Target: register as a new Model Package
# ---------------------------------------------------------------------------


def build_target_customer_metadata(
    source_customer_metadata: dict[str, str],
    *,
    source_environment: str,
    source_model_package_arn: str,
) -> dict[str, str]:
    """Carry over the source's own metadata (corpus version, hyperparameters,
    BLEU/chrF, run_id) unchanged, plus two new provenance fields. Does not
    mutate the input dict -- callers may reuse `source_customer_metadata`
    for logging/debugging after this returns.
    """
    metadata = dict(source_customer_metadata)
    metadata["promoted_from_environment"] = source_environment
    metadata["promoted_from_model_package_arn"] = source_model_package_arn
    return metadata


def register_promoted_model(
    sm_client: Any,
    *,
    model_package_group_name: str,
    model_data_url: str,
    image_uri: str,
    customer_metadata: dict[str, str],
) -> str:
    """Register the duplicated artifact as a new Model Package version in
    the target account's own group. Reuses
    `training.model_registry.ensure_model_package_group` -- same
    idempotency handling as every other registration path in this
    project. `ModelApprovalStatus` is unconditionally `Approved` (see
    module docstring) -- there is no `approval_status` parameter here by
    design.
    """
    ensure_model_package_group(
        sm_client,
        model_package_group_name,
        description=MODEL_PACKAGE_GROUP_DESCRIPTION,
    )

    response = sm_client.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus=APPROVAL_STATUS,
        InferenceSpecification={
            "Containers": [{"Image": image_uri, "ModelDataUrl": model_data_url}],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        CustomerMetadataProperties=customer_metadata,
    )
    return response["ModelPackageArn"]


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    source_profile = args.source_profile or build_profile_name(args.source_environment)
    target_profile = args.target_profile or build_profile_name(args.target_environment)

    # The first script in this project holding two accounts' credentials
    # live in one process at once -- both are the same human's own,
    # already-privileged SSO sessions (ADR 0010's Consequences), held only
    # for the duration of this one invocation.
    source_session = boto3.Session(profile_name=source_profile)
    target_session = boto3.Session(profile_name=target_profile)

    source_sm_client = source_session.client("sagemaker")
    source_sts_client = source_session.client("sts")

    source_account_id = get_account_id(source_sts_client)
    source_region = source_session.region_name or DEFAULT_REGION
    source_model_package_arn = build_model_package_arn(
        region=source_region,
        account_id=source_account_id,
        model_package_group_name=args.model_package_group_name,
        version=args.source_model_package_version,
    )

    description = describe_source_model_package(
        source_sm_client, model_package_arn=source_model_package_arn
    )
    try:
        ensure_approved(
            description, source_model_package_version=args.source_model_package_version
        )
    except SourceModelPackageNotApprovedError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    info = extract_source_artifact_info(description)

    target_cfn_client = target_session.client("cloudformation")
    target_outputs = resolve_stack_outputs(
        args.target_environment, cloudformation_client=target_cfn_client
    )
    target_bucket = target_outputs["TrainingDataBucketName"]

    run_id = info["customer_metadata"].get("run_id", "unknown-run")

    source_s3_client = source_session.client("s3")
    target_s3_client = target_session.client("s3")

    tmp_dir = Path(tempfile.mkdtemp(prefix="kaqchikel-promote-"))
    try:
        local_path = tmp_dir / "model.tar.gz"
        download_source_artifact(source_s3_client, info["model_data_url"], local_path)

        target_key = build_promoted_artifact_key(run_id)
        model_data_url = upload_promoted_artifact(
            target_s3_client, local_path, target_bucket, target_key
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    target_sm_client = target_session.client("sagemaker")
    customer_metadata = build_target_customer_metadata(
        info["customer_metadata"],
        source_environment=args.source_environment,
        source_model_package_arn=info["model_package_arn"],
    )
    model_package_arn = register_promoted_model(
        target_sm_client,
        model_package_group_name=args.model_package_group_name,
        model_data_url=model_data_url,
        image_uri=info["image_uri"],
        customer_metadata=customer_metadata,
    )

    print(f"Registered model package: {model_package_arn}")
    print(f"Promoted artifact: {model_data_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
