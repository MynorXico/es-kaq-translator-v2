"""Shared SageMaker Model Registry registration logic (issue #190).

Used by both:

- `training.train` -- self-registers a completed run's model package
  directly from inside the training container, right after its own
  `model_card.md` is written, with no download of the training artifact
  at all (see `training.train.register_model_from_training_job`).
- `training.submit_job` -- the `--register-existing` opt-in path for
  registering an already-completed job's artifact after the fact (e.g. one
  submitted with `--no-wait` before self-registration existed), which still
  needs to fetch the model card from the artifact itself.

Factored out here, rather than defined in either module, specifically to
avoid a circular import: `submit_job.py` already imports from
`training.train` (e.g. `DEFAULT_BASE_MODEL`), so `training.train` can't
import back from `training.submit_job`. This module depends on neither.

## Why registration always reads BLEU/chrF from a model card's *text*,
   never an in-memory metrics object

Both callers feed a model card's own rendered text through
`evaluation.model_card.parse_model_card_metrics` rather than passing
already-computed metrics directly. This guarantees the metadata sent to
the registry can never drift from what the model card itself (the
traceability artifact the metadata is describing) actually says -- one
rendering path, one parsing path, shared by every caller.
"""

from __future__ import annotations

from typing import Any

import botocore.exceptions

from evaluation.model_card import parse_model_card_metrics

DEFAULT_MODEL_PACKAGE_GROUP_NAME = "traductor-kaqchikel-es-cak"
DEFAULT_APPROVAL_STATUS = "PendingManualApproval"

MODEL_PACKAGE_GROUP_DESCRIPTION = (
    "Spanish<->Kaqchikel fine-tuned M2M100 checkpoints "
    "(ADR 0001/0003/0006). Trained weights are private (ADR 0002); "
    "only identifiers/metrics are recorded here."
)


def ensure_model_package_group(sm_client: Any, group_name: str, description: str) -> None:
    """Create the Model Package Group if it doesn't already exist.

    Idempotent: treats "already exists" as success. The real
    `CreateModelPackageGroup` API raises a generic `ValidationException`
    with the message "Model Package Group already exists" for this case
    (issue #76) -- caught here alongside `ResourceInUse`, in case some
    other AWS SDK version or code path uses that instead.
    """
    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription=description,
        )
    except sm_client.exceptions.ResourceInUse:
        pass
    except botocore.exceptions.ClientError as error:
        error_info = error.response.get("Error", {})
        is_already_exists = error_info.get(
            "Code"
        ) == "ValidationException" and "already exists" in error_info.get("Message", "")
        if not is_already_exists:
            raise


def build_customer_metadata(run_metadata: dict[str, Any], metrics: dict[str, str]) -> dict[str, str]:
    """Flatten run metadata + metrics into the string->string map
    `CustomerMetadataProperties` requires. Deliberately only
    identifiers/aggregate counts (corpus version tag, hyperparameters,
    BLEU/chrF) -- never raw corpus content (ADR 0002), matching
    `evaluation.model_card`'s own privacy constraint.
    """
    metadata = {
        "corpus_version": str(run_metadata["corpus_version"]),
        "direction": str(run_metadata["direction"]),
        "run_id": str(run_metadata["run_id"]),
    }
    for key, value in run_metadata.get("hyperparameters", {}).items():
        metadata[f"hp_{key}"] = str(value)
    metadata.update(metrics)
    return metadata


def register_model_package(
    sm_client: Any,
    *,
    model_package_group_name: str,
    model_data_url: str,
    image_uri: str,
    model_card_text: str,
    run_metadata: dict[str, Any],
    approval_status: str = DEFAULT_APPROVAL_STATUS,
) -> str:
    """Register a model package version: ensure the Model Package Group
    exists, then create a new Model Package pointing at `model_data_url`
    (never fetched or validated here -- see `training.train`'s module
    docstring for why it's safe for this to be a not-yet-existing S3 URI),
    carrying corpus version / hyperparameters / BLEU / chrF (parsed from
    `model_card_text`) as custom metadata.

    Defaults to `PendingManualApproval` -- a human should look at the eval
    metrics before a model can be approved for deployment.

    Returns the created Model Package's ARN.
    """
    ensure_model_package_group(
        sm_client,
        model_package_group_name,
        description=MODEL_PACKAGE_GROUP_DESCRIPTION,
    )

    metrics = parse_model_card_metrics(model_card_text)
    customer_metadata = build_customer_metadata(run_metadata, metrics)

    response = sm_client.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus=approval_status,
        InferenceSpecification={
            "Containers": [{"Image": image_uri, "ModelDataUrl": model_data_url}],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        CustomerMetadataProperties=customer_metadata,
    )
    return response["ModelPackageArn"]
