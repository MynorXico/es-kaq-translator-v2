"""Register a training run's model artifact as a new, *deployable*
SageMaker Model Package version (issue #8): repackage it with the custom
inference handler (`deployment.package_model`), upload the result, and
register it in the same Model Package Group `training.submit_job` already
uses (`traductor-kaqchikel-es-cak`) -- so Model Registry stays the single
place every version (training-only or deployable) is tracked, per ADR
0001's traceability requirement.

## Why this is a separate script from `training/submit_job.py`

`submit_job.py` registers the *training container's* image alongside the
raw model artifact right after a training job finishes -- useful for
traceability, but not directly servable (wrong container: a training DLC,
no custom inference code, GPU-oriented). This script takes an *already
registered* (or just any) training run's `model.tar.gz` and produces a
second, deployable Model Package version: a real Hugging Face **inference**
DLC (CPU -- SageMaker Serverless Inference doesn't support GPU) plus the
repackaged artifact with `code/inference.py` bundled in.

## Approval status default: `Approved`, not `PendingManualApproval`

Unlike `training.submit_job.register_model` (defaults to
`PendingManualApproval` -- a human should review BLEU/chrF before a
*training* run's model can be considered for deployment at all), this
script defaults to `Approved`. For issue #8, the project owner explicitly
authorized deploying the current best available model (BLEU 8.9 / chrF
31.2) as an "experimental" release, consciously overriding the originally
planned BLEU>=10 quality gate (issue #76) -- a UI disclaimer is being
added in parallel (issue #43) to make this honest to end users. Model
Registry's versioning is exactly what makes this a low-effort, reversible
choice: approving a new version and updating `infra/cdk/lib/
ml-hosting-stack.ts`'s pinned `MODEL_PACKAGE_ARN` constant is the whole
swap procedure once a better model exists.

## How the inference DLC version combination was determined

Mirrors `training/submit_job.py`'s module docstring for the *training*
combination, but for `image_scope="inference"` instead: the installed
`sagemaker` SDK's HuggingFace inference compatibility table tops out at
`transformers==4.49.0` (lower than the `4.56.2` training container used
to fine-tune this model -- the inference and training DLC lines are
released independently and don't always track each other). M2M100 has
been supported by `transformers` since well before 4.49, so this gap
doesn't affect correctness; this was still verified against the real,
deployed model (see `ml/README.md`'s "Serving" section) rather than
assumed. `instance_type` in the `retrieve()` call only steers which
container *variant* (CPU vs. GPU) gets resolved -- Serverless Inference
never actually provisions a persistent instance of that type.

```python
from sagemaker.image_uris import config_for_framework, retrieve
inference_versions = config_for_framework("huggingface")["inference"]["versions"]
sorted(inference_versions)  # -> [..., "4.48.0", "4.49.0", "4.51.3"]
inference_versions["4.49.0"]
# -> {"version_aliases": {"pytorch2.6": "pytorch2.6.0"},
#     "pytorch2.6.0": {"py_versions": ["py312"],
#                       "container_version": {"gpu": "cu124-ubuntu22.04", "cpu": "ubuntu22.04"}}}
retrieve(framework="huggingface", region="us-east-1", version="4.49.0",
         py_version="py312", instance_type="ml.m5.xlarge", image_scope="inference",
         base_framework_version="pytorch2.6.0")
# -> "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference
#     :2.6.0-transformers4.49.0-cpu-py312-ubuntu22.04"
```

Re-run this after any future `sagemaker` SDK upgrade to confirm the
constants below are still valid, same as `submit_job.py`.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import boto3
from sagemaker.image_uris import retrieve as _retrieve_image_uri

from deployment.package_model import build_inference_code_dir, repackage_model_artifact
from training.submit_job import (
    DEFAULT_MODEL_PACKAGE_GROUP_NAME,
    ensure_model_package_group,
    parse_model_card_metrics,
    resolve_stack_outputs,
)

# --- Inference DLC version combination (see module docstring) --------------
INFERENCE_TRANSFORMERS_VERSION = "4.49.0"
INFERENCE_PYTORCH_VERSION = "2.6.0"
INFERENCE_PY_VERSION = "py312"
# Only used to steer image_uris.retrieve() toward the CPU container variant
# (Serverless Inference is CPU-only); never used to provision a real instance.
DEFAULT_IMAGE_LOOKUP_INSTANCE_TYPE = "ml.m5.xlarge"

DEFAULT_REGION = "us-east-1"

# Deliberately Approved, not PendingManualApproval -- see module docstring.
DEFAULT_APPROVAL_STATUS = "Approved"

INFERENCE_ARTIFACT_PREFIX = "inference-artifacts"

_S3_URI_RE = re.compile(r"^s3://(?P<bucket>[^/]+)/(?P<key>.+)$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repackage a training run's model.tar.gz with the custom "
            "direction-tag-aware inference handler, and register it as a "
            "new, deployable SageMaker Model Package version (issue #8)."
        )
    )
    parser.add_argument(
        "--source-model-data-url",
        required=True,
        help="s3:// URI to the training run's model.tar.gz to deploy.",
    )
    parser.add_argument("--environment", default="dev")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--model-package-group-name", default=DEFAULT_MODEL_PACKAGE_GROUP_NAME
    )
    parser.add_argument(
        "--approval-status",
        choices=("PendingManualApproval", "Approved", "Rejected"),
        default=DEFAULT_APPROVAL_STATUS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve the DataStack bucket and the inference image URI and "
            "print the would-be config, without downloading/repackaging/"
            "uploading the (multi-GB) model artifact or registering anything."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Model card parsing (run identifiers, not just metrics -- submit_job.py's
# parse_model_card_metrics only extracts BLEU/chrF, since at submission time
# the run's other identifiers are already known from the CLI args used to
# submit the job; this script only has the artifact + its model card).
# ---------------------------------------------------------------------------

_TITLE_RUN_ID_RE = re.compile(r"^# Model card:\s*(?P<run_id>\S+)", re.MULTILINE)
_RUN_METADATA_FIELD_RE = {
    "base_model": re.compile(r"\*\*Base model\*\*:\s*(\S+)"),
    "direction": re.compile(r"\*\*Direction\*\*:\s*(\S+)"),
    "corpus_version": re.compile(r"\*\*Corpus version\*\*:\s*(\S+)"),
}


def parse_model_card_run_metadata(model_card_text: str) -> dict[str, str]:
    """Extract run identifiers (run_id, base_model, direction,
    corpus_version) from a rendered model card
    (`evaluation.model_card.render_model_card`'s output format) --
    everything `parse_model_card_metrics` doesn't already cover.
    """
    metadata: dict[str, str] = {}
    title_match = _TITLE_RUN_ID_RE.search(model_card_text)
    if title_match:
        metadata["run_id"] = title_match.group("run_id")
    for name, pattern in _RUN_METADATA_FIELD_RE.items():
        match = pattern.search(model_card_text)
        if match:
            metadata[name] = match.group(1)
    return metadata


def build_inference_artifact_key(run_id: str) -> str:
    """Where the repackaged, inference-ready artifact is uploaded --
    a distinct prefix from `training/submit_job.py`'s `model-artifacts/`,
    so the original training artifact (referenced by the training-container
    Model Package version) is never overwritten or confused with this one."""
    return f"{INFERENCE_ARTIFACT_PREFIX}/{run_id}/model.tar.gz"


def resolve_inference_image_uri(
    region: str = DEFAULT_REGION,
    *,
    instance_type: str = DEFAULT_IMAGE_LOOKUP_INSTANCE_TYPE,
) -> str:
    """Resolve the real HuggingFace **inference** DLC URI for this
    project's pinned version combination -- see module docstring.
    """
    return _retrieve_image_uri(
        framework="huggingface",
        region=region,
        version=INFERENCE_TRANSFORMERS_VERSION,
        py_version=INFERENCE_PY_VERSION,
        instance_type=instance_type,
        image_scope="inference",
        base_framework_version=f"pytorch{INFERENCE_PYTORCH_VERSION}",
    )


def build_inference_customer_metadata(
    run_metadata: dict[str, str],
    metrics: dict[str, str],
    *,
    source_model_data_url: str,
) -> dict[str, str]:
    """Flatten run metadata + metrics + provenance into the string->string
    map `CustomerMetadataProperties` requires. `artifact_type: inference`
    and `source_model_data_url` distinguish this Model Package version
    from the training-container version `submit_job.py` registers for the
    same run, and trace it back to the original training artifact.
    """
    metadata: dict[str, str] = {
        "artifact_type": "inference",
        "source_model_data_url": source_model_data_url,
    }
    metadata.update(run_metadata)
    metadata.update(metrics)
    return metadata


def register_inference_model(
    sm_client: Any,
    *,
    model_package_group_name: str,
    model_data_url: str,
    image_uri: str,
    run_metadata: dict[str, str],
    metrics: dict[str, str],
    source_model_data_url: str,
    approval_status: str = DEFAULT_APPROVAL_STATUS,
) -> str:
    """Register the repackaged, inference-ready artifact as a new Model
    Package version. Reuses `training.submit_job.ensure_model_package_group`
    -- same group, same idempotency handling, no reinvented logic.
    """
    ensure_model_package_group(
        sm_client,
        model_package_group_name,
        description=(
            "Spanish<->Kaqchikel fine-tuned M2M100 checkpoints "
            "(ADR 0001/0003/0006). Trained weights are private (ADR 0002); "
            "only identifiers/metrics are recorded here."
        ),
    )

    customer_metadata = build_inference_customer_metadata(
        run_metadata, metrics, source_model_data_url=source_model_data_url
    )

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


# ---------------------------------------------------------------------------
# S3 helpers (thin wrappers -- kept separate so the orchestration in main()
# stays readable and each I/O boundary is easy to mock in tests).
# ---------------------------------------------------------------------------


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    match = _S3_URI_RE.match(uri)
    if not match:
        raise ValueError(f"Not an s3:// URI: {uri!r}")
    return match.group("bucket"), match.group("key")


def download_model_artifact(s3_client: Any, source_model_data_url: str, destination: Path) -> None:
    bucket, key = _parse_s3_uri(source_model_data_url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, str(destination))


def upload_inference_artifact(s3_client: Any, local_path: Path, bucket: str, key: str) -> str:
    s3_client.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def read_model_card_from_local_tar(tar_path: Path) -> str:
    with tarfile.open(tar_path, "r:gz") as tar:
        member = tar.extractfile("model_card.md")
        if member is None:
            raise FileNotFoundError(f"model_card.md not found in {tar_path}")
        return member.read().decode("utf-8")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _print_dry_run_config(outputs: dict[str, str], image_uri: str, args: argparse.Namespace) -> None:
    preview = {
        "resolved_stack_outputs": outputs,
        "source_model_data_url": args.source_model_data_url,
        "inference_image_uri": image_uri,
        "model_package_group_name": args.model_package_group_name,
        "approval_status": args.approval_status,
    }
    print(json.dumps(preview, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfn_client = boto3.client("cloudformation")
    outputs = resolve_stack_outputs(args.environment, cloudformation_client=cfn_client)
    bucket = outputs["TrainingDataBucketName"]

    image_uri = resolve_inference_image_uri(args.region)

    if args.dry_run:
        _print_dry_run_config(outputs, image_uri, args)
        return 0

    s3_client = boto3.client("s3")
    sm_client = boto3.client("sagemaker")

    tmp_dir = Path(tempfile.mkdtemp(prefix="kaqchikel-deploy-"))
    code_dir: Path | None = None
    try:
        source_tar_path = tmp_dir / "source-model.tar.gz"
        download_model_artifact(s3_client, args.source_model_data_url, source_tar_path)

        model_card_text = read_model_card_from_local_tar(source_tar_path)
        metrics = parse_model_card_metrics(model_card_text)
        run_metadata = parse_model_card_run_metadata(model_card_text)
        run_id = run_metadata.get("run_id", "unknown-run")

        code_dir = build_inference_code_dir()
        output_tar_path = tmp_dir / "inference-model.tar.gz"
        repackage_model_artifact(source_tar_path, code_dir, output_tar_path)

        output_key = build_inference_artifact_key(run_id)
        model_data_url = upload_inference_artifact(s3_client, output_tar_path, bucket, output_key)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if code_dir is not None:
            shutil.rmtree(code_dir.parent, ignore_errors=True)

    model_package_arn = register_inference_model(
        sm_client,
        model_package_group_name=args.model_package_group_name,
        model_data_url=model_data_url,
        image_uri=image_uri,
        run_metadata=run_metadata,
        metrics=metrics,
        source_model_data_url=args.source_model_data_url,
        approval_status=args.approval_status,
    )
    print(f"Registered inference model package: {model_package_arn}")
    print(f"Inference artifact: {model_data_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
