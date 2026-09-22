"""Submit and monitor a real SageMaker Training Job running `training/train.py`
(issue #66) -- the one step in this project that spends real money. Every
piece of AWS-touching logic here is a small, separately testable function so
`ml/tests/unit/test_submit_job.py` can mock `boto3`/`sagemaker` calls; this
module is never exercised against real AWS in this repo's test suite.

## What this does

1. `resolve_stack_outputs` -- looks up the already-deployed `DataStack`'s
   CloudFormation outputs (`TrainingDataBucketName`,
   `SageMakerExecutionRoleArn`) at runtime, per environment
   (`infra/cdk/lib/data-stack.ts`). Never hardcodes a bucket name, role ARN,
   or account id.
2. `build_source_bundle` -- assembles a temp directory containing `data/`,
   `evaluation/`, and `training/` together (see its own docstring for why
   this is necessary: `source_dir` pointed at `training/` alone flattens
   away the sibling packages `train.py` actually imports, which surfaced
   as a real `ModuleNotFoundError` the first time this ran for real).
   `build_job_config` / `build_estimator` -- build a
   `sagemaker.huggingface.HuggingFace` estimator wired to that bundle
   (`entry_point="training/train.py"`, `source_dir=<bundle path>`), with a
   `max_run` safety cap so a runaway job can't rack up unbounded cost.
3. `build_training_inputs` / `submit_training_job` -- point the `train`/
   `validation` channels at the exact private-corpus object keys and call
   `estimator.fit(...)`.
4. `register_model` -- after a job completes, registers the resulting model
   in SageMaker Model Registry (idempotent Model Package Group creation,
   then a new Model Package with corpus version/hyperparameters/BLEU/chrF
   as `CustomerMetadataProperties`), defaulting to
   `PendingManualApproval` so a human reviews the eval metrics before an
   approved model can be deployed.
5. `main` -- CLI entrypoint tying the above together, with a `--dry-run`
   flag that prints the would-be job config without ever calling `.fit()`.

## Why `sagemaker<3`

As of this writing, PyPI's `sagemaker` package has two incompatible major
lines installable side by side in name only: the classic SDK (`2.x`, last
release `2.257.6`), which provides `sagemaker.huggingface.HuggingFace`,
`sagemaker.inputs.TrainingInput`, and `sagemaker.image_uris`; and a new,
unrelated `3.x` "next-generation" SDK that does **not** expose any of
those (`sagemaker.huggingface` doesn't exist there at all -- see
`sagemaker/__init__.py`'s submodule list: `ai_registry`, `core`, `lineage`,
`mlops`, `serve`, `train`). Since this ticket's required interface
(`sagemaker.huggingface.HuggingFace`, `sagemaker.inputs.TrainingInput`) only
exists in the `2.x` line, `ml/pyproject.toml` pins `sagemaker>=2.257,<3`
deliberately, not out of caution about breaking changes within `2.x`.

## How the HuggingFace DLC version combination was determined

`transformers_version`/`pytorch_version`/`py_version` must name an image
that actually exists in AWS's HuggingFace Deep Learning Container
repository, or `HuggingFace.fit()` fails immediately with an image-not-found
error. Rather than guessing a tuple by hand, this was derived from the
installed SDK's own compatibility data:

```python
from sagemaker.image_uris import config_for_framework, retrieve
training_versions = config_for_framework("huggingface")["training"]["versions"]
sorted(training_versions)  # -> [..., "4.49.0", "4.55.0", "4.56.2"]
training_versions["4.56.2"]
# -> {"version_aliases": {"pytorch2.8": "pytorch2.8.0"},
#     "pytorch2.8.0": {"py_versions": ["py312"], ...,
#                       "container_version": {"gpu": "cu129-ubuntu22.04"}}}
retrieve(framework="huggingface", region="us-east-1", version="4.56.2",
         py_version="py312", instance_type="ml.g4dn.xlarge",
         image_scope="training", base_framework_version="pytorch2.8.0")
# -> "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training
#     :2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04"
```

`4.56.2` is the highest (most recent) `transformers` version in the
installed SDK's training compatibility table with a GPU container variant,
`2.8.0` is its paired PyTorch version, and `py312` is the only Python
version that combination ships. Re-run the snippet above after any future
`sagemaker` SDK upgrade to confirm the constants below are still valid
before submitting a real job -- an SDK upgrade can add newer combinations
or (per the `sagemaker<3` note above) remove the whole `HuggingFace`
estimator API.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import botocore.exceptions
from sagemaker.huggingface import HuggingFace
from sagemaker.inputs import TrainingInput

from training.direction import DIRECTION_CHOICES
from training.train import DEFAULT_BASE_MODEL

# --- Corpus location (ADR 0002: private ALMG corpus, versioned prefix) -----
# Never derived from corpus content -- bump this (and re-run against a new
# prefix) whenever the raw corpus is reprocessed. See ml/README.md.
CORPUS_VERSION = "almg-v1"
CORPUS_PREFIX = "corpus/almg/v1"
TRAIN_OBJECT_KEY = f"{CORPUS_PREFIX}/train.tsv"
VALIDATION_OBJECT_KEY = f"{CORPUS_PREFIX}/val.tsv"

# SageMaker's standard channel-path convention: a channel named "train"
# pointing at an S3 object named "train.tsv" is downloaded, inside the
# training container, to /opt/ml/input/data/<channel>/<basename>.
TRAIN_CONTAINER_PATH = "/opt/ml/input/data/train/train.tsv"
VALIDATION_CONTAINER_PATH = "/opt/ml/input/data/validation/val.tsv"

# Channel name for an optional previous run's model artifact to continue
# training from (issue #75) -- container path is computed from the given
# S3 URI's basename (see `_init_model_container_path`), since unlike the
# fixed corpus files, that basename varies (a training job's own artifact
# is always named model.tar.gz, but this stays robust either way).
INIT_MODEL_CHANNEL_NAME = "init-model"

# --- Cost/safety defaults ---------------------------------------------------
DEFAULT_INSTANCE_TYPE = "ml.g4dn.xlarge"
DEFAULT_MAX_RUN_SECONDS = 3 * 60 * 60  # 3 hours -- a runaway job can't run forever.
DEFAULT_MODEL_PACKAGE_GROUP_NAME = "traductor-kaqchikel-es-cak"
DEFAULT_APPROVAL_STATUS = "PendingManualApproval"

# --- HuggingFace DLC version combination (see module docstring) ------------
TRANSFORMERS_VERSION = "4.56.2"
PYTORCH_VERSION = "2.8.0"
PY_VERSION = "py312"

# train.py is run as part of a package tree (it imports `data.*`,
# `evaluation.*`, and `training.*` as siblings), so `source_dir` must be a
# bundle whose root contains all three top-level packages -- see
# `build_source_bundle`. entry_point is relative to that bundle root.
ENTRY_POINT = "training/train.py"
BUNDLED_PACKAGES = ("data", "evaluation", "training")

REQUIRED_STACK_OUTPUTS = ("TrainingDataBucketName", "SageMakerExecutionRoleArn")


def _generate_run_id() -> str:
    """Same convention as `training.train.run_training_job`'s default, so a
    job name / hyperparameter run-id / model card run_id can all agree even
    though this script generates it independently (it must exist before the
    job is submitted, to name the job and register the model afterwards).
    """
    return datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI for submitting a real training job. Every hyperparameter defaults
    to matching `training/train.py`'s own default, and stays overridable here
    so a submission-time experiment doesn't require editing this file.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Submit (and optionally register) a SageMaker Training Job running "
            "training/train.py against the private ALMG corpus (issue #66)."
        )
    )
    parser.add_argument(
        "--environment",
        default="dev",
        help="Environment whose DataStack CloudFormation outputs to resolve (dev/qa/prod).",
    )
    parser.add_argument(
        "--instance-type",
        default=DEFAULT_INSTANCE_TYPE,
        help=f"Training instance type (default: {DEFAULT_INSTANCE_TYPE}).",
    )
    parser.add_argument(
        "--max-run",
        type=int,
        default=DEFAULT_MAX_RUN_SECONDS,
        help=(
            "Hard wall-clock cap on the job, in seconds, so a runaway job "
            f"can't run (and bill) forever (default: {DEFAULT_MAX_RUN_SECONDS})."
        ),
    )
    parser.add_argument(
        "--corpus-version",
        default=CORPUS_VERSION,
        help=(
            "Fixed identifier for the corpus version to train against "
            f"(default: {CORPUS_VERSION!r}). Never derived from corpus content."
        ),
    )
    parser.add_argument("--direction", choices=DIRECTION_CHOICES, default="both")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Identifier for this run; auto-generated (UTC timestamp) if omitted.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--init-model-s3-uri",
        default=None,
        help=(
            "s3:// URI to a previous run's model.tar.gz artifact to continue "
            "training from (issue #75), instead of --base-model. Staged as "
            "an input channel exactly like the train/validation corpus "
            "files, since a SageMaker training container can't reach an "
            "arbitrary S3 URI at runtime otherwise."
        ),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help=(
            "Defaults to 0.0 (disabled) -- see train.py's --label-smoothing "
            "help for why: a real crash confirmed against the actual "
            "checkpoint, not assumed."
        ),
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--model-package-group-name",
        default=DEFAULT_MODEL_PACKAGE_GROUP_NAME,
        help=f"SageMaker Model Registry group name (default: {DEFAULT_MODEL_PACKAGE_GROUP_NAME}).",
    )
    parser.add_argument(
        "--approval-status",
        choices=("PendingManualApproval", "Approved", "Rejected"),
        default=DEFAULT_APPROVAL_STATUS,
        help="ModelApprovalStatus to register the model with (default: PendingManualApproval).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit the job without blocking until it completes (skips registration).",
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Don't stream CloudWatch Logs to stdout while waiting.",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Don't register the resulting model in SageMaker Model Registry.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve the DataStack outputs and print the would-be job config "
            "(estimator kwargs, hyperparameters, channel S3 URIs) without "
            "building an estimator or calling .fit() at all."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 1. Resolve DataStack outputs
# ---------------------------------------------------------------------------


def resolve_stack_outputs(
    environment: str, *, cloudformation_client: Any = None
) -> dict[str, str]:
    """Look up the `{Environment}-Data` CloudFormation stack's outputs at
    runtime -- never hardcode the real bucket name/role ARN in this repo.
    """
    client = cloudformation_client or boto3.client("cloudformation")
    stack_name = f"{environment.capitalize()}-Data"

    response = client.describe_stacks(StackName=stack_name)
    stacks = response.get("Stacks", [])
    if not stacks:
        raise ValueError(f"No CloudFormation stack found named {stack_name!r}")

    outputs = {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}
    missing = [key for key in REQUIRED_STACK_OUTPUTS if key not in outputs]
    if missing:
        raise KeyError(
            f"Stack {stack_name!r} is missing required output(s): {missing} "
            f"(found: {sorted(outputs)})"
        )
    return outputs


# ---------------------------------------------------------------------------
# 2. Build the job config / estimator
# ---------------------------------------------------------------------------


def build_source_bundle(ml_root: Path | None = None) -> Path:
    """Assemble a fresh temp directory containing `data/`, `evaluation/`,
    and `training/` (this file's own package root by default), plus a
    root-level `requirements.txt` copied from `training/requirements.txt`.

    Why this exists: `sagemaker.huggingface.HuggingFace`'s `source_dir`
    upload flattens *the contents of* the given directory into
    `/opt/ml/code/` inside the training container -- it does not preserve
    that directory's own name as a package. Pointing `source_dir` directly
    at `ml/training/` (as an earlier version of this script did) meant
    `train.py`'s `from data.corpus_io import ...` / `from training.direction
    import ...` (sibling-package imports) failed with `ModuleNotFoundError`
    the first time this was run for real -- a duck-typed unit test can't
    catch this, since it's purely about how the real SDK packages a real
    local directory. This bundle preserves `data/`, `evaluation/`, and
    `training/` as actual subdirectories of `source_dir`'s root, so those
    imports resolve exactly as they do when running `train.py` locally from
    `ml/`. `entry_point` is `"training/train.py"`, relative to this root.

    Caller owns cleanup (`shutil.rmtree`) once the bundle has been uploaded
    (i.e. after `estimator.fit()`/`HuggingFace(...)` construction returns).
    """
    ml_root = ml_root or Path(__file__).resolve().parent.parent
    bundle_dir = Path(tempfile.mkdtemp(prefix="kaqchikel-train-bundle-"))

    for package_name in BUNDLED_PACKAGES:
        shutil.copytree(
            ml_root / package_name,
            bundle_dir / package_name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    shutil.copy2(
        ml_root / "training" / "requirements.txt",
        bundle_dir / "requirements.txt",
    )

    return bundle_dir


def build_channel_uris(bucket: str, *, init_model_s3_uri: str | None = None) -> dict[str, str]:
    """The exact S3 object URIs for the train/validation channels -- not the
    whole corpus prefix, so the container-side paths are deterministic (see
    module docstring / ml/README.md). Includes the optional `init-model`
    channel (issue #75) when `init_model_s3_uri` is given.
    """
    channels = {
        "train": f"s3://{bucket}/{TRAIN_OBJECT_KEY}",
        "validation": f"s3://{bucket}/{VALIDATION_OBJECT_KEY}",
    }
    if init_model_s3_uri:
        channels[INIT_MODEL_CHANNEL_NAME] = init_model_s3_uri
    return channels


def _init_model_container_path(init_model_s3_uri: str) -> str:
    """The container-side path an `init-model` channel resolves to, per
    SageMaker's `/opt/ml/input/data/<channel>/<s3-object-basename>`
    convention (see module docstring).
    """
    basename = init_model_s3_uri.rsplit("/", 1)[-1]
    return f"/opt/ml/input/data/{INIT_MODEL_CHANNEL_NAME}/{basename}"


def build_hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    """Hyperparameters passed through to `train.py` inside the container.

    Keys match `train.py`'s CLI flags exactly (hyphenated, e.g.
    `"corpus-version"`, not `"corpus_version"`) since the SageMaker training
    toolkit turns each hyperparameter dict key into a literal `--<key>` CLI
    flag for the entry point script.
    """
    run_id = args.run_id or _generate_run_id()
    hyperparameters = {
        "train": TRAIN_CONTAINER_PATH,
        "validation": VALIDATION_CONTAINER_PATH,
        "corpus-version": args.corpus_version,
        "base-model": args.base_model,
        "direction": args.direction,
        "run-id": run_id,
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "learning-rate": args.learning_rate,
        "max-length": args.max_length,
        "seed": args.seed,
        "warmup-ratio": args.warmup_ratio,
        "weight-decay": args.weight_decay,
        "label-smoothing": args.label_smoothing,
        "gradient-accumulation-steps": args.gradient_accumulation_steps,
    }
    if args.init_model_s3_uri:
        hyperparameters["init-model"] = _init_model_container_path(args.init_model_s3_uri)
    return hyperparameters


def build_job_config(
    *, bucket: str, role: str, args: argparse.Namespace, source_dir: str
) -> dict[str, Any]:
    """Assemble every value the `HuggingFace` estimator needs, as a plain
    dict -- used both to build the real estimator (`build_estimator`) and to
    print the `--dry-run` preview, so the two can never drift apart.

    `source_dir` is required (not defaulted here) so this function stays a
    pure dict-builder with no filesystem side effects -- callers build the
    real bundle via `build_source_bundle()` (see `main`) and pass its path
    in; tests pass a fake placeholder string instead.
    """
    hyperparameters = build_hyperparameters(args)
    return {
        "role": role,
        "instance_type": args.instance_type,
        "instance_count": 1,
        "max_run": args.max_run,
        "output_path": f"s3://{bucket}/model-artifacts/",
        "transformers_version": TRANSFORMERS_VERSION,
        "pytorch_version": PYTORCH_VERSION,
        "py_version": PY_VERSION,
        "hyperparameters": hyperparameters,
        "base_job_name": f"traductor-kaqchikel-{hyperparameters['run-id']}",
        "channels": build_channel_uris(bucket, init_model_s3_uri=args.init_model_s3_uri),
        "model_package_group_name": args.model_package_group_name,
        "approval_status": args.approval_status,
        "source_dir": source_dir,
        "entry_point": ENTRY_POINT,
    }


def build_estimator(job_config: dict[str, Any], *, sagemaker_session: Any = None) -> HuggingFace:
    """Build the `sagemaker.huggingface.HuggingFace` estimator wired to
    `training/train.py`. Never calls `.fit()` -- see `submit_training_job`.
    """
    return HuggingFace(
        entry_point=job_config["entry_point"],
        source_dir=job_config["source_dir"],
        role=job_config["role"],
        instance_type=job_config["instance_type"],
        instance_count=job_config["instance_count"],
        max_run=job_config["max_run"],
        output_path=job_config["output_path"],
        transformers_version=job_config["transformers_version"],
        pytorch_version=job_config["pytorch_version"],
        py_version=job_config["py_version"],
        hyperparameters=job_config["hyperparameters"],
        base_job_name=job_config["base_job_name"],
        sagemaker_session=sagemaker_session,
    )


def build_training_inputs(
    bucket: str, *, init_model_s3_uri: str | None = None
) -> dict[str, TrainingInput]:
    """`train`/`validation` channels pointing at the exact corpus object
    keys (not the whole prefix) -- see module docstring. Includes the
    optional `init-model` channel (issue #75) when `init_model_s3_uri` is
    given.
    """
    channels = build_channel_uris(bucket, init_model_s3_uri=init_model_s3_uri)
    return {name: TrainingInput(s3_data=uri) for name, uri in channels.items()}


# ---------------------------------------------------------------------------
# 3. Submit
# ---------------------------------------------------------------------------


def submit_training_job(
    estimator: HuggingFace,
    inputs: dict[str, TrainingInput],
    *,
    wait: bool = True,
    logs: bool = True,
) -> HuggingFace:
    """Submit the job. `wait=True` (default) blocks until the job completes
    or fails, streaming CloudWatch Logs when `logs=True` -- matching issue
    #66's "monitor the job to completion" scope.
    """
    estimator.fit(inputs=inputs, wait=wait, logs=logs)
    return estimator


# ---------------------------------------------------------------------------
# 4. Register in SageMaker Model Registry
# ---------------------------------------------------------------------------

_S3_URI_RE = re.compile(r"^s3://(?P<bucket>[^/]+)/(?P<key>.+)$")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    match = _S3_URI_RE.match(uri)
    if not match:
        raise ValueError(f"Not an s3:// URI: {uri!r}")
    return match.group("bucket"), match.group("key")


def fetch_model_card_from_artifact(s3_client: Any, model_data_url: str) -> str:
    """Download the training job's `model.tar.gz` from S3 and return the
    text of `model_card.md` packaged inside it (written by
    `training.train.run_training_job` alongside the saved model, per
    ADR 0001's traceability requirement).
    """
    bucket, key = _parse_s3_uri(model_data_url)
    buffer = io.BytesIO()
    s3_client.download_fileobj(bucket, key, buffer)
    buffer.seek(0)
    with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
        member = tar.extractfile("model_card.md")
        if member is None:
            raise FileNotFoundError(f"model_card.md not found in {model_data_url}")
        return member.read().decode("utf-8")


_METRIC_LINE_RE = {
    "bleu": re.compile(r"\*\*BLEU\*\*:\s*([0-9.]+)"),
    "chrf": re.compile(r"\*\*chrF\*\*:\s*([0-9.]+)"),
}


def parse_model_card_metrics(model_card_text: str) -> dict[str, str]:
    """Extract BLEU/chrF from a rendered model card
    (`evaluation.model_card.render_model_card`'s output format). Returns
    string values, ready to use as SageMaker `CustomerMetadataProperties`
    (which only accepts strings).
    """
    metrics: dict[str, str] = {}
    for name, pattern in _METRIC_LINE_RE.items():
        match = pattern.search(model_card_text)
        if match:
            metrics[name] = match.group(1)
    return metrics


def ensure_model_package_group(sm_client: Any, group_name: str, description: str) -> None:
    """Create the Model Package Group if it doesn't already exist.

    Idempotent: treats "already exists" as success. This was assumed to
    surface as a `ResourceInUse` error, and unit tests mocked exactly
    that -- but the real `CreateModelPackageGroup` API actually raises a
    generic `ValidationException` with the message "Model Package Group
    already exists" for this case, which the original `except
    sm_client.exceptions.ResourceInUse` clause never caught. This wasn't
    caught by tests because the mock matched the assumption, not the real
    API -- it only surfaced on the second-ever real registration call for
    this project (issue #76's continuation run), once the group already
    existed from #66's first registration. Catching `ResourceInUse` too,
    in case some other AWS SDK version or code path does use it.
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


def _build_customer_metadata(run_metadata: dict[str, Any], metrics: dict[str, str]) -> dict[str, str]:
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


def register_model(
    sm_client: Any,
    s3_client: Any,
    *,
    model_package_group_name: str,
    model_data_url: str,
    image_uri: str,
    run_metadata: dict[str, Any],
    approval_status: str = DEFAULT_APPROVAL_STATUS,
) -> str:
    """Register a completed training run's model artifact in SageMaker
    Model Registry: ensure the Model Package Group exists, then create a
    new Model Package version pointing at `model_data_url`, carrying
    corpus version / hyperparameters / BLEU / chrF as custom metadata.

    Defaults to `PendingManualApproval` -- a human should look at the eval
    metrics before a model can be approved for deployment.

    Returns the created Model Package's ARN.
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

    model_card_text = fetch_model_card_from_artifact(s3_client, model_data_url)
    metrics = parse_model_card_metrics(model_card_text)
    customer_metadata = _build_customer_metadata(run_metadata, metrics)

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
# 5. CLI entrypoint
# ---------------------------------------------------------------------------


def _print_dry_run_config(outputs: dict[str, str], config: dict[str, Any]) -> None:
    preview = {
        "resolved_stack_outputs": outputs,
        "estimator_kwargs": {
            "entry_point": config["entry_point"],
            "source_dir": config["source_dir"],
            "role": config["role"],
            "instance_type": config["instance_type"],
            "instance_count": config["instance_count"],
            "max_run": config["max_run"],
            "output_path": config["output_path"],
            "transformers_version": config["transformers_version"],
            "pytorch_version": config["pytorch_version"],
            "py_version": config["py_version"],
            "base_job_name": config["base_job_name"],
            "hyperparameters": config["hyperparameters"],
        },
        "channel_s3_uris": config["channels"],
        "model_package_group_name": config["model_package_group_name"],
        "approval_status": config["approval_status"],
    }
    print(json.dumps(preview, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfn_client = boto3.client("cloudformation")
    outputs = resolve_stack_outputs(args.environment, cloudformation_client=cfn_client)
    bucket = outputs["TrainingDataBucketName"]
    role = outputs["SageMakerExecutionRoleArn"]

    bundle_dir = build_source_bundle()
    try:
        config = build_job_config(
            bucket=bucket, role=role, args=args, source_dir=str(bundle_dir)
        )

        if args.dry_run:
            _print_dry_run_config(outputs, config)
            return 0

        estimator = build_estimator(config)
        inputs = build_training_inputs(bucket, init_model_s3_uri=args.init_model_s3_uri)

        submit_training_job(estimator, inputs, wait=not args.no_wait, logs=not args.no_logs)
    finally:
        # Safe to remove once HuggingFace(...)/estimator.fit() has returned --
        # the source_dir tarball is uploaded to S3 synchronously during
        # estimator construction/fit, not read lazily afterward.
        shutil.rmtree(bundle_dir, ignore_errors=True)

    if args.no_wait or args.no_register:
        return 0

    run_metadata = {
        "corpus_version": config["hyperparameters"]["corpus-version"],
        "direction": config["hyperparameters"]["direction"],
        "run_id": config["hyperparameters"]["run-id"],
        "hyperparameters": {
            "epochs": config["hyperparameters"]["epochs"],
            "batch_size": config["hyperparameters"]["batch-size"],
            "learning_rate": config["hyperparameters"]["learning-rate"],
            "max_length": config["hyperparameters"]["max-length"],
            "seed": config["hyperparameters"]["seed"],
            "warmup_ratio": config["hyperparameters"]["warmup-ratio"],
            "weight_decay": config["hyperparameters"]["weight-decay"],
            "label_smoothing": config["hyperparameters"]["label-smoothing"],
            "gradient_accumulation_steps": config["hyperparameters"][
                "gradient-accumulation-steps"
            ],
        },
    }

    sm_client = boto3.client("sagemaker")
    s3_client = boto3.client("s3")
    model_package_arn = register_model(
        sm_client,
        s3_client,
        model_package_group_name=args.model_package_group_name,
        model_data_url=estimator.model_data,
        image_uri=estimator.image_uri,
        run_metadata=run_metadata,
        approval_status=args.approval_status,
    )
    print(f"Registered model package: {model_package_arn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
