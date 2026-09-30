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
   `sagemaker.train.ModelTrainer` wired to that bundle
   (`entry_script="training/train.py"`, `source_dir=<bundle path>`), with a
   `max_run` safety cap so a runaway job can't rack up unbounded cost.
3. `build_training_inputs` / `submit_training_job` -- point the `train`/
   `validation` channels at the exact private-corpus object keys and call
   `estimator.train(...)`.
4. `register_model` -- after a job completes, registers the resulting model
   in SageMaker Model Registry (idempotent Model Package Group creation,
   then a new Model Package with corpus version/hyperparameters/BLEU/chrF
   as `CustomerMetadataProperties`), defaulting to
   `PendingManualApproval` so a human reviews the eval metrics before an
   approved model can be deployed.
5. `main` -- CLI entrypoint tying the above together, with a `--dry-run`
   flag that prints the would-be job config without ever calling `.train()`.

## SageMaker Python SDK v3 migration (issue #155, GHSA-5r2p-pjr8-7fh7)

This module was originally written against the SDK's classic `2.x` line
(`sagemaker.huggingface.HuggingFace`, `sagemaker.inputs.TrainingInput`,
`sagemaker.image_uris`), pinned `sagemaker>=2.257,<3` because the `3.x`
"next-generation" SDK exposed none of those names. `2.x` never received a
patch for GHSA-5r2p-pjr8-7fh7 (`eval()` on attacker-controlled input in
JumpStart's `search_hub()`, fixed in `3.4.0`) -- this codebase never calls
`search_hub()`/any JumpStart search function, so the real exploitability
was low, but staying on an unpatched, unpatchable major version
indefinitely wasn't a real fix. `ml/pyproject.toml` now pins
`sagemaker>=3.4,<4`, and this module targets v3's actual replacement API,
verified against the real installed package (`sagemaker==3.23.0` at
migration time), not assumed:

- `sagemaker.huggingface.HuggingFace` -> `sagemaker.train.ModelTrainer`,
  configured with `sagemaker.core.training.configs.SourceCode` (source
  dir/entry script/requirements file -- replaces `entry_point`/
  `source_dir`), `Compute` (instance type/count), and
  `sagemaker.core.shapes.{StoppingCondition,OutputDataConfig}` (max
  runtime / output path). Unlike `HuggingFace`, `ModelTrainer` doesn't
  resolve a training container image from
  `transformers_version`/`pytorch_version`/`py_version` -- callers must
  resolve and pass `training_image` explicitly (`resolve_training_image_uri`
  does this, the same way `deployment/deploy.py` already did for the
  inference image).
- `sagemaker.inputs.TrainingInput` -> `sagemaker.core.training.configs.
  InputData(channel_name=..., data_source=<s3 uri string>)`, passed to
  `ModelTrainer.train(input_data_config=[...])` as a **list**, not a
  `{channel_name: TrainingInput}` dict -- `estimator.fit(inputs=...)`
  becomes `estimator.train(input_data_config=...)`.
- `sagemaker.image_uris` -> `sagemaker.core.image_uris` (same
  `retrieve`/`config_for_framework` functions, just relocated).
- Hyperparameters are still passed through to `train.py` as literal
  `--<key> <value>` CLI flags -- confirmed against v3's actual training
  driver (`sagemaker.train.container_drivers.distributed_drivers.
  basic_script_driver.hyperparameters_to_cli_args`), not assumed. No
  change needed to `build_hyperparameters`' hyphenated keys.

**Behavioral difference that matters for `--dry-run`:** constructing a
`ModelTrainer` with a `role` triggers an eager, real (though read-only)
`iam:SimulatePrincipalPolicy` AWS call to validate that role --
confirmed directly against the real SDK, where this raised an uncaught
`botocore.exceptions.ClientError` for a placeholder role ARN in a
nonexistent account. `HuggingFace` (v2) was side-effect-free at
construction. This is why `main()`'s `--dry-run` path builds and prints
the job config dict but, exactly as before, never calls `build_estimator`
at all -- doing so would make `--dry-run` sometimes fail or succeed
depending on the caller's ambient AWS credentials/role, which defeats the
point of a dry run.

After a waited-for `.train()` call, the resulting model artifact S3 URI
comes from `estimator._latest_training_job.model_artifacts.
s3_model_artifacts` (populated once the training job resource is
refreshed to a terminal state) rather than the old `estimator.model_data`
attribute; the image URI is simply `estimator.training_image`, since this
module resolves and passes it in explicitly rather than letting the
estimator resolve it lazily.

## How the HuggingFace DLC version combination was determined

`transformers_version`/`pytorch_version`/`py_version` must name an image
that actually exists in AWS's HuggingFace Deep Learning Container
repository, or the training job fails immediately with an
image-not-found error. Rather than guessing a tuple by hand, this was
derived from the installed SDK's own compatibility data:

```python
from sagemaker.core.image_uris import config_for_framework, retrieve
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

`4.56.2` is the highest `transformers` version in the installed SDK's
training compatibility table this project has validated end-to-end (a
newer `5.3.0` combination now also exists in the table, unrelated to this
migration -- switching to it is a separate, deliberate decision, not a
side effect of the SDK bump). `2.8.0` is `4.56.2`'s paired PyTorch
version, and `py312` is the only Python version that combination ships.
Re-run the snippet above after any future `sagemaker` SDK upgrade to
confirm the constants below are still valid before submitting a real job.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from sagemaker.core.image_uris import retrieve as _retrieve_image_uri
from sagemaker.core.shapes import OutputDataConfig, StoppingCondition
from sagemaker.core.training.configs import Compute, InputData, SourceCode
from sagemaker.train import ModelTrainer

# `parse_model_card_metrics`/`ensure_model_package_group` aren't referenced
# directly in this module any more (both moved to
# evaluation.model_card/training.model_registry, issue #190) -- imported
# here purely to re-export as `submit_job.parse_model_card_metrics`/
# `submit_job.ensure_model_package_group` for existing callers/tests.
from evaluation.model_card import parse_model_card_metrics  # noqa: F401
from training.direction import DIRECTION_CHOICES
from training.model_registry import (
    DEFAULT_APPROVAL_STATUS,
    DEFAULT_MODEL_PACKAGE_GROUP_NAME,
    ensure_model_package_group,  # noqa: F401
    register_model_package,
)
from training.subword_vocab import DEFAULT_VOCAB_SIZE as DEFAULT_SUBWORD_VOCAB_SIZE
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
# DEFAULT_MODEL_PACKAGE_GROUP_NAME/DEFAULT_APPROVAL_STATUS now live in
# training.model_registry (issue #190) -- imported above, re-exported here
# unchanged so existing callers/tests referencing submit_job.DEFAULT_* keep
# working.

# Single source of truth for the region this project lives in (matches
# infra/cdk/bin/app.ts and deployment/deploy.py's DEFAULT_REGION) -- needed
# explicitly now that this module resolves its own training container image
# URI instead of relying on the (removed) HuggingFace estimator to do it.
DEFAULT_REGION = "us-east-1"

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
# `build_source_bundle` always copies `training/requirements.txt` to this
# fixed, bundle-root-relative path -- ModelTrainer's `SourceCode.requirements`
# wants a path *within* source_dir, not an absolute one.
REQUIREMENTS_FILENAME = "requirements.txt"

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
        "--region",
        default=DEFAULT_REGION,
        help=(
            f"AWS region to resolve the training container image URI for "
            f"(default: {DEFAULT_REGION}, matching this project's single region)."
        ),
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
    parser.add_argument(
        "--corpus-prefix",
        default=None,
        help=(
            "S3 prefix (under the environment's training-data bucket) to read "
            f"train.tsv/val.tsv from, overriding the hardcoded default "
            f"({CORPUS_PREFIX!r}) for a one-off comparison run -- e.g. issue "
            "#199's cleaned-corpus vs. baseline comparison. Always pass a "
            "matching --corpus-version alongside this so the model card/Model "
            "Registry label doesn't silently disagree with what was actually "
            "trained on (see ml/README.md)."
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
        "--subword-vocab-size",
        type=int,
        default=DEFAULT_SUBWORD_VOCAB_SIZE,
        help=(
            "Target vocabulary size for the Kaqchikel-only SentencePiece/"
            "Unigram subword vocabulary merged into the tokenizer (issue "
            f"#82). Default: {DEFAULT_SUBWORD_VOCAB_SIZE}. See "
            "train.py's --subword-vocab-size help for the full rationale."
        ),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help=(
            "Override the model's general dropout probability (issue #182). "
            "Defaults to None (untouched, matching train.py's own default) "
            "-- see train.py's --dropout help for the full rationale."
        ),
    )
    parser.add_argument(
        "--bpe-dropout-alpha",
        type=float,
        default=None,
        help=(
            "SentencePiece subword-sampling alpha for the training corpus "
            "('BPE-dropout' / subword regularization, issue #182). "
            "Defaults to None (disabled, matching train.py's own default) "
            "-- see train.py's --bpe-dropout-alpha help for the full "
            "rationale."
        ),
    )
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
        help=(
            "Submit the job without blocking until it completes. Issue "
            "#190: this no longer skips registration -- train.py "
            "self-registers from inside the container once the job "
            "finishes, regardless of whether this process waited for it."
        ),
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Don't stream CloudWatch Logs to stdout while waiting.",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help=(
            "Tell the submitted job not to self-register its model in "
            "SageMaker Model Registry (issue #190; train.py self-registers "
            "by default). Use --register-existing later to register such a "
            "job's artifact after the fact."
        ),
    )
    parser.add_argument(
        "--register-existing",
        metavar="TRAINING_JOB_NAME",
        default=None,
        help=(
            "Register an already-completed training job's artifact in "
            "Model Registry, by job name, instead of submitting a new job "
            "(issue #190's explicit opt-in path -- for a job that "
            "predates self-registration, or one submitted with "
            "--no-register/--no-wait that needs registering after the "
            "fact). Every other flag except --model-package-group-name/"
            "--approval-status is ignored when this is given."
        ),
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


def build_channel_uris(
    bucket: str, *, corpus_prefix: str | None = None, init_model_s3_uri: str | None = None
) -> dict[str, str]:
    """The exact S3 object URIs for the train/validation channels -- not the
    whole corpus prefix, so the container-side paths are deterministic (see
    module docstring / ml/README.md). Includes the optional `init-model`
    channel (issue #75) when `init_model_s3_uri` is given.

    `corpus_prefix` (issue #199) overrides the hardcoded `CORPUS_PREFIX`
    default -- e.g. to run a one-off comparison job against a different
    corpus reprocessing (a cleaned/filtered version) without editing this
    module's default, which stays whatever `CORPUS_PREFIX` says until a
    deliberate switchover decision (see ml/README.md's warning about
    `CORPUS_PREFIX`/`CORPUS_VERSION` drifting apart -- passing this without
    also passing a matching `--corpus-version` would have the exact same
    drift risk, so keep them in sync at the call site).
    """
    prefix = corpus_prefix or CORPUS_PREFIX
    channels = {
        "train": f"s3://{bucket}/{prefix}/train.tsv",
        "validation": f"s3://{bucket}/{prefix}/val.tsv",
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
        "subword-vocab-size": args.subword_vocab_size,
        # Issue #190: tells train.py whether to self-register this run's
        # completed model package from inside the training container --
        # the hyperparameter dict is serialized to string CLI args
        # regardless of value type (see module docstring), so this is a
        # literal "true"/"false" string, not a Python bool. Mirrors
        # `--no-register`'s existing meaning: register unless explicitly
        # opted out.
        "register-model": "false" if args.no_register else "true",
        "model-package-group-name": args.model_package_group_name,
        "approval-status": args.approval_status,
    }
    if args.init_model_s3_uri:
        hyperparameters["init-model"] = _init_model_container_path(args.init_model_s3_uri)
    # Issue #182: omitted (rather than always included, e.g. as `None`)
    # when not explicitly set -- a hyperparameter value of `None` would
    # serialize to the literal string "None" and get passed to train.py as
    # `--dropout None`, which argparse's `type=float` can't parse. This
    # also keeps a submission that never opts into either lever identical
    # to one submitted before this ticket existed.
    if args.dropout is not None:
        hyperparameters["dropout"] = args.dropout
    if args.bpe_dropout_alpha is not None:
        hyperparameters["bpe-dropout-alpha"] = args.bpe_dropout_alpha
    return hyperparameters


def resolve_training_image_uri(
    region: str = DEFAULT_REGION, *, instance_type: str = DEFAULT_INSTANCE_TYPE
) -> str:
    """Resolve the real HuggingFace **training** DLC URI for this project's
    pinned version combination (module docstring) via the v3 SDK's
    `sagemaker.core.image_uris.retrieve` -- a pure local lookup against the
    installed SDK's bundled compatibility tables, no AWS/network call.

    v2's `HuggingFace` estimator resolved its own training image internally
    from `transformers_version`/`pytorch_version`/`py_version`; v3's
    `ModelTrainer` requires the resolved `training_image` URI upfront (see
    `build_estimator`), so this module now does explicitly what
    `deployment/deploy.py`'s `resolve_inference_image_uri` already did for
    the inference image.
    """
    return _retrieve_image_uri(
        framework="huggingface",
        region=region,
        version=TRANSFORMERS_VERSION,
        py_version=PY_VERSION,
        instance_type=instance_type,
        image_scope="training",
        base_framework_version=f"pytorch{PYTORCH_VERSION}",
    )


def build_job_config(
    *, bucket: str, role: str, args: argparse.Namespace, source_dir: str, training_image: str
) -> dict[str, Any]:
    """Assemble every value the `ModelTrainer` estimator needs, as a plain
    dict -- used both to build the real estimator (`build_estimator`) and to
    print the `--dry-run` preview, so the two can never drift apart.

    `source_dir`/`training_image` are required (not defaulted here) so this
    function stays a pure dict-builder with no filesystem/network side
    effects -- callers build the real bundle via `build_source_bundle()` and
    resolve the real image via `resolve_training_image_uri()` (see `main`)
    and pass both in; tests pass fake placeholder strings instead.
    """
    hyperparameters = build_hyperparameters(args)
    output_path = f"s3://{bucket}/model-artifacts/"
    # Issue #190: train.py self-registers its own completed model package
    # from inside the training container, which requires knowing (a) where
    # SageMaker will eventually upload its artifact -- `output_path` here,
    # combined with this job's own SageMaker-assigned name (read from
    # SM_TRAINING_ENV at runtime, not knowable client-side) -- and (b) the
    # exact training container image URI to record on the registered Model
    # Package, which a container has no way to discover about itself.
    # Neither is otherwise available inside the container. Issue #196: a
    # real self-registration attempt failed with `NoRegionError` -- the
    # container has no ambient default region a bare `boto3.client()` call
    # can resolve the way a maintainer's own configured shell does, so
    # `--region` must be passed through too.
    hyperparameters["output-path"] = output_path
    hyperparameters["training-image"] = training_image
    hyperparameters["region"] = args.region
    return {
        "role": role,
        "instance_type": args.instance_type,
        "instance_count": 1,
        "max_run": args.max_run,
        "output_path": output_path,
        "transformers_version": TRANSFORMERS_VERSION,
        "pytorch_version": PYTORCH_VERSION,
        "py_version": PY_VERSION,
        "training_image": training_image,
        "hyperparameters": hyperparameters,
        "base_job_name": f"traductor-kaqchikel-{hyperparameters['run-id']}",
        "channels": build_channel_uris(
            bucket, corpus_prefix=args.corpus_prefix, init_model_s3_uri=args.init_model_s3_uri
        ),
        "model_package_group_name": args.model_package_group_name,
        "approval_status": args.approval_status,
        "source_dir": source_dir,
        "entry_point": ENTRY_POINT,
    }


def build_estimator(job_config: dict[str, Any], *, sagemaker_session: Any = None) -> ModelTrainer:
    """Build the v3 SDK's `sagemaker.train.ModelTrainer` wired to
    `training/train.py`. Never calls `.train()` -- see `submit_training_job`.

    Note: constructing `ModelTrainer` with a real `role` triggers an eager,
    read-only `iam:SimulatePrincipalPolicy` AWS call to validate that role
    -- a real behavioral difference from v2's `HuggingFace` estimator,
    which was side-effect-free at construction time (see module docstring).
    This is why `main()`'s `--dry-run` path never calls this function.
    """
    source_code = SourceCode(
        source_dir=job_config["source_dir"],
        entry_script=job_config["entry_point"],
        requirements=REQUIREMENTS_FILENAME,
    )
    compute = Compute(
        instance_type=job_config["instance_type"],
        instance_count=job_config["instance_count"],
    )
    stopping_condition = StoppingCondition(max_runtime_in_seconds=job_config["max_run"])
    output_data_config = OutputDataConfig(s3_output_path=job_config["output_path"])
    return ModelTrainer(
        training_image=job_config["training_image"],
        source_code=source_code,
        role=job_config["role"],
        base_job_name=job_config["base_job_name"],
        compute=compute,
        stopping_condition=stopping_condition,
        output_data_config=output_data_config,
        hyperparameters=job_config["hyperparameters"],
        sagemaker_session=sagemaker_session,
    )


def build_training_inputs(
    bucket: str, *, corpus_prefix: str | None = None, init_model_s3_uri: str | None = None
) -> list[InputData]:
    """`train`/`validation` channels pointing at the exact corpus object
    keys (not the whole prefix) -- see module docstring. Includes the
    optional `init-model` channel (issue #75) when `init_model_s3_uri` is
    given. `corpus_prefix` overrides the hardcoded default (issue #199,
    see `build_channel_uris`).

    Returns a **list** of `InputData`, not a `{channel_name: TrainingInput}`
    dict -- `ModelTrainer.train(input_data_config=...)`'s v3 shape, unlike
    v2's `HuggingFace.fit(inputs=...)`.
    """
    channels = build_channel_uris(
        bucket, corpus_prefix=corpus_prefix, init_model_s3_uri=init_model_s3_uri
    )
    return [InputData(channel_name=name, data_source=uri) for name, uri in channels.items()]


# ---------------------------------------------------------------------------
# 3. Submit
# ---------------------------------------------------------------------------


def submit_training_job(
    estimator: ModelTrainer,
    inputs: list[InputData],
    *,
    wait: bool = True,
    logs: bool = True,
) -> ModelTrainer:
    """Submit the job. `wait=True` (default) blocks until the job completes
    or fails, streaming CloudWatch Logs when `logs=True` -- matching issue
    #66's "monitor the job to completion" scope.
    """
    estimator.train(input_data_config=inputs, wait=wait, logs=logs)
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

    Streams the download to a temp file on disk rather than an in-memory
    buffer (issue #188): this artifact bundles per-epoch checkpoints
    alongside the final model (issue #182/#187) and was measured at 30.5 GB
    on a real run -- downloading that into memory just to read one small
    text file inside it caused a real out-of-memory kill. Peak process
    memory here stays roughly constant regardless of artifact size.
    """
    bucket, key = _parse_s3_uri(model_data_url)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp_file:
        s3_client.download_file(bucket, key, tmp_file.name)
        with tarfile.open(tmp_file.name, mode="r:gz") as tar:
            member = tar.extractfile("model_card.md")
            if member is None:
                raise FileNotFoundError(f"model_card.md not found in {model_data_url}")
            return member.read().decode("utf-8")


# `ensure_model_package_group`/`parse_model_card_metrics` now live in
# training.model_registry/evaluation.model_card (issue #190) -- imported
# above, re-exported here unchanged so existing callers/tests referencing
# submit_job.ensure_model_package_group / submit_job.parse_model_card_metrics
# keep working.


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
    """Register an already-completed training run's model artifact in
    SageMaker Model Registry -- the `--register-existing` opt-in path (issue
    #190; see this module's docstring for why the default, per-submission
    registration path no longer works this way).

    Unlike `training.model_registry.register_model_package` (which this
    delegates to), this fetches the model card from the artifact itself
    first (`fetch_model_card_from_artifact`, streamed to disk -- issue
    #188), since a maintainer using this path has no other way to read a
    completed job's own model card. Returns the created Model Package's
    ARN.
    """
    model_card_text = fetch_model_card_from_artifact(s3_client, model_data_url)
    return register_model_package(
        sm_client,
        model_package_group_name=model_package_group_name,
        model_data_url=model_data_url,
        image_uri=image_uri,
        model_card_text=model_card_text,
        run_metadata=run_metadata,
        approval_status=approval_status,
    )


def _reconstruct_run_metadata_from_job_hyperparameters(
    hyperparameters: dict[str, str],
) -> dict[str, Any]:
    """Rebuild the `run_metadata` dict `register_model_package` needs from
    a completed training job's own submitted hyperparameters, as returned
    by `describe_training_job` -- used by `register_existing_job` so a
    maintainer registering an existing job doesn't have to re-type its
    corpus version/direction/hyperparameters by hand.

    Every value in `describe_training_job`'s `HyperParameters` response
    comes back as a string (SageMaker's own convention for that field) --
    recorded as-is rather than guessing back the original type;
    `CustomerMetadataProperties` only accepts strings anyway.
    """
    # Excluded: channel paths and fields that are either not
    # hyperparameters worth recording as metadata (corpus-version/
    # direction/run-id are pulled out into their own top-level fields
    # below) or plumbing specific to *submitting* a job (never meaningful
    # metadata about the resulting model).
    excluded = {
        "train",
        "validation",
        "corpus-version",
        "direction",
        "run-id",
        "output-path",
        "training-image",
        "region",
        "register-model",
        "model-package-group-name",
        "approval-status",
        "init-model",
        # Tracked as its own top-level run_metadata["base_model"] field by
        # the self-registration path in train.py, never as a
        # "hyperparameters" entry -- excluded here so this reconstructed
        # path doesn't record a hp_base_model field the primary
        # self-registration path never produces (code review on PR #191).
        "base-model",
    }
    hyperparameters_for_metadata = {
        key.replace("-", "_"): value
        for key, value in hyperparameters.items()
        if key not in excluded
    }
    return {
        "corpus_version": hyperparameters.get("corpus-version", "unknown"),
        "direction": hyperparameters.get("direction", "unknown"),
        "run_id": hyperparameters.get("run-id", "unknown"),
        "hyperparameters": hyperparameters_for_metadata,
    }


def register_existing_job(
    job_name: str,
    *,
    sm_client: Any,
    s3_client: Any,
    model_package_group_name: str,
    approval_status: str,
) -> str:
    """Register an already-completed training job's artifact, by job name
    (issue #190's explicit opt-in path -- `submit_job.py --register-existing
    <job-name>`). For a job that predates self-registration entirely, or
    one submitted with `--no-register`/`--no-wait` that a maintainer now
    wants registered after the fact.

    Looks up the job's own model artifact URI/training image/submitted
    hyperparameters via `describe_training_job` -- a maintainer never needs
    to re-type any of that by hand. Still downloads the full
    `model.tar.gz` (streamed to disk, issue #188) to read its model card:
    unlike the eliminated per-submission client-side download this
    replaces, this is a rare, explicit, one-off action, not every real
    job's default path.

    Re-running this against a job that already self-registered creates a
    redundant (but harmless) additional Model Package version pointing at
    the same artifact -- this is intentionally for jobs that never
    self-registered in the first place.
    """
    description = sm_client.describe_training_job(TrainingJobName=job_name)
    model_data_url = description["ModelArtifacts"]["S3ModelArtifacts"]
    image_uri = description["AlgorithmSpecification"]["TrainingImage"]
    job_hyperparameters = description.get("HyperParameters", {})
    run_metadata = _reconstruct_run_metadata_from_job_hyperparameters(job_hyperparameters)

    return register_model(
        sm_client,
        s3_client,
        model_package_group_name=model_package_group_name,
        model_data_url=model_data_url,
        image_uri=image_uri,
        run_metadata=run_metadata,
        approval_status=approval_status,
    )


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
            "training_image": config["training_image"],
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

    if args.register_existing:
        # Explicit opt-in path (issue #190): register an already-completed
        # job's artifact, by name, instead of submitting anything new.
        sm_client = boto3.client("sagemaker")
        s3_client = boto3.client("s3")
        model_package_arn = register_existing_job(
            args.register_existing,
            sm_client=sm_client,
            s3_client=s3_client,
            model_package_group_name=args.model_package_group_name,
            approval_status=args.approval_status,
        )
        print(f"Registered model package: {model_package_arn}")
        return 0

    cfn_client = boto3.client("cloudformation")
    outputs = resolve_stack_outputs(args.environment, cloudformation_client=cfn_client)
    bucket = outputs["TrainingDataBucketName"]
    role = outputs["SageMakerExecutionRoleArn"]

    # Pure local lookup (module docstring) -- safe to resolve unconditionally,
    # including in --dry-run, unlike building the estimator itself below.
    training_image = resolve_training_image_uri(args.region, instance_type=args.instance_type)

    bundle_dir = build_source_bundle()
    try:
        config = build_job_config(
            bucket=bucket,
            role=role,
            args=args,
            source_dir=str(bundle_dir),
            training_image=training_image,
        )

        if args.dry_run:
            _print_dry_run_config(outputs, config)
            return 0

        # Only reached for a real (non-dry-run) submission: constructing
        # ModelTrainer makes a live IAM call to validate `role` (module
        # docstring) -- never do this on the --dry-run path.
        estimator = build_estimator(config)
        inputs = build_training_inputs(
            bucket, corpus_prefix=args.corpus_prefix, init_model_s3_uri=args.init_model_s3_uri
        )

        submit_training_job(estimator, inputs, wait=not args.no_wait, logs=not args.no_logs)
    finally:
        # Safe to remove once ModelTrainer(...)/estimator.train() has
        # returned -- the source_dir tarball is uploaded to S3 synchronously
        # during training-job creation, not read lazily afterward.
        shutil.rmtree(bundle_dir, ignore_errors=True)

    # Issue #190: no client-side registration here any more.
    # `config["hyperparameters"]` already told the submitted job (via
    # `register-model`/`output-path`/`training-image`/
    # `model-package-group-name`/`approval-status`) everything it needs to
    # register its own completed model package from inside the container,
    # with no download of the training artifact -- see train.py's
    # `register_model_from_training_job`. This happens whether or not this
    # process waited for the job (`--no-wait`); only `--no-register`
    # disables it.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
