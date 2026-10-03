"""Unit tests for `training.submit_job` -- the SageMaker Training Job
submission CLI for issue #66. Every AWS-touching call (CloudFormation,
SageMaker, S3, and the `sagemaker` v3 SDK's `ModelTrainer`) is mocked here
-- this suite never makes a real AWS API call, never constructs a real
`boto3`/`sagemaker` session, and never spends money. See `ml/README.md`'s
"Job submission" section.

Migrated off `sagemaker.huggingface.HuggingFace` (removed in SageMaker
Python SDK v3, GHSA-5r2p-pjr8-7fh7 -- issue #155) onto v3's
`sagemaker.train.ModelTrainer`. `ModelTrainer` is never constructed for
real in this suite (only ever mocked): its constructor makes a live,
read-only `iam:SimulatePrincipalPolicy` AWS call to validate the given
role, a real behavioral difference from v2's `HuggingFace` estimator
(side-effect-free at construction) -- see `training/submit_job.py`'s
module docstring.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import botocore.exceptions
import pytest

from training import submit_job

# ---------------------------------------------------------------------------
# build_source_bundle
# ---------------------------------------------------------------------------


def _make_fake_ml_root(tmp_path: Path) -> Path:
    """A tiny stand-in for the real ml/ package root: data/, evaluation/,
    training/, each with a marker file plus a __pycache__ dir that must not
    be copied, and training/requirements.txt.
    """
    ml_root = tmp_path / "ml"
    for package_name in ("data", "evaluation", "training"):
        package_dir = ml_root / package_name
        package_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("")
        (package_dir / f"{package_name}_marker.py").write_text(f"# {package_name}\n")
        cache_dir = package_dir / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "stale.pyc").write_bytes(b"\x00")
    (ml_root / "training" / "train.py").write_text("# entry point\n")
    (ml_root / "training" / "requirements.txt").write_text("transformers>=4.40\n")
    return ml_root


def test_build_source_bundle_copies_all_three_packages(tmp_path):
    ml_root = _make_fake_ml_root(tmp_path)

    bundle_dir = submit_job.build_source_bundle(ml_root)

    assert (bundle_dir / "data" / "data_marker.py").exists()
    assert (bundle_dir / "evaluation" / "evaluation_marker.py").exists()
    assert (bundle_dir / "training" / "train.py").exists()


def test_build_source_bundle_excludes_pycache(tmp_path):
    ml_root = _make_fake_ml_root(tmp_path)

    bundle_dir = submit_job.build_source_bundle(ml_root)

    assert not (bundle_dir / "data" / "__pycache__").exists()
    assert not (bundle_dir / "training" / "__pycache__").exists()


def test_build_source_bundle_copies_requirements_txt_to_bundle_root(tmp_path):
    ml_root = _make_fake_ml_root(tmp_path)

    bundle_dir = submit_job.build_source_bundle(ml_root)

    root_requirements = bundle_dir / "requirements.txt"
    assert root_requirements.exists()
    assert root_requirements.read_text() == "transformers>=4.40\n"


def test_build_source_bundle_returns_a_fresh_directory_each_call(tmp_path):
    ml_root = _make_fake_ml_root(tmp_path)

    first = submit_job.build_source_bundle(ml_root)
    second = submit_job.build_source_bundle(ml_root)

    assert first != second


# ---------------------------------------------------------------------------
# resolve_stack_outputs
# ---------------------------------------------------------------------------


def test_resolve_stack_outputs_returns_bucket_and_role(monkeypatch):
    fake_client = MagicMock()
    fake_client.describe_stacks.return_value = {
        "Stacks": [
            {
                "Outputs": [
                    {"OutputKey": "TrainingDataBucketName", "OutputValue": "fake-bucket"},
                    {"OutputKey": "SageMakerExecutionRoleArn", "OutputValue": "arn:aws:iam::000000000000:role/fake-role"},
                ]
            }
        ]
    }

    outputs = submit_job.resolve_stack_outputs("dev", cloudformation_client=fake_client)

    fake_client.describe_stacks.assert_called_once_with(StackName="Dev-Data")
    assert outputs["TrainingDataBucketName"] == "fake-bucket"
    assert outputs["SageMakerExecutionRoleArn"] == "arn:aws:iam::000000000000:role/fake-role"


def test_resolve_stack_outputs_capitalizes_environment_name():
    fake_client = MagicMock()
    fake_client.describe_stacks.return_value = {
        "Stacks": [
            {
                "Outputs": [
                    {"OutputKey": "TrainingDataBucketName", "OutputValue": "b"},
                    {"OutputKey": "SageMakerExecutionRoleArn", "OutputValue": "r"},
                ]
            }
        ]
    }

    submit_job.resolve_stack_outputs("qa", cloudformation_client=fake_client)

    fake_client.describe_stacks.assert_called_once_with(StackName="Qa-Data")


def test_resolve_stack_outputs_raises_on_missing_required_output():
    fake_client = MagicMock()
    fake_client.describe_stacks.return_value = {
        "Stacks": [{"Outputs": [{"OutputKey": "TrainingDataBucketName", "OutputValue": "b"}]}]
    }

    with pytest.raises(KeyError):
        submit_job.resolve_stack_outputs("dev", cloudformation_client=fake_client)


def test_resolve_stack_outputs_raises_when_stack_not_found():
    fake_client = MagicMock()
    fake_client.describe_stacks.return_value = {"Stacks": []}

    with pytest.raises(ValueError):
        submit_job.resolve_stack_outputs("dev", cloudformation_client=fake_client)


# ---------------------------------------------------------------------------
# Channel URIs / hyperparameters / job config (pure logic, no AWS)
# ---------------------------------------------------------------------------


def test_build_channel_uris_points_at_exact_corpus_objects():
    channels = submit_job.build_channel_uris("fake-bucket")

    assert channels == {
        "train": "s3://fake-bucket/corpus/almg/v1/train.tsv",
        "validation": "s3://fake-bucket/corpus/almg/v1/val.tsv",
    }


def test_build_channel_uris_accepts_a_corpus_prefix_override():
    # Issue #199: a one-off comparison run against a different corpus
    # version (e.g. a cleaned/filtered reprocessing) shouldn't require
    # editing this module's hardcoded default -- see ml/README.md's
    # existing warning about CORPUS_PREFIX/CORPUS_VERSION drifting apart.
    channels = submit_job.build_channel_uris("fake-bucket", corpus_prefix="corpus/almg/v3")

    assert channels == {
        "train": "s3://fake-bucket/corpus/almg/v3/train.tsv",
        "validation": "s3://fake-bucket/corpus/almg/v3/val.tsv",
    }


def test_parse_args_corpus_prefix_defaults_to_none():
    args = submit_job.parse_args(["--run-id", "run-test"])

    assert args.corpus_prefix is None


def test_build_hyperparameters_includes_container_side_paths():
    args = submit_job.parse_args(["--run-id", "run-test"])

    hyperparameters = submit_job.build_hyperparameters(args)

    assert hyperparameters["train"] == "/opt/ml/input/data/train/train.tsv"
    assert hyperparameters["validation"] == "/opt/ml/input/data/validation/val.tsv"


def test_build_hyperparameters_passes_through_cli_overridable_values():
    args = submit_job.parse_args(
        [
            "--run-id",
            "run-test",
            "--direction",
            "es->cak",
            "--epochs",
            "5",
            "--batch-size",
            "16",
            "--learning-rate",
            "0.0001",
            "--max-length",
            "64",
            "--seed",
            "7",
            "--warmup-ratio",
            "0.1",
            "--weight-decay",
            "0.02",
            "--label-smoothing",
            "0.2",
            "--gradient-accumulation-steps",
            "8",
            "--subword-vocab-size",
            "12000",
        ]
    )

    hyperparameters = submit_job.build_hyperparameters(args)

    assert hyperparameters["corpus-version"] == "almg-v1"
    assert hyperparameters["direction"] == "es->cak"
    assert hyperparameters["run-id"] == "run-test"
    assert hyperparameters["epochs"] == 5
    assert hyperparameters["batch-size"] == 16
    assert hyperparameters["learning-rate"] == 0.0001
    assert hyperparameters["max-length"] == 64
    assert hyperparameters["seed"] == 7
    assert hyperparameters["warmup-ratio"] == 0.1
    assert hyperparameters["weight-decay"] == 0.02
    assert hyperparameters["label-smoothing"] == 0.2
    assert hyperparameters["gradient-accumulation-steps"] == 8
    assert hyperparameters["subword-vocab-size"] == 12000


def test_build_hyperparameters_includes_regularization_defaults():
    args = submit_job.parse_args([])

    hyperparameters = submit_job.build_hyperparameters(args)

    assert hyperparameters["warmup-ratio"] == 0.05
    assert hyperparameters["weight-decay"] == 0.01
    # 0.0 (disabled): >0 crashes against the real M2M100 checkpoint with
    # this transformers version -- see train.py's --label-smoothing help.
    assert hyperparameters["label-smoothing"] == 0.0
    assert hyperparameters["gradient-accumulation-steps"] == 4


def test_build_hyperparameters_includes_subword_vocab_size_default():
    # Matches training.subword_vocab.DEFAULT_VOCAB_SIZE (issue #82) --
    # every submitted job records this for traceability (ADR 0001) even
    # when not explicitly overridden.
    args = submit_job.parse_args([])

    hyperparameters = submit_job.build_hyperparameters(args)

    assert hyperparameters["subword-vocab-size"] == 8000


def test_build_hyperparameters_includes_embedding_init_strategy_default():
    # Issue #218: always recorded, unlike dropout/bpe-dropout-alpha below --
    # "mean" is a real, meaningful default value, not an "untouched" state
    # to omit, the same way subword-vocab-size above always is.
    args = submit_job.parse_args([])

    hyperparameters = submit_job.build_hyperparameters(args)

    assert hyperparameters["embedding-init-strategy"] == "mean"


def test_build_hyperparameters_includes_embedding_init_strategy_override():
    args = submit_job.parse_args(["--embedding-init-strategy", "compositional"])

    hyperparameters = submit_job.build_hyperparameters(args)

    assert hyperparameters["embedding-init-strategy"] == "compositional"


def test_build_hyperparameters_generates_run_id_when_not_given():
    args = submit_job.parse_args([])

    hyperparameters = submit_job.build_hyperparameters(args)

    assert hyperparameters["run-id"]  # non-empty, auto-generated


def test_build_hyperparameters_omits_init_model_by_default():
    args = submit_job.parse_args([])

    hyperparameters = submit_job.build_hyperparameters(args)

    assert "init-model" not in hyperparameters


def test_build_hyperparameters_omits_dropout_and_bpe_dropout_alpha_by_default():
    # Issue #182: both default to "untouched"/"disabled" -- landing this
    # ticket's plumbing must not itself change any submitted job's
    # hyperparameters unless explicitly opted into.
    args = submit_job.parse_args([])

    hyperparameters = submit_job.build_hyperparameters(args)

    assert "dropout" not in hyperparameters
    assert "bpe-dropout-alpha" not in hyperparameters


def test_build_hyperparameters_includes_dropout_and_bpe_dropout_alpha_when_given():
    args = submit_job.parse_args(
        ["--dropout", "0.3", "--bpe-dropout-alpha", "0.1"]
    )

    hyperparameters = submit_job.build_hyperparameters(args)

    assert hyperparameters["dropout"] == 0.3
    assert hyperparameters["bpe-dropout-alpha"] == 0.1


def test_build_hyperparameters_includes_init_model_container_path_when_given():
    args = submit_job.parse_args(
        [
            "--init-model-s3-uri",
            "s3://fake-bucket/model-artifacts/prior-run/output/model.tar.gz",
        ]
    )

    hyperparameters = submit_job.build_hyperparameters(args)

    assert hyperparameters["init-model"] == "/opt/ml/input/data/init-model/model.tar.gz"


def test_build_channel_uris_includes_init_model_channel_when_given():
    channels = submit_job.build_channel_uris(
        "fake-bucket", init_model_s3_uri="s3://other-bucket/prior/model.tar.gz"
    )

    assert channels["init-model"] == "s3://other-bucket/prior/model.tar.gz"


def test_build_channel_uris_omits_init_model_channel_by_default():
    channels = submit_job.build_channel_uris("fake-bucket")

    assert "init-model" not in channels


def test_build_training_inputs_includes_init_model_channel_when_given():
    inputs = submit_job.build_training_inputs(
        "fake-bucket", init_model_s3_uri="s3://other-bucket/prior/model.tar.gz"
    )

    assert {i.channel_name for i in inputs} == {"train", "validation", "init-model"}
    init_model_uri = next(i.data_source for i in inputs if i.channel_name == "init-model")
    assert init_model_uri == "s3://other-bucket/prior/model.tar.gz"


def test_build_job_config_has_a_cost_safety_cap_and_valid_image_versions():
    args = submit_job.parse_args(["--run-id", "run-test"])

    config = submit_job.build_job_config(
        bucket="fake-bucket",
        role="fake-role",
        args=args,
        source_dir="fake-bundle",
        training_image="fake-training-image",
    )

    assert config["role"] == "fake-role"
    assert config["instance_type"] == "ml.g4dn.xlarge"
    assert config["instance_count"] == 1
    assert config["max_run"] == 10800
    assert config["output_path"] == "s3://fake-bucket/model-artifacts/"
    # These three must agree with each other in the HuggingFace DLC
    # compatibility table -- see submit_job.py's module docstring for how
    # this exact combination was derived from the installed SDK.
    assert config["transformers_version"]
    assert config["pytorch_version"]
    assert config["py_version"]
    assert config["training_image"] == "fake-training-image"
    assert config["channels"] == {
        "train": "s3://fake-bucket/corpus/almg/v1/train.tsv",
        "validation": "s3://fake-bucket/corpus/almg/v1/val.tsv",
    }


def test_build_job_config_includes_a_checkpoint_s3_uri_separate_from_the_model_artifact():
    """Issue #187: per-epoch checkpoints must sync to their own S3
    location, distinct from `output_path` (where the final model.tar.gz
    artifact lands) -- so the registration-time artifact never bundles
    them again. Scoped per run-id so concurrent/successive runs' checkpoints
    never collide.
    """
    args = submit_job.parse_args(["--run-id", "run-test"])

    config = submit_job.build_job_config(
        bucket="fake-bucket",
        role="fake-role",
        args=args,
        source_dir="fake-bundle",
        training_image="fake-training-image",
    )

    assert config["checkpoint_s3_uri"] == "s3://fake-bucket/model-checkpoints/run-test/"
    assert config["checkpoint_s3_uri"] != config["output_path"]


def test_build_job_config_respects_custom_instance_type_and_max_run():
    args = submit_job.parse_args(["--instance-type", "ml.p3.2xlarge", "--max-run", "3600"])

    config = submit_job.build_job_config(
        bucket="b", role="r", args=args, source_dir="fake-bundle", training_image="fake-image"
    )

    assert config["instance_type"] == "ml.p3.2xlarge"
    assert config["max_run"] == 3600


def test_build_job_config_passes_through_source_dir_and_entry_point():
    args = submit_job.parse_args([])

    config = submit_job.build_job_config(
        bucket="b",
        role="r",
        args=args,
        source_dir="/tmp/fake-bundle-dir",
        training_image="fake-image",
    )

    assert config["source_dir"] == "/tmp/fake-bundle-dir"
    assert config["entry_point"] == "training/train.py"


# ---------------------------------------------------------------------------
# resolve_training_image_uri
# ---------------------------------------------------------------------------


def test_resolve_training_image_uri_uses_the_pinned_version_combination(monkeypatch):
    fake_retrieve = MagicMock(return_value="fake-training-image-uri")
    monkeypatch.setattr(submit_job, "_retrieve_image_uri", fake_retrieve)

    image_uri = submit_job.resolve_training_image_uri("us-east-1")

    assert image_uri == "fake-training-image-uri"
    _, kwargs = fake_retrieve.call_args
    assert kwargs["framework"] == "huggingface"
    assert kwargs["region"] == "us-east-1"
    assert kwargs["image_scope"] == "training"
    assert kwargs["version"] == submit_job.TRANSFORMERS_VERSION
    assert kwargs["base_framework_version"] == f"pytorch{submit_job.PYTORCH_VERSION}"


# ---------------------------------------------------------------------------
# build_estimator / build_training_inputs / submit_training_job
# ---------------------------------------------------------------------------


def test_build_estimator_constructs_model_trainer_with_expected_kwargs(monkeypatch):
    fake_model_trainer_cls = MagicMock()
    monkeypatch.setattr(submit_job, "ModelTrainer", fake_model_trainer_cls)

    args = submit_job.parse_args(["--run-id", "run-test"])
    config = submit_job.build_job_config(
        bucket="fake-bucket",
        role="fake-role",
        args=args,
        source_dir="/tmp/fake-bundle",
        training_image="fake-training-image",
    )

    estimator = submit_job.build_estimator(config)

    assert estimator is fake_model_trainer_cls.return_value
    _, kwargs = fake_model_trainer_cls.call_args
    assert kwargs["training_image"] == "fake-training-image"
    assert kwargs["role"] == "fake-role"
    assert kwargs["base_job_name"] == config["base_job_name"]
    assert kwargs["hyperparameters"]["corpus-version"] == "almg-v1"

    source_code = kwargs["source_code"]
    assert source_code.source_dir == "/tmp/fake-bundle"
    assert source_code.entry_script == "training/train.py"
    assert source_code.requirements == "requirements.txt"

    compute = kwargs["compute"]
    assert compute.instance_type == "ml.g4dn.xlarge"
    assert compute.instance_count == 1

    assert kwargs["stopping_condition"].max_runtime_in_seconds == 10800
    assert kwargs["output_data_config"].s3_output_path == "s3://fake-bucket/model-artifacts/"

    # Issue #187: checkpoints sync to their own, separate S3 location via
    # ModelTrainer's native checkpoint_config -- never bundled into
    # output_data_config's model.tar.gz artifact again.
    checkpoint_config = kwargs["checkpoint_config"]
    assert checkpoint_config.s3_uri == "s3://fake-bucket/model-checkpoints/run-test/"
    assert checkpoint_config.local_path == submit_job.DEFAULT_CHECKPOINT_DIR


def test_build_training_inputs_uses_train_and_validation_channel_names():
    inputs = submit_job.build_training_inputs("fake-bucket")

    assert {i.channel_name for i in inputs} == {"train", "validation"}
    by_name = {i.channel_name: i.data_source for i in inputs}
    assert by_name["train"] == "s3://fake-bucket/corpus/almg/v1/train.tsv"
    assert by_name["validation"] == "s3://fake-bucket/corpus/almg/v1/val.tsv"


def test_submit_training_job_calls_train_with_inputs_and_wait_logs_flags():
    fake_estimator = MagicMock()
    fake_inputs = [MagicMock(), MagicMock()]

    submit_job.submit_training_job(fake_estimator, fake_inputs, wait=False, logs=False)

    fake_estimator.train.assert_called_once_with(
        input_data_config=fake_inputs, wait=False, logs=False
    )


def test_submit_training_job_defaults_to_waiting_and_streaming_logs():
    fake_estimator = MagicMock()
    fake_inputs = [MagicMock(), MagicMock()]

    submit_job.submit_training_job(fake_estimator, fake_inputs)

    fake_estimator.train.assert_called_once_with(
        input_data_config=fake_inputs, wait=True, logs=True
    )


# ---------------------------------------------------------------------------
# Model card metrics extraction + Model Registry registration
# ---------------------------------------------------------------------------


def _make_model_tarball(model_card_text: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        data = model_card_text.encode("utf-8")
        info = tarfile.TarInfo(name="model_card.md")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


SAMPLE_MODEL_CARD = """# Model card: run-test

- **Timestamp**: 2026-09-19T00:00:00+00:00
- **Base model**: facebook/m2m100_418M
- **Direction**: both
- **Corpus version**: almg-v1
- **Training sentences**: 33,616
- **Validation sentences**: 3,735

## Hyperparameters

- **epochs**: 3

## Metrics (validation set)

- **BLEU**: 12.3
- **chrF**: 34.5
- **Sentences evaluated**: 3,735
"""


def test_fetch_model_card_from_artifact_extracts_text_from_tarball():
    tarball_bytes = _make_model_tarball(SAMPLE_MODEL_CARD)
    fake_s3 = MagicMock()

    def fake_download_file(bucket, key, filename):
        assert bucket == "fake-bucket"
        assert key == "model-artifacts/run-test/output/model.tar.gz"
        Path(filename).write_bytes(tarball_bytes)

    fake_s3.download_file.side_effect = fake_download_file

    text = submit_job.fetch_model_card_from_artifact(
        fake_s3, "s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz"
    )

    assert "BLEU" in text


def test_fetch_model_card_from_artifact_streams_to_disk_not_memory():
    """Regression test for issue #188: a real model.tar.gz artifact was
    measured at 30.5 GB (per-epoch checkpoints bundled in, see #187) --
    downloading it into an in-memory buffer just to read one small text
    file inside it caused a real out-of-memory kill. This asserts the S3
    client's disk-streaming API (`download_file`) is used instead of the
    in-memory one (`download_fileobj`), so peak memory stays roughly
    constant regardless of artifact size.
    """
    tarball_bytes = _make_model_tarball(SAMPLE_MODEL_CARD)
    fake_s3 = MagicMock()

    def fake_download_file(bucket, key, filename):
        Path(filename).write_bytes(tarball_bytes)

    fake_s3.download_file.side_effect = fake_download_file

    submit_job.fetch_model_card_from_artifact(
        fake_s3, "s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz"
    )

    fake_s3.download_file.assert_called_once()
    fake_s3.download_fileobj.assert_not_called()


def test_fetch_model_card_from_artifact_cleans_up_its_temp_file():
    tarball_bytes = _make_model_tarball(SAMPLE_MODEL_CARD)
    fake_s3 = MagicMock()
    captured_path: dict[str, str] = {}

    def fake_download_file(bucket, key, filename):
        captured_path["path"] = filename
        Path(filename).write_bytes(tarball_bytes)

    fake_s3.download_file.side_effect = fake_download_file

    submit_job.fetch_model_card_from_artifact(
        fake_s3, "s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz"
    )

    assert not Path(captured_path["path"]).exists()


def test_parse_model_card_metrics_extracts_bleu_and_chrf():
    metrics = submit_job.parse_model_card_metrics(SAMPLE_MODEL_CARD)

    assert metrics["bleu"] == "12.3"
    assert metrics["chrf"] == "34.5"


def test_ensure_model_package_group_creates_when_absent():
    fake_sm = MagicMock()
    fake_sm.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})

    submit_job.ensure_model_package_group(fake_sm, "traductor-kaqchikel", "desc")

    fake_sm.create_model_package_group.assert_called_once_with(
        ModelPackageGroupName="traductor-kaqchikel",
        ModelPackageGroupDescription="desc",
    )


def test_ensure_model_package_group_is_idempotent_when_already_exists():
    fake_sm = MagicMock()
    resource_in_use = type("ResourceInUse", (Exception,), {})
    fake_sm.exceptions.ResourceInUse = resource_in_use
    fake_sm.create_model_package_group.side_effect = resource_in_use("already exists")

    # Should not raise.
    submit_job.ensure_model_package_group(fake_sm, "traductor-kaqchikel", "desc")


def test_ensure_model_package_group_is_idempotent_on_the_real_validation_exception():
    # Reproduces the actual bug found in production (issue #76): the real
    # CreateModelPackageGroup API raises a generic ValidationException with
    # this message when the group already exists, not ResourceInUse.
    fake_sm = MagicMock()
    fake_sm.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm.create_model_package_group.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "ValidationException", "Message": "Model Package Group already exists"}},
        "CreateModelPackageGroup",
    )

    # Should not raise.
    submit_job.ensure_model_package_group(fake_sm, "traductor-kaqchikel", "desc")


def test_ensure_model_package_group_reraises_other_validation_exceptions():
    # A ValidationException for a genuinely different reason must not be
    # swallowed -- only "already exists" is safe to ignore.
    fake_sm = MagicMock()
    fake_sm.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm.create_model_package_group.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "ValidationException", "Message": "Some other real problem"}},
        "CreateModelPackageGroup",
    )

    with pytest.raises(botocore.exceptions.ClientError):
        submit_job.ensure_model_package_group(fake_sm, "traductor-kaqchikel", "desc")


def test_register_model_creates_model_package_with_metadata(monkeypatch):
    tarball_bytes = _make_model_tarball(SAMPLE_MODEL_CARD)
    fake_sm = MagicMock()
    fake_sm.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm.create_model_package.return_value = {
        "ModelPackageArn": "arn:aws:sagemaker:us-east-1:000000000000:model-package/foo/1"
    }
    fake_s3 = MagicMock()
    fake_s3.download_file.side_effect = lambda b, k, filename: Path(filename).write_bytes(tarball_bytes)

    run_metadata = {
        "corpus_version": "almg-v1",
        "direction": "both",
        "run_id": "run-test",
        "hyperparameters": {"epochs": 3, "batch_size": 8},
    }

    arn = submit_job.register_model(
        fake_sm,
        fake_s3,
        model_package_group_name="traductor-kaqchikel",
        model_data_url="s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz",
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:fake",
        run_metadata=run_metadata,
    )

    assert arn == "arn:aws:sagemaker:us-east-1:000000000000:model-package/foo/1"
    fake_sm.create_model_package_group.assert_called_once()
    _, kwargs = fake_sm.create_model_package.call_args
    assert kwargs["ModelPackageGroupName"] == "traductor-kaqchikel"
    assert kwargs["ModelApprovalStatus"] == "PendingManualApproval"
    assert kwargs["InferenceSpecification"]["Containers"][0]["ModelDataUrl"] == (
        "s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz"
    )
    metadata = kwargs["CustomerMetadataProperties"]
    assert metadata["corpus_version"] == "almg-v1"
    assert metadata["bleu"] == "12.3"
    assert metadata["chrf"] == "34.5"


def test_register_model_respects_approval_status_override(monkeypatch):
    tarball_bytes = _make_model_tarball(SAMPLE_MODEL_CARD)
    fake_sm = MagicMock()
    fake_sm.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm.create_model_package.return_value = {"ModelPackageArn": "arn:fake"}
    fake_s3 = MagicMock()
    fake_s3.download_file.side_effect = lambda b, k, filename: Path(filename).write_bytes(tarball_bytes)

    submit_job.register_model(
        fake_sm,
        fake_s3,
        model_package_group_name="traductor-kaqchikel",
        model_data_url="s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz",
        image_uri="fake-image",
        run_metadata={"corpus_version": "almg-v1", "direction": "both", "run_id": "run-test", "hyperparameters": {}},
        approval_status="Approved",
    )

    _, kwargs = fake_sm.create_model_package.call_args
    assert kwargs["ModelApprovalStatus"] == "Approved"


def test_main_records_dropout_and_bpe_dropout_alpha_in_registry_metadata_when_given(
    monkeypatch,
):
    """Regression test restored by ADR 0009 (originally added for issue
    #182's code review, PR #183): `build_hyperparameters` (tested directly
    above) already conditionally includes `dropout`/`bpe-dropout-alpha` in
    the *training job's own* hyperparameters, but `main()`'s separate
    `run_metadata["hyperparameters"]` reconstruction (used for
    `register_model`'s `CustomerMetadataProperties`) once hardcoded a fixed
    key list that predated both flags. A real run using either would train
    correctly but leave the registry entry with no record of which
    experiment produced it. This exercises `main()` itself (every other
    test in this module exercises `build_hyperparameters`/`register_model`
    individually) so the reconstruction step in between can't silently drop
    a hyperparameter again without a test noticing.
    """
    fake_cfn_client = MagicMock()
    fake_cfn_client.describe_stacks.return_value = {
        "Stacks": [
            {
                "Outputs": [
                    {"OutputKey": "TrainingDataBucketName", "OutputValue": "fake-bucket"},
                    {"OutputKey": "SageMakerExecutionRoleArn", "OutputValue": "fake-role-arn"},
                ]
            }
        ]
    }
    fake_sm_client = MagicMock()
    fake_sm_client.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm_client.create_model_package.return_value = {"ModelPackageArn": "arn:fake"}
    fake_s3_client = MagicMock()
    tarball_bytes = _make_model_tarball(SAMPLE_MODEL_CARD)
    fake_s3_client.download_file.side_effect = lambda b, k, filename: Path(filename).write_bytes(
        tarball_bytes
    )

    def fake_boto3_client(service, **kwargs):
        return {
            "cloudformation": fake_cfn_client,
            "sagemaker": fake_sm_client,
            "s3": fake_s3_client,
        }[service]

    monkeypatch.setattr(submit_job.boto3, "client", fake_boto3_client)
    monkeypatch.setattr(submit_job, "_retrieve_image_uri", lambda **kw: "fake-training-image")

    fake_estimator = MagicMock()
    fake_estimator.training_image = "fake-training-image"
    fake_estimator._latest_training_job.model_artifacts.s3_model_artifacts = (
        "s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz"
    )
    monkeypatch.setattr(submit_job, "ModelTrainer", MagicMock(return_value=fake_estimator))

    exit_code = submit_job.main(
        ["--run-id", "run-test", "--dropout", "0.3", "--bpe-dropout-alpha", "0.1"]
    )

    assert exit_code == 0
    _, register_kwargs = fake_sm_client.create_model_package.call_args
    metadata = register_kwargs["CustomerMetadataProperties"]
    assert metadata["hp_dropout"] == "0.3"
    assert metadata["hp_bpe_dropout_alpha"] == "0.1"


def test_main_records_embedding_init_strategy_in_registry_metadata(monkeypatch):
    """Issue #218: unlike dropout/bpe-dropout-alpha above, this is always
    recorded (mirroring subword-vocab-size) -- a real, meaningful default
    value exists ("mean"), so `main()`'s `run_metadata["hyperparameters"]`
    reconstruction must always include it, not just when overridden.
    """
    fake_cfn_client = MagicMock()
    fake_cfn_client.describe_stacks.return_value = {
        "Stacks": [
            {
                "Outputs": [
                    {"OutputKey": "TrainingDataBucketName", "OutputValue": "fake-bucket"},
                    {"OutputKey": "SageMakerExecutionRoleArn", "OutputValue": "fake-role-arn"},
                ]
            }
        ]
    }
    fake_sm_client = MagicMock()
    fake_sm_client.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm_client.create_model_package.return_value = {"ModelPackageArn": "arn:fake"}
    fake_s3_client = MagicMock()
    tarball_bytes = _make_model_tarball(SAMPLE_MODEL_CARD)
    fake_s3_client.download_file.side_effect = lambda b, k, filename: Path(filename).write_bytes(
        tarball_bytes
    )

    def fake_boto3_client(service, **kwargs):
        return {
            "cloudformation": fake_cfn_client,
            "sagemaker": fake_sm_client,
            "s3": fake_s3_client,
        }[service]

    monkeypatch.setattr(submit_job.boto3, "client", fake_boto3_client)
    monkeypatch.setattr(submit_job, "_retrieve_image_uri", lambda **kw: "fake-training-image")

    fake_estimator = MagicMock()
    fake_estimator.training_image = "fake-training-image"
    fake_estimator._latest_training_job.model_artifacts.s3_model_artifacts = (
        "s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz"
    )
    monkeypatch.setattr(submit_job, "ModelTrainer", MagicMock(return_value=fake_estimator))

    exit_code = submit_job.main(
        ["--run-id", "run-test", "--embedding-init-strategy", "compositional"]
    )

    assert exit_code == 0
    _, register_kwargs = fake_sm_client.create_model_package.call_args
    metadata = register_kwargs["CustomerMetadataProperties"]
    assert metadata["hp_embedding_init_strategy"] == "compositional"


# ---------------------------------------------------------------------------
# register_existing_job / _reconstruct_run_metadata_from_job_hyperparameters
# (--register-existing opt-in path)
# ---------------------------------------------------------------------------


def test_reconstruct_run_metadata_pulls_out_identifiers_and_keeps_the_rest_as_hyperparameters():
    metadata = submit_job._reconstruct_run_metadata_from_job_hyperparameters(
        {
            "corpus-version": "almg-v1",
            "direction": "both",
            "run-id": "prior-run",
            "epochs": "3",
            "batch-size": "8",
            # Plumbing fields that must never leak into "hyperparameters".
            "train": "/opt/ml/input/data/train/train.tsv",
            "validation": "/opt/ml/input/data/validation/val.tsv",
            "output-path": "s3://bucket/model-artifacts/",
            "training-image": "fake-image",
            "region": "us-east-1",
            "register-model": "true",
            "model-package-group-name": "traductor-kaqchikel",
            "approval-status": "PendingManualApproval",
        }
    )

    assert metadata["corpus_version"] == "almg-v1"
    assert metadata["direction"] == "both"
    assert metadata["run_id"] == "prior-run"
    assert metadata["hyperparameters"] == {"epochs": "3", "batch_size": "8"}


def test_reconstruct_run_metadata_excludes_base_model():
    """`base-model` is tracked as its own top-level `run_metadata["base_model"]`
    field by the self-registration path in `train.py`, never as a
    `hyperparameters` entry (see `run_training_job`'s own `hyperparameters`
    dict) -- excluded here too so `--register-existing`'s reconstructed
    metadata doesn't record a `hp_base_model` field the primary
    self-registration path never produces (code review on PR #191).
    """
    metadata = submit_job._reconstruct_run_metadata_from_job_hyperparameters(
        {
            "corpus-version": "almg-v1",
            "direction": "both",
            "run-id": "prior-run",
            "base-model": "facebook/m2m100_418M",
            "epochs": "3",
        }
    )

    assert "base_model" not in metadata["hyperparameters"]
    assert metadata["hyperparameters"] == {"epochs": "3"}


def test_reconstruct_run_metadata_defaults_missing_identifiers_to_unknown():
    metadata = submit_job._reconstruct_run_metadata_from_job_hyperparameters({})

    assert metadata["corpus_version"] == "unknown"
    assert metadata["direction"] == "unknown"
    assert metadata["run_id"] == "unknown"


def test_register_existing_job_looks_up_the_job_and_delegates_to_register_model():
    fake_sm = MagicMock()
    fake_sm.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm.describe_training_job.return_value = {
        "ModelArtifacts": {"S3ModelArtifacts": "s3://fake-bucket/model-artifacts/prior-run/output/model.tar.gz"},
        "AlgorithmSpecification": {"TrainingImage": "fake-training-image"},
        "HyperParameters": {"corpus-version": "almg-v1", "direction": "both", "run-id": "prior-run"},
    }
    fake_sm.create_model_package.return_value = {"ModelPackageArn": "arn:fake"}
    fake_s3 = MagicMock()
    tarball_bytes = _make_model_tarball(SAMPLE_MODEL_CARD)
    fake_s3.download_file.side_effect = lambda b, k, filename: Path(filename).write_bytes(tarball_bytes)

    arn = submit_job.register_existing_job(
        "prior-run",
        sm_client=fake_sm,
        s3_client=fake_s3,
        model_package_group_name="traductor-kaqchikel",
        approval_status="Approved",
    )

    assert arn == "arn:fake"
    fake_sm.describe_training_job.assert_called_once_with(TrainingJobName="prior-run")
    _, kwargs = fake_sm.create_model_package.call_args
    assert kwargs["ModelApprovalStatus"] == "Approved"
    assert kwargs["InferenceSpecification"]["Containers"][0]["Image"] == "fake-training-image"
    assert kwargs["InferenceSpecification"]["Containers"][0]["ModelDataUrl"] == (
        "s3://fake-bucket/model-artifacts/prior-run/output/model.tar.gz"
    )
    assert kwargs["CustomerMetadataProperties"]["run_id"] == "prior-run"
