"""Unit tests for `training.submit_job` -- the SageMaker Training Job
submission CLI for issue #66. Every AWS-touching call (CloudFormation,
SageMaker, S3, and the `sagemaker` SDK's `HuggingFace` estimator) is
mocked here -- this suite never makes a real AWS API call, never
constructs a real `boto3`/`sagemaker` session, and never spends money.
See `ml/README.md`'s "Job submission" section.
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


def test_build_hyperparameters_generates_run_id_when_not_given():
    args = submit_job.parse_args([])

    hyperparameters = submit_job.build_hyperparameters(args)

    assert hyperparameters["run-id"]  # non-empty, auto-generated


def test_build_hyperparameters_omits_init_model_by_default():
    args = submit_job.parse_args([])

    hyperparameters = submit_job.build_hyperparameters(args)

    assert "init-model" not in hyperparameters


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

    assert set(inputs.keys()) == {"train", "validation", "init-model"}
    init_model_uri = inputs["init-model"].config["DataSource"]["S3DataSource"]["S3Uri"]
    assert init_model_uri == "s3://other-bucket/prior/model.tar.gz"


def test_build_job_config_has_a_cost_safety_cap_and_valid_image_versions():
    args = submit_job.parse_args(["--run-id", "run-test"])

    config = submit_job.build_job_config(
        bucket="fake-bucket", role="fake-role", args=args, source_dir="fake-bundle"
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
    assert config["channels"] == {
        "train": "s3://fake-bucket/corpus/almg/v1/train.tsv",
        "validation": "s3://fake-bucket/corpus/almg/v1/val.tsv",
    }


def test_build_job_config_respects_custom_instance_type_and_max_run():
    args = submit_job.parse_args(["--instance-type", "ml.p3.2xlarge", "--max-run", "3600"])

    config = submit_job.build_job_config(bucket="b", role="r", args=args, source_dir="fake-bundle")

    assert config["instance_type"] == "ml.p3.2xlarge"
    assert config["max_run"] == 3600


def test_build_job_config_passes_through_source_dir_and_entry_point():
    args = submit_job.parse_args([])

    config = submit_job.build_job_config(
        bucket="b", role="r", args=args, source_dir="/tmp/fake-bundle-dir"
    )

    assert config["source_dir"] == "/tmp/fake-bundle-dir"
    assert config["entry_point"] == "training/train.py"


# ---------------------------------------------------------------------------
# build_estimator / build_training_inputs / submit_training_job
# ---------------------------------------------------------------------------


def test_build_estimator_constructs_huggingface_estimator_with_expected_kwargs(monkeypatch):
    fake_huggingface_cls = MagicMock()
    monkeypatch.setattr(submit_job, "HuggingFace", fake_huggingface_cls)

    args = submit_job.parse_args(["--run-id", "run-test"])
    config = submit_job.build_job_config(
        bucket="fake-bucket", role="fake-role", args=args, source_dir="/tmp/fake-bundle"
    )

    estimator = submit_job.build_estimator(config)

    assert estimator is fake_huggingface_cls.return_value
    _, kwargs = fake_huggingface_cls.call_args
    assert kwargs["entry_point"] == "training/train.py"
    assert kwargs["source_dir"] == "/tmp/fake-bundle"
    assert kwargs["role"] == "fake-role"
    assert kwargs["instance_type"] == "ml.g4dn.xlarge"
    assert kwargs["instance_count"] == 1
    assert kwargs["max_run"] == 10800
    assert kwargs["output_path"] == "s3://fake-bucket/model-artifacts/"
    assert kwargs["hyperparameters"]["corpus-version"] == "almg-v1"


def test_build_training_inputs_uses_train_and_validation_channel_names():
    inputs = submit_job.build_training_inputs("fake-bucket")

    assert set(inputs.keys()) == {"train", "validation"}
    train_uri = inputs["train"].config["DataSource"]["S3DataSource"]["S3Uri"]
    val_uri = inputs["validation"].config["DataSource"]["S3DataSource"]["S3Uri"]
    assert train_uri == "s3://fake-bucket/corpus/almg/v1/train.tsv"
    assert val_uri == "s3://fake-bucket/corpus/almg/v1/val.tsv"


def test_submit_training_job_calls_fit_with_inputs_and_wait_logs_flags():
    fake_estimator = MagicMock()
    fake_inputs = {"train": MagicMock(), "validation": MagicMock()}

    submit_job.submit_training_job(fake_estimator, fake_inputs, wait=False, logs=False)

    fake_estimator.fit.assert_called_once_with(inputs=fake_inputs, wait=False, logs=False)


def test_submit_training_job_defaults_to_waiting_and_streaming_logs():
    fake_estimator = MagicMock()
    fake_inputs = {"train": MagicMock(), "validation": MagicMock()}

    submit_job.submit_training_job(fake_estimator, fake_inputs)

    fake_estimator.fit.assert_called_once_with(inputs=fake_inputs, wait=True, logs=True)


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

    def fake_download_fileobj(bucket, key, fileobj):
        assert bucket == "fake-bucket"
        assert key == "model-artifacts/run-test/output/model.tar.gz"
        fileobj.write(tarball_bytes)

    fake_s3.download_fileobj.side_effect = fake_download_fileobj

    text = submit_job.fetch_model_card_from_artifact(
        fake_s3, "s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz"
    )

    assert "BLEU" in text


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
    fake_s3.download_fileobj.side_effect = lambda b, k, f: f.write(tarball_bytes)

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
    fake_s3.download_fileobj.side_effect = lambda b, k, f: f.write(tarball_bytes)

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
