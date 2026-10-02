"""Fast smoke test of `training.submit_job`'s full CLI wiring: argument
parsing -> CloudFormation output resolution -> job config -> estimator
construction -> submission -> Model Registry registration.

Every AWS/`sagemaker` SDK touchpoint is mocked (`boto3.client`,
`training.submit_job.ModelTrainer`, `training.submit_job._retrieve_image_uri`)
-- this never makes a real AWS call, a real training job, or spends any
money, per issue #66's scope (this repo's test suite only covers the
submission *code*; the maintainer runs the real, billable job separately
after review).

Migrated off `sagemaker.huggingface.HuggingFace` onto the v3 SDK's
`sagemaker.train.ModelTrainer` (issue #155, GHSA-5r2p-pjr8-7fh7) --
`ModelTrainer` is never constructed for real here (see
`tests/unit/test_submit_job.py`'s module docstring for why: its
constructor makes a live IAM call to validate the given role).
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

from training import submit_job

SAMPLE_MODEL_CARD = """# Model card: run-test

- **Corpus version**: almg-v1

## Metrics (validation set)

- **BLEU**: 12.3
- **chrF**: 34.5
- **Sentences evaluated**: 3,735
"""


def _make_model_tarball(model_card_text: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        data = model_card_text.encode("utf-8")
        info = tarfile.TarInfo(name="model_card.md")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _fake_cfn_client() -> MagicMock:
    client = MagicMock()
    client.describe_stacks.return_value = {
        "Stacks": [
            {
                "Outputs": [
                    {"OutputKey": "TrainingDataBucketName", "OutputValue": "fake-bucket"},
                    {"OutputKey": "SageMakerExecutionRoleArn", "OutputValue": "fake-role-arn"},
                ]
            }
        ]
    }
    return client


def test_dry_run_resolves_config_end_to_end_without_touching_fit(monkeypatch, capsys):
    fake_cfn_client = _fake_cfn_client()
    monkeypatch.setattr(submit_job.boto3, "client", lambda service, **kw: fake_cfn_client)

    fake_model_trainer_cls = MagicMock()
    monkeypatch.setattr(submit_job, "ModelTrainer", fake_model_trainer_cls)
    monkeypatch.setattr(submit_job, "_retrieve_image_uri", lambda **kw: "fake-training-image")

    exit_code = submit_job.main(["--dry-run", "--run-id", "run-test", "--direction", "es->cak"])

    assert exit_code == 0
    fake_model_trainer_cls.assert_not_called()
    captured = capsys.readouterr()
    assert "fake-bucket" in captured.out
    assert "fake-role-arn" in captured.out
    assert "corpus/almg/v1/train.tsv" in captured.out
    assert "es->cak" in captured.out
    assert "fake-training-image" in captured.out
    # Issue #187: the checkpoint sync destination is visible in the dry-run
    # preview too, distinct from the model-artifact output path.
    assert "model-checkpoints/run-test" in captured.out


def test_full_submission_wires_fit_and_model_registry_registration(monkeypatch):
    """ADR 0009: reverts issue #190's in-container self-registration, which
    was structurally broken (issue #201 -- `CreateModelPackage` validates
    S3 object existence at call time, before the training artifact is
    ever uploaded). `main()` itself registers a waited-for run's model
    client-side again, after `.train()` completes, using the disk-streamed
    (not in-memory, issues #188/#189) artifact download -- no
    self-registration hyperparameters are passed to the submitted job.
    """
    fake_cfn_client = _fake_cfn_client()
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
    fake_model_trainer_cls = MagicMock(return_value=fake_estimator)
    monkeypatch.setattr(submit_job, "ModelTrainer", fake_model_trainer_cls)

    exit_code = submit_job.main(["--run-id", "run-test", "--dropout", "0.3"])

    assert exit_code == 0
    # Estimator was built with the resolved role/bucket, not placeholders.
    _, estimator_kwargs = fake_model_trainer_cls.call_args
    assert estimator_kwargs["role"] == "fake-role-arn"
    assert estimator_kwargs["output_data_config"].s3_output_path == (
        "s3://fake-bucket/model-artifacts/"
    )

    fake_estimator.train.assert_called_once()
    _, train_kwargs = fake_estimator.train.call_args
    assert {i.channel_name for i in train_kwargs["input_data_config"]} == {"train", "validation"}
    assert train_kwargs["wait"] is True
    assert train_kwargs["logs"] is True

    # No self-registration hyperparameters are passed to the submitted job
    # any more -- ADR 0009 reverted issue #190's in-container path.
    hyperparameters = estimator_kwargs["hyperparameters"]
    assert "register-model" not in hyperparameters
    assert "output-path" not in hyperparameters
    assert "training-image" not in hyperparameters
    assert "region" not in hyperparameters
    assert "model-package-group-name" not in hyperparameters
    assert "approval-status" not in hyperparameters
    assert hyperparameters["dropout"] == 0.3

    # Client-side registration, restored: fetches the artifact (streamed to
    # disk) and registers it, same as the --register-existing path already
    # did.
    fake_sm_client.create_model_package_group.assert_called_once()
    fake_s3_client.download_file.assert_called_once()
    _, register_kwargs = fake_sm_client.create_model_package.call_args
    assert register_kwargs["CustomerMetadataProperties"]["bleu"] == "12.3"
    assert register_kwargs["CustomerMetadataProperties"]["chrf"] == "34.5"
    assert register_kwargs["CustomerMetadataProperties"]["hp_dropout"] == "0.3"
    assert register_kwargs["InferenceSpecification"]["Containers"][0]["Image"] == (
        "fake-training-image"
    )
    assert register_kwargs["InferenceSpecification"]["Containers"][0]["ModelDataUrl"] == (
        "s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz"
    )


def test_no_register_tells_main_not_to_register_client_side(monkeypatch):
    """ADR 0009: `--no-register` disables `main()`'s own client-side
    registration call after a waited-for run completes -- there is no
    submitted-job hyperparameter for this any more (issue #190 reverted).
    """
    fake_cfn_client = _fake_cfn_client()
    fake_sm_client = MagicMock()
    monkeypatch.setattr(
        submit_job.boto3,
        "client",
        lambda service, **kw: {"cloudformation": fake_cfn_client, "sagemaker": fake_sm_client}[service],
    )
    monkeypatch.setattr(submit_job, "_retrieve_image_uri", lambda **kw: "fake-training-image")

    fake_estimator = MagicMock()
    fake_model_trainer_cls = MagicMock(return_value=fake_estimator)
    monkeypatch.setattr(submit_job, "ModelTrainer", fake_model_trainer_cls)

    exit_code = submit_job.main(["--run-id", "run-test", "--no-register"])

    assert exit_code == 0
    fake_estimator.train.assert_called_once()
    _, estimator_kwargs = fake_model_trainer_cls.call_args
    assert "register-model" not in estimator_kwargs["hyperparameters"]
    fake_sm_client.create_model_package.assert_not_called()


def test_no_wait_also_skips_client_side_registration(monkeypatch):
    """ADR 0009: `--no-wait` skips `main()`'s own client-side registration
    too -- unlike issue #190's (reverted) in-container self-registration,
    there is nothing to register client-side until the job's model
    artifact actually exists, which only happens once the job finishes.
    """
    fake_cfn_client = _fake_cfn_client()
    fake_sm_client = MagicMock()
    monkeypatch.setattr(
        submit_job.boto3,
        "client",
        lambda service, **kw: {"cloudformation": fake_cfn_client, "sagemaker": fake_sm_client}[service],
    )
    monkeypatch.setattr(submit_job, "_retrieve_image_uri", lambda **kw: "fake-training-image")

    fake_estimator = MagicMock()
    fake_model_trainer_cls = MagicMock(return_value=fake_estimator)
    monkeypatch.setattr(submit_job, "ModelTrainer", fake_model_trainer_cls)

    exit_code = submit_job.main(["--run-id", "run-test", "--no-wait"])

    assert exit_code == 0
    fake_estimator.train.assert_called_once()
    _, train_kwargs = fake_estimator.train.call_args
    assert train_kwargs["wait"] is False
    fake_sm_client.create_model_package.assert_not_called()


# ---------------------------------------------------------------------------
# --register-existing (issue #190's explicit opt-in path)
# ---------------------------------------------------------------------------


def test_register_existing_looks_up_the_job_and_registers_without_submitting_anything(monkeypatch):
    fake_sm_client = MagicMock()
    fake_sm_client.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm_client.describe_training_job.return_value = {
        "ModelArtifacts": {
            "S3ModelArtifacts": "s3://fake-bucket/model-artifacts/prior-run/output/model.tar.gz"
        },
        "AlgorithmSpecification": {"TrainingImage": "fake-training-image"},
        "HyperParameters": {
            "corpus-version": "almg-v1",
            "direction": "both",
            "run-id": "prior-run",
            "epochs": "3",
        },
    }
    fake_sm_client.create_model_package.return_value = {"ModelPackageArn": "arn:fake"}
    fake_s3_client = MagicMock()
    tarball_bytes = _make_model_tarball(SAMPLE_MODEL_CARD)
    fake_s3_client.download_file.side_effect = lambda b, k, filename: Path(filename).write_bytes(
        tarball_bytes
    )

    def fake_boto3_client(service, **kwargs):
        return {"sagemaker": fake_sm_client, "s3": fake_s3_client}[service]

    monkeypatch.setattr(submit_job.boto3, "client", fake_boto3_client)
    fake_model_trainer_cls = MagicMock()
    monkeypatch.setattr(submit_job, "ModelTrainer", fake_model_trainer_cls)

    exit_code = submit_job.main(["--register-existing", "prior-run"])

    assert exit_code == 0
    fake_sm_client.describe_training_job.assert_called_once_with(TrainingJobName="prior-run")
    fake_model_trainer_cls.assert_not_called()  # never submits a new job
    _, register_kwargs = fake_sm_client.create_model_package.call_args
    assert register_kwargs["InferenceSpecification"]["Containers"][0]["ModelDataUrl"] == (
        "s3://fake-bucket/model-artifacts/prior-run/output/model.tar.gz"
    )
    assert register_kwargs["CustomerMetadataProperties"]["corpus_version"] == "almg-v1"
    assert register_kwargs["CustomerMetadataProperties"]["run_id"] == "prior-run"
    assert register_kwargs["CustomerMetadataProperties"]["hp_epochs"] == "3"
    assert register_kwargs["CustomerMetadataProperties"]["bleu"] == "12.3"
