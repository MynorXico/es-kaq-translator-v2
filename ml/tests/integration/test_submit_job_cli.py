"""Fast smoke test of `training.submit_job`'s full CLI wiring: argument
parsing -> CloudFormation output resolution -> job config -> estimator
construction -> submission -> Model Registry registration.

Every AWS/`sagemaker` SDK touchpoint is mocked (`boto3.client`,
`training.submit_job.HuggingFace`) -- this never makes a real AWS call, a
real training job, or spends any money, per issue #66's scope (this repo's
test suite only covers the submission *code*; the maintainer runs the real,
billable job separately after review).
"""

from __future__ import annotations

import io
import tarfile
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

    fake_huggingface_cls = MagicMock()
    monkeypatch.setattr(submit_job, "HuggingFace", fake_huggingface_cls)

    exit_code = submit_job.main(["--dry-run", "--run-id", "run-test", "--direction", "es->cak"])

    assert exit_code == 0
    fake_huggingface_cls.assert_not_called()
    captured = capsys.readouterr()
    assert "fake-bucket" in captured.out
    assert "fake-role-arn" in captured.out
    assert "corpus/almg/v1/train.tsv" in captured.out
    assert "es->cak" in captured.out


def test_full_submission_wires_fit_and_model_registry_registration(monkeypatch):
    fake_cfn_client = _fake_cfn_client()
    fake_sm_client = MagicMock()
    fake_sm_client.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm_client.create_model_package.return_value = {"ModelPackageArn": "arn:fake"}
    fake_s3_client = MagicMock()
    tarball_bytes = _make_model_tarball(SAMPLE_MODEL_CARD)
    fake_s3_client.download_fileobj.side_effect = lambda b, k, f: f.write(tarball_bytes)

    def fake_boto3_client(service, **kwargs):
        return {
            "cloudformation": fake_cfn_client,
            "sagemaker": fake_sm_client,
            "s3": fake_s3_client,
        }[service]

    monkeypatch.setattr(submit_job.boto3, "client", fake_boto3_client)

    fake_estimator = MagicMock()
    fake_estimator.model_data = "s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz"
    fake_estimator.image_uri = "fake-image"
    fake_huggingface_cls = MagicMock(return_value=fake_estimator)
    monkeypatch.setattr(submit_job, "HuggingFace", fake_huggingface_cls)

    exit_code = submit_job.main(["--run-id", "run-test"])

    assert exit_code == 0
    # Estimator was built with the resolved role/bucket, not placeholders.
    _, estimator_kwargs = fake_huggingface_cls.call_args
    assert estimator_kwargs["role"] == "fake-role-arn"
    assert estimator_kwargs["output_path"] == "s3://fake-bucket/model-artifacts/"

    fake_estimator.fit.assert_called_once()
    _, fit_kwargs = fake_estimator.fit.call_args
    assert set(fit_kwargs["inputs"].keys()) == {"train", "validation"}
    assert fit_kwargs["wait"] is True
    assert fit_kwargs["logs"] is True

    fake_sm_client.create_model_package_group.assert_called_once()
    _, register_kwargs = fake_sm_client.create_model_package.call_args
    assert register_kwargs["CustomerMetadataProperties"]["bleu"] == "12.3"
    assert register_kwargs["CustomerMetadataProperties"]["chrf"] == "34.5"


def test_no_wait_skips_registration(monkeypatch):
    fake_cfn_client = _fake_cfn_client()
    fake_sm_client = MagicMock()
    monkeypatch.setattr(
        submit_job.boto3,
        "client",
        lambda service, **kw: {"cloudformation": fake_cfn_client, "sagemaker": fake_sm_client}[service],
    )

    fake_estimator = MagicMock()
    fake_huggingface_cls = MagicMock(return_value=fake_estimator)
    monkeypatch.setattr(submit_job, "HuggingFace", fake_huggingface_cls)

    exit_code = submit_job.main(["--run-id", "run-test", "--no-wait"])

    assert exit_code == 0
    fake_estimator.fit.assert_called_once()
    fake_sm_client.create_model_package.assert_not_called()
