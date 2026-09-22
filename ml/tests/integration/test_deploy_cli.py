"""Fast smoke test of `deployment.deploy`'s full CLI wiring: argument
parsing -> CloudFormation output resolution -> inference image resolution
-> download -> repackage -> upload -> Model Registry registration.

Every AWS/`sagemaker` SDK touchpoint is mocked -- this never makes a real
AWS call, never downloads/uploads a real (multi-GB) model artifact, and
never spends any money. The maintainer runs the real, one-off deployment
separately (see `ml/README.md`'s "Serving" section) after this suite is
green.
"""

from __future__ import annotations

import io
import tarfile
from unittest.mock import MagicMock

from deployment import deploy

SAMPLE_MODEL_CARD = """# Model card: run-20260922T141956Z

- **Base model**: facebook/m2m100_418M
- **Direction**: both
- **Corpus version**: almg-v1

## Metrics (validation set)

- **BLEU**: 8.9
- **chrF**: 31.2
- **Sentences evaluated**: 7,218
"""


def _make_fake_model_tarball_bytes(model_card_text: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in [
            ("model_card.md", model_card_text),
            ("config.json", "{}"),
        ]:
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
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


def test_dry_run_resolves_config_without_downloading_or_registering(monkeypatch, capsys):
    fake_cfn_client = _fake_cfn_client()
    monkeypatch.setattr(deploy.boto3, "client", lambda service, **kw: fake_cfn_client)
    monkeypatch.setattr(deploy, "_retrieve_image_uri", lambda **kw: "fake-inference-image")

    exit_code = deploy.main(
        [
            "--dry-run",
            "--source-model-data-url",
            "s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "fake-bucket" in captured.out
    assert "fake-inference-image" in captured.out
    assert "model-artifacts/run-test" in captured.out


def test_full_run_downloads_repackages_uploads_and_registers(monkeypatch, tmp_path):
    fake_cfn_client = _fake_cfn_client()
    fake_sm_client = MagicMock()
    fake_sm_client.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm_client.create_model_package.return_value = {
        "ModelPackageArn": "arn:aws:sagemaker:us-east-1:000000000000:model-package/foo/3"
    }

    tarball_bytes = _make_fake_model_tarball_bytes(SAMPLE_MODEL_CARD)
    fake_s3_client = MagicMock()

    def fake_download_file(bucket, key, local_path):
        assert bucket == "fake-bucket"
        assert key == "model-artifacts/run-test/output/model.tar.gz"
        with open(local_path, "wb") as f:
            f.write(tarball_bytes)

    uploaded: dict[str, object] = {}

    def fake_upload_file(local_path, bucket, key):
        # Mirrors real S3's synchronous read-then-upload: the caller
        # cleans up its temp directory right after this call returns, so
        # the fake must capture the bytes *now*, not just the path.
        uploaded["bucket"] = bucket
        uploaded["key"] = key
        with open(local_path, "rb") as f:
            uploaded["bytes"] = f.read()

    fake_s3_client.download_file.side_effect = fake_download_file
    fake_s3_client.upload_file.side_effect = fake_upload_file

    def fake_boto3_client(service, **kwargs):
        return {
            "cloudformation": fake_cfn_client,
            "sagemaker": fake_sm_client,
            "s3": fake_s3_client,
        }[service]

    monkeypatch.setattr(deploy.boto3, "client", fake_boto3_client)
    monkeypatch.setattr(deploy, "_retrieve_image_uri", lambda **kw: "fake-inference-image")

    exit_code = deploy.main(
        [
            "--source-model-data-url",
            "s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz",
        ]
    )

    assert exit_code == 0
    assert uploaded["bucket"] == "fake-bucket"
    assert uploaded["key"] == "inference-artifacts/run-20260922T141956Z/model.tar.gz"

    # The uploaded artifact is the *repackaged* one: original files plus code/.
    with tarfile.open(fileobj=io.BytesIO(uploaded["bytes"]), mode="r:gz") as tar:
        names = set(tar.getnames())
    assert "config.json" in names
    assert "code/inference.py" in names
    assert "code/training/direction.py" in names

    fake_sm_client.create_model_package_group.assert_called_once()
    _, register_kwargs = fake_sm_client.create_model_package.call_args
    assert register_kwargs["ModelApprovalStatus"] == "Approved"
    metadata = register_kwargs["CustomerMetadataProperties"]
    assert metadata["bleu"] == "8.9"
    assert metadata["chrf"] == "31.2"
    assert metadata["run_id"] == "run-20260922T141956Z"
    assert metadata["artifact_type"] == "inference"
    assert metadata["source_model_data_url"] == (
        "s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz"
    )
    container = register_kwargs["InferenceSpecification"]["Containers"][0]
    assert container["Image"] == "fake-inference-image"
    assert container["ModelDataUrl"] == (
        "s3://fake-bucket/inference-artifacts/run-20260922T141956Z/model.tar.gz"
    )
