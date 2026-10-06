"""Fast smoke test of `deployment.promote_model`'s full CLI wiring: argument
parsing -> two independent per-account `boto3.Session`s -> DescribeModelPackage
on the source -> the hard-fail approval guard -> download -> upload into the
target environment's bucket -> Model Registry registration there, per ADR
0010 (`docs/adr/0010-cross-account-model-promotion.md`).

Every AWS touchpoint is mocked -- this never makes a real AWS call, never
authenticates against a real SSO profile, never downloads/uploads a real
model artifact, and never spends any money. The maintainer runs the real,
one-off promotion separately (see `docs/runbooks/model-promotion.md`)
after this suite is green.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from deployment import promote_model

SOURCE_MODEL_PACKAGE_ARN = "arn:aws:sagemaker:us-east-1:111111111111:model-package/traductor-kaqchikel-es-cak/19"


def _fake_source_description(approval_status: str = "Approved") -> dict:
    return {
        "ModelPackageArn": SOURCE_MODEL_PACKAGE_ARN,
        "ModelApprovalStatus": approval_status,
        "InferenceSpecification": {
            "Containers": [
                {
                    "Image": "fake-inference-image",
                    "ModelDataUrl": "s3://dev-bucket/inference-artifacts/run-test/model.tar.gz",
                }
            ]
        },
        "CustomerMetadataProperties": {
            "run_id": "run-test",
            "corpus_version": "almg-v1",
            "bleu": "18.3",
            "chrf": "40.7",
        },
    }


def _fake_target_cfn_client() -> MagicMock:
    client = MagicMock()
    client.describe_stacks.return_value = {
        "Stacks": [
            {
                "Outputs": [
                    {"OutputKey": "TrainingDataBucketName", "OutputValue": "qa-bucket"},
                    {"OutputKey": "SageMakerExecutionRoleArn", "OutputValue": "fake-role-arn"},
                ]
            }
        ]
    }
    return client


def _install_fake_sessions(monkeypatch, *, source_sm, source_s3, target_sm, target_s3, target_cfn):
    sessions_by_profile = {
        "translator-dev": _fake_session(sagemaker=source_sm, s3=source_s3),
        "translator-qa": _fake_session(sagemaker=target_sm, s3=target_s3, cloudformation=target_cfn),
    }

    def fake_session_factory(*, profile_name):
        return sessions_by_profile[profile_name]

    monkeypatch.setattr(promote_model.boto3, "Session", fake_session_factory)


def _fake_session(**clients_by_service) -> MagicMock:
    session = MagicMock()
    session.client.side_effect = lambda service, **kw: clients_by_service[service]
    return session


def test_hard_fails_on_a_non_approved_source_model_package_before_any_io(monkeypatch, capsys):
    source_sm = MagicMock()
    source_sm.describe_model_package.return_value = _fake_source_description(
        approval_status="PendingManualApproval"
    )
    source_s3 = MagicMock()
    target_sm = MagicMock()
    target_s3 = MagicMock()
    target_cfn = _fake_target_cfn_client()

    _install_fake_sessions(
        monkeypatch,
        source_sm=source_sm,
        source_s3=source_s3,
        target_sm=target_sm,
        target_s3=target_s3,
        target_cfn=target_cfn,
    )

    exit_code = promote_model.main(
        [
            "--source-environment",
            "dev",
            "--source-model-package-version",
            "19",
            "--target-environment",
            "qa",
        ]
    )

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "PendingManualApproval" in captured.err or "PendingManualApproval" in captured.out

    # Non-negotiable per the devops review on PR #236: a typo'd version
    # must not silently promote a non-Approved package -- so nothing past
    # the DescribeModelPackage call may have run.
    source_s3.download_file.assert_not_called()
    target_s3.upload_file.assert_not_called()
    target_sm.create_model_package.assert_not_called()
    target_cfn.describe_stacks.assert_not_called()


def test_full_promotion_downloads_from_source_and_registers_in_target(monkeypatch, tmp_path):
    source_sm = MagicMock()
    source_sm.describe_model_package.return_value = _fake_source_description()

    source_s3 = MagicMock()

    def fake_download_file(bucket, key, local_path):
        assert bucket == "dev-bucket"
        assert key == "inference-artifacts/run-test/model.tar.gz"
        with open(local_path, "wb") as f:
            f.write(b"fake model bytes")

    source_s3.download_file.side_effect = fake_download_file

    target_s3 = MagicMock()
    uploaded: dict[str, object] = {}

    def fake_upload_file(local_path, bucket, key):
        uploaded["bucket"] = bucket
        uploaded["key"] = key
        with open(local_path, "rb") as f:
            uploaded["bytes"] = f.read()

    target_s3.upload_file.side_effect = fake_upload_file

    target_sm = MagicMock()
    target_sm.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    target_sm.create_model_package.return_value = {
        "ModelPackageArn": "arn:aws:sagemaker:us-east-1:222222222222:model-package/traductor-kaqchikel-es-cak/2"
    }

    target_cfn = _fake_target_cfn_client()

    _install_fake_sessions(
        monkeypatch,
        source_sm=source_sm,
        source_s3=source_s3,
        target_sm=target_sm,
        target_s3=target_s3,
        target_cfn=target_cfn,
    )

    exit_code = promote_model.main(
        [
            "--source-environment",
            "dev",
            "--source-model-package-version",
            "19",
            "--target-environment",
            "qa",
        ]
    )

    assert exit_code == 0

    source_sm.describe_model_package.assert_called_once_with(
        ModelPackageName="traductor-kaqchikel-es-cak/19"
    )

    assert uploaded["bucket"] == "qa-bucket"
    assert uploaded["key"] == "promoted-artifacts/run-test/model.tar.gz"
    assert uploaded["bytes"] == b"fake model bytes"

    target_sm.create_model_package_group.assert_called_once()
    _, register_kwargs = target_sm.create_model_package.call_args
    assert register_kwargs["ModelPackageGroupName"] == "traductor-kaqchikel-es-cak"
    assert register_kwargs["ModelApprovalStatus"] == "Approved"
    container = register_kwargs["InferenceSpecification"]["Containers"][0]
    assert container["Image"] == "fake-inference-image"
    assert container["ModelDataUrl"] == "s3://qa-bucket/promoted-artifacts/run-test/model.tar.gz"

    metadata = register_kwargs["CustomerMetadataProperties"]
    assert metadata["run_id"] == "run-test"
    assert metadata["bleu"] == "18.3"
    assert metadata["promoted_from_environment"] == "dev"
    assert metadata["promoted_from_model_package_arn"] == SOURCE_MODEL_PACKAGE_ARN


def test_default_profiles_follow_the_translator_env_convention(monkeypatch):
    # No --source-profile/--target-profile given -- must resolve to
    # translator-dev/translator-qa (docs/runbooks/aws-account-bootstrap.md),
    # not require them to be passed explicitly every time.
    source_sm = MagicMock()
    source_sm.describe_model_package.return_value = _fake_source_description(
        approval_status="Rejected"
    )
    source_s3 = MagicMock()
    target_sm = MagicMock()
    target_s3 = MagicMock()
    target_cfn = _fake_target_cfn_client()

    seen_profiles = []

    def fake_session_factory(*, profile_name):
        seen_profiles.append(profile_name)
        if profile_name == "translator-dev":
            return _fake_session(sagemaker=source_sm, s3=source_s3)
        return _fake_session(sagemaker=target_sm, s3=target_s3, cloudformation=target_cfn)

    monkeypatch.setattr(promote_model.boto3, "Session", fake_session_factory)

    promote_model.main(
        [
            "--source-environment",
            "dev",
            "--source-model-package-version",
            "19",
            "--target-environment",
            "qa",
        ]
    )

    assert seen_profiles == ["translator-dev", "translator-qa"]
