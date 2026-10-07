"""Unit tests for `deployment.promote_model` -- duplicates an already-
Approved SageMaker Model Package from one environment's account into
another environment's own Model Registry, per ADR 0010
(`docs/adr/0010-cross-account-model-promotion.md`).

Every AWS touchpoint here is mocked (fake `sagemaker`/`s3` clients) --
this suite never makes a real AWS call, never constructs a real
`boto3.Session` against a real SSO profile, and never spends money or
touches a real account. See `tests/unit/test_deploy.py` for the same
mocking approach this mirrors.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deployment import promote_model

# ---------------------------------------------------------------------------
# build_profile_name
# ---------------------------------------------------------------------------


def test_build_profile_name_matches_the_translator_env_convention():
    # docs/runbooks/aws-account-bootstrap.md's profile naming convention.
    assert promote_model.build_profile_name("dev") == "translator-dev"
    assert promote_model.build_profile_name("qa") == "translator-qa"
    assert promote_model.build_profile_name("prod") == "translator-prod"


# ---------------------------------------------------------------------------
# resolve_source_model_package_name / describe_source_model_package
# ---------------------------------------------------------------------------


def test_resolve_source_model_package_name_is_group_slash_version():
    name = promote_model.resolve_source_model_package_name("traductor-kaqchikel-es-cak", 19)

    assert name == "traductor-kaqchikel-es-cak/19"


def test_describe_source_model_package_calls_describe_model_package():
    fake_sm = MagicMock()
    fake_sm.describe_model_package.return_value = {"ModelApprovalStatus": "Approved"}

    description = promote_model.describe_source_model_package(
        fake_sm, "traductor-kaqchikel-es-cak", 19
    )

    fake_sm.describe_model_package.assert_called_once_with(
        ModelPackageName="traductor-kaqchikel-es-cak/19"
    )
    assert description == {"ModelApprovalStatus": "Approved"}


# ---------------------------------------------------------------------------
# ensure_approved -- the hard-fail guard (devops review on PR #236)
# ---------------------------------------------------------------------------


def test_ensure_approved_passes_silently_for_an_approved_package():
    promote_model.ensure_approved(
        {"ModelApprovalStatus": "Approved", "ModelPackageArn": "arn:fake/19"},
        source_model_package_version=19,
    )


@pytest.mark.parametrize("status", ["PendingManualApproval", "Rejected", None])
def test_ensure_approved_raises_for_any_non_approved_status(status):
    with pytest.raises(promote_model.SourceModelPackageNotApprovedError) as excinfo:
        promote_model.ensure_approved(
            {"ModelApprovalStatus": status, "ModelPackageArn": "arn:fake/19"},
            source_model_package_version=19,
        )

    message = str(excinfo.value)
    assert str(status) in message or "None" in message
    assert "arn:fake/19" in message


# ---------------------------------------------------------------------------
# extract_source_artifact_info
# ---------------------------------------------------------------------------


def test_extract_source_artifact_info_reads_image_url_and_metadata():
    description = {
        "ModelPackageArn": "arn:aws:sagemaker:us-east-1:111111111111:model-package/foo/19",
        "InferenceSpecification": {
            "Containers": [
                {
                    "Image": "fake-inference-image",
                    "ModelDataUrl": "s3://dev-bucket/inference-artifacts/run-test/model.tar.gz",
                }
            ]
        },
        "CustomerMetadataProperties": {"run_id": "run-test", "bleu": "18.3"},
    }

    info = promote_model.extract_source_artifact_info(description)

    assert info["image_uri"] == "fake-inference-image"
    assert info["model_data_url"] == "s3://dev-bucket/inference-artifacts/run-test/model.tar.gz"
    assert info["customer_metadata"] == {"run_id": "run-test", "bleu": "18.3"}
    assert info["model_package_arn"] == (
        "arn:aws:sagemaker:us-east-1:111111111111:model-package/foo/19"
    )


def test_extract_source_artifact_info_defaults_missing_metadata_to_empty_dict():
    description = {
        "ModelPackageArn": "arn:fake/19",
        "InferenceSpecification": {
            "Containers": [{"Image": "fake-image", "ModelDataUrl": "s3://bucket/key"}]
        },
    }

    info = promote_model.extract_source_artifact_info(description)

    assert info["customer_metadata"] == {}


# ---------------------------------------------------------------------------
# build_promoted_artifact_key
# ---------------------------------------------------------------------------


def test_build_promoted_artifact_key_is_namespaced_by_run_id():
    key = promote_model.build_promoted_artifact_key("run-test")

    assert key == "promoted-artifacts/run-test/model.tar.gz"


# ---------------------------------------------------------------------------
# build_target_customer_metadata
# ---------------------------------------------------------------------------


def test_build_target_customer_metadata_carries_over_source_metadata():
    metadata = promote_model.build_target_customer_metadata(
        {"run_id": "run-test", "bleu": "18.3", "chrf": "40.7"},
        source_environment="dev",
        source_model_package_arn="arn:aws:sagemaker:us-east-1:111111111111:model-package/foo/19",
    )

    assert metadata["run_id"] == "run-test"
    assert metadata["bleu"] == "18.3"
    assert metadata["chrf"] == "40.7"


def test_build_target_customer_metadata_adds_promotion_provenance_fields():
    metadata = promote_model.build_target_customer_metadata(
        {"run_id": "run-test"},
        source_environment="dev",
        source_model_package_arn="arn:aws:sagemaker:us-east-1:111111111111:model-package/foo/19",
    )

    assert metadata["promoted_from_environment"] == "dev"
    assert metadata["promoted_from_model_package_arn"] == (
        "arn:aws:sagemaker:us-east-1:111111111111:model-package/foo/19"
    )


def test_build_target_customer_metadata_does_not_mutate_the_source_dict():
    source_metadata = {"run_id": "run-test"}

    promote_model.build_target_customer_metadata(
        source_metadata,
        source_environment="dev",
        source_model_package_arn="arn:fake/19",
    )

    assert source_metadata == {"run_id": "run-test"}


# ---------------------------------------------------------------------------
# register_promoted_model
# ---------------------------------------------------------------------------


def test_register_promoted_model_registers_approved_unconditionally():
    fake_sm = MagicMock()
    fake_sm.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm.create_model_package.return_value = {
        "ModelPackageArn": "arn:aws:sagemaker:us-east-1:222222222222:model-package/foo/2"
    }

    arn = promote_model.register_promoted_model(
        fake_sm,
        model_package_group_name="traductor-kaqchikel-es-cak",
        model_data_url="s3://qa-bucket/promoted-artifacts/run-test/model.tar.gz",
        image_uri="fake-inference-image",
        customer_metadata={
            "run_id": "run-test",
            "promoted_from_environment": "dev",
            "promoted_from_model_package_arn": "arn:fake/19",
        },
    )

    assert arn == "arn:aws:sagemaker:us-east-1:222222222222:model-package/foo/2"
    fake_sm.create_model_package_group.assert_called_once()
    _, kwargs = fake_sm.create_model_package.call_args
    assert kwargs["ModelPackageGroupName"] == "traductor-kaqchikel-es-cak"
    # Unconditionally Approved -- promotion is not a second quality review
    # (ADR 0010's Decision section).
    assert kwargs["ModelApprovalStatus"] == "Approved"
    container = kwargs["InferenceSpecification"]["Containers"][0]
    assert container["Image"] == "fake-inference-image"
    assert container["ModelDataUrl"] == "s3://qa-bucket/promoted-artifacts/run-test/model.tar.gz"
    assert kwargs["CustomerMetadataProperties"]["promoted_from_environment"] == "dev"


# ---------------------------------------------------------------------------
# download_source_artifact / upload_promoted_artifact (thin S3 wrappers)
# ---------------------------------------------------------------------------


def test_download_source_artifact_parses_bucket_and_key(tmp_path):
    fake_s3 = MagicMock()
    destination = tmp_path / "nested" / "model.tar.gz"

    promote_model.download_source_artifact(
        fake_s3, "s3://dev-bucket/inference-artifacts/run-test/model.tar.gz", destination
    )

    fake_s3.download_file.assert_called_once_with(
        "dev-bucket", "inference-artifacts/run-test/model.tar.gz", str(destination)
    )


def test_upload_promoted_artifact_returns_the_new_s3_uri(tmp_path):
    fake_s3 = MagicMock()
    local_file = tmp_path / "model.tar.gz"
    local_file.write_bytes(b"fake")

    uri = promote_model.upload_promoted_artifact(
        fake_s3, local_file, "qa-bucket", "promoted-artifacts/run-test/model.tar.gz"
    )

    assert uri == "s3://qa-bucket/promoted-artifacts/run-test/model.tar.gz"
    fake_s3.upload_file.assert_called_once_with(
        str(local_file), "qa-bucket", "promoted-artifacts/run-test/model.tar.gz"
    )
