"""Unit tests for `training.model_registry` -- the SageMaker Model Registry
registration logic shared between `training.train` (self-registration from
inside the training container, issue #190) and `training.submit_job` (the
`--register-existing` opt-in path for an already-completed job). Every
`sagemaker` client call here is mocked; this suite never touches real AWS.

Factored out of `training.submit_job` specifically to avoid a circular
import: `submit_job.py` already imports from `training.train` (e.g.
`DEFAULT_BASE_MODEL`), so `training.train` can't import back from
`training.submit_job` -- this module has no dependency on either.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import botocore.exceptions
import pytest

from training import model_registry

SAMPLE_MODEL_CARD = """# Model card: run-test

- **Corpus version**: almg-v1

## Metrics (validation set)

- **BLEU**: 12.3
- **chrF**: 34.5
- **Sentences evaluated**: 3,735
"""


def test_ensure_model_package_group_creates_when_absent():
    fake_sm = MagicMock()
    fake_sm.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})

    model_registry.ensure_model_package_group(fake_sm, "traductor-kaqchikel", "desc")

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
    model_registry.ensure_model_package_group(fake_sm, "traductor-kaqchikel", "desc")


def test_ensure_model_package_group_is_idempotent_on_the_real_validation_exception():
    # Reproduces issue #76's real production bug: the real
    # CreateModelPackageGroup API raises a generic ValidationException with
    # this message when the group already exists, not ResourceInUse.
    fake_sm = MagicMock()
    fake_sm.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm.create_model_package_group.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "ValidationException", "Message": "Model Package Group already exists"}},
        "CreateModelPackageGroup",
    )

    # Should not raise.
    model_registry.ensure_model_package_group(fake_sm, "traductor-kaqchikel", "desc")


def test_ensure_model_package_group_reraises_other_validation_exceptions():
    fake_sm = MagicMock()
    fake_sm.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm.create_model_package_group.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "ValidationException", "Message": "Some other real problem"}},
        "CreateModelPackageGroup",
    )

    with pytest.raises(botocore.exceptions.ClientError):
        model_registry.ensure_model_package_group(fake_sm, "traductor-kaqchikel", "desc")


def test_build_customer_metadata_flattens_run_metadata_and_metrics():
    run_metadata = {
        "corpus_version": "almg-v1",
        "direction": "both",
        "run_id": "run-test",
        "hyperparameters": {"epochs": 3, "batch_size": 8},
    }

    metadata = model_registry.build_customer_metadata(run_metadata, {"bleu": "12.3", "chrf": "34.5"})

    assert metadata == {
        "corpus_version": "almg-v1",
        "direction": "both",
        "run_id": "run-test",
        "hp_epochs": "3",
        "hp_batch_size": "8",
        "bleu": "12.3",
        "chrf": "34.5",
    }


def test_register_model_package_creates_model_package_with_metadata_from_card_text():
    fake_sm = MagicMock()
    fake_sm.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm.create_model_package.return_value = {
        "ModelPackageArn": "arn:aws:sagemaker:us-east-1:000000000000:model-package/foo/1"
    }
    run_metadata = {
        "corpus_version": "almg-v1",
        "direction": "both",
        "run_id": "run-test",
        "hyperparameters": {"epochs": 3},
    }

    arn = model_registry.register_model_package(
        fake_sm,
        model_package_group_name="traductor-kaqchikel",
        model_data_url="s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz",
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:fake",
        model_card_text=SAMPLE_MODEL_CARD,
        run_metadata=run_metadata,
    )

    assert arn == "arn:aws:sagemaker:us-east-1:000000000000:model-package/foo/1"
    fake_sm.create_model_package_group.assert_called_once()
    _, kwargs = fake_sm.create_model_package.call_args
    assert kwargs["ModelPackageGroupName"] == "traductor-kaqchikel"
    assert kwargs["ModelApprovalStatus"] == model_registry.DEFAULT_APPROVAL_STATUS
    assert kwargs["InferenceSpecification"]["Containers"][0]["ModelDataUrl"] == (
        "s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz"
    )
    metadata = kwargs["CustomerMetadataProperties"]
    assert metadata["corpus_version"] == "almg-v1"
    assert metadata["bleu"] == "12.3"
    assert metadata["chrf"] == "34.5"


def test_register_model_package_respects_approval_status_override():
    fake_sm = MagicMock()
    fake_sm.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm.create_model_package.return_value = {"ModelPackageArn": "arn:fake"}

    model_registry.register_model_package(
        fake_sm,
        model_package_group_name="traductor-kaqchikel",
        model_data_url="s3://fake-bucket/model-artifacts/run-test/output/model.tar.gz",
        image_uri="fake-image",
        model_card_text=SAMPLE_MODEL_CARD,
        run_metadata={"corpus_version": "almg-v1", "direction": "both", "run_id": "run-test", "hyperparameters": {}},
        approval_status="Approved",
    )

    _, kwargs = fake_sm.create_model_package.call_args
    assert kwargs["ModelApprovalStatus"] == "Approved"


def test_register_model_package_never_downloads_anything():
    """The whole point of issue #190: `register_model_package` takes
    `model_card_text` as a plain string, never an S3 client/model data URL
    to fetch it from -- callers (train.py, submit_job.py) are responsible
    for supplying that text however is cheapest for their situation.
    """
    import inspect

    signature = inspect.signature(model_registry.register_model_package)
    assert "s3_client" not in signature.parameters
