"""Unit tests for `deployment.deploy` -- registers a training run's
artifact (repackaged with the custom inference handler) as a new,
deployable SageMaker Model Package version (issue #8). Every AWS/
`sagemaker` SDK touchpoint is mocked, mirroring
`tests/unit/test_submit_job.py`'s approach -- this suite never makes a
real AWS call or resolves a real container image.

`deploy.py`'s only `sagemaker` SDK dependency is `image_uris.retrieve`,
relocated (not renamed) from `sagemaker.image_uris` to
`sagemaker.core.image_uris` in SDK v3 (issue #155, GHSA-5r2p-pjr8-7fh7) --
`deploy._retrieve_image_uri` is mocked here exactly as before.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from deployment import deploy

SAMPLE_MODEL_CARD = """# Model card: run-20260922T141956Z

- **Timestamp**: 2026-09-22T17:03:33.557483+00:00
- **Base model**: facebook/m2m100_418M
- **Direction**: both
- **Corpus version**: almg-v1
- **Training sentences**: 32,906
- **Validation sentences**: 7,218

## Hyperparameters

- **epochs**: 5
- **subword_vocab_size**: 8000

## Metrics (validation set)

- **BLEU**: 8.9
- **chrF**: 31.2
- **Sentences evaluated**: 7,218

## Notes

Continued training from a previous checkpoint.
"""


# ---------------------------------------------------------------------------
# parse_model_card_run_metadata
# ---------------------------------------------------------------------------


def test_parse_model_card_run_metadata_extracts_run_id_from_the_title():
    metadata = deploy.parse_model_card_run_metadata(SAMPLE_MODEL_CARD)

    assert metadata["run_id"] == "run-20260922T141956Z"


def test_parse_model_card_run_metadata_extracts_corpus_version_and_direction():
    metadata = deploy.parse_model_card_run_metadata(SAMPLE_MODEL_CARD)

    assert metadata["corpus_version"] == "almg-v1"
    assert metadata["direction"] == "both"
    assert metadata["base_model"] == "facebook/m2m100_418M"


# ---------------------------------------------------------------------------
# build_inference_artifact_key
# ---------------------------------------------------------------------------


def test_build_inference_artifact_key_is_namespaced_by_run_id():
    key = deploy.build_inference_artifact_key("run-test")

    assert key == "inference-artifacts/run-test/model.tar.gz"


# ---------------------------------------------------------------------------
# resolve_inference_image_uri
# ---------------------------------------------------------------------------


def test_resolve_inference_image_uri_uses_the_pinned_version_combination(monkeypatch):
    fake_retrieve = MagicMock(return_value="fake-cpu-inference-image-uri")
    monkeypatch.setattr(deploy, "_retrieve_image_uri", fake_retrieve)

    image_uri = deploy.resolve_inference_image_uri("us-east-1")

    assert image_uri == "fake-cpu-inference-image-uri"
    _, kwargs = fake_retrieve.call_args
    assert kwargs["framework"] == "huggingface"
    assert kwargs["region"] == "us-east-1"
    assert kwargs["image_scope"] == "inference"
    assert kwargs["version"] == deploy.INFERENCE_TRANSFORMERS_VERSION
    assert kwargs["base_framework_version"] == f"pytorch{deploy.INFERENCE_PYTORCH_VERSION}"


def test_resolve_inference_image_uri_defaults_to_a_cpu_instance_type(monkeypatch):
    # Serverless Inference only supports CPU -- the instance_type passed to
    # `retrieve` here only steers which container *variant* is resolved,
    # it is never used to actually provision a real instance.
    fake_retrieve = MagicMock(return_value="fake-uri")
    monkeypatch.setattr(deploy, "_retrieve_image_uri", fake_retrieve)

    deploy.resolve_inference_image_uri("us-east-1")

    _, kwargs = fake_retrieve.call_args
    assert "gpu" not in kwargs["instance_type"]


# ---------------------------------------------------------------------------
# build_inference_customer_metadata / register_inference_model
# ---------------------------------------------------------------------------


def test_build_inference_customer_metadata_tags_artifact_type_and_source():
    metadata = deploy.build_inference_customer_metadata(
        {"run_id": "run-test", "corpus_version": "almg-v1"},
        {"bleu": "8.9", "chrf": "31.2"},
        source_model_data_url="s3://bucket/model-artifacts/run-test/output/model.tar.gz",
    )

    assert metadata["artifact_type"] == "inference"
    assert metadata["source_model_data_url"] == (
        "s3://bucket/model-artifacts/run-test/output/model.tar.gz"
    )
    assert metadata["run_id"] == "run-test"
    assert metadata["bleu"] == "8.9"
    assert metadata["chrf"] == "31.2"


def test_register_inference_model_creates_model_package_with_expected_fields():
    fake_sm = MagicMock()
    fake_sm.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm.create_model_package.return_value = {
        "ModelPackageArn": "arn:aws:sagemaker:us-east-1:000000000000:model-package/foo/3"
    }

    arn = deploy.register_inference_model(
        fake_sm,
        model_package_group_name="traductor-kaqchikel-es-cak",
        model_data_url="s3://bucket/inference-artifacts/run-test/model.tar.gz",
        image_uri="fake-inference-image",
        run_metadata={"run_id": "run-test", "corpus_version": "almg-v1", "direction": "both"},
        metrics={"bleu": "8.9", "chrf": "31.2"},
        source_model_data_url="s3://bucket/model-artifacts/run-test/output/model.tar.gz",
        approval_status="Approved",
    )

    assert arn == "arn:aws:sagemaker:us-east-1:000000000000:model-package/foo/3"
    fake_sm.create_model_package_group.assert_called_once()
    _, kwargs = fake_sm.create_model_package.call_args
    assert kwargs["ModelPackageGroupName"] == "traductor-kaqchikel-es-cak"
    assert kwargs["ModelApprovalStatus"] == "Approved"
    container = kwargs["InferenceSpecification"]["Containers"][0]
    assert container["Image"] == "fake-inference-image"
    assert container["ModelDataUrl"] == "s3://bucket/inference-artifacts/run-test/model.tar.gz"
    assert kwargs["CustomerMetadataProperties"]["artifact_type"] == "inference"
    assert kwargs["CustomerMetadataProperties"]["bleu"] == "8.9"


def test_register_inference_model_defaults_to_approved():
    # Unlike training.submit_job.register_model (defaults to
    # PendingManualApproval), this defaults to Approved: the project owner
    # explicitly authorized deploying this experimental model without the
    # BLEU>=10 gate (issue #8) -- see module docstring.
    fake_sm = MagicMock()
    fake_sm.exceptions.ResourceInUse = type("ResourceInUse", (Exception,), {})
    fake_sm.create_model_package.return_value = {"ModelPackageArn": "arn:fake"}

    deploy.register_inference_model(
        fake_sm,
        model_package_group_name="traductor-kaqchikel-es-cak",
        model_data_url="s3://bucket/inference-artifacts/run-test/model.tar.gz",
        image_uri="fake-image",
        run_metadata={"run_id": "run-test"},
        metrics={},
        source_model_data_url="s3://bucket/x",
    )

    _, kwargs = fake_sm.create_model_package.call_args
    assert kwargs["ModelApprovalStatus"] == "Approved"
