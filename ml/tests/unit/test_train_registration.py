"""Unit tests for `training.train`'s self-registration wiring (issue #190):
resolving this job's own SageMaker-assigned name, computing its future
model artifact S3 URI before it exists, and calling through to
`training.model_registry.register_model_package` -- with no client-side
download of the training artifact anywhere in this path.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from training import train

# ---------------------------------------------------------------------------
# resolve_training_job_name
# ---------------------------------------------------------------------------


def test_resolve_training_job_name_reads_job_name_from_sm_training_env():
    env = {"SM_TRAINING_ENV": json.dumps({"job_name": "traductor-kaqchikel-run-test-2026-09-28"})}

    job_name = train.resolve_training_job_name(env)

    assert job_name == "traductor-kaqchikel-run-test-2026-09-28"


def test_resolve_training_job_name_raises_a_clear_error_when_env_var_missing():
    with pytest.raises(KeyError):
        train.resolve_training_job_name({})


# ---------------------------------------------------------------------------
# compute_model_artifact_s3_uri
# ---------------------------------------------------------------------------


def test_compute_model_artifact_s3_uri_matches_sagemakers_own_upload_convention():
    uri = train.compute_model_artifact_s3_uri(
        "s3://fake-bucket/model-artifacts/", "traductor-kaqchikel-run-test"
    )

    assert uri == "s3://fake-bucket/model-artifacts/traductor-kaqchikel-run-test/output/model.tar.gz"


def test_compute_model_artifact_s3_uri_handles_output_path_without_trailing_slash():
    uri = train.compute_model_artifact_s3_uri(
        "s3://fake-bucket/model-artifacts", "traductor-kaqchikel-run-test"
    )

    assert uri == "s3://fake-bucket/model-artifacts/traductor-kaqchikel-run-test/output/model.tar.gz"


# ---------------------------------------------------------------------------
# register_model_from_training_job
# ---------------------------------------------------------------------------


def _base_args(**overrides):
    args = train.parse_args(
        [
            "--train",
            "train.tsv",
            "--validation",
            "val.tsv",
            "--corpus-version",
            "almg-v1",
        ]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_register_model_from_training_job_is_a_noop_when_register_model_is_false(tmp_path):
    model_card_path = tmp_path / "model_card.md"
    model_card_path.write_text("# Model card: run-test\n", encoding="utf-8")
    args = _base_args(register_model=False)

    result = train.register_model_from_training_job(
        model_card_path, {"corpus_version": "almg-v1", "direction": "both", "run_id": "run-test"}, args
    )

    assert result is None


def test_register_model_from_training_job_raises_when_output_path_missing(tmp_path):
    model_card_path = tmp_path / "model_card.md"
    model_card_path.write_text("# Model card: run-test\n", encoding="utf-8")
    args = _base_args(register_model=True, output_path=None, training_image="fake-image")

    with pytest.raises(ValueError):
        train.register_model_from_training_job(
            model_card_path,
            {"corpus_version": "almg-v1", "direction": "both", "run_id": "run-test"},
            args,
        )


def test_register_model_from_training_job_raises_when_training_image_missing(tmp_path):
    model_card_path = tmp_path / "model_card.md"
    model_card_path.write_text("# Model card: run-test\n", encoding="utf-8")
    args = _base_args(
        register_model=True, output_path="s3://fake-bucket/model-artifacts/", training_image=None
    )

    with pytest.raises(ValueError):
        train.register_model_from_training_job(
            model_card_path,
            {"corpus_version": "almg-v1", "direction": "both", "run_id": "run-test"},
            args,
        )


def test_register_model_from_training_job_raises_when_region_missing(tmp_path):
    """Regression test for issue #196: a real training job submitted with
    `--register-model true` failed with `NoRegionError: You must specify a
    region` -- `_boto3_client_for_registration` constructed a `boto3.client`
    with no explicit region, and the training container has no ambient
    default region resolvable the way a maintainer's own shell does. Region
    must be required the same way `--output-path`/`--training-image` are.
    """
    model_card_path = tmp_path / "model_card.md"
    model_card_path.write_text("# Model card: run-test\n", encoding="utf-8")
    args = _base_args(
        register_model=True,
        output_path="s3://fake-bucket/model-artifacts/",
        training_image="fake-image",
        region=None,
    )

    with pytest.raises(ValueError):
        train.register_model_from_training_job(
            model_card_path,
            {"corpus_version": "almg-v1", "direction": "both", "run_id": "run-test"},
            args,
        )


def test_register_model_from_training_job_never_downloads_the_artifact(tmp_path, monkeypatch):
    """The whole point of issue #190: reads `model_card_path` directly off
    local disk (the file `run_training_job` just wrote) -- no S3 client, no
    tarball extraction, anywhere in this path.
    """
    model_card_text = (
        "# Model card: run-test\n\n"
        "## Metrics (validation set)\n\n"
        "- **BLEU**: 12.3\n"
        "- **chrF**: 34.5\n"
    )
    model_card_path = tmp_path / "model_card.md"
    model_card_path.write_text(model_card_text, encoding="utf-8")

    monkeypatch.setenv(
        "SM_TRAINING_ENV", json.dumps({"job_name": "traductor-kaqchikel-run-test"})
    )

    captured = {}

    class FakeSmClient:
        pass

    def fake_boto3_client(service, region):
        assert service == "sagemaker"
        assert region == "us-east-1"
        return FakeSmClient()

    def fake_register_model_package(sm_client, **kwargs):
        captured["sm_client"] = sm_client
        captured.update(kwargs)
        return "arn:fake"

    monkeypatch.setattr(train, "_boto3_client_for_registration", fake_boto3_client)
    monkeypatch.setattr(train, "register_model_package", fake_register_model_package)

    args = _base_args(
        register_model=True,
        output_path="s3://fake-bucket/model-artifacts/",
        training_image="fake-training-image",
        region="us-east-1",
        model_package_group_name="traductor-kaqchikel-es-cak",
        approval_status="PendingManualApproval",
    )
    run_metadata = {
        "corpus_version": "almg-v1",
        "direction": "both",
        "run_id": "run-test",
        "hyperparameters": {"epochs": 3},
    }

    arn = train.register_model_from_training_job(model_card_path, run_metadata, args)

    assert arn == "arn:fake"
    assert isinstance(captured["sm_client"], FakeSmClient)
    assert captured["model_data_url"] == (
        "s3://fake-bucket/model-artifacts/traductor-kaqchikel-run-test/output/model.tar.gz"
    )
    assert captured["image_uri"] == "fake-training-image"
    assert captured["model_card_text"] == model_card_text
    assert captured["run_metadata"] == run_metadata
    assert captured["model_package_group_name"] == "traductor-kaqchikel-es-cak"
    assert captured["approval_status"] == "PendingManualApproval"


def test_register_model_from_training_job_defaults_group_name_and_approval_status(
    tmp_path, monkeypatch
):
    """When submit_job.py's own hyperparameters weren't given (e.g. a
    maintainer runs train.py directly with --register-model true), fall
    back to the same defaults training.model_registry itself defines.
    """
    model_card_path = tmp_path / "model_card.md"
    model_card_path.write_text("# Model card: run-test\n", encoding="utf-8")
    monkeypatch.setenv(
        "SM_TRAINING_ENV", json.dumps({"job_name": "traductor-kaqchikel-run-test"})
    )
    monkeypatch.setattr(train, "_boto3_client_for_registration", lambda service, region: object())

    captured = {}

    def fake_register_model_package(sm_client, **kwargs):
        captured.update(kwargs)
        return "arn:fake"

    monkeypatch.setattr(train, "register_model_package", fake_register_model_package)

    args = _base_args(
        register_model=True,
        output_path="s3://fake-bucket/model-artifacts/",
        training_image="fake-training-image",
        region="us-east-1",
        model_package_group_name=None,
        approval_status=None,
    )

    train.register_model_from_training_job(
        model_card_path,
        {"corpus_version": "almg-v1", "direction": "both", "run_id": "run-test"},
        args,
    )

    from training.model_registry import DEFAULT_APPROVAL_STATUS, DEFAULT_MODEL_PACKAGE_GROUP_NAME

    assert captured["model_package_group_name"] == DEFAULT_MODEL_PACKAGE_GROUP_NAME
    assert captured["approval_status"] == DEFAULT_APPROVAL_STATUS


# ---------------------------------------------------------------------------
# _boto3_client_for_registration
# ---------------------------------------------------------------------------


def test_boto3_client_for_registration_passes_an_explicit_region(monkeypatch):
    """Regression test for issue #196: constructing a `boto3.client` with no
    explicit region fails with `NoRegionError: You must specify a region`
    inside the training container, which has no ambient default region the
    way a maintainer's own configured shell does. `_boto3_client_for_
    registration` does a local `import boto3` (see its own docstring), so
    this patches the real `boto3` module's `client` attribute directly --
    that local import still binds to the same, now-patched module object.
    """
    fake_client = MagicMock()
    monkeypatch.setattr("boto3.client", fake_client)

    train._boto3_client_for_registration("sagemaker", "us-east-1")

    fake_client.assert_called_once_with("sagemaker", region_name="us-east-1")
