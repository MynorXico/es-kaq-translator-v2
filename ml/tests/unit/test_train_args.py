"""Unit tests for `training.train`'s argument parsing -- pure logic, no
torch/transformers import required (verifies `parse_args` doesn't need a
heavy import to just read CLI args, which matters for fast unit testing).
"""

import pytest

from training.train import parse_args


def test_parse_args_reads_required_and_default_values(monkeypatch):
    monkeypatch.delenv("SM_MODEL_DIR", raising=False)
    monkeypatch.delenv("SM_OUTPUT_DATA_DIR", raising=False)

    args = parse_args(
        [
            "--train",
            "s3://private-bucket/train.tsv",
            "--validation",
            "s3://private-bucket/val.tsv",
            "--corpus-version",
            "almg-v1",
        ]
    )

    assert args.train == "s3://private-bucket/train.tsv"
    assert args.validation == "s3://private-bucket/val.tsv"
    assert args.corpus_version == "almg-v1"
    # Direction defaults to "both" -- a single multilingual, direction-tagged
    # checkpoint (see training/direction.py) rather than two checkpoints.
    assert args.direction == "both"
    assert args.base_model == "facebook/m2m100_418M"
    assert args.epochs >= 1
    assert args.run_id is None
    # Not resuming from a checkpoint by default (issue #75).
    assert args.init_model is None
    # Regularization/schedule defaults (issue #79).
    assert args.warmup_ratio == 0.05
    assert args.weight_decay == 0.01
    # 0.0 (disabled): >0 crashes against the real M2M100 checkpoint with
    # this transformers version -- see train.py's --label-smoothing help.
    assert args.label_smoothing == 0.0
    assert args.gradient_accumulation_steps == 4
    # Target size for the Kaqchikel-only SentencePiece/Unigram subword
    # vocabulary trained fresh each run (issue #82) -- matches
    # training.subword_vocab.DEFAULT_VOCAB_SIZE.
    assert args.subword_vocab_size == 8000
    # Issue #182: both new levers default to "off"/"untouched" so landing
    # this ticket doesn't itself change any past run's default behavior --
    # actually turning them on is a separate, maintainer-approved
    # experiment.
    assert args.dropout is None
    assert args.bpe_dropout_alpha is None
    # Issue #190: self-registration is opt-in and disabled by default --
    # every local/test invocation of this script that never wires these is
    # unaffected.
    assert args.output_path is None
    assert args.training_image is None
    assert args.register_model is False
    assert args.model_package_group_name is None
    assert args.approval_status is None


def test_parse_args_reads_self_registration_overrides():
    args = parse_args(
        [
            "--train",
            "train.tsv",
            "--validation",
            "val.tsv",
            "--corpus-version",
            "almg-v1",
            "--output-path",
            "s3://fake-bucket/model-artifacts/",
            "--training-image",
            "fake-training-image",
            "--register-model",
            "true",
            "--model-package-group-name",
            "custom-group",
            "--approval-status",
            "Approved",
        ]
    )

    assert args.output_path == "s3://fake-bucket/model-artifacts/"
    assert args.training_image == "fake-training-image"
    assert args.register_model is True
    assert args.model_package_group_name == "custom-group"
    assert args.approval_status == "Approved"


def test_parse_args_register_model_accepts_false_string():
    args = parse_args(
        [
            "--train",
            "train.tsv",
            "--validation",
            "val.tsv",
            "--corpus-version",
            "almg-v1",
            "--register-model",
            "false",
        ]
    )

    assert args.register_model is False


def test_parse_args_rejects_unrecognized_register_model_value():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--train",
                "train.tsv",
                "--validation",
                "val.tsv",
                "--corpus-version",
                "almg-v1",
                "--register-model",
                "maybe",
            ]
        )


def test_parse_args_reads_dropout_override():
    args = parse_args(
        [
            "--train",
            "train.tsv",
            "--validation",
            "val.tsv",
            "--corpus-version",
            "almg-v1",
            "--dropout",
            "0.3",
        ]
    )

    assert args.dropout == 0.3


def test_parse_args_reads_bpe_dropout_alpha_override():
    args = parse_args(
        [
            "--train",
            "train.tsv",
            "--validation",
            "val.tsv",
            "--corpus-version",
            "almg-v1",
            "--bpe-dropout-alpha",
            "0.1",
        ]
    )

    assert args.bpe_dropout_alpha == 0.1


def test_parse_args_reads_subword_vocab_size_override():
    args = parse_args(
        [
            "--train",
            "train.tsv",
            "--validation",
            "val.tsv",
            "--corpus-version",
            "almg-v1",
            "--subword-vocab-size",
            "12000",
        ]
    )

    assert args.subword_vocab_size == 12000


def test_parse_args_reads_init_model_path():
    args = parse_args(
        [
            "--train",
            "train.tsv",
            "--validation",
            "val.tsv",
            "--corpus-version",
            "almg-v1",
            "--init-model",
            "/opt/ml/input/data/init-model/model.tar.gz",
        ]
    )

    assert args.init_model == "/opt/ml/input/data/init-model/model.tar.gz"


def test_parse_args_defaults_model_dir_and_output_dir_from_sagemaker_env(monkeypatch):
    monkeypatch.setenv("SM_MODEL_DIR", "/opt/ml/model")
    monkeypatch.setenv("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

    args = parse_args(
        [
            "--train",
            "/opt/ml/input/data/train/train.tsv",
            "--validation",
            "/opt/ml/input/data/validation/val.tsv",
            "--corpus-version",
            "almg-v1",
        ]
    )

    assert args.model_dir == "/opt/ml/model"
    assert args.output_data_dir == "/opt/ml/output/data"


def test_parse_args_rejects_unknown_direction():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--train",
                "train.tsv",
                "--validation",
                "val.tsv",
                "--corpus-version",
                "almg-v1",
                "--direction",
                "en->fr",
            ]
        )


def test_parse_args_requires_corpus_version():
    with pytest.raises(SystemExit):
        parse_args(["--train", "train.tsv", "--validation", "val.tsv"])
