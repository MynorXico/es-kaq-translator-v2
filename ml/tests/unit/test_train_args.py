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


def test_parse_args_defaults_checkpoint_dir_to_sagemakers_own_checkpoint_path():
    """Issue #187: per-epoch checkpoints must land in SageMaker's own,
    separate checkpoint-sync directory (`/opt/ml/checkpoints` -- confirmed
    against AWS's own "SageMaker AI environment variables and the default
    paths for training storage locations" reference: this is the *one* row
    in that table with no dedicated `SM_*` environment variable, so it must
    be a hardcoded literal, not read from the environment like
    `--model-dir`/`--output-data-dir` are), never inside `SM_MODEL_DIR`
    (`--model-dir`) -- that directory is what SageMaker tars into the
    registered `model.tar.gz` artifact, which must stay small (model +
    tokenizer + model_card.md only).
    """
    args = parse_args(
        [
            "--train",
            "train.tsv",
            "--validation",
            "val.tsv",
            "--corpus-version",
            "almg-v1",
        ]
    )

    assert args.checkpoint_dir == "/opt/ml/checkpoints"


def test_parse_args_reads_checkpoint_dir_override():
    args = parse_args(
        [
            "--train",
            "train.tsv",
            "--validation",
            "val.tsv",
            "--corpus-version",
            "almg-v1",
            "--checkpoint-dir",
            "/tmp/custom-checkpoints",
        ]
    )

    assert args.checkpoint_dir == "/tmp/custom-checkpoints"


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
