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
