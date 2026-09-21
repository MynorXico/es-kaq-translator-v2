"""Unit tests for `training.train.resolve_model_source` -- pure filesystem/
tarfile logic, no torch/transformers import required (see
`test_train_args.py`'s docstring for why that matters for fast tests).
"""

import tarfile
from pathlib import Path

from training.train import resolve_model_source


def test_resolve_model_source_passes_through_a_hub_model_id():
    assert resolve_model_source("facebook/m2m100_418M") == "facebook/m2m100_418M"


def test_resolve_model_source_passes_through_a_local_directory(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    assert resolve_model_source(str(checkpoint_dir)) == str(checkpoint_dir)


def test_resolve_model_source_extracts_a_local_tar_gz(tmp_path):
    # A previous run's saved checkpoint, packaged the way SageMaker
    # packages a training job's model artifact.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "config.json").write_text('{"model_type": "m2m_100"}', encoding="utf-8")

    tarball_path = tmp_path / "model.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(source_dir / "config.json", arcname="config.json")

    resolved = resolve_model_source(str(tarball_path))

    assert resolved != str(tarball_path)
    resolved_path = Path(resolved)
    assert resolved_path.is_dir()
    assert (resolved_path / "config.json").read_text(encoding="utf-8") == '{"model_type": "m2m_100"}'


def test_resolve_model_source_extracts_to_a_fresh_directory_each_call(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "config.json").write_text("{}", encoding="utf-8")
    tarball_path = tmp_path / "model.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(source_dir / "config.json", arcname="config.json")

    first = resolve_model_source(str(tarball_path))
    second = resolve_model_source(str(tarball_path))

    assert first != second
