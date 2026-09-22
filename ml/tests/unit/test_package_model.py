"""Unit tests for `deployment.package_model` -- turning a training run's
raw `model.tar.gz` (model + tokenizer files only) into a self-contained,
SageMaker-deployable artifact with a top-level `code/` directory holding
the custom inference handler (issue #8). No AWS/S3 involved here -- these
operate on local tarballs/directories only, mirroring
`tests/unit/test_submit_job.py`'s `build_source_bundle` tests.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from deployment.package_model import build_inference_code_dir, repackage_model_artifact


def _make_fake_ml_root(tmp_path: Path) -> Path:
    ml_root = tmp_path / "ml"

    deployment_dir = ml_root / "deployment"
    deployment_dir.mkdir(parents=True)
    (deployment_dir / "inference.py").write_text("# inference entry point\n")
    (deployment_dir / "requirements.txt").write_text("sentencepiece>=0.2\n")

    training_dir = ml_root / "training"
    training_dir.mkdir()
    (training_dir / "__init__.py").write_text("")
    (training_dir / "direction.py").write_text("# direction tags\n")

    return ml_root


def _make_fake_model_tarball(tmp_path: Path) -> Path:
    """A tiny stand-in for a real training run's model.tar.gz: a couple of
    marker files at the root, no `code/` directory (that's what this
    module adds).
    """
    source_dir = tmp_path / "source_model"
    source_dir.mkdir()
    (source_dir / "config.json").write_text("{}")
    (source_dir / "model_card.md").write_text("# Model card: run-test\n")

    tar_path = tmp_path / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for member in source_dir.iterdir():
            tar.add(member, arcname=member.name)
    return tar_path


# ---------------------------------------------------------------------------
# build_inference_code_dir
# ---------------------------------------------------------------------------


def test_build_inference_code_dir_includes_the_inference_entry_point(tmp_path):
    ml_root = _make_fake_ml_root(tmp_path)

    code_dir = build_inference_code_dir(ml_root)

    assert (code_dir / "inference.py").read_text() == "# inference entry point\n"


def test_build_inference_code_dir_includes_requirements_txt(tmp_path):
    ml_root = _make_fake_ml_root(tmp_path)

    code_dir = build_inference_code_dir(ml_root)

    assert (code_dir / "requirements.txt").read_text() == "sentencepiece>=0.2\n"


def test_build_inference_code_dir_includes_the_training_direction_module(tmp_path):
    # deployment/inference.py does `from training.direction import ...`, so
    # the packaged code/ dir needs `training/` as a real sibling package,
    # exactly like submit_job.build_source_bundle bundles data/evaluation/
    # training together for the training entry point.
    ml_root = _make_fake_ml_root(tmp_path)

    code_dir = build_inference_code_dir(ml_root)

    assert (code_dir / "training" / "__init__.py").exists()
    assert (code_dir / "training" / "direction.py").read_text() == "# direction tags\n"


def test_build_inference_code_dir_is_named_code(tmp_path):
    ml_root = _make_fake_ml_root(tmp_path)

    code_dir = build_inference_code_dir(ml_root)

    assert code_dir.name == "code"


def test_build_inference_code_dir_returns_a_fresh_directory_each_call(tmp_path):
    ml_root = _make_fake_ml_root(tmp_path)

    first = build_inference_code_dir(ml_root)
    second = build_inference_code_dir(ml_root)

    assert first != second


# ---------------------------------------------------------------------------
# repackage_model_artifact
# ---------------------------------------------------------------------------


def test_repackage_model_artifact_preserves_original_model_files(tmp_path):
    source_tar = _make_fake_model_tarball(tmp_path)
    ml_root = _make_fake_ml_root(tmp_path)
    code_dir = build_inference_code_dir(ml_root)
    output_tar = tmp_path / "output" / "model.tar.gz"

    repackage_model_artifact(source_tar, code_dir, output_tar)

    with tarfile.open(output_tar, "r:gz") as tar:
        names = set(tar.getnames())
    assert "config.json" in names
    assert "model_card.md" in names


def test_repackage_model_artifact_adds_a_top_level_code_directory(tmp_path):
    source_tar = _make_fake_model_tarball(tmp_path)
    ml_root = _make_fake_ml_root(tmp_path)
    code_dir = build_inference_code_dir(ml_root)
    output_tar = tmp_path / "output" / "model.tar.gz"

    repackage_model_artifact(source_tar, code_dir, output_tar)

    with tarfile.open(output_tar, "r:gz") as tar:
        names = set(tar.getnames())
    assert "code/inference.py" in names
    assert "code/requirements.txt" in names
    assert "code/training/direction.py" in names


def test_repackage_model_artifact_creates_output_parent_directories(tmp_path):
    source_tar = _make_fake_model_tarball(tmp_path)
    ml_root = _make_fake_ml_root(tmp_path)
    code_dir = build_inference_code_dir(ml_root)
    output_tar = tmp_path / "deeply" / "nested" / "output" / "model.tar.gz"

    repackage_model_artifact(source_tar, code_dir, output_tar)

    assert output_tar.exists()
