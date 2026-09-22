"""Turn a training run's raw `model.tar.gz` artifact (the output of
`training.train.save_model_and_tokenizer`: model + tokenizer files only)
into a self-contained artifact the SageMaker Hugging Face Inference
Toolkit can serve with our custom direction-tag-aware handler (issue #8).

## Why repackage rather than pointing `source_dir`/`entry_point` at code
elsewhere

`sagemaker.huggingface.HuggingFaceModel`'s `entry_point`/`source_dir`
kwargs are designed for the `Estimator -> Model` flow: the SDK uploads
`source_dir` as a *separate* code tarball and points the container at it
via a `SAGEMAKER_SUBMIT_DIRECTORY` environment variable. SageMaker Model
Registry model packages (`InferenceSpecification.Containers[]`) have no
first-class field for a separate code location -- only `Image`,
`ModelDataUrl`, and `Environment` -- so a registry-based deployment (this
project's approach, for the traceability ADR 0001 requires) needs the
inference code bundled *inside* `ModelDataUrl`'s own tarball instead. The
SageMaker Hugging Face Inference Toolkit auto-detects this: if the
extracted model artifact has a top-level `code/inference.py`, it's loaded
as the custom handler with no extra environment variables needed. This
module builds exactly that layout.
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

CODE_SUBDIR_NAME = "code"


def build_inference_code_dir(ml_root: Path | None = None) -> Path:
    """Assemble a fresh temp directory named `code/` containing
    `inference.py`, `requirements.txt` (this file's own package, `ml/
    deployment/` by default), and the sibling `training` package
    `deployment/inference.py` imports from (`training.direction`) --
    mirrors `training.submit_job.build_source_bundle`'s exact reasoning:
    the entry point imports a sibling package, so that package must be a
    real subdirectory alongside it, not merged/flattened away.

    Caller owns cleanup (`shutil.rmtree(code_dir.parent)`) once the
    directory has been consumed by `repackage_model_artifact`.
    """
    ml_root = ml_root or Path(__file__).resolve().parent.parent
    bundle_root = Path(tempfile.mkdtemp(prefix="kaqchikel-inference-code-"))
    code_dir = bundle_root / CODE_SUBDIR_NAME
    code_dir.mkdir()

    shutil.copy2(ml_root / "deployment" / "inference.py", code_dir / "inference.py")
    shutil.copy2(ml_root / "deployment" / "requirements.txt", code_dir / "requirements.txt")

    training_dir = code_dir / "training"
    training_dir.mkdir()
    shutil.copy2(ml_root / "training" / "__init__.py", training_dir / "__init__.py")
    shutil.copy2(ml_root / "training" / "direction.py", training_dir / "direction.py")

    return code_dir


def repackage_model_artifact(source_tar_path: Path, code_dir: Path, output_tar_path: Path) -> None:
    """Extract `source_tar_path` (a training run's `model.tar.gz`), add
    `code_dir` as a top-level `code/` directory, and re-tar everything to
    `output_tar_path` (parent directories created as needed).

    `source_tar_path` is always this project's own training-job output
    (never arbitrary/untrusted input), so a plain `extractall()` is safe
    here.
    """
    extract_dir = Path(tempfile.mkdtemp(prefix="kaqchikel-inference-extract-"))
    try:
        with tarfile.open(source_tar_path, "r:gz") as tar:
            # `filter="data"` (PEP 706) is only available on Python 3.12+;
            # this project supports 3.11+, so it's used opportunistically
            # to silence the extraction-safety deprecation warning on newer
            # interpreters without breaking 3.11 with an unknown kwarg.
            # `source_tar_path` is always our own training-job output, so
            # this is a defense-in-depth hardening, not a real threat model.
            extractall_kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            tar.extractall(extract_dir, **extractall_kwargs)

        shutil.copytree(code_dir, extract_dir / CODE_SUBDIR_NAME)

        output_tar_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output_tar_path, "w:gz") as tar:
            for member_path in sorted(extract_dir.iterdir()):
                tar.add(member_path, arcname=member_path.name)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
