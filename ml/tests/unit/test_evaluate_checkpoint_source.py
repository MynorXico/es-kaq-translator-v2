"""Unit tests for `evaluation.evaluate_checkpoint.resolve_checkpoint_source`
-- pure filesystem/tarfile/S3-download logic, no torch/transformers import
required (mirrors `tests/unit/test_resolve_model_source.py`'s reasoning for
why that matters for fast tests).

`resolve_checkpoint_source` is this script's one piece of genuinely new
logic (everything else is thin wiring around already-tested functions, per
issue #108's scope): it adds S3-download support on top of
`training.train.resolve_model_source`, which only ever needed to handle a
local directory, a local `.tar.gz`, or a Hugging Face Hub model id --
never an `s3://` URI, since `--init-model` is always staged as a local
SageMaker channel path already.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from evaluation.evaluate_checkpoint import resolve_checkpoint_source


class FakeS3Client:
    """Duck-typed fake standing in for a `boto3` S3 client: only
    `download_file(bucket, key, filename)`, the one method
    `resolve_checkpoint_source` needs.
    """

    def __init__(self, objects: dict[tuple[str, str], bytes]):
        self._objects = objects
        self.download_calls: list[tuple[str, str, str]] = []

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.download_calls.append((bucket, key, filename))
        Path(filename).write_bytes(self._objects[(bucket, key)])


def _make_tar_gz_bytes(files: dict[str, str]) -> bytes:
    import io

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_resolve_checkpoint_source_passes_through_a_local_directory(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    resolved = resolve_checkpoint_source(str(checkpoint_dir))

    assert resolved == str(checkpoint_dir)


def test_resolve_checkpoint_source_extracts_a_local_tar_gz(tmp_path):
    tarball_path = tmp_path / "model.tar.gz"
    tarball_path.write_bytes(_make_tar_gz_bytes({"config.json": '{"model_type": "m2m_100"}'}))

    resolved = resolve_checkpoint_source(str(tarball_path))

    resolved_path = Path(resolved)
    assert resolved_path.is_dir()
    assert (resolved_path / "config.json").read_text(encoding="utf-8") == (
        '{"model_type": "m2m_100"}'
    )


def test_resolve_checkpoint_source_downloads_an_s3_uri_then_extracts_it(tmp_path):
    tar_bytes = _make_tar_gz_bytes({"config.json": '{"model_type": "m2m_100"}'})
    fake_s3 = FakeS3Client({("my-bucket", "model-artifacts/run-1/output/model.tar.gz"): tar_bytes})

    resolved = resolve_checkpoint_source(
        "s3://my-bucket/model-artifacts/run-1/output/model.tar.gz",
        s3_client=fake_s3,
    )

    assert fake_s3.download_calls == [
        (
            "my-bucket",
            "model-artifacts/run-1/output/model.tar.gz",
            fake_s3.download_calls[0][2],
        )
    ]
    resolved_path = Path(resolved)
    assert resolved_path.is_dir()
    assert (resolved_path / "config.json").read_text(encoding="utf-8") == (
        '{"model_type": "m2m_100"}'
    )


def test_resolve_checkpoint_source_extracts_to_a_fresh_directory_each_call(tmp_path):
    tar_bytes = _make_tar_gz_bytes({"config.json": "{}"})
    fake_s3 = FakeS3Client({("bucket", "key/model.tar.gz"): tar_bytes})

    first = resolve_checkpoint_source("s3://bucket/key/model.tar.gz", s3_client=fake_s3)
    second = resolve_checkpoint_source("s3://bucket/key/model.tar.gz", s3_client=fake_s3)

    assert first != second
