"""Reading/writing tab-separated Spanish-Kaqchikel sentence pairs.

Accepts either a local filesystem path or an `s3://bucket/key` URI so
callers (the pipeline CLI, notebooks, SageMaker jobs) can point at the
private ALMG corpus or the public community corpus -- both live in S3,
under distinct prefixes/buckets per ADR 0002 -- without this module ever
hardcoding a bucket name or baking in which corpus is "the" corpus.
"""

from __future__ import annotations

from collections.abc import Iterable

SentencePair = tuple[str, str]

_S3_PREFIX = "s3://"


def _get_s3_client():
    import boto3

    return boto3.client("s3")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    without_prefix = uri[len(_S3_PREFIX) :]
    bucket, _, key = without_prefix.partition("/")
    return bucket, key


def _read_text(path_or_uri: str) -> str:
    if path_or_uri.startswith(_S3_PREFIX):
        bucket, key = _parse_s3_uri(path_or_uri)
        client = _get_s3_client()
        response = client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read().decode("utf-8")

    with open(path_or_uri, encoding="utf-8") as f:
        return f.read()


def _write_text(text: str, path_or_uri: str) -> None:
    if path_or_uri.startswith(_S3_PREFIX):
        bucket, key = _parse_s3_uri(path_or_uri)
        client = _get_s3_client()
        client.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))
        return

    with open(path_or_uri, "w", encoding="utf-8") as f:
        f.write(text)


def read_tsv_pairs(path_or_uri: str) -> list[SentencePair]:
    """Read (source, target) pairs from a two-column TSV file or S3 object.

    Blank lines are skipped. A line that doesn't split into exactly two
    tab-separated fields raises `ValueError` -- silently dropping
    malformed rows would hide corpus alignment problems rather than
    surfacing them.
    """
    text = _read_text(path_or_uri)
    pairs: list[SentencePair] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(
                f"{path_or_uri}:{line_number}: expected 2 tab-separated fields, "
                f"got {len(fields)}"
            )
        es, cak = fields
        pairs.append((es, cak))
    return pairs


def write_tsv_pairs(pairs: Iterable[SentencePair], path_or_uri: str) -> None:
    """Write (source, target) pairs as a two-column TSV file or S3 object."""
    lines = [f"{es}\t{cak}" for es, cak in pairs]
    text = "\n".join(lines)
    if text:
        text += "\n"
    _write_text(text, path_or_uri)
