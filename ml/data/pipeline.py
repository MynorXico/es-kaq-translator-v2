"""Pipeline wiring for corpus cleaning and split-integrity validation.

Composable building blocks (normalize, dedup, length-filter, split-integrity
live in their own modules and are independently unit-tested; this module
just chains them for real usage. Every path/URI is a parameter -- nothing
here hardcodes a bucket, prefix, or local path to real corpus data, and
this module never distinguishes "private" vs. "public" corpus internally:
that separation is enforced by the caller passing distinct source URIs
(see ADR 0002 / docs/data-governance.md), not by anything in this code.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable

from data.corpus_io import SentencePair, read_tsv_pairs, write_tsv_pairs
from data.dedup import deduplicate_pairs
from data.length_filter import LengthFilterConfig, filter_pairs_by_length
from data.normalize import normalize_pair
from data.split_integrity import SplitIntegrityReport, validate_split_integrity


def clean_pairs(
    pairs: Iterable[SentencePair], length_config: LengthFilterConfig | None = None
) -> list[SentencePair]:
    """Normalize, deduplicate, and length-filter a set of sentence pairs.

    Order matters: normalize first (so duplicates that only differ by
    whitespace/encoding are recognized as duplicates), then dedupe, then
    length-filter (so filtering thresholds apply to the normalized text).
    """
    normalized = [normalize_pair(es, cak) for es, cak in pairs]
    deduped = deduplicate_pairs(normalized)
    return filter_pairs_by_length(deduped, config=length_config)


def clean_corpus_file(
    input_path: str, output_path: str, length_config: LengthFilterConfig | None = None
) -> list[SentencePair]:
    """Read a raw TSV corpus, clean it, and write the result to `output_path`."""
    raw_pairs = read_tsv_pairs(input_path)
    cleaned = clean_pairs(raw_pairs, length_config=length_config)
    write_tsv_pairs(cleaned, output_path)
    return cleaned


def validate_split_files(
    train_path: str, val_path: str, raise_on_overlap: bool = True
) -> SplitIntegrityReport:
    """Read a train/val TSV pair and validate there's no leakage between them."""
    train_pairs = read_tsv_pairs(train_path)
    val_pairs = read_tsv_pairs(val_path)
    return validate_split_integrity(train_pairs, val_pairs, raise_on_overlap=raise_on_overlap)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml.data.pipeline",
        description="Clean a Spanish-Kaqchikel corpus file, or validate a train/val split.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean_parser = subparsers.add_parser(
        "clean", help="Normalize, dedupe, and length-filter a raw corpus TSV."
    )
    clean_parser.add_argument("input", help="Local path or s3:// URI of the raw TSV corpus.")
    clean_parser.add_argument("output", help="Local path or s3:// URI to write the cleaned TSV.")
    clean_parser.add_argument("--min-length", type=int, default=LengthFilterConfig().min_length)
    clean_parser.add_argument("--max-length", type=int, default=LengthFilterConfig().max_length)
    clean_parser.add_argument("--max-ratio", type=float, default=LengthFilterConfig().max_ratio)

    validate_parser = subparsers.add_parser(
        "validate-split", help="Check a train/val TSV pair for overlapping sentence pairs."
    )
    validate_parser.add_argument("train", help="Local path or s3:// URI of the train TSV.")
    validate_parser.add_argument("val", help="Local path or s3:// URI of the val TSV.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "clean":
        config = LengthFilterConfig(
            min_length=args.min_length, max_length=args.max_length, max_ratio=args.max_ratio
        )
        cleaned = clean_corpus_file(args.input, args.output, length_config=config)
        print(f"Wrote {len(cleaned)} cleaned pairs to {args.output}")
        return 0

    if args.command == "validate-split":
        report = validate_split_files(args.train, args.val, raise_on_overlap=False)
        if report.has_exact_overlap:
            print(
                f"FAIL: {len(report.exact_pair_overlap)} exact pair(s) leaked "
                "between train and val.",
                file=sys.stderr,
            )
            return 1
        print("OK: no exact-pair overlap between train and val.")
        if report.source_overlap or report.target_overlap:
            print(
                f"Warning: {len(report.source_overlap)} source-side and "
                f"{len(report.target_overlap)} target-side one-sided overlaps found."
            )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
