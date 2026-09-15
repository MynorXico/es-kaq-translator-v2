# ml

Machine learning pipeline for the Spanish<->Kaqchikel translation model.

- `data/` — corpus preprocessing/cleaning scripts. **Never commit raw
  corpus files here** — see [`docs/data-governance.md`](../docs/data-governance.md).
  Raw data lives in a private S3 bucket. Composable functions:
  - `data/normalize.py` — whitespace/Unicode (NFC) cleanup. Never
    lowercases (see module docstring for why).
  - `data/dedup.py` — exact-duplicate `(source, target)` pair removal.
  - `data/length_filter.py` — drops empty/too-short/too-long pairs and
    pairs with an outlier source/target length ratio (configurable
    thresholds via `LengthFilterConfig`).
  - `data/split_integrity.py` — validates a train/val split has no
    overlapping sentence pairs (raises `SplitIntegrityError` on exact-pair
    leakage; also reports weaker one-sided source/target overlap).
  - `data/corpus_io.py` — reads/writes two-column TSV sentence pairs from
    a local path or an `s3://` URI. Callers pass the URI; this module
    never hardcodes a bucket or distinguishes the private ALMG corpus
    from the public community corpus (see ADR 0002) — that separation is
    the caller's responsibility, by pointing at distinct S3
    buckets/prefixes for each.
  - `data/pipeline.py` — chains the above into `clean_corpus_file` and
    `validate_split_files`, plus a CLI: `uv run python -m data.pipeline
    clean <input> <output>` or `... validate-split <train> <val>`. Never
    run this against real corpus data in this repo/CI — only against S3
    paths from an authorized environment.
- `training/` — SageMaker training job entrypoints, tokenizer/vocabulary
  extension for Kaqchikel. Not yet scaffolded.
- `evaluation/` — BLEU/chrF evaluation harness and model card generation.
  Not yet scaffolded.

Tests: `uv sync && uv run pytest` (unit tests in `tests/unit/`, a fast
pipeline-wiring smoke test in `tests/integration/`, both against tiny
synthetic fixtures under `tests/fixtures/` — never the real corpus). Lint:
`uv run ruff check .`. See [`docs/testing.md`](../docs/testing.md) for the
`ml/` test pyramid.

See [`docs/adr/0001-initial-architecture.md`](../docs/adr/0001-initial-architecture.md)
for the modeling approach.
