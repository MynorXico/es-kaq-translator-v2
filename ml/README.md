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
- `training/` — SageMaker training job entrypoint (`training/train.py`,
  see below) and tokenizer/vocabulary extension for Kaqchikel (see below).
- `evaluation/` — BLEU/chrF evaluation harness and model card generation
  (see below).

This directory manages its own Python environment with
[uv](https://docs.astral.sh/uv/):

```sh
cd ml
uv sync
```

## Commands

```sh
uv run pytest          # run tests (unit/ + integration/)
uv run ruff check .    # lint
```

Tests are split `tests/unit/` (pure functions, no I/O, e.g. metric/model
card/data-cleaning logic) and `tests/integration/` (fast smoke runs of
pipeline wiring against tiny fixture data under `tests/fixtures/`). See
[`docs/testing.md`](../docs/testing.md) for the full pyramid and TDD
process. **No script here ever reads the real private corpus or
validation set directly in a test** — those live in a private S3 bucket
(see [`docs/data-governance.md`](../docs/data-governance.md)) and are only
touched by real training/evaluation runs, not by the automated test
suite.

## Evaluation harness (`evaluation/`)

- `evaluation/metrics.py` — `compute_metrics(hypotheses, references)`
  wraps [`sacrebleu`](https://github.com/mjpost/sacrebleu) to compute
  corpus-level BLEU and chrF; returns a `MetricsResult(bleu, chrf,
  num_sentences)`.
- `evaluation/model_card.py` — `render_model_card(ModelCardData)` renders
  a Markdown model card from a training run's metadata (run id,
  timestamp, base model, translation direction, corpus version
  identifier, sentence counts, hyperparameters, and metrics). Deliberately
  records only identifiers/aggregate counts, never raw corpus content —
  the private ALMG corpus must never leak into what is effectively a
  public-facing artifact (ADR 0002).
- `evaluation/run.py` — `run_evaluation(predictions_path, references_path,
  run_metadata, output_path)` ties the two together: reads a predictions
  file and a references file (one sentence per line, aligned by line
  order), computes metrics, and writes a Markdown model card to
  `output_path`. Every path is caller-supplied — nothing here hardcodes a
  location for the real corpus.
- Per ADR 0001, every training run should be traceable: register the
  generated model card (corpus version + hyperparameters + metrics)
  alongside the model in SageMaker Model Registry rather than producing
  an untraceable artifact.

BLEU/chrF are quality metrics, not pass/fail tests — see
`docs/testing.md`. The unit tests for `evaluation/metrics.py` sanity-check
that the wrapper calls `sacrebleu` correctly (identical hypothesis/
reference scores near 100, an unrelated hypothesis scores low), not that
the model itself translates well.

See [`docs/adr/0001-initial-architecture.md`](../docs/adr/0001-initial-architecture.md)
for the modeling approach and
[`docs/adr/0003-base-model-license-verification.md`](../docs/adr/0003-base-model-license-verification.md)
for the base model choice (M2M100).

## Tokenizer/vocabulary extension for Kaqchikel (`training/`)

M2M100's pretrained vocabulary was not trained on Kaqchikel, so it needs to
be extended before fine-tuning (per ADR 0003's consequences). This is kept
as pure, unit-testable logic, separate from actually running a training job:

- `training/vocab_gap.py` — `find_missing_characters` / `find_missing_words`
  identify which characters/word-forms in sample Kaqchikel text are not
  already representable by a base tokenizer's vocabulary. Kaqchikel shares
  the Latin alphabet with Spanish, so the real gap is narrower than "the
  whole alphabet" — mainly a handful of extra vowels (e.g. "ä") and the
  plain apostrophe marking glottalized consonants (k', tz', ch', q').
- `training/vocab_extension.py` — `select_new_tokens` / `extend_vocab`
  extend a base vocab dict with new tokens, deterministically and without
  ever duplicating an existing entry. `resize_embedding_matrix` returns a
  resized `(vocab_size + N, dim)` NumPy array given a base embedding matrix
  and a count of new tokens, initializing new rows at the mean of the
  existing rows plus a small amount of noise (a warm start, rather than
  zeros/random, so new tokens start in-distribution but can still diverge
  from each other during fine-tuning).
- `training/tokenizer_extension.py` — the thin integration layer wiring the
  above to a real `transformers.M2M100Tokenizer`/model.
  `extend_tokenizer_vocab`, which only needs `get_vocab()`/`add_tokens()`,
  is unit-tested against a fake tokenizer that duck-types those two
  methods. `resize_embeddings_for_new_tokens` is exercised against the real
  `facebook/m2m100_418M` checkpoint in
  `tests/integration/test_tokenizer_extension_real_model.py`.

All fixtures here are a handful of hand-written Kaqchikel sentences
(`tests/fixtures/sample_kaqchikel_text.txt`) — never the real private ALMG
corpus (ADR 0002).

### Real-checkpoint integration test

`tests/integration/test_tokenizer_extension_real_model.py` downloads and
loads the actual `facebook/m2m100_418M` tokenizer + model (ADR 0003) — the
only test in this repo that touches a real pretrained checkpoint. It
requires `torch`, `transformers`, and `sentencepiece` (real `ml/`
dependencies, unlike the rest of this pipeline's data/logic code, which
stays framework-free). First run downloads ~1.9GB from the Hugging Face
Hub to `~/.cache/huggingface` (`$HF_HOME` if set); CI caches that directory
across runs (see `.github/workflows/ci.yml`), keyed on this test file's
contents so a model/config change busts the cache.

This test caught a real bug during development: a real M2M100 checkpoint's
embedding matrix is pre-padded a few rows beyond its tokenizer's raw vocab
size (128112 rows for a 128104-token `facebook/m2m100_418M` vocab).
`resize_embeddings_for_new_tokens` originally inferred how many new rows
to add from `len(tokenizer) - model_embedding_row_count`, which silently
no-op'd whenever a small vocab extension fit inside that existing padding
— leaving new token ids pointing at untrained padding rows instead of
warm-started ones. It now takes the actual new-token count directly
(`len(extend_tokenizer_vocab(...))`), sidestepping the padding mismatch
entirely. This is exactly the class of bug the duck-typed fake in the unit
tests can't catch, since a fake tokenizer/embedding pair has no reason to
reproduce a real checkpoint's padding quirks.

## Training entrypoint (`training/train.py`)

`training/train.py` is the SageMaker Training Job entrypoint that wires
everything above (`data/`, `training/tokenizer_extension.py`,
`evaluation/`) into an actual fine-tuning run:

1. Reads the training/validation corpora via `data.corpus_io.read_tsv_pairs`
   from paths/S3 URIs passed in as `--train`/`--validation` — never a
   hardcoded bucket. In a real SageMaker Training Job, point these at the
   container's channel paths (e.g.
   `/opt/ml/input/data/train/train.tsv`), populated from whichever S3
   prefix the job configures; keeping the private ALMG corpus and public
   community corpus physically separate on S3 (ADR 0002) is the caller's
   responsibility, not this script's.
2. Builds direction-tagged training/eval examples
   (`training/direction.py`) — see "Direction handling" below.
3. Loads `facebook/m2m100_418M` (ADR 0003), extends its vocabulary and
   resizes its embeddings for Kaqchikel plus the direction tag tokens,
   using `training/tokenizer_extension.py` (#34/#62) against the actual
   training corpus text.
4. Fine-tunes via `transformers.Seq2SeqTrainer`.
5. Saves the fine-tuned model + tokenizer to `SM_MODEL_DIR`
   (`--model-dir`) — this artifact is **never published**; it's only
   uploaded by SageMaker as a private training-job artifact and served
   through the project's own hosted API (ADR 0002, CLAUDE.md).
6. Generates validation-set translations, computes BLEU/chrF, and writes
   a model card (`evaluation/run.py`) to `SM_MODEL_DIR/model_card.md`,
   alongside the saved model artifact, for registration in SageMaker
   Model Registry per ADR 0001's traceability requirement.

Run `uv run python -m training.train --help` for the full CLI (corpus
paths, `--direction`, hyperparameters, `--corpus-version`, `--run-id`).
`training/requirements.txt` lists the extra dependencies the SageMaker
training container needs at runtime (kept in sync by hand with
`pyproject.toml`; `torch` is intentionally omitted there since SageMaker's
PyTorch/HuggingFace framework containers already provide a
CUDA-compatible build).

**No real training run happens in this repo's test suite** — issue #66
covers the first real (billable) training job.
`tests/integration/test_train_pipeline.py` exercises the full wiring
(argument parsing → corpus loading → direction-tagged example building →
vocab/embedding extension → save → evaluate → model card) against tiny
fixture data, a duck-typed fake tokenizer/model (no download), and fake
`trainer`/`translator` callables standing in for the real (heavy)
`fine_tune`/`generate_translations` functions — the same "duck-type and
fixture" approach as the tokenizer-extension tests above.

### Direction handling: one multilingual model, tagged

ADR 0001 left open whether to train one multilingual model (distinguishing
directions via a tag) or two separate per-direction checkpoints. **This
project trains one multilingual checkpoint** for both `es->cak` and
`cak->es`, using an explicit direction tag token (`__es__` / `__cak__`)
prepended to the source text and used as the generation-time
`forced_bos_token_id`, rather than M2M100's built-in language-code
mechanism (which doesn't cover Kaqchikel). See
[`docs/adr/0006-translation-direction-handling.md`](../docs/adr/0006-translation-direction-handling.md)
for the full rationale — in short: the corpus is small enough that
splitting it in two would likely hurt more than any cross-direction
interference would, Kaqchikel has no pretrained skill in either direction
to protect via separation, and one checkpoint is cheaper to
register/serve. `train.py --direction` can still be set to a single
direction for a comparison run; this is a starting decision to revisit
with real per-direction BLEU/chrF once training runs (#66) exist, not a
permanent one.
