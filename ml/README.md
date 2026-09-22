# ml

Machine learning pipeline for the Spanish<->Kaqchikel translation model.

- `data/` — corpus preprocessing/cleaning scripts. **Never commit raw
  corpus files here** — see [`docs/data-governance.md`](../docs/data-governance.md).
  Raw data lives in a private, versioned S3 bucket provisioned by
  `DataStack` (`infra/cdk/lib/data-stack.ts`), one per environment
  (dev/qa/prod); the bucket name/ARN is only ever referenced via CDK
  cross-stack refs or `CfnOutput`, never hardcoded here. Composable
  functions:
  - `data/normalize.py` — whitespace/Unicode (NFC) cleanup. Never
    lowercases (see module docstring for why).
  - `data/dedup.py` — exact-duplicate `(source, target)` pair removal.
  - `data/length_filter.py` — drops empty/too-short/too-long pairs and
    pairs with an outlier source/target length ratio (configurable
    thresholds via `LengthFilterConfig`).
  - `data/split_integrity.py` — validates a train/val split has no
    overlapping sentence pairs (raises `SplitIntegrityError` on exact-pair
    leakage; also reports weaker one-sided source/target overlap). This
    caught a real issue when the private corpus (`v1`) was first uploaded:
    97 exact pairs — mostly short dictionary/glossary-style entries
    (single words, numbers, technical terms) — appeared in both `train`
    and `val`, because the original train/val split predates this check
    and wasn't deduplicated first. Fixed by removing those 97 pairs from
    `val.tsv` only (every one was already present in `train.tsv`, so
    nothing was lost); `val` is now 3,609 pairs, `train` unchanged at
    32,906. Re-run `validate-split` (below) after any future re-upload or
    reprocessing of the corpus to confirm this hasn't regressed.
  - `data/corpus_io.py` — reads/writes two-column TSV sentence pairs from
    a local path or an `s3://` URI. Callers pass the URI; this module
    never hardcodes a bucket or distinguishes the private ALMG corpus
    from the public community corpus (see ADR 0002) — that separation is
    the caller's responsibility, by pointing at distinct S3
    buckets/prefixes for each.

    **File format** (`read_tsv_pairs`/`write_tsv_pairs`, and therefore
    every corpus file in the training bucket): plain UTF-8 text, one
    sentence pair per line, **two fields separated by a literal tab
    character** — **no header row** (a header line would itself be
    parsed as a malformed data row and raise `ValueError`, since it
    won't split into exactly two fields the way real header text
    usually looks). Column order is **Spanish first, then Kaqchikel**:

    ```
    Buenos días	Utz sq'ij
    ¿Cómo estás?	La utz awäch?
    ```

    Blank lines are skipped; any line that isn't exactly two
    tab-separated fields raises an error rather than being silently
    dropped, so a malformed upload fails loudly instead of quietly
    losing data.

    The private ALMG corpus lives under a versioned prefix,
    `s3://<training-bucket>/corpus/almg/v1/{train,val}.tsv` (bucket name
    from `DataStack`, see above) — bump the version segment on any future
    reprocessing of the raw source data, rather than overwriting `v1` in
    place, so a given training run's corpus version stays traceable
    (ADR 0001).
  - `data/pipeline.py` — chains the above into `clean_corpus_file` and
    `validate_split_files`, plus a CLI: `uv run python -m data.pipeline
    clean <input> <output>` or `... validate-split <train> <val>`. Never
    run this against real corpus data in this repo/CI — only against S3
    paths from an authorized environment.
  - `data/dialect_signal.py` — diagnostic tool for issue #51 (the ALMG
    corpus mixes two unlabeled Kaqchikel dialectal variants, per
    `docs/data-governance.md`/project memory). `analyze_dialect_signal`
    computes two purely computational, aggregate-only signals over the
    Kaqchikel side of a corpus: (1) an unsupervised 2-way cluster split
    over character n-gram frequencies (`build_ngram_vectors` +
    `kmeans_two_clusters`, deterministic k-means with a "farthest pair"
    init), and (2) frequency counts for orthographic features known from
    Kaqchikel dialectology to vary across dialects/orthographic eras
    (`feature_frequencies`): tense/long vowel doubling ("aa", "ee", ...),
    central vowel marks ("ä", "ë", "ï", "ö", "ü"), apostrophe-marked
    glottalized consonants ("k'", "tz'", ...), and word-initial "h" (a
    proxy for older Mayan-language orthographies that used "h" where the
    modern ALMG standard uses "j"). **Every public function returns only
    counts/frequencies/cluster sizes — never sentence text** (ADR 0002):
    `render_report`'s output is safe to paste into a public GitHub issue
    as-is, and `tests/integration/test_dialect_signal_cli.py` asserts this
    holds for the real CLI output, not just by convention.

    This is a diagnostic hypothesis-generator, not a labeling authority —
    an n-gram cluster split can reflect topic/register/source-document
    differences as easily as dialect, and the feature list is grounded in
    general Kaqchikel dialectology literature (Kaqchikel has ~11
    recognized regional dialects; see module docstring for sources), not
    validated against this specific corpus. Treat its output as a starting
    point for human/expert review, not a ground-truth split to act on
    directly.

    Run against a real corpus file from an environment with S3 access:
    `uv run python -m data.dialect_signal s3://<bucket>/corpus/almg/v1/train.tsv`
    (or a local path). Never run this in CI/this repo's test suite against
    real data — `tests/unit/test_dialect_signal.py` and
    `tests/integration/test_dialect_signal_cli.py` only ever use tiny,
    hand-written synthetic fixtures with a designed two-group signal.
- `training/` — SageMaker training job entrypoint (`training/train.py`,
  see below), the code that submits/monitors a real training job and
  registers it in SageMaker Model Registry (`training/submit_job.py`, see
  "Job submission" below), and tokenizer/vocabulary extension for
  Kaqchikel (see below).
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
- `training/subword_vocab.py` — trains a fresh, small SentencePiece/Unigram
  model on the **Kaqchikel-only** side of the corpus and diffs its subword
  pieces against a base tokenizer's vocab to find high-value multi-character
  subwords it doesn't already have. See "Kaqchikel subword vocabulary
  (`training/subword_vocab.py`)" below for the full rationale (issue #82).
- `training/tokenizer_extension.py` — the thin integration layer wiring the
  above to a real `transformers.M2M100Tokenizer`/model.
  `extend_tokenizer_vocab` (character/whole-word gaps) and
  `extend_tokenizer_vocab_with_subwords` (subword gaps), which only need
  `get_vocab()`/`add_tokens()`, are unit-tested against a fake tokenizer
  that duck-types those two methods. `resize_embeddings_for_new_tokens` is
  exercised against the real `facebook/m2m100_418M` checkpoint in
  `tests/integration/test_tokenizer_extension_real_model.py`, alongside a
  real-vocab diff for the subword step too (see "Real-checkpoint
  integration test" below).

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

`tests/integration/test_tokenizer_extension_real_model.py` also covers
`extend_tokenizer_vocab_with_subwords` against the real vocab: it confirms
that a SentencePiece model trained on a handful of real Kaqchikel sentences
finds genuine high-value subwords `facebook/m2m100_418M`'s actual
pretrained vocabulary doesn't already have -- issue #82's core acceptance
criterion, verified against the real checkpoint rather than assumed from a
synthetic fake dict.

## Kaqchikel subword vocabulary (`training/subword_vocab.py`)

Three real training runs (#66, #76, and its continuation) showed BLEU
gains shrinking sharply (+3.9, then +1.7 across two equal 5-epoch batches)
while training loss kept dropping (4.45 → 1.97) -- the signature of a
representational ceiling, not undertraining. The suspected cause (issue
#82): run #66's whole-word extension
(`training/vocab_gap.py`/`find_missing_words`, ~61,901 tokens) covers
exact word-forms the model has seen, but says nothing about the *subword*
structure underneath. Kaqchikel is agglutinative (ergative/absolutive
person marking, noun incorporation), so a word-form the model hasn't
memorized verbatim still gets fragmented by M2M100's generic multilingual
SentencePiece model into long, awkward multi-token chains it has no reason
to compose cleanly at inference.

`training/subword_vocab.py` addresses this by training a **second**,
small SentencePiece/Unigram model on the **Kaqchikel-only** side of the
corpus (never mixed with Spanish, which M2M100 already tokenizes
natively), then diffing its resulting subword pieces against the base
tokenizer's vocab:

- `train_subword_model(texts, vocab_size, model_type)` — trains fully in
  memory (`io.BytesIO`, via SentencePiece's `sentence_iterator`/
  `model_writer` kwargs), never writing the trained model proto to disk.
  Passes `hard_vocab_limit=False`: the requested `vocab_size` is a
  *target*, not a hard requirement -- SentencePiece's unigram trainer
  otherwise raises a hard `RuntimeError` when the requested size can't be
  reached from the given text (true for every small fixture this repo's
  fast test suite uses, and a real risk on a real corpus too). With this
  flag, an unreachable request is silently capped instead of crashing a
  potentially multi-hour, billable job at the vocab-training step.
- `extract_vocab_pieces(model_proto)` — every subword piece the trained
  model knows, excluding SentencePiece's own reserved control tokens
  (`<unk>`, `<s>`, `</s>`, `<pad>`).
- `select_high_value_subwords(pieces, min_subword_length=2)` — filters out
  pieces that are just a single character once SentencePiece's leading
  word-boundary marker ("▁") is stripped -- single-character coverage is
  already `training/vocab_gap.py`'s job, so re-adding single characters
  here would be redundant, not "high value".
- `compute_new_subword_tokens(texts, base_vocab, ...)` — orchestrates the
  above, then diffs the high-value pieces against `base_vocab` using the
  **same** `training/vocab_extension.py`'s `select_new_tokens` already
  used for whole words/characters, rather than inventing a second diff
  mechanism.

`training/tokenizer_extension.py`'s `extend_tokenizer_vocab_with_subwords`
wires this to a real tokenizer, mirroring `extend_tokenizer_vocab`'s shape
exactly (compute new tokens against the tokenizer's current vocab, call
`add_tokens`, return what was added). `training/train.py`'s
`extend_vocabulary_for_examples` runs both extension steps in sequence --
character/whole-word gaps first, then subword gaps (diffed against the
vocab *after* step 1, so it never re-proposes something step 1 already
added) -- and resizes the model's embeddings once for their combined
total. The Kaqchikel-only text fed to the subword step comes from
`training/direction.py`'s `collect_texts_for_language(examples,
KAQCHIKEL)`, which reads from whichever side (source or target) of each
direction-tagged example is actually Kaqchikel.

Deliberately **not** a fully separate tokenizer: a separate tokenizer
would sever M2M100's pretrained multilingual embedding alignment, which is
the model's main transfer-learning advantage for a low-resource language
like Kaqchikel. New subword tokens go through the exact same
add-tokens-then-warm-start-embeddings mechanism as any other new token in
this pipeline.

`--subword-vocab-size` (default 8000, both `train.py` and
`submit_job.py`) controls the target vocabulary size for this step and is
recorded in the model card's hyperparameters (and in `submit_job.py`'s
Model Registry metadata) for traceability (ADR 0001).

**The real, controlled training run evaluating this change's BLEU/chrF
impact is a deliberate follow-up, not part of this ticket (#82).** Per the
project's process, every real (billable) SageMaker Training Job submission
requires the project owner's explicit go-ahead -- this ticket is code +
tests only. A "controlled" run should change *only* `--subword-vocab-size`
(from effectively absent, pre-#82, to its new default) and nothing else,
so any BLEU/chrF delta is cleanly attributable to the vocabulary change
alone -- unlike #76's continuation run, which changed epoch count and
regularization hyperparameters together.

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
   using `training/tokenizer_extension.py`'s character/whole-word
   extension (#34/#62) against the actual training corpus text, then its
   Kaqchikel-only subword extension (#82, see "Kaqchikel subword
   vocabulary" above).
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

`generate_translations` moves each batch's encoded tensors to
`model.device` before calling `model.generate(...)` —
`tokenizer(..., return_tensors="pt")` always returns CPU tensors
regardless of where the model lives, and this hand-written generation
loop (unlike `Seq2SeqTrainer`'s own training batches) has to do that
placement itself. Missing this crashed the 3rd real training run after
~3 hours of otherwise-successful training, on the very last step, with
`RuntimeError: ... but got index is on cpu, different from other tensors
on cuda:0`. This is a GPU-only failure mode — CI's CPU-only runners can't
reproduce the crash itself (`model.device` is always `"cpu"` there too),
so `tests/unit/test_generate_translations.py` instead directly asserts
the `.to(model.device)` call happened, via a spy on a fake encoding
object, so this can't silently regress even without real GPU hardware.

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

## Job submission (`training/submit_job.py`)

`training/submit_job.py` is the code that actually submits a real
(billable) SageMaker Training Job running `training/train.py` against the
private ALMG corpus, and registers the resulting model in SageMaker Model
Registry -- issue #66. **This repo's automated test suite never runs a
real training job or calls real AWS**; every AWS/`sagemaker` SDK call in
`submit_job.py` is exercised only against mocks
(`tests/unit/test_submit_job.py`, `tests/integration/test_submit_job_cli.py`).
Actually submitting the real job is a separate, deliberate, maintainer-run
action, gated on an AWS quota increase.

Run it from `ml/` (so `build_source_bundle()`'s default `ml_root` resolves):

```sh
cd ml
uv run python -m training.submit_job --dry-run          # sanity-check first
uv run python -m training.submit_job                     # the real, billable submission
```

**Source packaging**: `train.py` imports sibling packages (`data.*`,
`evaluation.*`, `training.*`), but `sagemaker.huggingface.HuggingFace`'s
`source_dir` upload flattens *the contents of* whatever directory you give
it into `/opt/ml/code/` inside the container -- it does not preserve that
directory's own name. Pointing `source_dir` directly at `training/` (an
earlier version of this script did exactly that) meant those sibling
imports failed with `ModuleNotFoundError: No module named 'data'` the
first time this ran for real -- a bug a mocked unit test can't catch,
since it's about how the real SDK packages a real local directory, not
about this script's own logic. `build_source_bundle()` fixes this by
assembling a temp directory containing `data/`, `evaluation/`, and
`training/` together before building the estimator, so
`entry_point="training/train.py"` and its imports resolve exactly as they
do when running `train.py` locally from `ml/`.

CLI flags (all optional, defaulting to `training/train.py`'s own defaults
where applicable): `--environment` (dev/qa/prod, selects which `DataStack`
to resolve), `--instance-type` (default `ml.g4dn.xlarge`), `--max-run`
(hard wall-clock cap in seconds, default 10800 = 3h, so a runaway job can't
bill forever), `--corpus-version`, `--direction`, `--run-id`, `--base-model`,
`--init-model-s3-uri` (continue training from a previous run's artifact --
see "Continuing training from a checkpoint" below), `--epochs`,
`--batch-size`, `--learning-rate`, `--max-length`, `--seed`,
`--warmup-ratio`, `--weight-decay`, `--label-smoothing`,
`--gradient-accumulation-steps` (regularization/schedule settings, issue
#79 -- see "Regularization and schedule settings" below),
`--subword-vocab-size` (default 8000, issue #82 -- see "Kaqchikel subword
vocabulary" above),
`--model-package-group-name`, `--approval-status` (default
`PendingManualApproval` -- a human reviews BLEU/chrF before approving),
`--no-wait` (submit without blocking/monitoring; also skips registration,
since there's nothing to register until the job finishes), `--no-logs`
(don't stream CloudWatch Logs while waiting), `--no-register` (skip Model
Registry registration even after a successful, waited-for run).

### Continuing training from a checkpoint

`train.py --init-model <path>` (issue #75) loads a previously fine-tuned
checkpoint instead of `--base-model`, so a second (or third...) training
pass doesn't re-pay for epochs already trained. Accepts a local directory
(an already-extracted `save_pretrained()` output) or a local
`model.tar.gz` path, extracted automatically
(`training.train.resolve_model_source`). The checkpoint's vocabulary is
already extended from the first run, so
`extend_vocabulary_for_examples`/`extend_tokenizer_vocab` becomes a safe
no-op when nothing new is missing -- no special-casing needed.

To submit a real continuation job, pass `submit_job.py --init-model-s3-uri
s3://<bucket>/model-artifacts/<prior-run-name>/output/model.tar.gz`: this
stages that artifact as an `init-model` input channel (SageMaker training
containers can't reach an arbitrary S3 URI at runtime otherwise, same
reasoning as the `train`/`validation` channels), and passes
`--init-model /opt/ml/input/data/init-model/model.tar.gz` through to
`train.py` automatically.

`--base-model` still gets recorded in the model card for traceability even
when resuming -- it names what the checkpoint chain ultimately started
from, not literally what this run loaded. The model card's `notes` field
flags when a run was a continuation, and its `train_sentence_count`/
`hyperparameters` describe only that run's additional training, not the
full cumulative history across every continuation.

### Regularization and schedule settings

Issue #66's first run (3 epochs, no regularization) scored BLEU 3.0/chrF
20.4; issue #76's continuation (+5 epochs, same bare-minimum config)
reached BLEU 6.9/chrF 28.6 -- clearly still undertrained, and a code
review flagged the training config itself as a likely contributor: zero
LR warmup interacts badly with the ~62K freshly cold-started embedding
rows from vocab extension, and there was no weight decay, label
smoothing, or gradient accumulation at all (effective batch size 8 is
small/noisy for a vocab this size). Issue #79 adds `--warmup-ratio`
(default 0.05), `--weight-decay` (default 0.01), `--label-smoothing`
(default **0.0**, disabled -- see below), and
`--gradient-accumulation-steps` (default 4, raising effective batch size
to `--batch-size * this`) to close that gap. `fp16=True` is also now
hardcoded in `build_training_arguments` (a fixed real-GPU speed/cost
optimization, not a per-run experiment).

**`--label-smoothing` defaults to 0.0 (disabled), not the "cheap win"
value it started as.** The first real run using these settings crashed
immediately: `ValueError: You cannot specify both decoder_input_ids and
decoder_inputs_embeds at the same time`, from `label_smoothing_factor >
0`'s interaction with M2M100's forward signature under the installed
`transformers` version -- reproduced locally against the real checkpoint
with a tiny dataset (not assumed), isolated by testing each new setting
independently: warmup/weight-decay/gradient-accumulation all work fine
together, only label smoothing crashes.
`tests/integration/test_fine_tune_real_checkpoint.py` guards against this
regressing silently again. Cost of the real run that surfaced this: ~$0.08
(371s billed, killed almost immediately).

`--warmup-ratio` is converted to an absolute `warmup_steps` count inside
`build_training_arguments` rather than passed straight through --
the installed `transformers` version's `Seq2SeqTrainingArguments` no
longer accepts a `warmup_ratio` kwarg at all, confirmed by a real
`TypeError` when this was first written directly. `accelerate` was also
added as a real dependency (`ml/pyproject.toml`,
`training/requirements.txt`): this transformers version requires it even
just to construct `TrainingArguments` with `fp16=True`, confirmed the
same way.

`build_training_arguments` (the `Seq2SeqTrainingArguments` construction,
previously inline in `fine_tune`) is now its own function specifically so
these settings have real, direct unit test coverage
(`tests/unit/test_build_training_arguments.py`) -- constructing
`Seq2SeqTrainingArguments` is cheap and CPU-safe, unlike the rest of
`fine_tune`, which stays untestable in CI for the reasons documented in
its own docstring.

These fixes are a config-only pass -- a separate, bigger redesign of the
whole-word vocab-extension strategy (`ml/training/vocab_gap.py`, which
adds entire Kaqchikel word-forms as atomic tokens rather than subwords,
likely a bigger factor for an agglutinative language) was flagged as a
follow-up, not addressed here. **That follow-up is issue #82** -- see
"Kaqchikel subword vocabulary (`training/subword_vocab.py`)" above.

**`--dry-run`** resolves the real `{Environment}-Data` CloudFormation stack
outputs (a free, read-only call) and prints the full would-be job config --
resolved bucket/role, instance type, image version combination,
hyperparameters, and channel S3 URIs -- as JSON, without ever constructing
a `HuggingFace` estimator or calling `.fit()`. This is what the maintainer
runs first to sanity-check before the real submission.

**Channel path convention**: the `train`/`validation` channels point at the
exact corpus object keys, `s3://<bucket>/corpus/almg/v1/train.tsv` and
`.../val.tsv` -- not the whole prefix. Combined with SageMaker's standard
`/opt/ml/input/data/<channel>/<s3-object-basename>` download convention,
this makes the container-side paths deterministic:
`/opt/ml/input/data/train/train.tsv` and
`/opt/ml/input/data/validation/val.tsv`. `submit_job.py` passes those exact
paths as the `--train`/`--validation` hyperparameters to `train.py`.

`submit_job.py`'s module docstring documents why `ml/pyproject.toml` pins
`sagemaker>=2.257,<3` (the `HuggingFace` estimator / `TrainingInput` API
this ticket requires only exists in the SageMaker Python SDK's `2.x` line;
the installed `3.x` line is an unrelated, incompatible rewrite) and how the
`transformers_version`/`pytorch_version`/`py_version` combination
(`4.56.2`/`2.8.0`/`py312`) was derived from the installed SDK's own
HuggingFace DLC compatibility table rather than guessed by hand -- re-check
that derivation after any future `sagemaker` upgrade.
