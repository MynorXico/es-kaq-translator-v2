# ml

Machine learning pipeline for the Spanish<->Kaqchikel translation model.

- `data/` — corpus preprocessing/cleaning scripts. **Never commit raw
  corpus files here** — see [`docs/data-governance.md`](../docs/data-governance.md).
  Raw data lives in a private, versioned S3 bucket provisioned by
  `DataStack` (`infra/cdk/lib/data-stack.ts`), one per environment
  (dev/qa/prod); the bucket name/ARN is only ever referenced via CDK
  cross-stack refs or `CfnOutput`, never hardcoded here. Composable
  functions:
  - `data/normalize.py` — whitespace/Unicode (NFC) cleanup, plus
    glottal-stop/apostrophe look-alike normalization (`normalize_glottal_marks`,
    issue #90): maps confirmed Unicode look-alike codepoints for the
    Kaqchikel glottal stop/glottalized consonants to the plain ASCII
    apostrophe (`'`, U+0027) that ALMG orthography and the corpus's
    majority convention already use. An aggregate scan of the real
    corpus (32,906 training sentences, counts only — ADR 0002) confirmed
    U+02C8 (MODIFIER LETTER VERTICAL LINE, "ˈ") in active use as a
    look-alike, almost certainly a font/OCR/typesetting artifact from
    digitizing different source documents; U+02BC and U+2019 were also
    checked as candidates but had **zero** occurrences, so — per the
    issue's "verify before assuming" guidance — only U+02C8 is mapped.
    Never lowercases (see module docstring for why).
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
    `s3://<training-bucket>/corpus/almg/{version}/{train,val}.tsv` (bucket
    name from `DataStack`, see above) — bump the version segment on any
    future reprocessing of the raw source data, rather than overwriting a
    prior version in place, so a given training run's corpus version stays
    traceable (ADR 0001). Current version: **`v2`** — `v1` with
    `data/normalize.py`'s glottal-stop look-alike normalization (issue
    #90, see above) applied to every pair via `normalize_pair`. This was a
    normalization-only reprocessing: no dedup or length-filtering was run
    (the `~19x` sentence-length/register gap between the two issue #51
    clusters is a separate, explicitly out-of-scope question — running the
    full `data/pipeline.py clean` pipeline here would have conflated the
    two). Sentence counts are therefore identical to `v1` (32,906 train /
    3,609 val); only text content changed, on ~21.5% of train pairs and
    ~20.8% of val pairs. `validate-split` (below) was re-run against `v2`
    and shows the same one-sided overlap counts as `v1` (977 source-side /
    726 target-side, zero exact-pair leakage) — confirming the
    normalization introduced no split-integrity regression. `v1` is left
    in place, unmodified, for traceability of any run already registered
    against it.

    **`training/submit_job.py` still defaults to `v1`** (its
    `CORPUS_PREFIX`/`CORPUS_VERSION` module constants, see below) — issue
    #90 is a data-preparation ticket only, deliberately not switching the
    training pipeline over. Note for whoever does that switchover next
    (issue #90's documented follow-up sequence, step 2/3): `CORPUS_PREFIX`
    (which S3 objects are actually read) and the `--corpus-version` CLI
    flag/`CORPUS_VERSION` constant (the label recorded in the model card
    and Model Registry) are two **independent** values today — bumping
    only the CLI default without also updating `CORPUS_PREFIX` would train
    against `v1` data while the model card claims `v2`, silently breaking
    ADR 0001 traceability. Update both together (or derive `CORPUS_PREFIX`
    from `--corpus-version` instead of hardcoding it) when that switchover
    happens.
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
    `uv run python -m data.dialect_signal s3://<bucket>/corpus/almg/v2/train.tsv`
    (or a local path). Never run this in CI/this repo's test suite against
    real data — `tests/unit/test_dialect_signal.py` and
    `tests/integration/test_dialect_signal_cli.py` only ever use tiny,
    hand-written synthetic fixtures with a designed two-group signal.

    **Issue #90 update**: `analyze_dialect_signal` now runs
    `data.normalize.normalize_text` over each Kaqchikel sentence before
    computing any feature — previously it read raw corpus text directly,
    so both the glottal-stop look-alike issue above and, in principle,
    precomposed-vs-decomposed Unicode variants of the central-vowel-mark
    characters could confound `glottal_apostrophe`/`central_vowel_marks`.
    A real-corpus check found zero raw NFD-decomposed central-vowel-mark
    sequences already present (so that particular confound wasn't actually
    live in this corpus), but the glottal-stop one was: re-running against
    the real corpus after this fix (both against `v1`, normalized
    on-the-fly, and the new `v2` object, normalized at rest — identical
    results either way, as expected) narrows the previously-reported
    ~10x-looking `glottal_apostrophe` gap between the two issue #51
    clusters (95.6 vs. 0.3 matches/1,000 chars using only the ASCII
    apostrophe, before this fix) down to 87.79 vs. 52.02 matches/1,000
    chars on `train.tsv` (87.39 vs. 51.85 on `val.tsv`) — in line with the
    issue's predicted combined-density figure (~95.9 vs. ~52.3). Cluster
    sizes are unchanged (train: 24,196/8,710, ~73.5%/26.5%; val:
    2,677/932, ~74.2%/25.8%), confirming this was purely an
    encoding-normalization fix, not a change to which sentences cluster
    together.
  - `data/glued_word_signal.py` — full-corpus OCR/digitization glued-word
    artifact sweep (issue #199, see "Full-corpus data curation sweep"
    below for the real-corpus results). Two aggregate-only signals: an
    exact, case-insensitive sweep for the concrete strings issue #116/#125
    found glued together in generated model output ("dedios", "queestá",
    "deguatemala", ...), and a conservative general heuristic that flags a
    rare word as a glued-artifact candidate only if it starts with a
    common Spanish clitic *and* the remainder is itself independently
    frequent in the corpus.
  - `data/alignment_signal.py` — embedding-based source/target
    alignment-quality signal (issue #199, see "Full-corpus data curation
    sweep" below). Pure, unit-tested logic
    (`cosine_similarities`/`auc_separation`/
    `build_shuffled_control_indices`/`analyze_alignment_signal`) compares
    real-pair embedding similarity against a shuffled-control baseline,
    turning "does this embedding model have any usable signal for this
    language pair" into an empirical, AUC-based question. The real-model
    embedding step is a maintainer-run diagnostic (requires `torch`/
    `transformers` and downloads a real checkpoint) never exercised in
    this repo's test suite.
- `training/` — SageMaker training job entrypoint (`training/train.py`,
  see below), the code that submits/monitors a real training job and
  registers it in SageMaker Model Registry (`training/submit_job.py`, see
  "Job submission" below), tokenizer/vocabulary extension for
  Kaqchikel (see below), and a post-hoc checkpoint-averaging utility
  (`training/checkpoint_averaging.py`, issue #182 -- see "Cheap-tier
  quality-experiment prep" below).
- `evaluation/` — BLEU/chrF evaluation harness and model card generation
  (see below).
- `deployment/` — custom SageMaker inference handler and the script that
  registers a training run's artifact as a deployable Model Package
  version (issue #8, see "Serving" below).

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
  run_metadata, output_path, *, directions_path=None)` ties the two
  together: reads a predictions file and a references file (one sentence
  per line, aligned by line order), computes metrics, and writes a
  Markdown model card to `output_path`. Every path is caller-supplied —
  nothing here hardcodes a location for the real corpus.
- **Per-direction BLEU/chrF (issue #178).** Every combined-direction
  number reported before this ticket mixed es->cak and cak->es into one
  score, hiding whether one direction was starving the other under
  shared, direction-tagged training — the concrete evidence
  [ADR 0006](../docs/adr/0006-translation-direction-handling.md) (still
  "Proposed") needs to decide one-model-vs-two.
  `evaluation.metrics.compute_metrics_by_direction(hypotheses, references,
  directions)` buckets hypotheses/references by an aligned direction label
  (e.g. `"es->cak"`/`"cak->es"`) and scores each bucket separately.
  `run_evaluation`'s optional `directions_path` (one direction label per
  line, aligned with `predictions_path`/`references_path`) wires this into
  the model card automatically via a new `## Metrics by direction` section
  on `ModelCardData`/`render_model_card`. Both `training.train.
  run_training_job` and `evaluation.evaluate_checkpoint.
  run_checkpoint_evaluation` now write a `directions.txt` alongside their
  `predictions.txt`/`references.txt` and pass it through, so every run's
  model card going forward reports both directions, not just a one-off
  check.
- Per ADR 0001, every training run should be traceable: register the
  generated model card (corpus version + hyperparameters + metrics)
  alongside the model in SageMaker Model Registry rather than producing
  an untraceable artifact.

BLEU/chrF are quality metrics, not pass/fail tests — see
`docs/testing.md`. The unit tests for `evaluation/metrics.py` sanity-check
that the wrapper calls `sacrebleu` correctly (identical hypothesis/
reference scores near 100, an unrelated hypothesis scores low), not that
the model itself translates well.

**Known bug affecting every real BLEU/chrF number reported before issue
#106 (fixed): es->cak hypotheses were contaminated by a leaked direction
tag.** `training.train.generate_translations` — the function that
generates the hypotheses these metrics are computed against for every
real training run (#66, #76, the hyperparameter run, #82) — had the same
`__cak__`-leak bug independently found and fixed in
`deployment/inference.py`'s real-time serving path for issue #8 (see
"Real deployment verification" below): every es->cak hypothesis came out
as e.g. `"__cak__ Utz awäch?"` instead of `"Utz awäch?"`, because
`__cak__` is an ordinary added-vocab token (not a real special token like
`__es__`), so `tokenizer.batch_decode(..., skip_special_tokens=True)`
never stripped it. cak->es was unaffected. A spurious non-matching
leading token can only ever hurt n-gram precision (BLEU) / character-level
match (chrF), never inflate them, so **every combined-direction score
reported so far (BLEU 3.0/6.9/8.6/8.9, chrF 20.4/28.6/31.2/31.2 — see
"Regularization and schedule settings" and "Real deployment verification"
below) is a likely underestimate for the es->cak-contributed half of that
number, not an overestimate.** True model quality is probably somewhat
higher than recorded. The fix
(`training.direction.strip_leading_direction_tag`, shared by both
`generate_translations` and `deployment/inference.py`'s `translate()` so
they can't drift apart again) is a pure bugfix, applied without
retraining — **none of the numbers above have been re-measured with the
fix applied**; re-evaluating the current best checkpoint's true BLEU/chrF
is a separate, tracked follow-up (issue #108's real run, see "Eval-only
checkpoint scoring" below), not done as part of #106.

See [`docs/adr/0001-initial-architecture.md`](../docs/adr/0001-initial-architecture.md)
for the modeling approach and
[`docs/adr/0003-base-model-license-verification.md`](../docs/adr/0003-base-model-license-verification.md)
for the base model choice (M2M100).

## Eval-only checkpoint scoring (`evaluation/evaluate_checkpoint.py`)

`evaluation/evaluate_checkpoint.py` scores an *already-trained* checkpoint's
BLEU/chrF against a validation set, without running any training step
(issue #108). It exists because `training.train.run_training_job`
unconditionally calls `fine_tune()` before evaluating — there was
previously no way to re-score an existing checkpoint (e.g. after a bugfix
to `generate_translations()` itself, like issue #106's direction-tag-leak
fix above) without paying for a full, billable retrain.

It reuses the exact same, already-tested pieces `run_training_job`
composes for its own post-training evaluation step
(`training.direction.build_direction_examples`,
`training.train.generate_translations`, `evaluation.run.run_evaluation`),
minus everything training-only:

- **No `fine_tune()` call** — this script never touches the model's weights.
- **No `extend_vocabulary_for_examples()` call** — a checkpoint saved by a
  prior training run already has its *post-extension* tokenizer saved
  alongside it, so re-running vocabulary extension here would be wrong (at
  best a no-op, at worst redundant tokens with cold, never-trained
  embeddings). It's a training-time-only step by construction — its whole
  purpose is to warm-start embeddings *before* fine-tuning trains them,
  which doesn't apply to an eval-only run.
- **No new SageMaker Model Registry model package is registered** — this
  re-scores the same already-registered weights, it doesn't produce new ones.

`--checkpoint` accepts a local directory, a local `model.tar.gz`, or an
`s3://` URI to a `model.tar.gz` (downloaded via `resolve_checkpoint_source`,
then extracted the same way `training.train.resolve_model_source` extracts
a local one — the one piece of genuinely new logic here, unit-tested
against a fake S3 client in `tests/unit/test_evaluate_checkpoint_source.py`
without ever touching real AWS).

**`--train` (required): reconstructing issue #116's word-boundary-spacing
fix for a reloaded checkpoint.** A checkpoint's saved tokenizer never
records which added tokens were whole words/characters (`training.
tokenizer_extension.extend_tokenizer_vocab`) versus subword pieces
(`extend_tokenizer_vocab_with_subwords`) — `_mark_word_boundary_
tokens`'s effect (issue #116's fix) lives only on the live tokenizer
*instance* that originally called it, and `save_pretrained()` only ever
writes a flat `added_tokens.json` list. Simply reloading a checkpoint here
and generating translations would therefore silently **not** apply #116's
fix at all, even after that fix landed on `main` — discovered when
attempting to re-evaluate the deployed checkpoint (`run-20260922T141956Z`)
against #116's fix without retraining. `--train` must name the exact
training corpus the checkpoint's *own* training history used (from that
run's own model card/hyperparameters — not necessarily the same corpus
this script's own `--validation` scores against). There's deliberately no
separate `--train-direction` flag: the reconstructed token set is provably
identical regardless of direction, since every direction mode's sample
text reduces to the same underlying *set* of both columns of every pair
(confirmed empirically — an earlier version of this flag was dropped as
dead weight once this was noticed in review, PR #128). Given `--train`,
`training.tokenizer_extension.patch_word_boundary_decoding_for_checkpoint`
deterministically recomputes the same whole-word boundary token set the
checkpoint's training run(s) actually added — no SentencePiece retraining,
no randomness, safe even across a chain of `--init-model` continuation
runs (`training.tokenizer_extension`'s module docstring has the full
argument, confirmed empirically against the real deployed checkpoint's
own continuation chain) — and patches the reloaded tokenizer's decode
behavior in place before any translation is generated. The reconstructed
token count is recorded in the model card's hyperparameters
(`word_boundary_tokens_reconstructed`).

**Provenance-aware reconstruction (issue #143).** Issue #125 scoped
`training.train.extend_vocabulary_for_examples`'s whole-word/character step
to Kaqchikel-only text instead of every pair's both columns. Reconstructing
against the wrong construction for a given checkpoint would misleadingly
trip the over-reconstruction diagnostic below, so this script picks the
construction to use from the checkpoint's *own* recorded provenance rather
than a caller-supplied flag: `training.train.run_training_job` records a
fixed `vocab_extension_scoping` hyperparameter
(`training.train.VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY`) in every model
card it writes going forward, and `_read_checkpoint_vocab_extension_
scoping` reads it back from the checkpoint's own saved `model_card.md`
(best-effort, never fatal). If present, `_build_train_sample_texts` uses
the new Kaqchikel-only column only; if absent — true for every checkpoint
trained before this field existed, including the currently-deployed
`run-20260922T141956Z` (Model Package v4) — it falls back to the legacy
"both columns" construction unchanged. A present-but-unrecognized value
(neither `kaqchikel_only` nor absent — e.g. some future scoping scheme
this function hasn't been updated for) prints a distinct WARNING rather
than silently being folded into the "absent" case, and still falls back
to the legacy construction as the safest available guess (PR #144 review).

**Lineage, not just the checkpoint's own last run (PR #144 review).** A
checkpoint's real vocabulary is the union of every run in its
`--init-model` continuation chain, not just the final run's own
contribution. `training.train.run_training_job` accounts for this at
write time, not read time: it only records `vocab_extension_scoping` in
its own model card when continuing from an ancestor whose *own* model
card also recorded it (`_resolve_vocab_extension_scoping_for_model_card`)
-- an inductive, one-hop check, since the ancestor's own recorded value
already reflects *its* whole lineage. Continuing from any ancestor that
predates issue #125 (or has no readable model card at all) means the
field is omitted from every descendant's model card too, so
`evaluation.evaluate_checkpoint` keeps correctly falling back to the safe,
superset "both columns" reconstruction for the whole chain -- it never
needs to (and structurally cannot, since a checkpoint's saved files don't
record its ancestor's S3 URI) walk continuation lineage itself.

Two independent diagnostics then run and print to stderr (never raise) if
something looks wrong — see `_diagnose_word_boundary_reconstruction`'s own
docstring for the full reasoning:

- **Over-reconstruction**: any reconstructed token that isn't actually in
  the checkpoint's vocabulary — unambiguous evidence `--train`/
  `--base-model` don't match what the checkpoint was trained with.
- **Under-reconstruction** (the more dangerous failure mode, since a
  too-small `--train` still makes every reconstructed token trivially
  present in the checkpoint's larger real vocab, so the over-reconstruction
  check alone can never catch it): the checkpoint's *exact* set of
  ambiguous added tokens (a real vocab diff against the pristine base
  model, independent of `--train`'s content) is cross-checked against the
  reconstructed set. The gap between them is expected to be non-empty in
  the normal case (genuine subword continuation pieces, which must
  correctly stay unmarked), but it mathematically cannot exceed the
  checkpoint's own recorded `new_tokens_added` (read best-effort from its
  saved `model_card.md`) if reconstruction is correct — a larger gap means
  some real whole-word tokens were missed.

**Provenance**: a model card produced by this script is *not* a new
training run's own eval, and is marked as such so it can't be confused
with one — `--source-run-id` (required) names the training run whose
checkpoint is being re-scored, and the rendered model card records
`reevaluation: True` plus `source_run_id`/`source_checkpoint` in its
hyperparameters section, a `reeval-<UTC timestamp>` run-id prefix (instead
of `training.train`'s `run-<UTC timestamp>`) if `--run-id` is omitted, and
a `train_sentence_count` of `0` with a note pointing back at the original
run's own model card for that number, since no training happened here.

Example (run from `ml/`, once a checkpoint exists in S3 — see the module's
own `--help` for every flag):

```sh
uv run python -m evaluation.evaluate_checkpoint \
  --checkpoint s3://<training-data-bucket>/model-artifacts/<run-id>/output/model.tar.gz \
  --validation s3://<training-data-bucket>/corpus/almg/v1/val.tsv \
  --train s3://<training-data-bucket>/corpus/almg/v1/train.tsv \
  --corpus-version almg-v1 \
  --source-run-id <run-id> \
  --output-dir ./eval-output \
  --num-beams 5
```

`--num-beams` (issue #180) defaults to `training.train.DEFAULT_NUM_BEAMS`
(5, matching the base model's own `generation_config.json`) and is passed
straight through to `generate_translations`/`model.generate()`, then
recorded in the rendered model card's hyperparameters for traceability.
Before issue #180, beam width was never a parameter anywhere in this
project's code at all -- it was silently inherited from whatever
`generation_config.json` a given checkpoint happened to carry. See
"Decode configuration" under the training entrypoint section below for why
this now matters and the real at-scale validation result.

Test coverage follows this project's "duck-type and fixture" convention
(`tests/integration/test_evaluate_checkpoint_pipeline.py`, mirroring
`tests/integration/test_train_pipeline.py`): a fake checkpoint loader, a
fake base-tokenizer-vocab loader, and a fake `generate_translations` stand
in for the real (heavy) model load/generate calls, and one test asserts
the fake tokenizer's `add_tokens` is never called, guarding against
vocabulary extension being reintroduced here by mistake. Separate tests
cover the word-boundary reconstruction wiring itself (a real, non-zero
token count reaches the model card; over-reconstruction and
under-reconstruction each trigger their own distinct stderr diagnostic,
the latter both with and without a `model_card.md` available for the
cross-check) and, against the real `facebook/m2m100_418M` tokenizer,
`tests/integration/test_tokenizer_extension_reload_reconstruction.py`
proves the underlying save/reload/reconstruct round-trip actually works
(extend → save → reload loses issue #116's fix → reconstruct + patch
recovers it, byte-for-byte matching the never-saved original's decode
output). **No real checkpoint is downloaded and no real SageMaker job is
submitted in this repo's test suite** — actually running this against the
real checkpoint is a separate, maintainer-run action (issue #108's
follow-up), same as `training/submit_job.py`'s real training job
submission.

### Decode-time length-bias rectification for label-smoothed checkpoints (`evaluation/length_bias.py`, issue #193)

Issue #182's label-smoothing experiment (Model Package v9, `--label-smoothing
0.1`, 13 epochs, otherwise identical to the v7 baseline) scored BLEU 13.2 /
chrF 36.4 against v7's baseline BLEU 14.0 / chrF 36.5, and was rejected in
Model Registry on a raw comparison. But Liang, Wang & Cao, "The Implicit
Length Bias of Label Smoothing on Beam Search Decoding"
([arXiv:2205.00659](https://arxiv.org/abs/2205.00659)), show label smoothing
implicitly biases beam search toward *shorter* outputs (their Sec. 3.1: beam
search scores a sequence by `sum(log(p_hat))`, where `p_hat = (1 - alpha) *
q + alpha / V` is the label-smoothed model's learned prediction, which adds
an implicit `log(1 - alpha)` penalty to every generated token relative to
the true `log(q)`), and that a decode-time rectification recovers real BLEU
a raw comparison masks. `evaluation/length_bias.py` implements their exact
proposed correction (their Eq. 4, not a generic length-penalty heuristic):

```
p_db_i = ReLU(p_hat_i - delta) / sum_j(ReLU(p_hat_j - delta))
```

applied to the model's own predicted next-token distribution at every beam
search decode step. `delta = alpha / V` (`V` = vocab size) is the paper's
theoretically exact inverse of label smoothing's interpolation (their Eq.
3); their own experiments (Table 1) found a *larger* `delta = 1/V` gives
near-peak BLEU across every language pair they tested at beam size 4
(closest to this project's own `--num-beams` default of 5), i.e. stronger-
than-theoretically-justified debiasing was empirically beneficial in their
setup too. See the module's own docstring for the full derivation.

Two independent implementations exist, tested against each other
(`tests/unit/test_length_bias_logits_processor.py`) so they can't silently
drift apart: `rectify_probabilities` is a pure NumPy reference
implementation of Eq. 4 (`tests/unit/test_length_bias.py`, including a test
that rectifying a perfectly label-smoothed distribution exactly recovers the
original ground-truth distribution -- Eq. 3 as a special case of Eq. 4), and
`build_length_bias_logits_processor` is the real `transformers.
LogitsProcessor`-based implementation actually used at decode time (`torch`-
native, since it has to operate on live GPU-resident tensors at every decode
step without a per-step CPU round-trip).

`evaluate_checkpoint.py --debias-delta-multiplier <k>` (default `0.0`,
disabled -- no `logits_processor` reaches `generate_translations` at all,
so every prior re-evaluation's decode behavior is completely unchanged)
computes `delta = k / vocab_size` from the checkpoint's own (already
vocab-extended) tokenizer and threads a real logits processor through
`generate_translations`/`model.generate()`. Passing the checkpoint's own
`--label-smoothing` training value (e.g. `0.1`) is the theoretically exact
rectification; `1.0` is the paper's own empirically near-peak value at small
beam widths and the recommended first value to try. Recorded in the
rendered model card's hyperparameters (`debias_delta_multiplier`, and when
enabled, the resolved `debias_delta`/`debias_vocab_size`) for traceability.

```sh
uv run python -m evaluation.evaluate_checkpoint \
  --checkpoint s3://<training-data-bucket>/model-artifacts/<v9-run-id>/output/model.tar.gz \
  --validation s3://<training-data-bucket>/corpus/almg/v1/val.tsv \
  --train s3://<training-data-bucket>/corpus/almg/v1/train.tsv \
  --corpus-version almg-v1 \
  --source-run-id <v9-run-id> \
  --output-dir ./eval-output-debiased \
  --num-beams 5 \
  --debias-delta-multiplier 1.0
```

**Real re-scoring result (issue #193): rectification does not close the
gap -- a fully-settled "label smoothing doesn't help here" finding.**
Re-scored against real weights (v9's own actual checkpoint, downloaded from
its S3 training artifact -- not a reconstruction) and v7's own real
checkpoint, on the *same* 200-pair (400-example, balanced both directions)
sample of the real `almg-v1` validation set, same `--num-beams 5`, same
word-boundary reconstruction -- mirroring issue #180's own "n=200/direction,
same pairs across every setting compared" methodology, since a full
7,218-example run of three separate decode configs on this project's local
4GB GPU was not practical within this diagnostic's own scope:

| checkpoint | `--debias-delta-multiplier` | BLEU | chrF | es->cak BLEU/chrF | cak->es BLEU/chrF |
|---|---|---|---|---|---|
| v7 (baseline) | 0.0 (n/a, no label smoothing) | 15.9 | 37.8 | 19.4/40.9 | 9.9/32.8 |
| v9 | 0.0 (undebiased, as originally rejected) | 14.8 | 37.1 | 19.1/40.1 | 8.4/32.3 |
| v9 | 0.1 (theoretically exact, `delta = alpha/V`, `alpha=0.1`) | 14.2 | 36.9 | 18.2/40.0 | 8.1/32.0 |
| v9 | 1.0 (paper's own empirical near-peak at small beam) | 14.3 | 37.1 | 18.3/40.1 | 8.1/32.1 |

(This sample's absolute numbers differ somewhat from the full-validation-set
numbers registered in Model Registry -- expected sample variance at n=200
pairs vs. n=3,609 -- but the ordering matches: v7 > v9 undebiased on both
samples, confirming this smaller matched sample is a reasonable stand-in for
the question this ticket asks.)

**Neither debiasing configuration improves on v9's own undebiased score,
let alone closes the gap to v7.** Both the theoretically exact rectification
and the paper's own stronger, empirically-recommended value are flat-to
-slightly-worse than `delta=0` on this checkpoint, on the exact same 400
examples. v9's near-baseline chrF (36.4 vs. v7's 36.5, full-set numbers) is
**not** evidence of a masked gain in this case -- correcting for the
mechanism the cited paper describes reveals no hidden quality improvement.
This closes the question issue #193 opened: **label smoothing (`alpha=0.1`,
13 epochs, otherwise matching the v7 baseline) does not help this project's
fine-tuning setup**, even after accounting for its documented beam-search
length bias. Re-affirms, rather than overturns, Model Package v9's original
`Rejected` status in Model Registry -- no change to the deployed checkpoint
or its decode configuration. Per issue #193's own scope, this is a pure
re-evaluation: no retraining, no new Model Registry model package.

### Verifying the reconstruction against a real checkpoint (maintainer step)

The claim that reconstruction exactly matches a real checkpoint's own
`added_tokens.json` (confirmed for the deployed checkpoint,
`run-20260922T141956Z`, during this feature's development: 61,901
reconstructed whole-word/character tokens against corpus `almg-v1`,
exactly matching that checkpoint's real vocabulary) is independently
re-checkable without retraining, mirroring issue #108's "real re-eval is a
separate maintainer step" pattern rather than a one-off, unrepeatable
claim:

```sh
# 1. Download the real checkpoint's tokenizer files only (not the model
#    weights) to inspect its added vocabulary directly.
aws s3 cp s3://<training-data-bucket>/model-artifacts/<run-id>/output/model.tar.gz ./model.tar.gz
tar xzf model.tar.gz added_tokens.json vocab.json model_card.md

# 2. Download that run's exact training corpus (from the run's own
#    hyperparameters/model card -- --corpus-version, --train channel URI).
aws s3 cp s3://<training-data-bucket>/corpus/almg/v1/train.tsv ./train.tsv

# 3. Recompute the whole-word/character token set the same way
#    evaluate_checkpoint.py does, using the real facebook/m2m100_418M
#    tokenizer's pristine vocab (never the checkpoint's own).
uv run python -c "
from data.corpus_io import read_tsv_pairs
from training.direction import ALL_DIRECTION_TAG_TOKENS
from training.tokenizer_extension import reconstruct_whole_word_boundary_tokens
from transformers import M2M100Tokenizer

base_vocab = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M').get_vocab()
pairs = read_tsv_pairs('train.tsv')
sample_texts = [s for s, _ in pairs] + [t for _, t in pairs] + list(ALL_DIRECTION_TAG_TOKENS)
reconstructed = reconstruct_whole_word_boundary_tokens(base_vocab, sample_texts)
print(len(reconstructed))
"

# 4. Compare against the real checkpoint's saved vocab: every reconstructed
#    token should be a key in added_tokens.json (step 1), and
#    len(vocab.json) + len(added_tokens.json) - len(base_vocab) should
#    equal len(reconstructed) + <subword tokens actually added, if any --
#    see the checkpoint's own model_card.md's subword_vocab_size/
#    new_tokens_added hyperparameters>.
```

If the checkpoint under test is itself the result of an `--init-model`
continuation chain (like the real deployed checkpoint), step 2's corpus
must be the corpus *every* run in that chain used (traceable via each
training job's own `describe-training-job` hyperparameters/`init-model`
channel, chained backward) -- see `training.tokenizer_extension`'s module
docstring for why this still produces an exact match regardless of chain
depth.

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

**Issue #125 finding:** run #66's whole-word step above wasn't actually
isolated to Kaqchikel the way this subword step always has been --
`find_missing_words` checks whether a word's exact surface form is
already a single vocab *key*, not whether the base tokenizer can already
represent it via existing subwords, so it flagged the large majority of
ordinary Spanish words in the real corpus as "missing" too and registered
a new standalone token for each one. A real classification (run against
the real `facebook/m2m100_418M` tokenizer, since the private corpus isn't
available outside a training job -- see issue #125 for the full
methodology and numbers) found roughly half of that kind of whole-word
token set attributable to Spanish, and that essentially all of the
Spanish-attributed tokens sampled were already fully representable by the
base tokenizer's existing subwords (i.e. not a genuine gap). Fixed in
`training.train.extend_vocabulary_for_examples`, which now scopes the
whole-word/character step to Kaqchikel-only text too, matching this
subword step. This does **not** retroactively change the currently
deployed checkpoint's already-baked-in vocabulary (see
`training/tokenizer_extension.py`'s `reconstruct_whole_word_boundary_tokens`
docstring, which still deliberately reconstructs the old, unscoped
behavior to match that checkpoint's real training history) -- only a
future retraining run picks up the fix.

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

`train.py` itself needs zero AWS credentials or SDK calls -- registration
in SageMaker Model Registry is `submit_job.py`'s job, client-side, after a
waited-for run completes (see "Job submission" below). Issue #190 briefly
moved registration into this script (self-registering from inside the
training container); ADR 0009 reverted that entirely once issue #201
confirmed it was structurally unable to succeed (`CreateModelPackage`
validates S3 object existence at call time, before the training
container's own toolkit has uploaded the artifact).

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

### Decode configuration: beam search, confirmed (issue #178)

Neither `generate_translations` (here) nor `evaluate_checkpoint.py`'s
generation call ever passes `num_beams` explicitly to `model.generate()`
— only `forced_bos_token_id` and `max_length`. That means whatever decode
strategy is actually in effect comes entirely from the checkpoint's own
saved `generation_config.json`, not from anything in this repo's code.
Issue #178 confirmed this directly against a real checkpoint (v7,
`run-20260926T060000Z`) rather than assuming it either way: `tar xzf
model.tar.gz generation_config.json` shows `"num_beams": 5,
"early_stopping": true`, inherited unmodified from `facebook/m2m100_418M`'s
own `generation_config.json` through every `save_pretrained`/
`from_pretrained` round trip in the fine-tuning pipeline. Loading the real
checkpoint and calling `model.generate()` exactly the way
`generate_translations` does (no `num_beams` kwarg) produces output that
measurably differs from an explicit `num_beams=1` (greedy) call on the
same input, confirming beam search is genuinely active at generation time,
not just present-but-ignored in the config file. **Generation has been
using real beam search (width 5) all along** — this was not a
silently-greedy fallback bug.

A decode-parameter sweep (`num_beams` in {3, 5, 8}, plus
`length_penalty`/`no_repeat_ngram_size` variations) was run against the
same v7 checkpoint on a small sample of the validation set (n=80); see
issue #125's comment thread for the full numeric results. That sweep's own
author flagged it explicitly as **not yet validated enough to adopt**:
`num_beams=8` looked like a >1 BLEU win (12.42 vs. 11.03), driven almost
entirely by cak->es nearly doubling (3.33 → 6.47) — but the sample was
tiny and the sweep ran under this same local GPU's severe thermal
throttling (see "Real at-scale validation" below).

### `num_beams`: now configurable, and issue #178's sweep does not hold up at scale (issue #180)

Issue #180 did two things: made `num_beams` a real, explicit, tested
parameter on `generate_translations`, `evaluate_checkpoint.py`'s CLI
(`--num-beams`, defaulting to `training.train.DEFAULT_NUM_BEAMS`), and
`deployment.inference.translate` (`deployment.inference.DEFAULT_NUM_BEAMS`)
— rather than a value silently inherited from whatever
`generation_config.json` a given checkpoint happened to carry — and then
used that new parameter to re-run issue #178's `num_beams=8` finding at a
much larger, balanced sample against the real v7 checkpoint.

**Real at-scale validation.** This local GPU (a 4GB card with no active
cooling) throttles too severely to run the full 7,218-example validation
set in one sitting — confirmed directly, not assumed: temperature pinned
at 92-95°C throughout, clock oscillating between roughly 300MHz and
1700MHz, with wildly variable per-batch generation time as a direct
result. Per this issue's own guidance, the comparison instead used 200
pairs (400 examples) per direction, **the same 200 pairs scored under both
`num_beams` settings** for a real apples-to-apples comparison (issue
#178's own n=80 sweep mixed both directions into one combined-and-grouped
sample, which — as this validation separately discovered — is a
methodological trap: `training.direction.build_direction_examples("both")`
groups every es->cak example before any cak->es example, so a
time-boxed/truncated run can silently end up scoring only one direction):

| direction | num_beams | BLEU | chrF | n |
|---|---|---|---|---|
| es->cak | 5 (current default) | 16.56 | 38.65 | 200 |
| es->cak | 8 | 16.37 | 38.21 | 200 |
| cak->es | 5 (current default) | 10.06 | 30.82 | 200 |
| cak->es | 8 | **9.62** | 30.84 | 200 |

**`num_beams=8` does not hold up at this larger, balanced sample size** —
it's flat-to-slightly-worse on both directions, including cak->es, the
exact direction issue #178's small sample reported nearly doubling.
Issue #178's finding looks like small-sample noise, not a real effect —
exactly the risk its own author called out before this validation ran.

**Recommendation: no-go.** The current default (`num_beams=5`) is kept
unchanged in `deployment/inference.py`. There's no quality upside to
justify adopting `num_beams=8`, and there's a real downside: beam width 8
also produced a genuine `CUDA out of memory` error at this evaluation
script's batch size of 8 on this 4GB card (recovered by dropping to batch
size 4 for that one run) — a concrete data point that higher beam widths
cost more generation memory, on top of the latency/cost reasoning below.

**Latency/cost, reasoned from this environment's numbers (an estimate, not
a SageMaker measurement — no real endpoint invocation was in this ticket's
scope).** At matched batch size (cak->es, batch size 8 for both settings),
`num_beams=8` took 332.0s vs. `num_beams=5`'s 206.1s for the same 200
examples — **~1.6x** the wall time, consistent with issue #178's own
"roughly proportional to beam count" pattern (8/5 = 1.6). `/v1/translate`
is still a synchronous call bound by API Gateway's 29-second timeout (ADR
0008's async fix isn't implemented yet), and SageMaker Serverless bills
per invocation duration — so adopting `num_beams=8` would mean paying
~1.6x more per request for a quality change that, at this sample size, is
not actually positive. Real single-request (batch size 1) serving latency
on actual SageMaker hardware was not measured directly and would likely
differ from this batched, thermally-throttled local number in absolute
terms, but the *relative* ~1.6x multiplier is the more portable takeaway,
and it points the same direction as the quality result: don't adopt.

This was a pure diagnostic and configurability change — no retraining, no
new SageMaker Model Registry model package, and (per the no-go
recommendation above) no change to the deployed decode configuration
either. Findings posted in full, with the same numbers as above, as a
comment on issue #125.

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

**Registration happens client-side, after a waited-for run completes
(ADR 0009).** `main()` downloads the completed run's `model.tar.gz`
artifact (streamed to disk, never buffered in memory -- issues #188/#189)
just far enough to extract `model_card.md`, then calls
`CreateModelPackageGroup`/`CreateModelPackage` under the maintainer's own
SSO credentials. Issue #190 briefly moved this into `train.py` instead,
self-registering from inside the training container to avoid the
download entirely; ADR 0009 reverted that once issue #201 confirmed it
was structurally unable to succeed: `CreateModelPackage` validates S3
object existence at call time, and self-registration ran *before* the
training container's own toolkit had uploaded the artifact to S3 -- a
separate, post-exit lifecycle step the running script has no hook into.
See `docs/adr/0009-model-registry-registration-timing.md` for the full
decision record. Every real run submitted before issue #187's fix had its
per-epoch checkpoints bundled into this same artifact, measured at
28.4-30.5 GB -- every registration (and every `--register-existing` call,
below) paid a 45min-1hr+ download purely to read a few-KB `model_card.md`.
Issue #187 fixed this at the source (see "Checkpoint storage" under
"Checkpoint averaging" below): a run submitted after that fix registers a
small (model + tokenizer + model_card.md only, roughly ~2GB) artifact
instead, syncing checkpoints to their own separate S3 location during
training. The streamed-to-disk download (issues #188/#189) is kept
regardless of artifact size -- it costs nothing extra for a small artifact
and stays correct for any older, pre-#187 run still being registered.
`train.py` itself needs zero AWS credentials or SDK calls.

Run it from `ml/` (so `build_source_bundle()`'s default `ml_root` resolves):

```sh
cd ml
uv run python -m training.submit_job --dry-run          # sanity-check first
uv run python -m training.submit_job                     # the real, billable submission
```

**Source packaging**: `train.py` imports sibling packages (`data.*`,
`evaluation.*`, `training.*`), but `sagemaker.train.ModelTrainer`'s
`SourceCode.source_dir` upload flattens *the contents of* whatever
directory you give it into `/opt/ml/code/` inside the container -- it does
not preserve that directory's own name. Pointing `source_dir` directly at
`training/` (an earlier version of this script did exactly that) meant
those sibling imports failed with `ModuleNotFoundError: No module named
'data'` the first time this ran for real -- a bug a mocked unit test can't
catch, since it's about how the real SDK packages a real local directory,
not about this script's own logic. `build_source_bundle()` fixes this by
assembling a temp directory containing `data/`, `evaluation/`, and
`training/` together before building the estimator, so
`entry_script="training/train.py"` and its imports resolve exactly as they
do when running `train.py` locally from `ml/`.

CLI flags (all optional, defaulting to `training/train.py`'s own defaults
where applicable): `--environment` (dev/qa/prod, selects which `DataStack`
to resolve), `--instance-type` (default `ml.g4dn.xlarge`), `--max-run`
(hard wall-clock cap in seconds, default 10800 = 3h, so a runaway job can't
bill forever), `--corpus-version`, `--corpus-prefix` (override the S3
corpus prefix a job trains against, e.g. to run against a candidate
cleaned corpus version -- defaults to the hardcoded `CORPUS_PREFIX`
constant, `corpus/almg/v1`, independent of `--corpus-version`, which is
just a metadata string recorded in the model card; see "Full-corpus data
curation sweep" below), `--direction`, `--run-id`, `--base-model`,
`--init-model-s3-uri` (continue training from a previous run's artifact --
see "Continuing training from a checkpoint" below), `--epochs`,
`--batch-size`, `--learning-rate`, `--max-length`, `--seed`,
`--warmup-ratio`, `--weight-decay`, `--label-smoothing`,
`--gradient-accumulation-steps` (regularization/schedule settings, issue
#79 -- see "Regularization and schedule settings" below),
`--subword-vocab-size` (default 8000, issue #82 -- see "Kaqchikel subword
vocabulary" above), `--dropout`, `--bpe-dropout-alpha` (cheap-tier quality
levers prepared -- not yet evaluated -- by issue #182; see "Cheap-tier
quality-experiment prep" below),
`--model-package-group-name`, `--approval-status` (default
`PendingManualApproval` -- a human reviews BLEU/chrF before approving),
`--no-wait` (submit without blocking/monitoring; also skips registration,
since there's nothing to register client-side until the job's model
artifact exists, which only happens once the job finishes), `--no-logs`
(don't stream CloudWatch Logs while waiting), `--no-register` (skip Model
Registry registration even after a successful, waited-for run),
`--register-existing TRAINING_JOB_NAME` (see below).

### Registering an existing job's artifact after the fact (`--register-existing`)

`submit_job.py --register-existing <training-job-name>` looks up an
already-completed training job by name (`describe_training_job`) and
registers its artifact, instead of submitting anything new --
`--model-package-group-name`/`--approval-status` still apply, every other
flag is ignored. Useful for a job submitted with `--no-register`/
`--no-wait` that a maintainer now wants registered after the fact, without
re-typing its hyperparameters by hand. It downloads the full
`model.tar.gz` (streamed to disk, issue #188) to read the job's model
card, same as `main()`'s own default registration path. Re-running it
against a job that's already registered creates a redundant (but
harmless) extra Model Package version pointing at the same artifact --
it's meant for jobs that were never registered at all, not to re-register
an already-registered one.

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
reached BLEU 6.9/chrF 28.6 -- clearly still undertrained (and, per the
"Known bug" note under "Evaluation harness" above, computed before issue
#106's direction-tag-leak fix, so the es->cak half of each of these
numbers is a likely underestimate), and a code
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

**`--label-smoothing` defaulted to 0.0 (disabled) from issue #79 through
issue #182, not the "cheap win" value it started as.** The first real run
using these settings crashed immediately: `ValueError: You cannot specify
both decoder_input_ids and decoder_inputs_embeds at the same time` --
reproduced locally against the real checkpoint with a tiny dataset (not
assumed), isolated by testing each new setting independently: warmup/
weight-decay/gradient-accumulation all worked fine together, only label
smoothing crashed. Cost of the real run that surfaced this: ~$0.08 (371s
billed, killed almost immediately).

**Issue #182 root-caused and fixed this crash properly, instead of leaving
the flag disabled indefinitely.** The error's own wording ("both ... at the
same time") is misleading -- tracing the actual call stack against the
installed `transformers` version showed the real defect is the opposite:
*neither* `decoder_input_ids` nor `decoder_inputs_embeds` was reaching
`M2M100Decoder.forward` at all when label smoothing was enabled.
`transformers.DataCollatorForSeq2Seq(tokenizer, model=model)` only builds
`decoder_input_ids` itself when
`hasattr(model, "prepare_decoder_input_ids_from_labels")` -- and
`M2M100ForConditionalGeneration` no longer defines that method at all in
the installed `transformers` version (confirmed directly: `hasattr(...)`
is `False`, though several other model families -- bart, t5, mbart,
pegasus, ... -- still define it). This was invisible without label
smoothing, because `M2M100ForConditionalGeneration.forward`'s own `if
labels is not None: decoder_input_ids = shift_tokens_right(...)` branch
built it internally in that case -- but `--label-smoothing > 0` makes
`Trainer` pop `labels` out of the batch *before* calling the model (so it
can apply smoothing itself against the raw logits), so that internal
fallback never ran either, leaving both decoder-input fields unset and
tripping `M2M100Decoder.forward`'s "exactly one of these must be given"
guard.

The fix (`training.train.build_data_collator`/
`attach_decoder_input_ids`/`shift_tokens_right`) computes
`decoder_input_ids` explicitly, unconditionally, in the data collator --
before `Trainer.compute_loss` ever gets a chance to pop `labels` -- using
a pure reimplementation of the exact shift-right transform the model's own
(now effectively dead, for this use) internal branch performed, so it's a
no-op change to what the decoder actually sees for every already-working
(no label smoothing) run. `tests/integration/test_fine_tune_real_checkpoint.py`
now runs a real training step with `label_smoothing_factor=0.2` and
confirms it no longer crashes -- see
`training.train.attach_decoder_input_ids`'s docstring and
`tests/unit/test_decoder_input_ids.py`'s module docstring for the full
traced root cause. `--label-smoothing` still defaults to 0.0: literature on
low-resource NMT fine-tuning (Sennrich & Zhang 2019, "Revisiting
Low-Resource Neural Machine Translation: A Case Study", ACL 2019 --
specifically studies larger label-smoothing factors as one of several
hyperparameters worth re-tuning for low-resource settings, as part of a
broader set of best practices for adapting NMT systems to small data
sizes) flags this as disproportionately helpful at small data sizes, but
actually changing the default is a separate, maintainer-approved
real-training-run experiment (issue #182's own scope is the code fix only,
not evaluating it).

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

### Cheap-tier quality-experiment prep (issue #182)

A research pass on further model-improvement options identified four
"cheap tier" levers -- each cheap to actually *run* (~$1.50-4 per real
training job) but, until this ticket, not even possible to try, since the
underlying code didn't support them. **This ticket is code-level prep
only: no training job was submitted as part of it.** Each lever below
still needs its own real, controlled, maintainer-approved training run to
actually measure a BLEU/chrF effect -- changing only that one lever
relative to the current best checkpoint's config, the same "isolate one
variable" discipline as issue #82's own subword-vocabulary follow-up.

1. **The label-smoothing crash is fixed** (not just left disabled) -- see
   "Regularization and schedule settings" above for the full root cause
   and fix. `--label-smoothing` still defaults to 0.0; the literature-
   informed ~0.1-0.2 experiment is a separate future run.
2. **`--dropout`** (default `None`, untouched) overrides the model's
   general (non-attention) dropout probability before fine-tuning --
   `training.train.apply_dropout_config`. Threading this through required
   more than setting `model.config.dropout`: `M2M100Encoder`/
   `M2M100Decoder`/`M2M100EncoderLayer`/`M2M100DecoderLayer` each copy
   `config.dropout` into their own `self.dropout` plain-float attribute
   once, at `__init__` time, so a loaded checkpoint's already-constructed
   submodules need their own `.dropout` attribute overwritten directly --
   confirmed against the real `facebook/m2m100_418M` checkpoint's actual
   module tree in `tests/integration/test_dropout_real_checkpoint.py`.
   Deliberately never touches `M2M100Attention`'s own `.dropout` attribute
   (a different value, `config.attention_dropout`) -- this ticket doesn't
   add a separate `--attention-dropout` flag. Identifies attention modules
   to skip via `isinstance(module, M2M100Attention)` (lazily imported from
   `transformers.models.m2m_100.modeling_m2m_100`), not a
   `type(module).__name__` string match -- PR #183 review flagged the
   name-based version as fragile against a future transformers version
   introducing an attention subclass (e.g. an SDPA/FlashAttention variant),
   which `isinstance` still correctly recognizes via the class hierarchy.
3. **`--bpe-dropout-alpha`** (default `None`, disabled) enables
   SentencePiece subword sampling ("BPE-dropout" / subword regularization,
   Kudo 2018, "Subword Regularization") for the *training* dataset's
   tokenization only, via `training.train.enable_subword_sampling`. Rather
   than baking `enable_sampling=True` into the shared tokenizer instance at
   construction time (`sp_model_kwargs`, which would also make post-
   training validation-set generation and any later re-evaluation
   non-deterministic), this monkeypatches `tokenizer._tokenize` only for
   the lifetime of each training-time encode call inside
   `TranslationDataset.__getitem__`, restoring the original immediately
   after -- every other user of the same shared tokenizer instance (vocab
   extension, `generate_translations`, `evaluation.evaluate_checkpoint`)
   is unaffected. Confirmed against the real `facebook/m2m100_418M`
   tokenizer that sampling genuinely produces different segmentations
   across repeated calls at a real alpha
   (`tests/integration/test_train_dataset_real_tokenizer.py`), not just
   that the right kwargs reach a fake. Deliberately independent of
   `training/vocab_extension.py`/`vocab_gap.py`/`subword_vocab.py`: none
   of those call `tokenizer.sp_model.encode` on the live fine-tuning
   tokenizer at all, so this needed no changes to that pipeline.
4. **`save_strategy="epoch"` is the new default** (previously `"no"` --
   no intermediate checkpoint ever existed for any past training run,
   making checkpoint averaging structurally impossible: there was nothing
   to average). Checkpoints write to `--checkpoint-dir`
   (`/opt/ml/checkpoints` by default), **not** under `--model-dir` -- see
   "Checkpoint storage: separated from the registered artifact (issue
   #187)" below for why that separation matters and how it's wired up.
   Paired with `save_total_limit=DEFAULT_SAVE_TOTAL_LIMIT`
   (`training.checkpoint_averaging.DEFAULT_AVERAGE_N + 2`, derived from --
   not independently hardcoded next to -- the averaging utility's own
   default `N`) so per-epoch checkpoints (each a full model + optimizer +
   scheduler + rng-state save) don't grow unbounded across a longer run
   and risk filling a training instance's disk; `transformers`' own FIFO
   eviction always keeps the most recent checkpoints, exactly the ones
   averaging needs. `training/checkpoint_averaging.py` is the new,
   independent post-hoc utility this enables: `find_checkpoint_dirs` /
   `select_last_n_checkpoints` locate a training run's last N
   `checkpoint-<step>` directories (sorted numerically, not
   lexicographically), and `average_checkpoints` elementwise-averages
   their `model.safetensors` weights (via `average_state_dicts`) into a
   new, standalone checkpoint directory, copying over the
   checkpoint-invariant non-weight files (`config.json`,
   `generation_config.json`, ...) from the most recent one and explicitly
   excluding training-progress files (`optimizer.pt`, `scheduler.pt`,
   `scaler.pt`, `rng_state.pth`, `trainer_state.json`,
   `training_args.bin`) that describe training progress, not the
   averaged weights themselves.
   `average_state_dicts` processes checkpoints one at a time via a running
   sum (accepting any `Iterable`, not just a `list`) rather than loading
   every checkpoint's full weights into memory simultaneously -- PR #183
   review flagged the original list-based version's peak memory scaling
   with N as a real risk for a utility meant to be run casually (e.g. a
   laptop) once N real ~2GB checkpoints are involved;
   `average_checkpoints` passes a generator expression, never a
   pre-built list, and a real, tracked-allocation test
   (`tests/integration/test_checkpoint_averaging_pipeline.py`) confirms an
   earlier checkpoint's weights are actually garbage-collected before a
   later one is loaded, not just that the final result looks correct.
   **Does not copy tokenizer files** -- `Seq2SeqTrainer`'s
   own per-checkpoint saves never include them (only
   `training.train.save_model_and_tokenizer`'s one final save, to the
   run's own `--model-dir`, does); copy those separately before loading an
   averaged checkpoint with `from_pretrained`. Only supports single-file
   (non-sharded) `model.safetensors` checkpoints -- a real, current
   limitation given every real training run's checkpoint fits comfortably
   under the sharding threshold for this model's size, not a hypothetical
   one designed around preemptively. Tested against tiny fixture
   checkpoint directories (`tests/unit/test_checkpoint_averaging.py`,
   `tests/integration/test_checkpoint_averaging_pipeline.py`), never a
   real checkpoint -- this item has no retraining cost by itself, so
   there's nothing that needs verifying against real weights the way, say,
   `--dropout`'s submodule-walking logic did. Run it directly:
   ```sh
   uv run python -m training.checkpoint_averaging \
     <model-dir>/checkpoints ./averaged-checkpoint --n 3
   ```

**All three tunable levers above (label smoothing, dropout, BPE-dropout)
default to whatever leaves existing behavior completely unchanged** (0.0
and `None`/`None` respectively) -- landing this ticket doesn't itself
change any future run's outcome unless a maintainer explicitly opts a
specific run into one of them. `submit_job.py --dropout`/
`--bpe-dropout-alpha` mirror `train.py`'s own flags and are omitted from a
submitted job's hyperparameters entirely (rather than passed through as a
literal `"None"`) when left at their defaults. `main()`'s own
`run_metadata["hyperparameters"]` reconstruction (used for
`register_model`'s `CustomerMetadataProperties`, ADR 0009) must mirror
`build_hyperparameters`' conditional inclusion of both flags exactly, or a
real run using either would train correctly but leave the registry entry
with no record of which experiment produced it -- PR #183 review caught
exactly this drift once already (a hardcoded key list that predated both
flags), and `tests/unit/test_submit_job.py`'s
`test_main_records_dropout_and_bpe_dropout_alpha_in_registry_metadata_when_given`
guards against it regressing again.

`training.train.shift_tokens_right` (the label-smoothing fix's core
building block) delegates to the real
`transformers.models.m2m_100.modeling_m2m_100.shift_tokens_right` rather
than maintaining an independent reimplementation of the same transform --
PR #183 review flagged the original reimplementation as a needless
drift risk against future transformers versions.

### Checkpoint averaging, evaluated against real checkpoints (issue #192)

Issue #182 built `training/checkpoint_averaging.py` as prep infrastructure
only -- it had never been run against a real, multi-epoch checkpoint set.
Real-world precedent (Xiao et al., "Revisiting Checkpoint Averaging for
NMT", AACL 2022, arXiv:2210.11803; the AmericasNLP 2024 shared task's
2nd/3rd-place teams for the closest comparable low-resource-Indigenous
translation setup) made this the next lever to actually evaluate, after
issue #182's own dropout/label-smoothing/bpe-dropout levers all
underperformed the v7 baseline (see v8/v9/v10 below).

**Phase 1: free-tier sanity check against v8's (`--dropout 0.3`)
already-existing per-epoch checkpoints.** `save_total_limit` (derived from
`checkpoint_averaging.DEFAULT_AVERAGE_N + 2`, see above) meant only the
last 5 of 13 epochs' checkpoints were actually retained in v8's training
artifact (steps 18513/20570/22627/24684/26741) -- `find_checkpoint_dirs`/
`select_last_n_checkpoints` correctly picked the last 3
(22627/24684/26741) out of those 5. Averaged via `average_checkpoints`,
then scored with `evaluation/evaluate_checkpoint.py` against the **same**
150-pair (300-example, both directions) random sample of `almg-v1`'s real
validation set for both the averaged and the plain final-epoch checkpoint,
so the comparison is apples-to-apples on identical methodology (not just
identical data):

| checkpoint | BLEU | chrF | es->cak BLEU/chrF | cak->es BLEU/chrF |
|---|---|---|---|---|
| final epoch only (n=300) | 11.3 | 33.8 | 12.5 / 34.8 | 9.3 / 32.4 |
| last-3 averaged (n=300) | 11.1 | 33.7 | 12.0 / 34.8 | 9.5 / 32.2 |

The final-epoch-only sample number (BLEU 11.3/chrF 33.8) closely tracks
v8's own full-validation-set (n=7,218) reported number (BLEU 11.7/chrF
34.3), confirming the 300-example sample is representative enough for a
sanity check. **The mechanics work end to end**: averaging ran, the
resulting checkpoint loaded and generated coherent-shaped output, and
issue #116/#125's word-boundary reconstruction matched exactly between the
two runs (30,996 tokens reconstructed both times, zero over/under-
reconstruction warnings once `model_card.md` was copied alongside the
averaged checkpoint -- see the bugfix note below). The averaged number
itself is flat-to-negligibly-lower than final-epoch-only, well within
n=300 sample noise -- **exactly what this phase was scoped to show
(mechanics, not a real signal): v8 is individually confounded by its own
already-negative `--dropout 0.3` lever**, so this is not read as evidence
against averaging in general.

**Two real, concrete findings from running this for real, not assumed:**

1. **`evaluate_checkpoint.py` needs `model_card.md` copied alongside a
   freshly-averaged checkpoint, or its word-boundary reconstruction
   silently uses the wrong (legacy, unscoped) construction.**
   `average_checkpoints` never copies `model_card.md` (a real
   `Seq2SeqTrainer` per-epoch checkpoint never has one to copy in the first
   place -- only the run's own final `--model-dir` does, see
   `training.train.save_model_and_tokenizer`), so
   `evaluate_checkpoint.py`'s `_read_checkpoint_vocab_extension_scoping`
   found nothing and fell back to the legacy "both columns" reconstruction
   -- for a `vocab_extension_scoping=kaqchikel_only` checkpoint like v8,
   this over-reconstructs (30,885 of 61,899 reconstructed tokens turned out
   to be missing from the checkpoint's real vocabulary), correctly tripping
   `_diagnose_word_boundary_reconstruction`'s over-reconstruction warning
   exactly as designed. Not a code bug (the diagnostic caught it as
   intended) but a real operational gap worth documenting here: **always
   copy the source run's own `model_card.md` into an averaged checkpoint's
   output directory, alongside the tokenizer files `average_checkpoints`
   already reminds you to copy**, before evaluating it.
2. **`training/checkpoint_averaging.py`'s `_EXCLUDED_FILENAMES` was
   missing `scaler.pt`** (the fp16 AMP grad-scaler's own state,
   confirmed present in a real v8 checkpoint -- every real GPU training run
   in this project sets `fp16=True`), so it was being silently copied into
   the averaged checkpoint's output directory alongside the correctly
   excluded `optimizer.pt`/`scheduler.pt`/`rng_state.pth`/
   `trainer_state.json`/`training_args.bin`. Harmless to a later
   `from_pretrained` load (an unrecognized file is simply ignored) but the
   same class of "training progress, not a fact about the averaged
   weights" leftover the exclusion set exists to keep out. Fixed
   test-first: `tests/integration/test_checkpoint_averaging_pipeline.py`'s
   fixture checkpoints now include a `scaler.pt`, and
   `test_average_checkpoints_never_copies_training_progress_files` asserts
   it's excluded too (confirmed red against the pre-fix code, then green
   after adding `"scaler.pt"` to `_EXCLUDED_FILENAMES`).

**Phase 2: clean matched-baseline run, isolating averaging as the only
variable.** A new training job (`run-20260929T000000Z-v11-matched-
baseline`) was submitted with a config verified byte-for-byte identical to
v7's (13 epochs, batch-size 8, learning-rate 5e-5, warmup-ratio 0.05,
weight-decay 0.01, gradient-accumulation-steps 4, subword-vocab-size 8000,
`almg-v1`, direction `both`, no dropout/label-smoothing/bpe-dropout
overrides) via `submit_job.py --dry-run`'s printed config before the real
submission, with `--max-run 32400` (9h, since v7 itself took ~6h11m and
the default 3h would have killed it mid-training). Completed in ~6h35m.
Scored final-epoch-only vs. last-3-averaged on the same fixed 150-pair
(300-example) sample of the real validation set used in Phase 1, same
decode config (`--num-beams 5`):

| checkpoint | BLEU | chrF | es->cak BLEU/chrF | cak->es BLEU/chrF |
|---|---|---|---|---|
| final epoch only (n=300) | 14.7 | 36.3 | 16.6 / 39.2 | 11.2 / 31.9 |
| last-3 averaged (n=300) | 13.7 | 35.7 | 15.7 / 39.1 | 10.1 / 30.5 |

**Conclusion: checkpoint averaging does not help here -- no-go.**
Final-epoch-only beats last-3-averaged on both metrics, both directions --
consistent in direction with Phase 1's sanity check, now with a clearer,
non-confounded gap on a genuinely clean baseline. This is a real, valuable
negative result (the same shape as issue #180's num_beams=8 finding), not
an inconclusive one: this project should **not** adopt checkpoint
averaging as a default post-training step.

### Checkpoint storage: separated from the registered artifact (issue #187)

Issue #192's conclusion above means checkpoint averaging is very unlikely
to actually be used going forward -- but until issue #187, the *cost* of
having per-epoch checkpoints available for it at all was being paid on
every single real training run, whether or not anyone ever averaged
anything: `training.train.build_training_arguments`'s `output_dir` (issue
#182's `save_strategy="epoch"`) was `<model_dir>/checkpoints`, i.e. *inside*
`SM_MODEL_DIR` -- the exact directory SageMaker tars whole into the
registered `model.tar.gz` artifact. Every per-epoch checkpoint (each a full
model + optimizer + scheduler + rng-state save, 2-3x the bare model's size
for AdamW) rode along into that artifact. Concretely, this caused:

- Every `submit_job.py` registration (`main()`'s default path, and every
  `--register-existing` call) downloading the full ~30GB artifact --
  measured at 28.4-30.5 GB across v8 through v16 -- just to read a few-KB
  `model_card.md`, costing 45min-1hr+ each time.
- A real deployment of issue #205's 20-epoch model (the first genuine
  quality improvement found since v7) failing outright: `deployment/
  package_model.py`'s `repackage_model_artifact` re-bundled the *entire*
  source artifact, checkpoints included, into the inference package, which
  the SageMaker inference container's disk couldn't decompress. Issue #211
  patched the symptom (excludes `checkpoints/` from the repackaged
  *inference* copy only) but explicitly left this issue open, since the
  *registration-time* artifact -- what `submit_job.py` downloads -- was
  still untouched and still ~30GB.
- Issue #192's own averaging evaluation above concluding averaging doesn't
  help this project's corpus/model size -- so the original justification
  for keeping multiple epoch checkpoints around at all was no longer even
  theoretically paying for itself.

**Fix: SageMaker's own native checkpoint-sync mechanism, not a hand-rolled
location inside the model artifact.** Verified against the real installed
SageMaker Python SDK v3 (`sagemaker==3.23.0`, the same `ModelTrainer` this
project already migrated to in #155/#172) rather than assumed from v2-era
documentation -- `ModelTrainer` accepts a `checkpoint_config`
(`sagemaker.core.training.configs.CheckpointConfig(s3_uri=..., local_path=
...)`). SageMaker Training Jobs continuously sync whatever local directory
`local_path` names to `s3_uri` *during* training, entirely independent of
the final `output_data_config`/`model.tar.gz` artifact upload -- confirmed
against AWS's own "SageMaker AI environment variables and the default
paths for training storage locations" reference page, which explicitly
documents `/opt/ml/checkpoints` as "Writes to S3 during training: Yes" /
"Writes to S3 when job is terminated: No" (i.e. never folded into the
final artifact at all) and, notably, as the *one* storage location in that
table with **no** dedicated `SM_*` environment variable -- every other row
(`SM_CHANNEL_*`, `SM_OUTPUT_DIR`, `SM_MODEL_DIR`) has one.

- `training/train.py` gained a `--checkpoint-dir` flag
  (`training.train.DEFAULT_CHECKPOINT_DIR`, hardcoded to
  `/opt/ml/checkpoints` -- not read from an `SM_*` environment variable,
  since none exists for this path, unlike `--model-dir`/`--output-data-dir`).
  `build_training_arguments`'s `output_dir` now points there instead of
  `<model_dir>/checkpoints`.
- `training/submit_job.py`'s `build_estimator` configures
  `ModelTrainer(checkpoint_config=CheckpointConfig(s3_uri=..., local_path=
  DEFAULT_CHECKPOINT_DIR))`, importing `DEFAULT_CHECKPOINT_DIR` from
  `training.train` rather than hardcoding a second, independent copy of the
  same literal that could silently drift apart from `train.py`'s own CLI
  default. The S3 destination (`build_checkpoint_s3_uri`) is
  `s3://<training-data-bucket>/model-checkpoints/<run-id>/` -- a prefix
  entirely separate from `model-artifacts/` (where the registered
  `model.tar.gz` lands), scoped per run-id so concurrent/successive runs'
  checkpoints never collide. Visible in `--dry-run`'s printed config too.
- `training/checkpoint_averaging.py`'s own docstring and CLI `--help` now
  point at this new S3 location (`aws s3 sync
  s3://<bucket>/model-checkpoints/<run-id>/ ./checkpoints` before running
  the averaging tool against the downloaded directory) instead of "inside
  the downloaded Model Registry artifact" -- the tool itself needed **no**
  code change, since `checkpoints_dir` was always a plain CLI argument, not
  hardcoded to the old `<model-dir>/checkpoints` layout. The tool stays
  fully available for a future revisit despite #192's negative result --
  this ticket doesn't remove it, only fixes where its input comes from.
- `deployment/package_model.py`'s `repackage_model_artifact` keeps its
  issue #211 `checkpoints/` exclusion (defensive, for any older artifact
  still being repackaged) -- confirmed via the existing
  `test_repackage_model_artifact_preserves_original_model_files` test (a
  fixture tarball with **no** `checkpoints/` subtree at all) that this is
  now a genuine no-op for any run submitted after this fix, not a
  load-bearing filter.

**Net effect**: a training job's registered `model.tar.gz` is back to
model + tokenizer + `model_card.md` only (roughly ~2GB, not ~30GB) for
every run submitted after this fix. Checkpoints are still real and
retrievable for a future averaging revisit -- just from
`s3://<bucket>/model-checkpoints/<run-id>/` instead. Verified test-first
and via `submit_job.py --dry-run`'s printed config (no real training job
submitted as part of this ticket -- see its own acceptance criteria: this
is artifact plumbing, not a model-quality change, so a real run wasn't
required to confirm it).

As a side benefit, this run's full-validation-set score (BLEU 13.5/chrF
36.6, registered as Model Package v11, then rejected -- it was a reference
run for this comparison, not a deployment candidate) gives the project's
first real estimate of run-to-run variance for this exact training config:
roughly 0.5 BLEU / 0.1 chrF of spread from v7's own numbers (14.0/36.5)
with identical hyperparameters including `--seed 42`.

This run also surfaced a real, separate bug: it was submitted with
`--register-model true` (issue #190/#191's self-registration path), but
self-registration failed inside the container with `NoRegionError: You
must specify a region` (`_boto3_client_for_registration` never passes an
explicit region) -- filed as issue #196. It failed *safely*: the training
job itself completed successfully rather than being marked `Failed`,
confirming PR #191's error-isolation fix works as designed. Registered
manually via `submit_job.py --register-existing` instead.

This project's local development GPU (the same 4GB GTX 1050 Ti referenced
under "Real at-scale validation" above) could not be used for this
ticket's evaluation runs at all: `torch.cuda.is_available()` returns
`False` in this environment, not because of thermal throttling this time,
but because the installed driver (CUDA 12.4) is older than what this
project's pinned `torch==2.14.0+cu130` build requires -- a different
failure mode than issue #180's throttling, confirmed directly (`torch.cuda
.is_available()` returns `False`, `nvidia-smi` still shows the card).
Both Phase 1 eval runs above ran on CPU instead; a 300-example sample took
roughly 30 minutes each, consistent with the ~12-hour full-validation-set
(7,218 examples) CPU runtime already documented in
`evaluation.evaluate_checkpoint._move_model_to_cuda_if_available`'s own
docstring (issue #170).

**`--dry-run`** resolves the real `{Environment}-Data` CloudFormation stack
outputs (a free, read-only call), resolves the training container image URI
(a free, local-only lookup against the installed SDK's compatibility
tables), and prints the full would-be job config -- resolved bucket/role,
instance type, image version combination/URI, hyperparameters, and channel
S3 URIs -- as JSON, without ever constructing a `ModelTrainer` or calling
`.train()`. This is deliberate, not just an optimization: constructing a
real `ModelTrainer` with a `role` triggers a live `iam:SimulatePrincipalPolicy`
AWS call to validate that role (see `submit_job.py`'s module docstring), so
building one during `--dry-run` could fail (or misleadingly succeed)
depending on the caller's ambient AWS credentials. This is what the
maintainer runs first to sanity-check before the real submission.

**Channel path convention**: the `train`/`validation` channels point at the
exact corpus object keys, `s3://<bucket>/corpus/almg/v1/train.tsv` and
`.../val.tsv` -- not the whole prefix. Combined with SageMaker's standard
`/opt/ml/input/data/<channel>/<s3-object-basename>` download convention,
this makes the container-side paths deterministic:
`/opt/ml/input/data/train/train.tsv` and
`/opt/ml/input/data/validation/val.tsv`. `submit_job.py` passes those exact
paths as the `--train`/`--validation` hyperparameters to `train.py`.

`submit_job.py`'s module docstring documents the SageMaker Python SDK v3
migration (issue #155, GHSA-5r2p-pjr8-7fh7 -- `ml/pyproject.toml` now pins
`sagemaker>=3.4,<4`): `sagemaker.huggingface.HuggingFace` ->
`sagemaker.train.ModelTrainer`, `sagemaker.inputs.TrainingInput` ->
`sagemaker.core.training.configs.InputData`, `sagemaker.image_uris` ->
`sagemaker.core.image_uris`, each verified against the real installed
package rather than assumed. It also documents how the
`transformers_version`/`pytorch_version`/`py_version` combination
(`4.56.2`/`2.8.0`/`py312`) was derived from the installed SDK's own
HuggingFace DLC compatibility table rather than guessed by hand -- re-check
that derivation after any future `sagemaker` upgrade.

## Full-corpus data curation sweep (issue #199)

After #182's three cheap-tier hyperparameter levers (dropout, label-
smoothing, bpe-dropout) and #192's checkpoint-averaging experiment all came
back no-go, follow-up research concluded the regularization/modeling-
tuning direction is exhausted for this dataset size and pointed at data
curation as the likelier next lever (the AmericasNLP 2024 shared task's own
stated conclusion for the same problem shape, and the specific edge the
BSC team credited for their own strongest AmericasNLP 2024 result: length-
based filtering, deduplication, and embedding-based alignment filtering).
This ticket ran that sweep for real, at full corpus scale (32,906 train /
3,609 val pairs, `almg-v1`), rather than against a sample.

**Status when this PR was opened: Phase 1 (sweep, research, decision, and
the cleaned corpus) is done; Phase 2 (the real training-run comparison) was
submitted and still in flight.** Mirrors issue #192's own Phase 1/Phase 2
PR split (#194/#197) -- see "Follow-up needed" at the end of this section.

### 1. OCR/digitization glued-word artifact sweep: no evidence found

Issues #116/#125 found systematic word-boundary space loss in *generated
model output* (e.g. "dedios", "queestá", "deguatemala", "parachoch",
"queestaban"), root-caused to a `tokenizer.add_tokens()` spacing bug and
fixed. That finding was against decoded output, never checked against the
raw corpus text itself. `data.glued_word_signal` (new this ticket) ran two
independent checks against the full real corpus (`corpus/almg/v1`, both
`train.tsv` and `val.tsv`):

1. **Known-pattern sweep** (the exact strings above): **zero occurrences**
   in either split. The previously observed glued-word artifacts are
   confirmed to be a pure decode-time phenomenon (already fixed by #116),
   not a property of the source corpus text.
2. **General heuristic** (a rare word starting with a common Spanish
   clitic, whose remainder after stripping that clitic is itself
   independently frequent in the corpus -- a conservative proxy for "clitic
   glued onto a word the corpus already knows well"): 22 candidates on
   `train.tsv`, 2 on `val.tsv`, out of 26,584 distinct Spanish word forms.
   Manually inspecting every one of these 24 candidates confirmed **all of
   them are ordinary, correctly-spelled Spanish words** that happen to
   start with a common clitic and coincidentally have an independently
   frequent remainder (the same shape as e.g. an invented illustrative
   case: "entierra" -- a real verb form -- superficially matching "en" +
   "tierra", both independently common, without being a glued artifact at
   all). Zero of the 24 are genuine OCR/digitization artifacts.

**Conclusion: no OCR/digitization glued-word contamination exists in the
raw corpus at full scale, by either check.** This refutes this ticket's own
opening premise (which conflated #116's decode-time bug with a corpus-data
quality issue) -- flagging that explicitly per this project's "say so
explicitly rather than overstating" convention, rather than inventing a
fix for a problem that isn't actually present in the data. No corpus text
is mechanically "de-glued" or dropped for this reason. `data/glued_word_signal.py`
is kept as a one-off diagnostic tool (like `data/dialect_signal.py`), not
wired into `data/pipeline.py`'s cleaning chain -- there is nothing for it
to fix.

### 2. Embedding-based alignment-quality filtering: no viable model found

Per the BSC AmericasNLP 2024 precedent, the plan was to score each pair's
source/target semantic similarity with a multilingual sentence embedding
model and flag/drop low-similarity (likely misaligned) pairs. The open
question issue #199 itself posed -- does any available multilingual
sentence embedding model have real Kaqchikel-language signal -- was
checked directly against primary sources, not assumed:

- **LaBSE** (Feng et al. 2022, 109 languages): Kaqchikel is not in its
  language list.
- **NLLB-200 / FLORES-200** (Meta AI, 200 languages): confirmed directly
  against the `facebookresearch/flores` `flores200` language table --
  Kaqchikel and every other Mayan language are absent. The closest
  indigenous-language coverage NLLB-200 has is Quechua and Guaraní (the
  same two languages BSC's own AmericasNLP 2024 embedding-filtering
  success was measured on -- languages the embedding model actually has
  training coverage for, unlike Kaqchikel).

Neither major embedding model family has ever seen Kaqchikel in training.
Rather than stop at "no confirmed coverage, skip it", `data.alignment_signal`
turned this into an empirical, real-corpus question: embed a random
300-pair sample of `almg-v1` train with `sentence-transformers/LaBSE`
(real weights, real corpus text -- a maintainer-run diagnostic, not part
of the test suite) and compare real-pair cosine similarity against a
shuffled-control baseline (each Spanish sentence paired with a random,
definitely-wrong Kaqchikel sentence instead of its real counterpart):

| | mean | median |
|---|---|---|
| Real es<->cak pairs | 0.334 | 0.338 |
| Shuffled control (definitely misaligned) | 0.249 | 0.261 |

**AUC (real pair similarity > shuffled control): 0.656** -- better than
chance (0.5), but far below what a model with genuine coverage of both
languages would show for this kind of check (LaBSE alignment-filtering
use on languages it was actually trained on typically separates real from
shuffled pairs with AUC well above 0.9). The weak-but-nonzero signal is
consistent with LaBSE picking up shared surface features (shared Spanish
loanwords/proper nouns, punctuation, length) rather than genuine
cross-lingual semantic understanding of Kaqchikel content.

**Conclusion: no viable embedding model with adequate Kaqchikel coverage
exists today.** The measured signal is real but too weak and unreliable to
use as a hard corpus filter -- at this separation level, a similarity
threshold would drop legitimately well-aligned but unusually-phrased pairs
about as often as it would catch genuine misalignment, a bad trade for an
already-small corpus. This project relies on corpus-internal consistency
checks instead -- the existing length-ratio filter (`data.length_filter`,
see below), matching the fallback issue #199 itself anticipated. If a
Kaqchikel-aware (or broader Mayan-language-aware) embedding model is ever
published, this diagnostic (`data.alignment_signal.analyze_alignment_signal`)
can be re-run against it directly to re-check this conclusion.

### 3. Revisiting #51's ~19x length/register gap

Issue #51 previously concluded (after ruling out a genuine dialect split)
that the corpus's ~19x sentence-length gap between two computationally-
clustered groups most likely reflects at least two differently-registered,
unlabeled source documents merged into the corpus (short glossary/
dictionary-style entries vs. long connected prose), deferred as "noted but
not shown to need action" while modeling-side gains were still available.
With that avenue now exhausted (per #182/#192's findings) and both (1) and
(2) above checked directly:

- **The OCR-artifact explanation is now positively ruled out**, not just
  unconsidered: (1) found zero evidence of glued-word corruption anywhere
  in the raw corpus, so the length/register gap cannot be an artifact of
  OCR/digitization noise. #51's original "two source-document
  registers" explanation stands.
- **Real length statistics** (`almg-v1` train, post-normalization): median
  sentence length is just 12 characters on both sides (consistent with
  #51's "short glossary-style entries" majority), but the distribution has
  a real long tail -- p95 is 161 characters (Spanish) / 287 characters
  (Kaqchikel), and the maximum on either side exceeds 1,600 characters.
  Source/target length ratio has median 1.50 and p90 2.64 (most pairs are
  reasonably length-matched), but 2.06% of pairs exceed a 4x ratio and
  0.09% exceed 10x -- a real, if small, tail of extreme outliers
  consistent with #51's reported ~19x figure.
- **This is a real, structural feature of the corpus, not itself evidence
  of misalignment.** A short, correctly-aligned dictionary-style entry
  (e.g. a single word or short phrase gloss) is still a valid, useful
  training pair for an already data-starved language -- dropping it just
  for being short/differently-registered from a prose sentence would
  discard real signal, not noise, the opposite of what issue #199's own
  "quality/quantity tradeoff" framing calls for.

**Decision: don't drop pairs for being short/long or differently
registered per se. Do apply the existing, already-built, never-yet-run-at-
full-scale length-ratio/length-bound filter** (`data.length_filter`'s
default `LengthFilterConfig`: `min_length=1`, `max_length=400`,
`max_ratio=4.0`) to catch the genuine extreme-outlier tail (very likely
real misalignment -- e.g. a one-word gloss paired with a full paragraph --
rather than legitimate register variation), via `data.pipeline`'s existing
`clean` command. This was deliberately *not* run when `almg-v2` was
produced (issue #90, normalization-only, specifically to avoid conflating
the normalization fix with this still-open length/register question) --
this ticket is the point where that question is resolved enough to run it.

### Decision: `almg-v3` -- normalize + dedup + length-filter, no other changes

Applied `data.pipeline`'s existing `clean_corpus_file` (normalize, dedup,
length-filter, in that order -- unchanged logic, first real run at full
corpus scale) to `almg-v1`'s raw train/val TSVs, uploaded as
`s3://<training-data-bucket>/corpus/almg/v3/{train,val}.tsv`:

| split | before | after dedup | after length-filter | total dropped |
|---|---|---|---|---|
| train | 32,906 | 32,906 (0 exact dupes) | 31,856 | 1,050 (3.2%) |
| val | 3,609 | 3,609 (0 exact dupes) | 3,497 | 112 (3.1%) |

Zero exact-duplicate `(source, target)` pairs existed at the pair level (a
real, useful negative result in itself -- dedup was worth actually running,
not assumed to be a no-op). The length-filter drop (~3.2%/~3.1%) removes
only the genuine extreme-outlier tail identified above, keeping the corpus
size reduction modest relative to an already-small 33k-sentence corpus.
`data.pipeline validate-split` confirms `almg-v3` still has zero exact-pair
leakage between train/val (958 source-side / 684 target-side one-sided
overlaps remain, in the same ballpark as `v1`/`v2`'s 977/726 -- expected,
since dedup/length-filtering isn't targeted at that kind of overlap).

**This is a normalize + dedup + length-filter reprocessing of `almg-v1`
(not built on top of `almg-v2`, though the two are equivalent in their
shared normalization step, since `data.normalize.normalize_text` is
idempotent)**, so `almg-v3`'s provenance is self-contained: raw `v1` input,
`data.pipeline clean`'s unchanged, already-tested logic, nothing else.

`training/submit_job.py` gained a new `--corpus-prefix` override (this
ticket) so a one-off comparison run like this doesn't require editing the
hardcoded `CORPUS_PREFIX` default -- production still defaults to `v1`
until a deliberate switchover decision, made only after Phase 2's real
result is in (see "Follow-up needed" below), avoiding the exact
`CORPUS_PREFIX`/`--corpus-version` drift this README already warns about
elsewhere.

### Phase 2: real training-run comparison

Submitted a training job identical to the v7 baseline's config in every
way except the corpus (verified via `submit_job.py --dry-run` before the
real submission, same discipline as #192's Phase 2): 13 epochs,
batch-size 8, learning-rate 5e-5, warmup-ratio 0.05, weight-decay 0.01,
gradient-accumulation-steps 4, subword-vocab-size 8000, direction `both`,
seed 42, no dropout/label-smoothing/bpe-dropout overrides -- corpus
changed from `almg-v1` to `almg-v3` only. Completed in ~5h55m, registered
as Model Package v12:

| Version | Corpus | BLEU | chrF |
|---|---|---|---|
| v7 (baseline) | `almg-v1` | 14.0 | 36.5 |
| v11 (clean reproduction, #192's Phase 2) | `almg-v1` | 13.5 | 36.6 |
| v12 (cleaned corpus) | `almg-v3` | 13.7 | 36.7 |

**Conclusion: the cleaned corpus does not show a clear improvement.**
v12's numbers fall inside the run-to-run variance band v11 already
established for this exact hyperparameter config (v11 exists specifically
to measure that variance -- see #192's Phase 2) -- chrF marginally higher,
BLEU marginally lower, no signal either direction. `almg-v1` remains the
corpus version in production use; `submit_job.py`'s `CORPUS_PREFIX`/
`CORPUS_VERSION` defaults are unchanged, `almg-v3` is not adopted. v12
rejected in Model Registry with this reasoning recorded; v7 remains the
deployed model.

**Acknowledged methodological gap, not fully resolved**: this compares
v12's `almg-v3`-val score against v7's original `almg-v1`-val score, not a
byte-for-byte identical validation set (`almg-v3`'s val set is 3,497 pairs
after Phase 1's length-filter dropped ~3.1% of `almg-v1`'s original 3,609).
A fully isolated re-score of v7 against `almg-v3`'s own validation set was
judged not worth the additional ~45min-1hr of work, since the effect size
here is small either way (well within the v7/v11 noise band) and very
unlikely to change the qualitative conclusion -- a deliberate, documented
tradeoff, not a hidden gap.

### A second, more serious self-registration failure surfaced by this run

This run's own self-registration attempt failed too, but with a
**different** error than #196's `NoRegionError` (which #198 already
fixed -- `region` correctly appeared in this run's hyperparameters):

```
WARNING: self-registration failed, but the training run itself succeeded
(model card at /opt/ml/model/model_card.md): An error occurred
(ValidationException) when calling the CreateModelPackage operation:
Cannot find S3 object: model-artifacts/.../output/model.tar.gz in bucket
traductor-kaqchikel-training-data-dev.
```

This reveals a structural flaw in #190's entire design, not a one-line
bug: `CreateModelPackage` validates S3 object existence at call time in
practice (contradicting the assumption #190's design rested on, which was
checked against AWS's static API reference during #191's review but not
against real runtime behavior) -- and self-registration runs *inside* the
training container's own process, before that process exits, which is
*before* SageMaker uploads the artifact to S3. This isn't fixable with
another quick patch; filed as issue #201 for an architecture decision.
Registered this run manually via `submit_job.py --register-existing` in
the meantime, same as #196 required for the previous run.

**Resolved by ADR 0009**: issue #190's in-container self-registration was
reverted entirely (never fixable, per the structural reason above) in
favor of restoring the pre-#190 client-side registration this section's
`--register-existing` fallback already exercised successfully on both
real runs. See
[`docs/adr/0009-model-registry-registration-timing.md`](../docs/adr/0009-model-registry-registration-timing.md)
for the full decision record and "Job submission" above for the current
(again client-side) registration flow.

## Compositional embedding initialization for new Kaqchikel tokens (issue #218)

Every tuning lever tried across five rounds (epochs, regularization,
checkpoint averaging, corpus cleaning) plateaued or came back negative (see
"Checkpoint averaging" and "Full-corpus data curation sweep" above). A
research pass identified a cheaper, less-tried lever: *how* a new token's
embedding row is initialized before fine-tuning, not just the training
config around it. A case study on adding an unseen low-resource language
(Limbum) to a multilingual NMT model's vocabulary found that embedding-
initialization strategy "significantly impacts translation quality", with
linguistically-informed initialization beating naive initialization on
BLEU/chrF -- the exact magnitude wasn't extractable from that source, so
this is a worthwhile cheap experiment, not a confident bet.

### What the current (and still default) strategy actually does

Checked directly, not assumed: `training.tokenizer_extension.
resize_embeddings_for_new_tokens` delegates to `training.vocab_extension.
resize_embedding_matrix`, which initializes **every** new token's row as
the mean of **all** existing rows, plus a small amount of noise -- one
global average applied uniformly to every new token regardless of its own
linguistic composition (which characters/subwords it's built from).

### The alternative: `resize_embedding_matrix_compositional`

`training.vocab_extension.resize_embedding_matrix_compositional` instead
decomposes each new token into smaller pieces already present in the base
vocab (`decompose_token_into_known_pieces`, a greedy longest-match
segmentation against the vocab's own keys -- deliberately simple and
deterministic, not a true Viterbi/unigram-optimal segmentation, but
sufficient to recover a composition-aware signal), and initializes its row
as the mean of those pieces' *own* existing embeddings.

**Same-batch siblings can compose from each other.** `training.vocab_gap.
find_missing_characters`/`find_missing_words` (wired together by
`extend_tokenizer_vocab`) always add a genuinely new character (e.g.
Kaqchikel "ä", entirely absent from M2M100's base vocab) in the *same*
batch as the longer whole-word tokens built out of it -- without
accounting for this, those words could never compose at all (the base
vocab alone has no row for "ä" yet), and nearly every realistic Kaqchikel
word would fall straight back to the global mean, defeating the point.
`resize_embedding_matrix_compositional` processes a batch's new tokens in
increasing-length order internally (not the order given), so a resolved
single character becomes a legitimate piece for any longer sibling token
in the same batch that contains it. A token that still can't be fully
decomposed -- even with sibling help -- falls back to the same
global-mean-plus-noise strategy, per token, so every new token still gets
a real, in-distribution starting row (never left at zero/uninitialized).

Both strategies are available side by side, never a silent replacement:
`--embedding-init-strategy` (`training/train.py`, `training/submit_job.py`)
defaults to `"mean"` (byte-for-byte the original behavior -- confirmed by
the full existing test suite passing unchanged), with `"compositional"` as
an explicit opt-in. Every run's model card records which strategy it used
(`embedding_init_strategy`, always present, unlike `dropout`/
`bpe_dropout_alpha`'s "omit unless set" convention -- this always has a
real, meaningful value) and, only for the compositional strategy,
`embedding_init_compositional_coverage` (`<decomposable>/<total new
tokens>`, via `training.vocab_extension.count_decomposable_tokens` --
itself sibling-aware, mirroring the real algorithm's own processing order
so it doesn't understate coverage). **Caveat (code review on PR #219):**
"decomposable" here means "fully segmentable into known piece keys",
*including* a same-batch sibling that itself fell back to the global mean
-- so this stat is an upper bound on "genuinely composed from real
pretrained pieces", not proof every counted token avoided the global mean
entirely (see `count_decomposable_tokens`'s own docstring for the full
explanation).

### Test coverage (test-first, per `docs/testing.md`)

Pure, framework-free unit tests (`tests/unit/test_vocab_extension.py`):
decomposition (longest-match preference, single-character fallback,
undecomposable-token handling, a token already equal to a vocab key),
`compositional_row_for_token`, `count_decomposable_tokens` (including the
sibling-assisted case), and `resize_embedding_matrix_compositional` itself
(shape/dtype/row-preservation parity with `resize_embedding_matrix`,
composed-vs-global-mean center selection, the sibling-composition case and
its order-independence, and the undecomposable-token fallback). Strategy
selection/validation in `training.tokenizer_extension.
resize_embeddings_for_new_tokens` (mean remains the default with zero
behavior change, compositional requires `added_tokens`/`base_vocab` and
rejects a token-count mismatch, an unknown strategy raises) is unit-tested
against a fake model using real `torch` tensors
(`tests/unit/test_tokenizer_extension.py`). `training.train`'s CLI/wiring
is covered in `tests/unit/test_train_args.py` and
`tests/integration/test_train_pipeline.py` (model card recording for both
strategies, and -- the real point of this ticket -- a test proving the two
strategies actually produce *different* embedding rows for the same run,
not just a label with no behavioral effect).

**Confirmed against the real `facebook/m2m100_418M` checkpoint, not just
synthetic fixtures** (`tests/integration/test_tokenizer_extension_real_model.py`):
extending the tokenizer with a small real Kaqchikel sample found **24 new
whole-word/character tokens, 24 of which (100%) were fully decomposable**
into known piece keys (with sibling assistance in several cases, e.g.
words containing "ä" or glottal-apostrophe-adjacent characters composing
once those characters' own new rows were resolved) -- confirming this
strategy finds real, structurally composable signal against M2M100's
actual vocabulary, not only a toy fixture's deliberately tiny one. As
`count_decomposable_tokens`'s own docstring caveats, "fully decomposable"
here is not the same claim as "entirely built from genuine pretrained
embeddings" -- a sibling-assisted decomposition can itself include a
token that fell back to the global mean, so 100% is an upper bound on
genuine composition, not a measurement of it. This is also a small
sample, not a claim that 100% coverage holds across the full
33k-sentence corpus's much more varied vocabulary -- the real
training-run comparison below is what actually measures quality impact.

### Controlled comparison: real training runs, same config except this one lever

Per this ticket's acceptance criteria, isolating the strategy as the only
variable against the established 13-epoch reference point (same discipline
as "Checkpoint averaging" and "Full-corpus data curation sweep" above):
13 epochs, batch-size 8, learning-rate 5e-5, warmup-ratio 0.05,
weight-decay 0.01, gradient-accumulation-steps 4, subword-vocab-size 8000
(issue #217's vocab-size sweep was still in progress at the time this
ticket ran, per its own explicit scope note -- using the current default
rather than waiting on or combining with that result), direction `both`,
seed 42, no dropout/label-smoothing/bpe-dropout overrides, corpus
`almg-v1` -- identical to the v7/v11 baseline config, changing only
`--embedding-init-strategy`. Both job configs were validated via
`submit_job.py --dry-run` (confirmed identical except that one
hyperparameter), then actually submitted one at a time, re-checking `aws
sagemaker list-training-jobs --status-equals InProgress` immediately before
each (issue #217's own vocab-size sweep -- concluded no change to the
8,000 default, see PR #230 -- had been occupying the project's single
`ml.g4dn.xlarge` slot when this ticket's implementation first finished;
submission resumed once that slot freed up):

```sh
cd ml
export AWS_PROFILE=translator-dev AWS_REGION=us-east-1
uv run python -m training.submit_job --no-wait \
  --run-id run-embinit-mean-13ep \
  --epochs 13 --subword-vocab-size 8000 --embedding-init-strategy mean \
  --max-run 32400
uv run python -m training.submit_job --no-wait \
  --run-id run-embinit-compositional-13ep \
  --epochs 13 --subword-vocab-size 8000 --embedding-init-strategy compositional \
  --max-run 32400
```

### Real results

Both jobs completed successfully (~6h17m and ~6h20m respectively) and were
registered in Model Registry (`--register-existing`):

| Version | `--embedding-init-strategy` | BLEU | chrF | es->cak BLEU/chrF | cak->es BLEU/chrF |
|---|---|---|---|---|---|
| v11 (prior reference, #192's Phase 2) | *(pre-#218, equivalent to `mean`)* | 13.5 | 36.6 | -- | -- |
| v22 (this ticket) | `mean` | 13.5 | 36.6 | 16.4/39.0 | 9.2/33.0 |
| v23 (this ticket) | `compositional` | **13.8** | **37.0** | **17.1/39.5** | 9.1/33.1 |

The `mean` run (v22) reproduces v11's combined BLEU/chrF exactly (13.5/36.6)
-- expected, since `--embedding-init-strategy mean` is bit-for-bit the same
code path as every pre-#218 run, and both used the identical seed (42) and
hyperparameters. This also means v22 is a near-zero-noise baseline for this
specific comparison (unlike v7-vs-v11's own 0.5 BLEU / 0.1 chrF spread,
which reflects whatever actually differed between those two runs).

The `compositional` run (v23) scores **+0.3 BLEU / +0.4 chrF** over v22,
isolating this ticket's one changed variable. The gain is concentrated in
es->cak (+0.7 BLEU / +0.5 chrF); cak->es is essentially flat (-0.1 BLEU /
+0.1 chrF) -- an asymmetry this single comparison doesn't explain.

**Real compositional coverage on the full corpus, not just a small
sample**: v23's own model card records `embedding_init_compositional_coverage:
36857/36859` -- effectively the entire real vocabulary extension (99.99%)
got a genuinely composed row, not a global-mean fallback. (Per this
stat's own documented caveat above, "decomposable" technically allows a
sibling-assisted token to inherit a fallback-to-mean piece transitively --
but at this coverage level, that caveat describes a vanishingly small edge
case here, not a material qualifier on the headline number.)

### Conclusion: promising, but not (yet) clearing this project's own bar for a default change

This is the first cheap-tier lever in this project's experiment history
(after label smoothing, dropout, bpe-dropout, checkpoint averaging, and
corpus cleaning -- all flat or negative) to move **both** BLEU and chrF in
the same, positive direction in a single comparison. That's a genuinely
encouraging result, and the near-universal real-corpus coverage confirms
the mechanism works as designed, not just on a small fixture.

**Being honest about what one comparison run can and can't show**: the
+0.3 BLEU delta sits inside the only run-to-run noise band this project has
previously measured for this exact hyperparameter config (v7 vs. v11: 0.5
BLEU) -- on BLEU alone, this result doesn't clearly clear the bar issue
#199's corpus-cleaning experiment used to reject a similarly-sized, mixed-
direction effect. The +0.4 chrF delta is larger than that same previously-
measured chrF noise (0.1), which is a more encouraging signal, but it's a
single data point, not a repeated-seed measurement for *this* comparison
specifically -- neural network training is not perfectly reproducible
across different initial embedding values even with a fixed data seed, so
part of this delta could still be ordinary training-trajectory variance
rather than a real effect of the init strategy itself. No repeated-seed
run was performed for either arm of this specific comparison (that would
be a second pair of ~6h, real-money jobs, and this ticket's scope was one
isolated comparison, not a variance study).

**Decision: keep `compositional` as a tested, available opt-in; do not
promote it to the default yet.** This is consistent with this project's
existing bar (every other experiment section above required clearing an
established noise band before changing a default, not just pointing in
the right direction) -- applied evenly rather than relaxed because this
result happens to be the first encouraging one. `--embedding-init-strategy`
keeps defaulting to `mean`; nothing about existing training behavior
changes as a result of this ticket. Recommended next step, **not executed
as part of this ticket** (a second real, billable training-job pair,
needing explicit go-ahead): re-run this same comparison with a different
seed to get an actual noise-floor estimate for the `compositional` arm
specifically, which is the missing piece needed to tell this result apart
from noise with confidence. If a second run confirms the same direction,
promoting `compositional` to the default would be well-supported;
`training.vocab_extension.EMBEDDING_INIT_COMPOSITIONAL`/`--embedding-
init-strategy compositional` already exist and are fully tested, so that
promotion would be a one-line default change whenever that evidence
exists -- not a new implementation effort.

## Serving (`deployment/`)

Issue #8 stands up the actual SageMaker Serverless Inference endpoint
(ADR 0001) serving a fine-tuned checkpoint. Two things had to be solved
that a default SageMaker Hugging Face deployment doesn't handle out of the
box: (1) this model's translation *direction* is controlled by a custom
tag mechanism (`training/direction.py`), not M2M100's built-in language
codes, so a default `text2text-generation` pipeline deployment would
silently ignore it; (2) SageMaker Model Registry model packages have no
first-class "separate code location" field, unlike the `Estimator ->
Model` flow's `source_dir`/`entry_point`, so a custom handler has to be
bundled *inside* the deployed model artifact itself.

### Request/response contract

`deployment/inference.py` implements the SageMaker Inference Toolkit's
`model_fn`/`input_fn`/`predict_fn`/`output_fn` hooks. This is the contract
issue #9's `apps/api` codes against:

Request (`Content-Type: application/json`):

```json
{"source_lang": "es", "target_lang": "cak", "text": "Buenos días"}
```

- `source_lang`/`target_lang`: `"es"` or `"cak"`, must differ from each
  other.
- `text`: non-empty string (surrounding whitespace is stripped) in the
  language named by `source_lang`.

Response (`Accept: application/json`):

```json
{"translated_text": "Utz sq'ij"}
```

Malformed/invalid requests (missing field, unsupported language code,
empty text, same source/target, wrong content type) raise `ValueError`
from `input_fn`/`parse_request`, which SageMaker surfaces as a client
error response -- confirmed against the real deployed endpoint, not
assumed (see "Real deployment verification" below).

`translate()`'s beam width (`num_beams`, issue #180) is a plain module-level
constant, `deployment.inference.DEFAULT_NUM_BEAMS` (currently 5, matching
`training.train.DEFAULT_NUM_BEAMS` and the base model's own
`generation_config.json`), not a request-level field -- there's no
per-request decode-quality/latency tradeoff exposed to callers, only a
single fixed decode config for the whole deployed endpoint. Adopting a
different value here (see "Decode configuration" under the training
entrypoint section for the real at-scale validation this was based on) is
a one-line change to this constant, requiring only re-registering/
re-packaging the same v7 weights -- no retraining, no new Model Package
version.

### Why the model artifact is repackaged (`deployment/package_model.py`)

The SageMaker SDK's `HuggingFaceModel` convenience class (removed
entirely in v3, but the same reasoning applied to it in v2 too) has
`entry_point`/`source_dir` kwargs designed for the `Estimator -> Model`
flow, which upload a *separate* code tarball referenced via a
`SAGEMAKER_SUBMIT_DIRECTORY` environment variable. A Model Registry model
package's
`InferenceSpecification.Containers[]` only has `Image`, `ModelDataUrl`,
and `Environment` fields -- no first-class separate-code-location concept
-- so this project's registry-based deployment instead bundles the
inference code *inside* `ModelDataUrl`'s own tarball, under a top-level
`code/` directory, which the SageMaker Hugging Face Inference Toolkit
auto-detects. `build_inference_code_dir` assembles `code/inference.py`,
`code/requirements.txt` (`sentencepiece`, the one real gap between the
inference DLC and what `M2M100Tokenizer` needs), and `code/training/`
(just `__init__.py` + `direction.py` -- the sibling package
`inference.py` imports `DIRECTION_TAGS`/`tag_source_text` from, mirroring
`training/submit_job.py`'s `build_source_bundle` reasoning exactly).
`repackage_model_artifact` extracts the original training artifact, adds
that `code/` directory, and re-tars everything -- the original
`model-artifacts/<run>/output/model.tar.gz` (registered separately by
`training/submit_job.py` against the *training* container image) is never
modified; the repackaged, inference-ready artifact is uploaded to a
distinct prefix, `s3://<bucket>/inference-artifacts/<run-id>/model.tar.gz`.

### Registration (`deployment/deploy.py`)

`deployment/deploy.py` ties the above together into a real (billable, but
cheap -- only S3 transfer + a Model Registry API call, no compute) CLI:
resolve the environment's `DataStack` bucket, resolve the real HuggingFace
**inference** DLC image URI (see the module's own docstring for how the
`transformers`/`pytorch`/`py_version` combination -- `4.49.0`/`2.6.0`/
`py312`, CPU variant since Serverless Inference doesn't support GPU --
was derived from the installed SDK's compatibility table, mirroring
`submit_job.py`'s equivalent derivation for the *training* combination),
download the source artifact, repackage it, upload it, and register it as
a new Model Package version in the same
`traductor-kaqchikel-es-cak` group `training/submit_job.py` already uses:

```sh
cd ml
uv run python -m deployment.deploy --dry-run \
  --source-model-data-url s3://<bucket>/model-artifacts/<run>/output/model.tar.gz
uv run python -m deployment.deploy \
  --source-model-data-url s3://<bucket>/model-artifacts/<run>/output/model.tar.gz
```

**Approval status defaults to `Approved`, not `PendingManualApproval`**
(unlike `training/submit_job.py`'s training-container registration) --
see `deploy.py`'s module docstring: the project owner explicitly
authorized deploying the current best available model as an
"experimental" release for issue #8, consciously overriding the
originally planned BLEU>=10 quality gate (issue #76). A UI disclaimer
communicating this to end users is being added in parallel (issue #43).
Model Registry's versioning is exactly what makes this reversible at low
effort: approving a better model's version later and bumping
`infra/cdk/lib/ml-hosting-stack.ts`'s `DEV_MODEL_PACKAGE_VERSION` constant
(in `app-stage.ts`) is the entire swap procedure.

### Infrastructure (`infra/cdk/lib/ml-hosting-stack.ts`)

`MlHostingStack` provisions the actual endpoint: a `AWS::SageMaker::Model`
referencing the approved Model Package (by ARN, *constructed* at synth
time from a hardcoded version number plus the stack's own account/region
tokens -- never a literal ARN in source, since that would embed a real
AWS account ID, forbidden by CLAUDE.md even in infra code), an
`AWS::SageMaker::EndpointConfig` with a `ServerlessConfig` (6144MB memory
-- the checkpoint's `model.safetensors` alone is ~2.1GB fp32 given the
~196k-token extended vocabulary, so this uses Serverless Inference's
maximum memory tier rather than risk cold-start OOM failures at a lower
one; `MaxConcurrency: 2`, generous enough for this project's low, bursty
traffic per ADR 0001's serverless rationale), and the
`AWS::SageMaker::Endpoint` itself, plus a least-privilege execution role
(read-only access to the environment's training-data bucket, CloudWatch
Logs write scoped to `/aws/sagemaker/Endpoints/*` only). The endpoint name
is exposed as a `CfnOutput` (`MlEndpointName`) for issue #9's `apps/api`
to resolve at deploy time rather than hardcode.

**Scoped to `dev` only, for now**: SageMaker Model Registry entries are
account-scoped, and only the `dev` account has a trained, registered,
approved model today (training only ever runs against `dev`'s corpus
bucket). Promoting a model to `qa`/`prod` (separate AWS accounts under
ADR 0001's per-environment-account layout) would need either a duplicated
registration there or cross-account Model Registry sharing -- neither is
addressed by ADR 0001, so `app-stage.ts` only instantiates `MlHostingStack`
for `environmentName === "dev"`. This is a real, un-designed gap flagged
for a follow-up ticket, not solved here.

### Real deployment verification

Because every environment's stacks are deployed exclusively through the
self-mutating CDK Pipeline (`infra/cdk/lib/pipeline-stack.ts`, triggered
by a push to `main` -- see `docs/runbooks/cdk-pipelines-bootstrap.md`),
`MlHostingStack`'s endpoint cannot actually go live until this ticket's PR
merges. To confirm the custom inference handler and the repackaged
artifact genuinely work against the real fine-tuned weights *before*
merging (per this project's "verify against real AWS, don't just assert
CDK synth succeeds" convention -- see `training/submit_job.py`'s own
real-run precedent), a temporary, manually-created SageMaker Serverless
Inference endpoint (same Model Package, same execution role shape,
distinct physical name so it can never collide with the one `MlHostingStack`
creates on merge) was deployed directly via `boto3`/the AWS CLI.

**This real verification found two real bugs**, both now fixed in
`deployment/inference.py` (see its `output_fn` docstring, and
`training.direction.strip_leading_direction_tag`'s docstring for the
direction-tag leak, for the full story) before registering the version
this project actually deploys:

1. `output_fn` originally returned a `(body, content_type)` tuple. The
   real `sagemaker-huggingface-inference-toolkit` doesn't use that
   convention -- it serializes whatever `output_fn` returns as the literal
   response body -- so the first real invocation returned
   `["{\"translated_text\": \"...\"}", "application/json"]` instead of the
   documented `{"translated_text": "..."}` contract. Fixed by returning
   just the serialized JSON string.
2. An es->cak request returned `"__cak__ q'ij"` instead of `"q'ij"`.
   `__es__` is one of M2M100's own pretrained special tokens (stripped
   automatically by `skip_special_tokens=True`); `__cak__` was added by
   this project's own vocabulary extension as an ordinary token, so it
   survived verbatim as the first word of every `cak`-target translation.
   Fixed with a defensive strip in `translate()`. **This same
   contamination affected `training.train.generate_translations`'s BLEU/
   chrF computation for every es->cak example** -- the model card's
   reported BLEU 8.9 / chrF 31.2 for this checkpoint was very likely
   computed against es->cak hypotheses with this same stray leading token,
   which the reference translations never have. At the time this was
   found (issue #8), fixing the training-side leak and re-evaluating was
   flagged as a follow-up rather than done immediately. **That follow-up
   is issue #106**: `generate_translations` now shares the exact same fix
   (`training.direction.strip_leading_direction_tag`, extracted so
   `deployment/inference.py` and `training/train.py` can't drift apart on
   this again) rather than a second independent implementation. #106 was
   a pure bugfix + regression test, deliberately **not** a retrain or
   re-evaluation -- see the "Known bug" note under "Evaluation harness"
   above for what is/isn't fixed as of that ticket. Every BLEU/chrF number
   in this README predates the fix and should be read with that caveat.

The first registered Model Package version (version 3) was rejected in
Model Registry (`ModelApprovalStatus: Rejected`, with a description
pointing at this section) once these bugs were found; version 4 --
registered with the fixes above -- is the one `MlHostingStack` actually
deploys.

With the fixes in place, real `InvokeEndpoint` calls against the temporary
verification endpoint confirmed:

- `{"source_lang": "es", "target_lang": "cak", "text": "Buenos días"}` ->
  `{"translated_text": "q'ij"}`
- `{"source_lang": "cak", "target_lang": "es", "text": "La utz awäch?"}` ->
  `{"translated_text": "bueno lado"}`
- `{"source_lang": "es", "target_lang": "es", "text": "hola"}` (invalid:
  same source/target) -> HTTP 400 with the `ValueError` message in the
  body, confirming the documented error-handling contract holds for real,
  not just in unit tests.

Translation quality itself is poor by inspection (`"bueno lado"` for "La
utz awäch?" is not a coherent Spanish translation) -- consistent with the
low BLEU 8.9 / chrF 31.2 already reported and the explicit "experimental"
framing this deployment ships under (issue #43's UI disclaimer). This
verification confirms the *pipeline* works end-to-end (request in,
direction-tag logic applied, real weights invoked, response out, matching
the documented contract), not that the model translates well -- those are
deliberately different claims (see `docs/testing.md`).

The temporary verification endpoint, endpoint configs, and models were all
deleted after this verification; nothing from it persists in AWS.
