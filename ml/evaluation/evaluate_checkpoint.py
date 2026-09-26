"""Eval-only checkpoint scoring: score an already-trained checkpoint's
BLEU/chrF against a validation set, without running any training (issue
#108).

## Why this exists

`training.train.run_training_job` unconditionally calls `fine_tune()`
before evaluating -- there was previously no way to re-score an existing
checkpoint (e.g. after a bugfix to `generate_translations()` itself, like
issue #106's direction-tag-leak fix, see `ml/README.md`) without paying
for a full, billable retrain. This script reuses the exact same,
already-tested pieces `run_training_job` composes for its own
post-training evaluation step
(`training.direction.build_direction_examples`,
`training.train.generate_translations`, `evaluation.run.run_evaluation`),
minus everything training-only:

- No `fine_tune()` call -- this script never touches the model's weights.
- No `extend_vocabulary_for_examples()` call: a checkpoint saved by a
  prior `run_training_job` run already has its *post-extension*
  tokenizer saved alongside it (`save_model_and_tokenizer` saves the
  tokenizer after `extend_vocabulary_for_examples` has already run), so
  re-running vocabulary extension here would be wrong -- at best a
  no-op (every token it would propose is already in the vocab), at worst
  it silently adds redundant tokens with cold, untrained embeddings that
  never see a gradient update in an eval-only run. It is a
  training-time-only step by construction: its whole purpose is to
  warm-start embeddings for brand-new tokens *before* fine-tuning trains
  them, which has nothing to do here.

## Reconstructing issue #116's word-boundary-spacing fix for a reloaded checkpoint

A checkpoint's saved tokenizer never persists which added tokens were
whole words/characters (`training.tokenizer_extension.
extend_tokenizer_vocab`) versus subword pieces
(`extend_tokenizer_vocab_with_subwords`) -- `_mark_word_boundary_tokens`'s
effect (issue #116's fix) lives only on the live tokenizer *instance* that
originally called it, and `save_pretrained()` only ever writes a flat
`added_tokens.json` list. Simply reloading a checkpoint's tokenizer here
and generating translations would therefore silently **not** apply #116's
fix at all, even after that fix landed -- see `training.
tokenizer_extension`'s module docstring ("Re-evaluating an already-trained
checkpoint") for the full argument.

This script recovers it: `--train` (required) must name the exact training
corpus the checkpoint's own training history was extended against (from
that run's own model card/hyperparameters -- *not* necessarily the same
corpus this script's own `--validation` scores against). Given that,
`training.tokenizer_extension.patch_word_boundary_decoding_for_checkpoint`
deterministically recomputes the same whole-word boundary token set the
checkpoint's training run(s) actually added (no SentencePiece retraining,
no randomness -- see that function's docstring) and patches the reloaded
tokenizer's decode behavior in place, before any translation is generated.
`--base-model` (already recorded for provenance) is also used here to load
the *pristine* base tokenizer vocab this reconstruction is computed
against.

There is deliberately no `--train-direction` flag: an earlier version of
this script had one, but `training.train.extend_vocabulary_for_examples`'s
`sample_texts` always includes every pair's *both* columns regardless of
`--direction` (`build_direction_examples("both")` doubles the example
count, but `[ex.source_text ...] + [ex.target_text ...]` still reduces to
the same underlying *set* of texts as any single-direction mode -- see
`training.direction.build_direction_examples`), so the reconstructed token
set is provably identical no matter what direction value would have been
passed. A flag that can never change its own output has no reason to
exist (PR #128 review).

**This "always both columns" construction is only valid for checkpoints
trained *before* issue #125's fix.** #125 found that
`extend_vocabulary_for_examples`'s whole-word/character step mixed
ordinary Spanish into what should be Kaqchikel-only vocabulary coverage,
and scoped it to Kaqchikel-only text going forward. For any checkpoint
retrained under the post-#125 behavior, reconstructing against the old
"both columns" set would over-generate boundary tokens (including
Spanish-derived ones the new checkpoint's vocab never actually has),
misleadingly tripping `_diagnose_word_boundary_reconstruction`'s
over-reconstruction check below as if `--train`/`--base-model` didn't
match the checkpoint, when the real cause would be reconstructing against
the wrong scoping for that checkpoint's own provenance.

Issue #143 closed this gap: `_build_train_sample_texts` picks the
construction to use based on the checkpoint's *own* recorded provenance,
never a caller-supplied flag. `training.train.run_training_job` records a
fixed `vocab_extension_scoping` hyperparameter
(`training.train.VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY`) in every model
card it writes, going forward -- issue #125's fix fully replaced the old
behavior in code, so there is no runtime choice to make at training time
any more, only a fixed value to record. `_read_checkpoint_vocab_
extension_scoping` reads this back from the checkpoint's own saved
`model_card.md` (best-effort, mirroring `_read_checkpoint_new_tokens_
added`'s "never raises" contract): if present, `_build_train_sample_texts`
uses the new Kaqchikel-only construction (`training.direction.
collect_texts_for_language`-equivalent: the target/Kaqchikel column of
every pair, since this project's TSV convention is always `es<TAB>cak`);
if absent -- which is implicitly true for every checkpoint trained before
this field existed, including the currently-deployed
`run-20260922T141956Z` (Model Package v4) -- it falls back to the legacy
"both columns" construction unchanged, so nothing about how already-
deployed checkpoints get re-evaluated changes.

The number of boundary tokens reconstructed/applied is recorded in the
model card's hyperparameters for traceability. Two independent checks
then run (see `_diagnose_word_boundary_reconstruction`'s own docstring for
the full reasoning) and warn on stderr (never raise, so an expensive real
run doesn't die on a diagnostic) if something looks wrong:

- **Over-reconstruction**: a reconstructed token that isn't actually in
  the checkpoint's vocabulary at all -- unambiguously means `--train`/
  `--base-model` don't match what the checkpoint was trained with.
- **Under-reconstruction**: fewer boundary tokens were reconstructed than
  are mathematically possible given the checkpoint's own recorded
  `new_tokens_added` (from its saved `model_card.md`, best-effort) --
  the more dangerous failure mode, since a naive "is every reconstructed
  token present in the checkpoint?" check can *never* catch it (every
  reconstructed token from a too-small `--train` is still trivially
  present in the checkpoint's larger real vocab).

## Provenance: distinguishing a re-evaluation from a training run's own eval

A model card produced by this script is **not** a new training run: it
re-scores the exact same, already-registered weights named by
`--source-run-id` (this script registers no new SageMaker Model Registry
model package -- see issue #108's explicit scope). To keep that
unambiguous downstream (e.g. skimming `ml/README.md`'s history of
reported BLEU/chrF numbers), this script:

- Records `reevaluation: True` and `source_run_id`/`source_checkpoint` in
  the model card's hyperparameters section (`ModelCardData.
  hyperparameters` is really "everything about the run other than the
  metrics themselves" -- see `evaluation.model_card`), so it's visible
  alongside the other run identifiers without needing a new schema field.
- Defaults `--run-id` to a `reeval-<UTC timestamp>` prefix, distinct from
  `training.train.run_training_job`'s `run-<UTC timestamp>` convention,
  so a re-evaluation run's own id is visually distinguishable at a glance.
- Sets `train_sentence_count` to `0` with an explanatory note: no
  training happened in this run, and the original run's training
  sentence count belongs on that run's own model card, not this one's.

## Testing

Same "duck-type and fixture" approach as `tests/integration/
test_train_pipeline.py`: `tests/integration/test_evaluate_checkpoint_pipeline.py`
exercises the wiring (checkpoint resolution -> tokenizer/model loading ->
direction-tagged example building -> translation -> evaluation -> model
card) against tiny fixture data and duck-typed fakes, never a real
checkpoint download or a real `model.generate()` call.
`resolve_checkpoint_source`'s S3-download + tar-extraction logic is
unit-tested directly against a fake S3 client
(`tests/unit/test_evaluate_checkpoint_source.py`), mirroring
`training.train.resolve_model_source`'s local-only equivalent.
`load_checkpoint_tokenizer_and_model`'s CUDA device placement (issue #170)
is unit-tested indirectly via its extracted `_move_model_to_cuda_if_available`
helper against a duck-typed fake model
(`tests/unit/test_evaluate_checkpoint_device.py`) -- see that helper's own
docstring for why the real `from_pretrained`-calling wrapper itself isn't
tested directly.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data.corpus_io import read_tsv_pairs
from evaluation.run import run_evaluation
from training.direction import ALL_DIRECTION_TAG_TOKENS, DIRECTION_CHOICES, build_direction_examples
from training.subword_vocab import WORD_BOUNDARY_MARKER
from training.tokenizer_extension import patch_word_boundary_decoding_for_checkpoint
from training.train import (
    DEFAULT_BASE_MODEL,
    VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY,
    generate_translations,
    resolve_model_source,
)

_S3_URI_RE = re.compile(r"^s3://(?P<bucket>[^/]+)/(?P<key>.+)$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for a one-off, local eval-only run against an
    existing checkpoint. Unlike `training.train.parse_args`, there is no
    `SM_*` environment variable fallback -- this script is meant to be run
    directly (locally, or as a one-off script inside a lightweight
    SageMaker Processing Job), not as a Training Job entrypoint.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Score an existing checkpoint's BLEU/chrF against a validation "
            "set, without any training step (issue #108)."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "The checkpoint to score: a local directory (an "
            "already-extracted save_pretrained() output), a local "
            "model.tar.gz, or an s3:// URI to a model.tar.gz (e.g. a prior "
            "training run's SageMaker model artifact)."
        ),
    )
    parser.add_argument(
        "--validation",
        required=True,
        help="Path or s3:// URI to the validation corpus TSV (source<TAB>target).",
    )
    parser.add_argument(
        "--train",
        required=True,
        help=(
            "Path or s3:// URI to the exact training corpus TSV the "
            "checkpoint's own training history was extended against (from "
            "that run's own model card/hyperparameters -- see this "
            "module's docstring, 'Reconstructing issue #116's "
            "word-boundary-spacing fix'). Required: without it, this "
            "script cannot correctly recover which of the checkpoint's "
            "added tokens must decode with a leading space, and would "
            "silently under-report or over-report issue #116's fix. "
            "There is deliberately no separate --train-direction flag: "
            "the reconstructed token set is provably identical regardless "
            "of direction (see this module's docstring)."
        ),
    )
    parser.add_argument(
        "--corpus-version",
        required=True,
        help=(
            "Identifier for the corpus version the validation set came "
            "from (e.g. 'almg-v1'), recorded in the model card for "
            "traceability (ADR 0001). Use the *same* corpus version the "
            "checkpoint's original training run was scored against, for an "
            "apples-to-apples comparison against that run's reported "
            "numbers. Never derived automatically from corpus content "
            "(ADR 0002)."
        ),
    )
    parser.add_argument(
        "--source-run-id",
        required=True,
        help=(
            "The run_id of the training run whose checkpoint this "
            "re-evaluates (e.g. from that run's own model card), recorded "
            "here for provenance. This script produces no new weights and "
            "registers no new Model Registry model package -- it only "
            "re-scores the already-registered weights from this run."
        ),
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=(
            "Hugging Face model id the checkpoint was originally "
            "fine-tuned from. Recorded in the model card for traceability, "
            "and also actually loaded (its *tokenizer* only, never its "
            "weights) to get the pristine, pre-extension vocab that issue "
            "#116's word-boundary reconstruction is computed against -- "
            "the checkpoint's own (already-extended) tokenizer must never "
            "be used for that base vocab, or the reconstruction would "
            "wrongly see everything as 'already covered'."
        ),
    )
    parser.add_argument(
        "--direction",
        choices=DIRECTION_CHOICES,
        default="both",
        help="Which translation direction(s) to evaluate. Default 'both'.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Identifier for this re-evaluation run; defaults to a "
            "'reeval-<UTC timestamp>' id if omitted (distinct from "
            "training.train's 'run-<UTC timestamp>' convention, so a "
            "re-evaluation run is visually distinguishable at a glance)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./eval-output",
        help="Directory to write predictions.txt, references.txt, and model_card.md to.",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)

    return parser.parse_args(argv)


def _get_s3_client() -> Any:
    import boto3

    return boto3.client("s3")


def resolve_checkpoint_source(source: str, *, s3_client: Any = None) -> str:
    """Resolve `source` to a local path `from_pretrained` can load
    directly: a local directory or local `.tar.gz` is delegated to
    `training.train.resolve_model_source` unchanged (so the two scripts
    can't drift apart on that logic); an `s3://` URI is downloaded to a
    fresh temp file first, then handed to `resolve_model_source` the same
    way.

    `training.train.resolve_model_source` never needed S3 support: its
    caller, a real SageMaker Training Job's `--init-model`, always names a
    local channel path SageMaker itself already staged from S3. This
    script has no such staging step -- it is meant to be pointed directly
    at a checkpoint's S3 URI (e.g. a prior run's `model.tar.gz` model
    artifact) from a plain local/CLI invocation.
    """
    if source.startswith("s3://"):
        match = _S3_URI_RE.match(source)
        if not match:
            raise ValueError(f"Not an s3:// URI: {source!r}")
        bucket, key = match.group("bucket"), match.group("key")

        client = s3_client or _get_s3_client()
        tmp_dir = Path(tempfile.mkdtemp(prefix="eval-checkpoint-"))
        local_tar_path = tmp_dir / "model.tar.gz"
        client.download_file(bucket, key, str(local_tar_path))
        source = str(local_tar_path)

    return resolve_model_source(source)


def _move_model_to_cuda_if_available(model: Any) -> Any:
    """Move `model` onto a CUDA device when one is available, otherwise
    leave it untouched (issue #170).

    Mirrors `training.train.build_training_arguments`'s existing
    `fp16=torch.cuda.is_available()` pattern, but for device placement:
    `training.train.generate_translations` already moves every encoded
    input batch to `model.device` before calling `model.generate` (see
    that function's docstring), but the model itself was never actually
    moved onto a GPU in the first place -- `from_pretrained` always loads
    onto CPU by default, so `model.device` stayed `"cpu"` regardless of
    the environment's real hardware, and the whole generation pass ran on
    CPU even in a GPU-capable environment. A real eval run (7,218
    validation examples, beam=5) took ~12 hours on a CPU-only environment
    because of this; a local smoke test confirmed the same generation is
    trivially GPU-capable (peak ~2.3GB at the script's real default
    `--batch-size 16`) once the model is explicitly moved here.

    On a CPU-only environment (including this repo's own CI runners),
    `torch.cuda.is_available()` is `False` and this is a no-op -- no
    behavior or performance change there.
    """
    import torch

    if torch.cuda.is_available():
        return model.to("cuda")
    return model


def load_checkpoint_tokenizer_and_model(source: str) -> tuple[Any, Any]:
    """Load the real `M2M100Tokenizer` + `M2M100ForConditionalGeneration`
    directly from `source` (a local checkpoint directory produced by a
    prior `training.train.save_model_and_tokenizer` call). Its vocabulary
    is already extended and its weights already fine-tuned -- nothing
    further to load or extend here. Lazily imports `transformers`, same
    laziness pattern as `training.train.load_base_model_and_tokenizer`.

    Moves the model to a CUDA device when one is available (issue #170) --
    see `_move_model_to_cuda_if_available`'s docstring for why this matters.
    """
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    tokenizer = M2M100Tokenizer.from_pretrained(source)
    model = M2M100ForConditionalGeneration.from_pretrained(source)
    model = _move_model_to_cuda_if_available(model)
    return tokenizer, model


def load_base_tokenizer_vocab(base_model: str) -> dict[str, int]:
    """Load *only* the vocab (`get_vocab()`) of a pristine base tokenizer
    (e.g. `facebook/m2m100_418M`, never one with any extension already
    applied). Used as the `base_vocab` argument to `training.
    tokenizer_extension.reconstruct_whole_word_boundary_tokens`/
    `patch_word_boundary_decoding_for_checkpoint` -- see this module's
    docstring, "Reconstructing issue #116's word-boundary-spacing fix".
    Lazily imports `transformers`, same laziness pattern as
    `load_checkpoint_tokenizer_and_model` above.
    """
    from transformers import M2M100Tokenizer

    return M2M100Tokenizer.from_pretrained(base_model).get_vocab()


_NEW_TOKENS_ADDED_RE = re.compile(r"-\s*\*\*new_tokens_added\*\*:\s*(\d+)")
_VOCAB_EXTENSION_SCOPING_RE = re.compile(r"-\s*\*\*vocab_extension_scoping\*\*:\s*(\S+)")

# Label recorded on *this script's own* re-evaluation model card for
# provenance, when the checkpoint being re-evaluated has no recorded
# `vocab_extension_scoping` of its own (i.e. it predates issue #125) --
# never compared against, purely a human-readable note of which
# construction this run actually used.
_LEGACY_UNSCOPED_VOCAB_EXTENSION_LABEL = "legacy_unscoped_pre_125"


def _read_checkpoint_model_card_text(checkpoint_dir: str) -> str | None:
    """Best-effort, single read of the checkpoint's own saved
    `model_card.md` text (`training.train.run_training_job` writes this
    alongside the model artifact, in the same directory
    `save_model_and_tokenizer` saves the tokenizer to). Returns `None`
    (never raises) if `checkpoint_dir` isn't a real local path or the file
    is missing/unreadable -- a missing/unreadable model card must never
    crash an otherwise-valid eval-only run over a best-effort diagnostic.

    Shared by every field-specific reader below (`_read_checkpoint_
    new_tokens_added` / `_read_checkpoint_vocab_extension_scoping`) and by
    `run_checkpoint_evaluation` directly, so a checkpoint's model card is
    only ever read from disk once per evaluation run rather than once per
    field (PR #144 review).
    """
    model_card_path = Path(checkpoint_dir) / "model_card.md"
    try:
        if not model_card_path.is_file():
            return None
        return model_card_path.read_text(encoding="utf-8")
    except OSError:
        return None


def _extract_new_tokens_added(model_card_text: str | None) -> int | None:
    """Parse the `new_tokens_added` hyperparameter out of already-read
    model card text (see `_read_checkpoint_model_card_text`). Used only as
    an auxiliary sanity bound for word-boundary reconstruction
    (`_diagnose_word_boundary_reconstruction`) -- never required.
    """
    if model_card_text is None:
        return None
    match = _NEW_TOKENS_ADDED_RE.search(model_card_text)
    return int(match.group(1)) if match else None


def _extract_vocab_extension_scoping(model_card_text: str | None) -> str | None:
    """Parse the `vocab_extension_scoping` hyperparameter out of
    already-read model card text (see `_read_checkpoint_model_card_text`).

    Returns `None` for a checkpoint trained before issue #125 (its own
    `training.train.run_training_job` run never wrote this field at all --
    this is the *expected*, common case for every checkpoint trained so
    far, including the currently-deployed `run-20260922T141956Z`), and
    `training.train.VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY` for one trained
    under issue #125's fix with a fully Kaqchikel-only-scoped
    `--init-model` lineage (see `training.train.run_training_job`'s own
    docstring on how it propagates this across continuation runs -- PR
    #144 review). See `_build_train_sample_texts`, which uses this to pick
    the correct `train_sample_texts` construction by the checkpoint's own
    provenance rather than a caller-supplied flag (issue #143).
    """
    if model_card_text is None:
        return None
    match = _VOCAB_EXTENSION_SCOPING_RE.search(model_card_text)
    return match.group(1) if match else None


def _read_checkpoint_new_tokens_added(checkpoint_dir: str) -> int | None:
    """Convenience wrapper: read + extract in one call. Prefer
    `_read_checkpoint_model_card_text` + `_extract_new_tokens_added`
    directly when also reading another field from the same checkpoint in
    the same call site, to avoid reading the file twice.
    """
    return _extract_new_tokens_added(_read_checkpoint_model_card_text(checkpoint_dir))


def _read_checkpoint_vocab_extension_scoping(checkpoint_dir: str) -> str | None:
    """Convenience wrapper: read + extract in one call. Prefer
    `_read_checkpoint_model_card_text` + `_extract_vocab_extension_scoping`
    directly when also reading another field from the same checkpoint in
    the same call site, to avoid reading the file twice.
    """
    return _extract_vocab_extension_scoping(_read_checkpoint_model_card_text(checkpoint_dir))


def _build_train_sample_texts(
    train_pairs: list[tuple[str, str]], vocab_extension_scoping: str | None
) -> list[str]:
    """Build the `sample_texts` issue #116's word-boundary reconstruction is
    computed against, choosing the construction that actually matches how
    the checkpoint being re-evaluated was trained (issue #143):

    - `vocab_extension_scoping == VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY`
      (a checkpoint trained under issue #125's fix): only the Kaqchikel
      (target) column of every pair -- this project's TSV convention is
      always `es<TAB>cak` (see `data.corpus_io.read_tsv_pairs`), so this is
      equivalent to `training.direction.collect_texts_for_language`
      restricted to Kaqchikel across a "both directions" example set,
      without needing to build `TranslationExample`s just for this.
    - Anything else, including `None` (a checkpoint trained before issue
      #125's fix ever existed -- the common case for every checkpoint
      trained so far): the legacy "every pair's both columns" construction,
      unchanged from this script's original behavior.

    Either way, this project's direction tag tokens (`training.direction.
    ALL_DIRECTION_TAG_TOKENS`) are always appended -- they need real,
    warm-started embedding rows regardless of which vocab-extension scoping
    produced them, and were never excluded by issue #125's fix.

    A `vocab_extension_scoping` that is *present but not recognized* (not
    `VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY`, and not `None`) is deliberately
    **not** silently folded into the `None`/legacy case (PR #144 review): a
    future scoping scheme this function hasn't been updated to know about
    would otherwise be treated as an ordinary pre-#125 checkpoint with zero
    signal anything's off, quietly reintroducing this issue's
    over-reconstruction bug the moment such a checkpoint is evaluated. This
    prints a distinct WARNING to stderr (never raises -- consistent with
    `_diagnose_word_boundary_reconstruction`'s "don't crash an expensive
    real run over a diagnostic" policy) and falls back to the legacy "both
    columns" construction as the broadest, safest available guess.
    """
    if vocab_extension_scoping == VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY:
        sample_texts = [target for _, target in train_pairs]
    else:
        if vocab_extension_scoping is not None:
            print(
                "WARNING: checkpoint recorded an unrecognized "
                f"vocab_extension_scoping value {vocab_extension_scoping!r} "
                "-- this script only knows how to build train_sample_texts "
                f"for {VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY!r} or no "
                "recorded value at all (a checkpoint predating issue #125). "
                "Falling back to the legacy 'both columns' construction, "
                "which may not correctly match what this checkpoint was "
                "actually trained with -- update _build_train_sample_texts "
                "if this is a genuinely new vocab-extension scoping scheme.",
                file=sys.stderr,
            )
        sample_texts = [source for source, _ in train_pairs] + [
            target for _, target in train_pairs
        ]
    sample_texts.extend(ALL_DIRECTION_TAG_TOKENS)
    return sample_texts


def _diagnose_word_boundary_reconstruction(
    boundary_tokens: list[str],
    tokenizer: Any,
    base_vocab: dict[str, int],
    checkpoint_new_tokens_added: int | None,
) -> None:
    """Cross-check the reconstructed whole-word boundary token set against
    the checkpoint's own real vocabulary, catching both over- and
    under-reconstruction.

    An earlier version of this check only asked "is every reconstructed
    token present in the checkpoint's vocabulary?" -- that can catch
    over-reconstruction, but can **never** catch under-reconstruction
    (PR #128 review): if `--train` points at a corpus that's a genuine
    *subset* of what the checkpoint was actually trained on, reconstruction
    silently returns an incomplete boundary-token set, and every one of
    those (fewer) tokens is still trivially present in the checkpoint's
    larger real vocab, so the old check's `missing` list is always empty.
    Some genuinely-whole-word tokens would then silently stay glued on
    decode -- partially re-introducing issue #116's bug with zero signal
    anything's wrong.

    This instead also computes the checkpoint's *exact* set of ambiguous
    added tokens directly from a real vocab diff (checkpoint vocab minus
    the pristine `base_vocab`, restricted to marker-less multi-character
    tokens -- word-initial subword pieces and single missing characters
    are self-describing and excluded) -- entirely independent of
    `--train`'s content, unlike the reconstruction itself. The gap between
    that exact set and the reconstructed `boundary_tokens` is *expected*
    to be non-empty in the normal case: that's exactly where genuine
    subword continuation pieces (`extend_tokenizer_vocab_with_subwords`)
    live, which must correctly stay unmarked. But the gap can't exceed the
    checkpoint's own recorded `new_tokens_added` (the *producing* training
    run's own count of tokens it added, from its saved `model_card.md`) if
    reconstruction correctly captured everything that run's own
    whole-word extension step contributed -- `new_tokens_added` upper-bounds
    "whole words added by that run" + "subword pieces added by that run",
    so a gap *larger* than it can only mean some whole-word tokens were
    missed. If it's larger, that's the under-reconstruction signal.

    This deliberately does **not** require an exact equality against
    `new_tokens_added`: for a checkpoint continued from a prior checkpoint
    via `--init-model` (like this project's real deployed checkpoint,
    `run-20260922T141956Z` -- a 3-run continuation chain), most whole-word
    tokens were added by an *earlier* run, so `new_tokens_added` on the
    checkpoint's own (last) model card only reflects that final run's own
    incremental contribution, not the full cumulative history -- an exact
    equality check would spuriously fire on every continuation checkpoint.
    The inequality bound above still holds regardless of how many
    continuation runs contributed, since it only relies on the *last*
    run's own whole-word step being a correctly-reconstructed subset of
    what it actually added (guaranteed by `--train` matching the corpus
    every run in the chain used, per this project's convention).

    Never raises: an expensive, hours-long real re-evaluation run
    shouldn't die on a diagnostic. Everything is reported to stderr.
    """
    if not hasattr(tokenizer, "get_vocab"):
        return
    checkpoint_vocab = tokenizer.get_vocab()
    boundary_token_set = set(boundary_tokens)

    over_reconstructed = sorted(
        token for token in boundary_token_set if token not in checkpoint_vocab
    )
    if over_reconstructed:
        print(
            f"WARNING: {len(over_reconstructed)} of {len(boundary_token_set)} "
            "reconstructed word-boundary tokens are missing from the "
            "checkpoint's own vocabulary -- --train/--base-model may not "
            f"match what this checkpoint was actually trained with. Sample: "
            f"{over_reconstructed[:10]!r}",
            file=sys.stderr,
        )

    ambiguous_checkpoint_tokens = {
        token
        for token in checkpoint_vocab
        if token not in base_vocab
        and len(token) > 1
        and not token.startswith(WORD_BOUNDARY_MARKER)
    }
    unaccounted = sorted(ambiguous_checkpoint_tokens - boundary_token_set)

    if not unaccounted:
        return

    if checkpoint_new_tokens_added is None:
        print(
            f"NOTE: {len(unaccounted)} marker-less, multi-character tokens in "
            "the checkpoint's vocabulary were not reconstructed as "
            "word-boundary tokens. Most are expected to be genuine subword "
            "continuation pieces (correctly left glued), but this could not "
            "be cross-checked against the checkpoint's own recorded "
            "new_tokens_added (no model_card.md found alongside the "
            "checkpoint) -- treat this count as informational only.",
            file=sys.stderr,
        )
    elif len(unaccounted) > checkpoint_new_tokens_added:
        print(
            f"WARNING: {len(unaccounted)} marker-less, multi-character "
            "tokens in the checkpoint's vocabulary were not reconstructed "
            "as word-boundary tokens, which exceeds the checkpoint's own "
            f"recorded new_tokens_added ({checkpoint_new_tokens_added}) -- "
            "mathematically, this can only happen if some genuine "
            "whole-word tokens were missed (under-reconstruction), most "
            "likely because --train doesn't fully reflect the checkpoint's "
            f"real training corpus. Sample unaccounted-for tokens: "
            f"{unaccounted[:10]!r}",
            file=sys.stderr,
        )


def run_checkpoint_evaluation(
    args: argparse.Namespace,
    *,
    resolve_source: Callable[[str], str] = resolve_checkpoint_source,
    model_loader: Callable[[str], tuple[Any, Any]] = load_checkpoint_tokenizer_and_model,
    base_vocab_loader: Callable[[str], dict[str, int]] = load_base_tokenizer_vocab,
    translator: Callable[..., list[str]] = generate_translations,
) -> Path:
    """Run the full eval-only job: load validation corpus -> load
    checkpoint -> reconstruct + patch issue #116's word-boundary decoding
    fix -> translate -> evaluate -> write model card. Returns the path to
    the written model card.

    Deliberately does **not** call `training.train.fine_tune` or
    `training.train.extend_vocabulary_for_examples` -- see this module's
    own docstring for why. `resolve_source`/`model_loader`/
    `base_vocab_loader`/`translator` default to the real implementations
    above; tests inject fakes in their place (see `tests/integration/
    test_evaluate_checkpoint_pipeline.py`).
    """
    run_id = args.run_id or datetime.now(UTC).strftime("reeval-%Y%m%dT%H%M%SZ")

    val_pairs = read_tsv_pairs(args.validation)
    val_examples = build_direction_examples(val_pairs, args.direction)

    model_source = resolve_source(args.checkpoint)
    tokenizer, model = model_loader(model_source)

    # Reconstruct + patch issue #116's word-boundary-spacing fix before
    # generating any translation -- see this module's docstring
    # ("Reconstructing issue #116's word-boundary-spacing fix") for why a
    # reloaded checkpoint's tokenizer can't just carry this itself. No
    # direction tagging needed here, since the resulting *set* of sample
    # texts is identical regardless of direction either way (see this
    # module's docstring, "no --train-direction flag"). Which columns
    # actually feed into this is picked by the checkpoint's own recorded
    # provenance, not a caller-supplied flag (issue #143) -- see
    # `_build_train_sample_texts`.
    train_pairs = read_tsv_pairs(args.train)
    # Single read of the checkpoint's own model card, shared by both fields
    # pulled from it below (PR #144 review: avoid reading the same file
    # twice).
    checkpoint_model_card_text = _read_checkpoint_model_card_text(model_source)
    vocab_extension_scoping = _extract_vocab_extension_scoping(checkpoint_model_card_text)
    train_sample_texts = _build_train_sample_texts(train_pairs, vocab_extension_scoping)
    base_vocab = base_vocab_loader(args.base_model)
    boundary_tokens = patch_word_boundary_decoding_for_checkpoint(
        tokenizer, base_vocab, train_sample_texts
    )
    checkpoint_new_tokens_added = _extract_new_tokens_added(checkpoint_model_card_text)
    _diagnose_word_boundary_reconstruction(
        boundary_tokens, tokenizer, base_vocab, checkpoint_new_tokens_added
    )

    hypotheses = translator(
        model, tokenizer, val_examples, max_length=args.max_length, batch_size=args.batch_size
    )
    references = [example.target_text for example in val_examples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.txt"
    references_path = output_dir / "references.txt"
    predictions_path.write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
    references_path.write_text("\n".join(references) + "\n", encoding="utf-8")

    run_metadata = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "base_model": args.base_model,
        "direction": args.direction,
        "corpus_version": args.corpus_version,
        # No training happened in this run -- see module docstring. The
        # original run's own model card is the source of truth for its
        # training sentence count.
        "train_sentence_count": 0,
        "hyperparameters": {
            "reevaluation": True,
            "source_run_id": args.source_run_id,
            "source_checkpoint": args.checkpoint,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "word_boundary_reconstruction_train": args.train,
            "word_boundary_tokens_reconstructed": len(boundary_tokens),
            "vocab_extension_scoping": (
                vocab_extension_scoping or _LEGACY_UNSCOPED_VOCAB_EXTENSION_LABEL
            ),
        },
        "notes": (
            f"Re-evaluation of an existing checkpoint from run "
            f"{args.source_run_id!r} -- no training occurred in this run "
            "(train_sentence_count above is not applicable; see that run's "
            "own model card for its training sentence count). Hypotheses "
            "were generated with the fixed training.train."
            "generate_translations() (issue #106's direction-tag-leak "
            "fix). This tokenizer was also patched with issue #116's "
            "word-boundary-spacing fix, reconstructed from --train/"
            "--train-direction/--base-model rather than persisted by the "
            f"checkpoint itself ({len(boundary_tokens)} whole-word "
            "boundary tokens reconstructed -- see training."
            "tokenizer_extension.patch_word_boundary_decoding_for_"
            "checkpoint), since a saved checkpoint's tokenizer never "
            "records which added tokens need it. This model card's "
            "BLEU/chrF therefore supersedes any earlier number reported "
            "for the same checkpoint, including a prior re-evaluation "
            "that only applied issue #106's fix without also correctly "
            "reconstructing issue #116's. This run produced no new "
            "weights and registered no new SageMaker Model Registry "
            f"model package -- it only re-scores the already-registered "
            f"weights from {args.source_run_id!r}."
        ),
    }

    model_card_path = output_dir / "model_card.md"
    _metrics, model_card_path = run_evaluation(
        predictions_path, references_path, run_metadata, model_card_path
    )
    return model_card_path


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_checkpoint_evaluation(args)


if __name__ == "__main__":
    main()
