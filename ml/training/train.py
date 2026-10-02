"""SageMaker Training Job entrypoint: fine-tune `facebook/m2m100_418M`
(ADR 0003) for Spanish<->Kaqchikel translation.

## What this script does

1. Reads the training and validation corpora as two-column TSV
   (source<TAB>target) files via `data.corpus_io.read_tsv_pairs`, from
   whatever local path or `s3://` URI is passed in as `--train`/
   `--validation` -- never a hardcoded bucket. When run inside a real
   SageMaker Training Job, point these at the container's channel paths
   (e.g. `/opt/ml/input/data/train/train.tsv`,
   `/opt/ml/input/data/validation/val.tsv`, populated from whichever S3
   prefix the job's `Estimator` call configures -- the private ALMG
   corpus and the public community corpus live under distinct prefixes
   per ADR 0002, and it is the *caller's* job -- not this script's -- to
   keep them physically separate on disk/S3).
2. Loads the `facebook/m2m100_418M` tokenizer + model.
3. Extends the tokenizer's vocabulary and resizes the model's embeddings
   to cover Kaqchikel (`training.tokenizer_extension`, built in #34/#62),
   using the actual training corpus text as the sample text the gap is
   computed against, plus this script's own direction tag tokens (see
   `training.direction`).
4. Fine-tunes the model via `transformers.Seq2SeqTrainer`.
5. Saves the fine-tuned model + tokenizer to `SM_MODEL_DIR` (`--model-dir`)
   for SageMaker to upload as the training job's model artifact.
6. Generates translations for the validation set, computes BLEU/chrF via
   `evaluation.metrics`, and writes a model card via
   `evaluation.model_card` alongside the saved model artifact
   (`SM_MODEL_DIR/model_card.md`), per ADR 0001's traceability
   requirement (register the resulting model card + config in SageMaker
   Model Registry, not just the raw weights).

## Direction handling: one multilingual checkpoint, tagged

ADR 0001 left open whether to train one multilingual checkpoint (with
direction tags) or two separate per-direction checkpoints. This script
implements **one multilingual checkpoint** -- see `training.direction`'s
module docstring and `docs/adr/0006-translation-direction-handling.md`
for the full rationale (short version: the corpus is small enough that
splitting it in two would hurt more than cross-direction interference
would, Kaqchikel has no pretrained skill in either direction to protect
via separation, and one checkpoint is cheaper to register/serve).
`--direction` defaults to `"both"`, training on both es->cak and cak->es
examples in the same run; it can be set to a single direction for
experimentation/comparison.

## Testing

This script is never run for real (no real training pass, no real
checkpoint download) in this repo's test suite -- that is issue #66's
job. `tests/integration/test_train_pipeline.py` exercises the wiring
(argument parsing -> corpus loading -> direction-tagged example building
-> vocab/embedding extension -> save -> evaluate -> model card) against
tiny fixture data, a duck-typed fake tokenizer/model, and fake
`trainer`/`translator` callables injected into `run_training_job` in place
of the real (heavy) `fine_tune`/`generate_translations` -- following the
same "duck-type and fixture" approach as `training/tokenizer_extension.py`
(#34/#62).
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# SageMaker's v3 `ModelTrainer` "Basic Script driver" invokes this file as
# `python3 training/train.py` with the bundle root (see
# `submit_job.build_source_bundle`) as `cwd` -- Python then puts only this
# script's own directory (`<bundle root>/training`) on `sys.path[0]`, not
# the bundle root itself, so the sibling-package imports below
# (`data.*`/`evaluation.*`) fail with `ModuleNotFoundError`. This worked
# under the old v2 script-mode toolkit, which put the code root on
# `PYTHONPATH` before invoking the entry script; the v3 SDK's driver does
# not (issue #185, discovered on the first real job submitted after the
# v3 migration). Adding the bundle root here makes this correct regardless
# of how the SDK's driver invokes this script.
_BUNDLE_ROOT = str(Path(__file__).resolve().parent.parent)
if _BUNDLE_ROOT not in sys.path:
    sys.path.insert(0, _BUNDLE_ROOT)

from data.corpus_io import read_tsv_pairs
from evaluation.run import run_evaluation
from training.checkpoint_averaging import DEFAULT_AVERAGE_N
from training.direction import (
    ALL_DIRECTION_TAG_TOKENS,
    DIRECTION_CHOICES,
    DIRECTION_TAGS,
    KAQCHIKEL,
    TranslationExample,
    build_direction_examples,
    collect_texts_for_language,
    strip_leading_direction_tag,
    tag_source_text,
)
from training.subword_vocab import DEFAULT_VOCAB_SIZE as DEFAULT_SUBWORD_VOCAB_SIZE
from training.tokenizer_extension import (
    extend_tokenizer_vocab,
    extend_tokenizer_vocab_with_subwords,
    resize_embeddings_for_new_tokens,
)

DEFAULT_BASE_MODEL = "facebook/m2m100_418M"

# Issue #182 code review: an uncapped per-epoch checkpoint (each a full
# model + optimizer + scheduler + rng-state save, on the order of ~2x model
# size just from AdamW's own optimizer state) risks filling up a training
# instance's disk on a longer run -- nothing bounded retention before this.
# Derived from `checkpoint_averaging.DEFAULT_AVERAGE_N` (the utility this
# checkpointing exists to feed) plus a small margin, rather than an
# independently hardcoded number, so the two constants can't silently drift
# apart and leave fewer checkpoints on disk than averaging actually needs.
DEFAULT_SAVE_TOTAL_LIMIT = DEFAULT_AVERAGE_N + 2

# Issue #187: per-epoch checkpoints must land in SageMaker's own, separate
# checkpoint-sync directory, never inside SM_MODEL_DIR (`--model-dir`) --
# the directory SageMaker tars whole into the registered `model.tar.gz`
# artifact. Before this fix, `build_training_arguments` wrote checkpoints to
# `<model_dir>/checkpoints`, so every real training run's registered
# artifact bundled the final model *and* several full epoch checkpoints
# (each including optimizer/scheduler/rng-state, typically 2-3x the bare
# model's size for AdamW) -- measured at 28.4-30.5 GB on a real run. Every
# downstream consumer of that artifact (`submit_job.py`'s registration-time
# download, `deployment/package_model.py`'s repackaging step) paid that cost
# repeatedly; see issue #187 for the full history.
#
# `/opt/ml/checkpoints` is the literal, hardcoded path SageMaker Training
# Jobs continuously sync to the S3 URI configured via `ModelTrainer`'s
# `checkpoint_config` (`sagemaker.core.training.configs.CheckpointConfig`,
# see `training/submit_job.py`'s `build_estimator`) -- entirely independent
# of the final model artifact upload. Confirmed against AWS's own
# "SageMaker AI environment variables and the default paths for training
# storage locations" reference page: unlike every other storage location in
# that table (`SM_CHANNEL_*`, `SM_OUTPUT_DIR`, `SM_MODEL_DIR`), the
# checkpoints row has **no** dedicated `SM_*` environment variable -- so,
# unlike `--model-dir`/`--output-data-dir` above, this can't be defaulted
# from an `os.environ.get(...)` call; it must be this literal. Also matches
# `sagemaker.core.training.configs.CheckpointConfig`'s own `local_path`
# default exactly (confirmed against the installed `sagemaker==3.23.0`
# package) -- `submit_job.py` imports this same constant rather than
# independently hardcoding a second copy that could silently drift apart
# from this one.
DEFAULT_CHECKPOINT_DIR = "/opt/ml/checkpoints"

# Matches facebook/m2m100_418M's own generation_config.json default
# (confirmed directly against the real v7 checkpoint artifact, issue #178).
# Kept here as an explicit, named default rather than left unset: before
# issue #180, `generate_translations`/`deployment.inference.translate` never
# passed `num_beams` to `model.generate()` at all, so beam width was
# whatever the checkpoint's own generation_config.json happened to say --
# an implicit, easy-to-miss dependency rather than a real, documented,
# overridable parameter of this project's own generation code.
DEFAULT_NUM_BEAMS = 5

# Issue #143: the value `run_training_job` records in a run's own model
# card's `vocab_extension_scoping` hyperparameter, when (and only when --
# see `_resolve_vocab_extension_scoping_for_model_card` below, PR #144
# review) that run's *entire* `--init-model` continuation lineage was
# scoped this way. Issue #125 fully replaced the old ("both languages
# mixed in") whole-word/character vocab-extension behavior in code -- every
# run from here on always uses this scoping *for its own step* -- so this
# is a fixed value, not a CLI-configurable one. Its purpose is purely
# provenance: a checkpoint's saved model card recording this value (or, for
# any checkpoint whose lineage includes a run that predates this field,
# *not* recording it at all) is how `evaluation.evaluate_checkpoint` tells
# a fully post-#125-scoped checkpoint apart from one with any pre-#125
# ancestry when reconstructing issue #116's word-boundary fix (see that
# module's docstring/`_build_train_sample_texts`).
VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY = "kaqchikel_only"

_VOCAB_EXTENSION_SCOPING_RE = re.compile(r"-\s*\*\*vocab_extension_scoping\*\*:\s*(\S+)")


def _read_ancestor_vocab_extension_scoping(model_source: str) -> str | None:
    """Best-effort read of the `vocab_extension_scoping` hyperparameter from
    an `--init-model` ancestor checkpoint's own saved `model_card.md`
    (mirrors `evaluation.evaluate_checkpoint`'s "never raises" reader
    contract for the same field). `model_source` here is already the
    *resolved* local directory of that ancestor checkpoint (see
    `resolve_model_source`) -- a plain local directory in the common case,
    or a Hugging Face Hub model id when there's no `--init-model` at all
    (never has a local `model_card.md`, so this correctly returns `None`).

    Returns `None` both when the ancestor genuinely predates issue #125
    (never wrote this field) and when there's no readable model card at
    all (e.g. a checkpoint not produced by this script) -- either way, the
    caller (`_resolve_vocab_extension_scoping_for_model_card`) must treat
    the whole lineage as not fully Kaqchikel-only scoped.
    """
    model_card_path = Path(model_source) / "model_card.md"
    try:
        if not model_card_path.is_file():
            return None
        text = model_card_path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _VOCAB_EXTENSION_SCOPING_RE.search(text)
    return match.group(1) if match else None


def _resolve_vocab_extension_scoping_for_model_card(
    *, init_model: str | None, model_source: str
) -> str | None:
    """Decide what (if anything) this run's own model card should record
    for `vocab_extension_scoping` (issue #143, PR #144 review).

    This run's *own* whole-word/character vocab-extension step always uses
    Kaqchikel-only scoping (issue #125 fully replaced the old behavior in
    code -- there's no other option any more). But a checkpoint's real
    vocabulary is the union of every run in its `--init-model` continuation
    chain, not just this run's own contribution: continuing from an
    ancestor whose *own* recorded scoping is anything other than
    `VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY` (including `None` -- an
    ancestor that predates issue #125 entirely, e.g. the currently-deployed
    `run-20260922T141956Z`) means the checkpoint being produced here still
    carries pre-#125, unscoped (Spanish-inclusive) whole-word tokens from
    that ancestor. Recording `kaqchikel_only` unconditionally here would
    tell `evaluation.evaluate_checkpoint` to reconstruct against the
    Kaqchikel-only column alone, which would never re-mark those
    inherited Spanish-derived tokens for issue #116's leading-space
    decoding fix -- silently reintroducing that exact bug for them.

    - No `--init-model` at all (`init_model` falsy): this run's checkpoint
      has no pre-existing vocabulary to inherit, so its whole lineage
      (i.e. just itself) is trivially fully Kaqchikel-only scoped.
    - `--init-model` given: only propagate `VOCAB_EXTENSION_SCOPING_
      KAQCHIKEL_ONLY` forward if the ancestor's *own* model card also
      recorded exactly that value (i.e. the ancestor's own lineage was
      already confirmed fully scoped, by the same inductive argument one
      level up). Anything else -- `None`, or some unrecognized value --
      means the field must be omitted from this run's own model card too,
      so a fully-mixed-lineage checkpoint keeps falling back to
      `evaluation.evaluate_checkpoint`'s legacy (safe, superset)
      reconstruction, exactly like a checkpoint that predates issue #125
      outright.

    `model_source` is the already-resolved local checkpoint directory (see
    `resolve_model_source`) when `init_model` is truthy.
    """
    if not init_model:
        return VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY

    ancestor_scoping = _read_ancestor_vocab_extension_scoping(model_source)
    if ancestor_scoping == VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY:
        return VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI args, matching the SageMaker Training Job convention of
    passing hyperparameters as `--key value` flags and reading channel/
    output directories from `SM_*` environment variables when not
    explicitly overridden.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune facebook/m2m100_418M for Spanish<->Kaqchikel "
            "translation (ADR 0001, ADR 0003)."
        )
    )
    parser.add_argument(
        "--train",
        required=True,
        help=(
            "Path or s3:// URI to the training corpus TSV (source<TAB>target). "
            "In a real SageMaker Training Job, point this at the 'train' "
            "channel's file, e.g. /opt/ml/input/data/train/train.tsv."
        ),
    )
    parser.add_argument(
        "--validation",
        required=True,
        help=(
            "Path or s3:// URI to the validation corpus TSV. In a real "
            "SageMaker Training Job, point this at the 'validation' "
            "channel's file, e.g. /opt/ml/input/data/validation/val.tsv."
        ),
    )
    parser.add_argument(
        "--corpus-version",
        required=True,
        help=(
            "Identifier/tag for the corpus version used (e.g. an S3 object "
            "version id or a dataset release tag), recorded in the model "
            "card for traceability (ADR 0001). Never derived automatically "
            "from corpus content, since that content must never leak into "
            "a run artifact (ADR 0002)."
        ),
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("SM_MODEL_DIR", "./model-output"),
        help="Output directory for the fine-tuned model+tokenizer (SM_MODEL_DIR).",
    )
    parser.add_argument(
        "--output-data-dir",
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output-data"),
        help="Output directory for non-model run artifacts, e.g. predictions/"
        "references used to compute metrics (SM_OUTPUT_DATA_DIR).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help=(
            "Local directory to write per-epoch checkpoints to (issue #187) "
            "-- separate from --model-dir, which SageMaker tars whole into "
            "the registered model.tar.gz artifact. Defaults to "
            f"{DEFAULT_CHECKPOINT_DIR!r}, the literal path SageMaker "
            "Training Jobs continuously sync to the S3 URI configured via "
            "ModelTrainer's checkpoint_config (see submit_job.py) -- not "
            "read from an SM_* environment variable, since (unlike "
            "--model-dir/--output-data-dir above) none exists for this "
            "path; see DEFAULT_CHECKPOINT_DIR's own module-level comment."
        ),
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Hugging Face model id to fine-tune (ADR 0003).",
    )
    parser.add_argument(
        "--init-model",
        default=None,
        help=(
            "Path to a previously fine-tuned checkpoint to continue training "
            "from, instead of --base-model. Accepts a local directory (an "
            "already-extracted save_pretrained() output) or a local "
            "model.tar.gz path (extracted automatically) -- e.g. a "
            "SageMaker channel path pointing at a prior run's model "
            "artifact. --base-model is still recorded in the model card for "
            "traceability even when resuming; the tokenizer/vocab-extension "
            "step becomes a no-op when the checkpoint's vocab already "
            "covers everything (see extend_vocabulary_for_examples)."
        ),
    )
    parser.add_argument(
        "--direction",
        choices=DIRECTION_CHOICES,
        default="both",
        help=(
            "Which translation direction(s) to train. Default 'both' trains "
            "a single multilingual checkpoint with direction tags -- see "
            "training/direction.py for why."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Identifier for this training run; defaults to a UTC timestamp if omitted.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help=(
            "Fraction of total training steps used to linearly warm up the "
            "learning rate (issue #79). Zero warmup interacts badly with the "
            "large fraction of freshly cold-started embedding rows from "
            "vocab extension (see extend_vocabulary_for_examples). Converted "
            "to an absolute warmup_steps count in build_training_arguments, "
            "since the installed transformers version's "
            "Seq2SeqTrainingArguments no longer accepts warmup_ratio directly."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="L2 weight decay applied by the optimizer (issue #79).",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help=(
            "Label smoothing factor for the cross-entropy loss (issue #79). "
            "Setting this above 0 used to crash (the real 5th "
            "real-hyperparameter training run hit `ValueError: You cannot "
            "specify both decoder_input_ids and decoder_inputs_embeds at "
            "the same time`) -- root-caused and fixed in issue #182: "
            "DataCollatorForSeq2Seq's usual decoder_input_ids construction "
            "silently didn't apply to M2M100 on the installed transformers "
            "version (see attach_decoder_input_ids's docstring for the full "
            "trace), so decoder_input_ids ended up unset specifically "
            "whenever label smoothing made Trainer pop `labels` out of the "
            "batch before calling the model. `build_data_collator` computes "
            "decoder_input_ids explicitly now, independent of that gap, and "
            "a real training step with label_smoothing_factor > 0 is "
            "exercised in tests/integration/test_fine_tune_real_checkpoint.py. "
            "Still defaults to 0.0 (disabled) here -- literature on "
            "low-resource NMT fine-tuning (Sennrich & Zhang 2019, "
            "'Revisiting Low-Resource Neural Machine Translation: A Case "
            "Study', which specifically studies larger label-smoothing "
            "factors as one of several hyperparameters worth re-tuning for "
            "low-resource settings) flags this as disproportionately "
            "helpful at small data sizes, but actually changing this "
            "default is a separate, deliberately maintainer-approved "
            "experiment (billable real training run), not bundled into "
            "this fix."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help=(
            "Accumulate gradients over this many steps before an optimizer "
            "update, raising the effective batch size (--batch-size * this) "
            "without needing more GPU memory (issue #79)."
        ),
    )
    parser.add_argument(
        "--subword-vocab-size",
        type=int,
        default=DEFAULT_SUBWORD_VOCAB_SIZE,
        help=(
            "Target vocabulary size for the Kaqchikel-only SentencePiece/"
            "Unigram subword model trained fresh from this run's training "
            "corpus and diffed against the base tokenizer's vocab (issue "
            "#82), to find high-value multi-character subwords that "
            "M2M100's generic multilingual tokenizer fragments Kaqchikel's "
            "agglutinative morphology into. This is a target, not a hard "
            "requirement: SentencePiece caps to whatever the corpus "
            "actually supports rather than failing (see "
            "training/subword_vocab.py's hard_vocab_limit=False)."
        ),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help=(
            "Override the model's general (non-attention) dropout "
            "probability before fine-tuning (issue #182) -- see "
            "apply_dropout_config's docstring for why this can't just be "
            "`model.config.dropout = ...` after loading. Defaults to `None` "
            "(untouched): facebook/m2m100_418M's own pretrained dropout "
            "(0.1) was never deliberately chosen for this fine-tuning "
            "regime, but actually tuning it away from that default is a "
            "separate, maintainer-approved experiment, not this flag's "
            "own default."
        ),
    )
    parser.add_argument(
        "--bpe-dropout-alpha",
        type=float,
        default=None,
        help=(
            "SentencePiece subword-sampling alpha for the training corpus "
            "only ('BPE-dropout' / subword regularization, issue #182; see "
            "enable_subword_sampling's docstring). A real value (e.g. "
            "0.1-0.2) makes each training example's source/target text get "
            "re-segmented into subwords with a different random sample "
            "each time it's encoded, instead of always the single "
            "deterministic 'best' SentencePiece segmentation -- a "
            "regularizer shown to help low-resource NMT fine-tuning "
            "(Kudo 2018, 'Subword Regularization'). Defaults to `None` "
            "(disabled, fully deterministic encoding, matching every past "
            "training run) -- turning this on is a separate, "
            "maintainer-approved experiment. Never applied to the "
            "validation dataset, which must stay deterministic (see "
            "TranslationDataset's docstring)."
        ),
    )
    return parser.parse_args(argv)


def resolve_model_source(source: str) -> str:
    """If `source` is a local `.tar.gz` file, extract it to a fresh temp
    directory and return that directory's path; otherwise return `source`
    unchanged (a Hugging Face Hub model id, or an already-extracted local
    directory).

    Lets `--init-model` accept either shape: a SageMaker channel path
    pointing directly at a previous run's `model.tar.gz` artifact (the
    common case -- `submit_job.py` stages it as a channel exactly like the
    train/validation corpus files), or a plain local directory (useful for
    local testing/development without a real SageMaker channel).
    """
    if source.endswith(".tar.gz") and Path(source).is_file():
        import tarfile

        extract_dir = Path(tempfile.mkdtemp(prefix="init-model-"))
        with tarfile.open(source, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")
        return str(extract_dir)
    return source


def apply_dropout_config(model: Any, dropout: float | None) -> None:
    """Override `model`'s general (non-attention) dropout probability
    in place, threading `--dropout` (issue #182) through to the
    already-constructed submodules a loaded checkpoint carries, not just
    its config object.

    `facebook/m2m100_418M` is fine-tuned with whatever dropout probability
    its own pretrained config happens to specify (`config.dropout`,
    0.1) -- never deliberately set for this project's fine-tuning regime.
    Simply assigning `model.config.dropout = dropout` after the model is
    already loaded would silently do nothing to actual training behavior:
    reading the installed transformers version's own `modeling_m2m_100.py`
    shows `M2M100Encoder`/`M2M100Decoder`/`M2M100EncoderLayer`/
    `M2M100DecoderLayer` each copy `config.dropout` into their own
    `self.dropout` plain-float attribute exactly once, at `__init__` time
    (`nn.functional.dropout(hidden_states, p=self.dropout, ...)`) -- and
    `model` here is always already constructed (via `from_pretrained`)
    before this function ever runs. This walks `model.modules()` and
    overwrites each submodule's own `.dropout` attribute directly, in
    addition to `model.config.dropout` (kept in sync mainly for
    provenance/introspection, e.g. if the checkpoint's config is ever
    re-read later).

    Deliberately skips every `M2M100Attention` instance (checked via
    `isinstance`, not a class-name string match -- code review on PR #183:
    a name match would silently stop excluding attention modules the
    moment a future transformers version introduces a subclass or
    alternate attention implementation, e.g. an SDPA/FlashAttention
    variant, which `isinstance` correctly still recognizes via the class
    hierarchy): its own `.dropout` attribute is a *different* config value
    (`config.attention_dropout`), not the general dropout this flag
    controls -- overwriting it too would silently change attention dropout
    as an undocumented side effect of `--dropout`. This project doesn't
    expose a separate `--attention-dropout` flag; only the general
    residual/hidden-state dropout is in scope for this ticket.

    No-op when `dropout` is `None` -- `--dropout`'s own default, so a run
    that never passes the flag leaves the checkpoint's pretrained dropout
    completely untouched (baseline behavior for every past run is
    unaffected unless someone explicitly opts in). Lazily imports
    `M2M100Attention` so this stays a no-cost no-op (no transformers import
    at all) on that common path, mirroring this module's existing
    lazy-import pattern (see `load_base_model_and_tokenizer`).
    """
    if dropout is None:
        return

    from transformers.models.m2m_100.modeling_m2m_100 import M2M100Attention

    model.config.dropout = dropout
    for module in model.modules():
        if isinstance(module, M2M100Attention):
            continue
        if isinstance(getattr(module, "dropout", None), float):
            module.dropout = dropout


def load_base_model_and_tokenizer(base_model: str) -> tuple[Any, Any]:
    """Load the real `M2M100Tokenizer` + `M2M100ForConditionalGeneration`
    from the Hugging Face Hub. Lazily imports `transformers` so this module
    can be imported (e.g. for `parse_args` unit tests) without requiring a
    full `transformers`/`torch` install, mirroring the laziness pattern in
    `training/tokenizer_extension.py`.
    """
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    tokenizer = M2M100Tokenizer.from_pretrained(base_model)
    model = M2M100ForConditionalGeneration.from_pretrained(base_model)
    return tokenizer, model


def extend_vocabulary_for_examples(
    tokenizer: Any,
    model: Any,
    examples: list[TranslationExample],
    *,
    seed: int | None = None,
    subword_vocab_size: int = DEFAULT_SUBWORD_VOCAB_SIZE,
) -> list[str]:
    """Extend `tokenizer`/`model` to cover the Kaqchikel text in `examples`,
    plus this script's own direction tag tokens (`training.direction`), so
    they get real, warm-started embedding rows too rather than falling back
    to whatever `add_tokens` would leave uninitialized.

    Two complementary extension steps, run in sequence (issue #82), both
    scoped to the Kaqchikel-only side of `examples`
    (`training.direction.collect_texts_for_language`) plus the direction
    tags -- never mixed with Spanish source/target text, which M2M100
    already tokenizes natively:

    1. `extend_tokenizer_vocab` -- character/whole-word gap coverage
       (#34/#62).
    2. `extend_tokenizer_vocab_with_subwords` -- a Kaqchikel-only
       SentencePiece/Unigram subword vocabulary (`training.subword_vocab`),
       diffed against the tokenizer's vocab *after* step 1 (so it never
       re-proposes a whole word/character already added there). This
       targets the subword-fragmentation ceiling three real training runs'
       diminishing BLEU-vs-loss returns pointed at (see
       `training/subword_vocab.py`'s module docstring for the full
       rationale).

    Step 1 originally also fed every example's Spanish source/target text
    into `extend_tokenizer_vocab` (issue #125): `find_missing_words`
    checks whether a word's exact literal surface form is already a single
    vocab *key*, not whether the tokenizer can already represent it at all
    via composition of existing subwords -- so it flagged the large
    majority of ordinary Spanish words in a real corpus as "missing" and
    registered a new standalone token for each one, even though the base
    M2M100 tokenizer already encodes/decodes them correctly (confirmed
    empirically against the real checkpoint: of a real sample of Spanish
    words this step flagged as missing, 100% were already fully
    representable with zero `<unk>` tokens). That contradicted this same
    function's own rationale for isolating Spanish out of step 2, and
    roughly doubled real vocabulary growth with tokens that don't fill a
    genuine gap -- see issue #125 for the full classification. Both steps
    are now scoped the same way.

    A single `resize_embeddings_for_new_tokens` call at the end covers the
    combined total from both steps, so the model's embedding matrix is
    resized exactly once per run.

    Returns the list of tokens actually added by either step (may be
    empty).
    """
    kaqchikel_texts = collect_texts_for_language(examples, KAQCHIKEL)

    sample_texts = list(kaqchikel_texts)
    sample_texts.extend(ALL_DIRECTION_TAG_TOKENS)

    added_tokens = extend_tokenizer_vocab(tokenizer, sample_texts)

    if kaqchikel_texts:
        added_tokens = added_tokens + extend_tokenizer_vocab_with_subwords(
            tokenizer, kaqchikel_texts, vocab_size=subword_vocab_size
        )

    resize_embeddings_for_new_tokens(model, len(added_tokens), seed=seed)
    return added_tokens


DEFAULT_SUBWORD_SAMPLING_NBEST_SIZE = -1


@contextlib.contextmanager
def enable_subword_sampling(
    tokenizer: Any, *, alpha: float, nbest_size: int = DEFAULT_SUBWORD_SAMPLING_NBEST_SIZE
) -> Iterator[Any]:
    """Context manager enabling SentencePiece subword sampling ("BPE-dropout"
    / subword regularization, issue #182) on `tokenizer`, for the lifetime
    of the `with` block only.

    ## Why this needs to be a temporary, call-scoped patch

    `M2M100Tokenizer._tokenize` calls `self.sp_model.encode(text,
    out_type=str)` with no sampling kwargs at all (confirmed by reading the
    installed transformers version's own `tokenization_m2m_100.py`).
    SentencePiece's own `enable_sampling`/`nbest_size`/`alpha` support (see
    the `sentencepiece` Python wrapper's `SentencePieceProcessor.Encode`
    signature) can be supplied per call, or baked into a `sp_model_kwargs`
    dict at tokenizer construction time -- but baking it in at construction
    would apply to *every* future call on that tokenizer instance,
    including this same training run's post-training validation-set
    generation (`generate_translations`) and any later re-evaluation
    (`evaluation.evaluate_checkpoint`), silently making those non-
    deterministic too. Training uses one shared tokenizer instance for
    vocabulary extension, fine-tuning, and generation (see `run_training_job`),
    so this instead monkeypatches `tokenizer._tokenize` just for the
    duration of this context, restoring the original afterward (even if the
    `with` block raises) -- every other caller of the same tokenizer
    instance, outside the `with` block, is completely unaffected and keeps
    getting deterministic, "best" SentencePiece segmentation.

    ## Pluggable without touching the vocab-extension pipeline

    Deliberately independent of `training.vocab_extension`/`vocab_gap`/
    `subword_vocab`: none of those call `tokenizer.sp_model.encode` on the
    live fine-tuning tokenizer at all -- they either operate on plain vocab
    dicts (`vocab_gap`/`vocab_extension`), or train an entirely separate,
    throwaway SentencePiece model on Kaqchikel-only text
    (`subword_vocab.train_subword_model`) whose *output* (which new tokens
    to add) is unaffected by whether sampling is enabled on a completely
    different, already-loaded tokenizer object. This function only ever
    wraps `TranslationDataset`'s own training-time encoding calls (see
    `TranslationDataset.__getitem__`).

    No-ops for tokenizer-like objects without a real `sp_model` (e.g. the
    duck-typed fakes used elsewhere in this test suite that only implement
    `get_vocab`/`add_tokens`) -- lets callers exercise the wiring without a
    full SentencePiece dependency.
    """
    if not hasattr(tokenizer, "sp_model"):
        yield tokenizer
        return

    sp_model = tokenizer.sp_model
    original_tokenize = tokenizer._tokenize

    def _sampled_tokenize(text: str) -> list[str]:
        return sp_model.encode(
            text, out_type=str, enable_sampling=True, nbest_size=nbest_size, alpha=alpha
        )

    tokenizer._tokenize = _sampled_tokenize
    try:
        yield tokenizer
    finally:
        tokenizer._tokenize = original_tokenize


class TranslationDataset:
    """Tokenizes `TranslationExample`s into `Seq2SeqTrainer`-ready dicts.

    Direction-tags each example's source text (`training.direction.
    tag_source_text`) before encoding, and builds labels that start with
    the same direction-tag token id used as `forced_bos_token_id` at
    generation time (`generate_translations`), so a single multilingual
    checkpoint can be trained on examples from both directions at once
    (see `training/direction.py`).

    Labels are built *without* the tokenizer's `text_target=` kwarg,
    deliberately. M2M100Tokenizer's `text_target=` path calls
    `_switch_to_target_mode()`, which requires `tokenizer.tgt_lang` to
    already be a valid entry in its own pretrained language table
    (`lang_code_to_id`) -- but this project's direction tags (ADR 0006)
    exist specifically *because* Kaqchikel isn't in that table, and
    `tgt_lang` is never set anywhere in this script. Calling
    `tokenizer(text_target=...)` therefore crashes with `KeyError: None`
    (confirmed against the real checkpoint -- this surfaced on the first
    real, billable training run, since a duck-typed fake tokenizer has no
    reason to reproduce this real API's internal state requirements).
    Encoding target text as plain (non-target) text and prepending our
    own direction-tag token id manually sidesteps that internal mechanism
    entirely, and also keeps training consistent with generation: the
    decoder is trained to predict the exact same first token
    (`forced_bos_token_id`) it will later be forced to start from.

    `subword_dropout_alpha` (issue #182, opt-in via `--bpe-dropout-alpha`,
    default `None`/disabled) wraps both the source and target encoding
    calls in `enable_subword_sampling`, so each call to `__getitem__`
    -- i.e. potentially every epoch, since `Seq2SeqTrainer`'s dataloader
    re-fetches each example every epoch -- can get a different sampled
    subword segmentation of the same underlying text (subword
    regularization / "BPE-dropout": Kudo 2018, "Subword Regularization").
    Deliberately a per-instance setting rather than always-on: only the
    *training* dataset should pass this (see `fine_tune`) -- the
    evaluation dataset must stay deterministic, since sampled segmentation
    would make validation-loss numbers vary run-to-run for reasons
    unrelated to model quality.
    """

    def __init__(
        self,
        examples: list[TranslationExample],
        tokenizer: Any,
        max_length: int,
        *,
        subword_dropout_alpha: float | None = None,
    ):
        self._examples = examples
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._subword_dropout_alpha = subword_dropout_alpha

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self._examples[index]
        tagged_source = tag_source_text(example.source_text, example.target_lang)
        target_tag_id = self._tokenizer.convert_tokens_to_ids(
            DIRECTION_TAGS[example.target_lang]
        )
        target_budget = max(1, self._max_length - 2)  # room for the tag + eos below

        with self._encoding_context():
            model_inputs = self._tokenizer(
                tagged_source, max_length=self._max_length, truncation=True
            )
            target_ids = self._tokenizer(
                example.target_text,
                add_special_tokens=False,
                max_length=target_budget,
                truncation=True,
            )["input_ids"]

        labels = [target_tag_id, *target_ids, self._tokenizer.eos_token_id]

        model_inputs["labels"] = labels
        return model_inputs

    def _encoding_context(self) -> contextlib.AbstractContextManager[Any]:
        if self._subword_dropout_alpha is None:
            return contextlib.nullcontext(self._tokenizer)
        return enable_subword_sampling(self._tokenizer, alpha=self._subword_dropout_alpha)


def shift_tokens_right(labels: Any, pad_token_id: int, decoder_start_token_id: int) -> Any:
    """Shift `labels` one position to the right to build the decoder's
    teacher-forcing input, prepending `decoder_start_token_id` and
    replacing any `-100` loss-ignore padding with `pad_token_id`.

    Delegates to the real
    `transformers.models.m2m_100.modeling_m2m_100.shift_tokens_right` (the
    exact function `M2M100ForConditionalGeneration.forward`'s own `labels
    is not None` branch already calls internally) rather than maintaining
    an independent, from-scratch reimplementation of the same transform --
    code review on PR #183: a hand-copied reimplementation could silently
    drift from upstream if a future transformers version changes this
    function's semantics, whereas delegating means any such change is
    picked up automatically and this project's own behavior stays exactly
    in sync with what the model's own forward pass would have done anyway.
    See `attach_decoder_input_ids`'s docstring for why this needs to be
    precomputed explicitly rather than left to that internal branch alone
    (issue #182's label-smoothing crash fix). Lazily imports `transformers`,
    mirroring this module's existing lazy-import pattern (see
    `load_base_model_and_tokenizer`).
    """
    from transformers.models.m2m_100.modeling_m2m_100 import (
        shift_tokens_right as real_shift_tokens_right,
    )

    return real_shift_tokens_right(labels, pad_token_id, decoder_start_token_id)


def attach_decoder_input_ids(
    batch: dict[str, Any], pad_token_id: int, decoder_start_token_id: int
) -> dict[str, Any]:
    """Mutate `batch` in place, adding a `decoder_input_ids` field computed
    from its `labels` field via `shift_tokens_right` -- unless `batch` has
    no (or a `None`) `labels` field, in which case this is a no-op.

    ## Why this can't just be left to `DataCollatorForSeq2Seq`/the model

    `DataCollatorForSeq2Seq(tokenizer, model=model)` only builds
    `decoder_input_ids` itself when
    `hasattr(model, "prepare_decoder_input_ids_from_labels")` -- a
    convenience hook the installed `transformers` version's
    `M2M100ForConditionalGeneration` no longer implements at all
    (confirmed directly: `hasattr(...)` is `False` for this model class in
    this version, though several other model families still define it).
    Without label smoothing this silently doesn't matter, because
    `Trainer.compute_loss` leaves `"labels"` in the batch it hands to
    `model(**inputs)`, and `M2M100ForConditionalGeneration.forward`'s own
    `if labels is not None: decoder_input_ids = shift_tokens_right(...)`
    branch builds it internally instead. But `--label-smoothing > 0` makes
    `Trainer` set a `label_smoother`, and `Trainer.compute_loss` then pops
    `"labels"` out of the batch *before* calling the model (so it can
    apply smoothing itself against the raw logits) -- so that internal
    branch never fires, both `decoder_input_ids` and `decoder_inputs_embeds`
    end up `None`, and `M2M100Decoder.forward`'s own "exactly one of these
    two must be given" guard raises `ValueError: You cannot specify both
    decoder_input_ids and decoder_inputs_embeds at the same time` -- a
    real, confirmed (not assumed) incompatibility whose wording describes
    the "both given" half of that guard, not the "neither given" half that
    actually fires here. See `tests/unit/test_decoder_input_ids.py`'s
    module docstring for the full traced call stack.

    Computing `decoder_input_ids` here, unconditionally, *before*
    `Trainer.compute_loss` gets a chance to pop `"labels"`, fixes this for
    every value of `--label-smoothing` at once, without depending on a
    model-specific hook this transformers version doesn't provide for
    M2M100. It changes nothing for the existing (already-working, no
    label smoothing) path: `shift_tokens_right` here computes the exact
    same tensor the model's own internal branch would have, so passing it
    in explicitly is a no-op change to what the decoder actually sees --
    confirmed with a real training step in
    `tests/integration/test_fine_tune_real_checkpoint.py`.
    """
    labels = batch.get("labels")
    if labels is None:
        return batch
    batch["decoder_input_ids"] = shift_tokens_right(labels, pad_token_id, decoder_start_token_id)
    return batch


def build_data_collator(tokenizer: Any, model: Any) -> Callable[[list[dict[str, Any]]], Any]:
    """Build the `data_collator` callable `fine_tune` passes to
    `Seq2SeqTrainer`: the standard `transformers.DataCollatorForSeq2Seq`,
    plus an explicit `attach_decoder_input_ids` pass (issue #182 -- see its
    own docstring for the full label-smoothing-crash root cause) so
    `decoder_input_ids` is always present in the collated batch, regardless
    of whether `--label-smoothing` is enabled.
    """
    from transformers import DataCollatorForSeq2Seq

    base_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    pad_token_id = model.config.pad_token_id
    decoder_start_token_id = model.config.decoder_start_token_id

    def _collate(features: list[dict[str, Any]]) -> Any:
        batch = base_collator(features)
        return attach_decoder_input_ids(batch, pad_token_id, decoder_start_token_id)

    return _collate


def build_training_arguments(args: argparse.Namespace, num_train_examples: int) -> Any:
    """Build the `transformers.Seq2SeqTrainingArguments` for `fine_tune`.

    Pulled out into its own function specifically so it's directly unit
    testable (see `tests/unit/test_build_training_arguments.py`) without
    needing a real model/tokenizer/forward pass -- `Seq2SeqTrainingArguments`
    is a cheap, CPU-safe dataclass to construct on its own, unlike the rest
    of `fine_tune`. This closes a real gap: before issue #79, the
    regularization/schedule settings added here (warmup, weight decay,
    label smoothing, gradient accumulation) had zero test coverage, since
    `fine_tune` as a whole is never exercised directly by the CPU-only test
    suite (see `fine_tune`'s own docstring).

    `--warmup-ratio` is converted to an absolute `warmup_steps` count here
    rather than passed straight through: the installed `transformers`
    version's `Seq2SeqTrainingArguments` no longer accepts a `warmup_ratio`
    kwarg at all (only `warmup_steps`) -- confirmed by a real `TypeError`
    when this was first written directly, not assumed from documentation.
    `num_train_examples` is needed to compute total optimizer steps
    (`ceil(num_train_examples / (batch_size * gradient_accumulation_steps))
    * epochs`) that the ratio is a fraction of.
    """
    import torch
    from transformers import Seq2SeqTrainingArguments

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(num_train_examples / effective_batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = round(args.warmup_ratio * total_steps)

    return Seq2SeqTrainingArguments(
        # Issue #187: SageMaker's own separate checkpoint-sync directory,
        # never nested under args.model_dir -- see DEFAULT_CHECKPOINT_DIR's
        # module-level comment for the full rationale (SageMaker tars
        # args.model_dir whole into the registered model.tar.gz artifact;
        # bundling per-epoch checkpoints into *that* is exactly what grew a
        # real run's artifact to 28.4-30.5 GB before this fix).
        output_dir=args.checkpoint_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        # Not a CLI flag -- a real-GPU speed/cost optimization, not
        # something to experiment with per run. Conditional on actual CUDA
        # availability (rather than always True) so this same function
        # works correctly for the real GPU training job *and* for a real
        # local/CI CPU run (see tests/integration/test_fine_tune_real_checkpoint.py)
        # -- fp16=True with no CUDA fails at Trainer.train() time, not at
        # construction, so this can't just be hardcoded and left to callers
        # to override.
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        # Issue #182: per-epoch checkpoints, not "no" -- previously no
        # intermediate checkpoint ever existed for any past run, which made
        # checkpoint averaging (a cheap-tier quality lever worth measuring
        # in its own dedicated future run) structurally impossible: there
        # was nothing to average. `training/checkpoint_averaging.py`
        # consumes the resulting `checkpoint-<step>` directories under
        # `args.checkpoint_dir` (issue #187 moved this out from under
        # `args.model_dir/checkpoints/` -- see DEFAULT_CHECKPOINT_DIR's own
        # comment). Uses the epoch-based strategy
        # (rather than a steps-based one) since "average the final N
        # epochs" is this ticket's explicit framing and epoch boundaries
        # are already meaningful checkpoints for this project's small
        # per-run epoch counts (3-8 in every real run to date) -- a
        # steps-based schedule would need its own tuning to land on
        # comparable boundaries and isn't needed to make averaging
        # possible.
        save_strategy="epoch",
        # Code review on PR #183: bounds disk usage regardless of epoch
        # count -- see DEFAULT_SAVE_TOTAL_LIMIT's own module-level comment.
        # transformers' own FIFO eviction (oldest checkpoint deleted first)
        # is exactly what checkpoint averaging wants: it only ever needs the
        # *most recent* N checkpoints, which are always the last ones
        # standing under this policy regardless of how many epochs a run
        # trains for.
        save_total_limit=DEFAULT_SAVE_TOTAL_LIMIT,
        report_to=[],
    )


def fine_tune(
    model: Any,
    tokenizer: Any,
    train_examples: list[TranslationExample],
    eval_examples: list[TranslationExample],
    args: argparse.Namespace,
) -> Any:
    """Fine-tune `model` via `transformers.Seq2SeqTrainer`.

    This is the one real gradient-descent step in this script, and is
    deliberately never exercised directly by the automated test suite --
    `tests/integration/test_train_pipeline.py` injects a fake in its place
    via `run_training_job`'s `trainer` parameter, since a real forward/
    backward pass is out of scope for a fast wiring test (and running a
    real multi-epoch fine-tune here would be issue #66's job, not this
    one's). `build_training_arguments` above is tested directly instead.
    """
    from transformers import Seq2SeqTrainer

    train_dataset = TranslationDataset(
        train_examples,
        tokenizer,
        args.max_length,
        subword_dropout_alpha=args.bpe_dropout_alpha,
    )
    eval_dataset = (
        TranslationDataset(eval_examples, tokenizer, args.max_length) if eval_examples else None
    )

    training_args = build_training_arguments(args, len(train_examples))
    data_collator = build_data_collator(tokenizer, model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    return model


def generate_translations(
    model: Any,
    tokenizer: Any,
    examples: list[TranslationExample],
    *,
    max_length: int = 128,
    batch_size: int = 16,
    num_beams: int = DEFAULT_NUM_BEAMS,
    logits_processor: Any = None,
) -> list[str]:
    """Generate hypothesis translations for `examples` using `model.generate`.

    Groups examples by target language so each batch uses the correct
    `forced_bos_token_id` (the direction tag token for that target
    language -- see `training.direction`), then generates with beam search.

    `num_beams` (issue #180) is now an explicit, named parameter of this
    function, defaulting to `DEFAULT_NUM_BEAMS` (5) -- matching the base
    model's own `generation_config.json` value, so default behavior is
    unchanged. Before this, no `num_beams` (or other decode-strategy kwarg)
    was ever passed to `model.generate()` here -- decode strategy came
    entirely from whatever the loaded checkpoint's own
    `generation_config.json` happened to specify, an implicit, easy-to-miss
    dependency rather than a real, overridable parameter of this project's
    own generation code. This function's docstring previously (incorrectly)
    said "generates greedily"; issue #178 confirmed directly against the
    real v7 checkpoint that generation is actually beam search (`num_beams:
    5`, inherited unmodified from `facebook/m2m100_418M`'s own
    `generation_config.json` through every fine-tuning save/reload), not
    greedy decoding -- see `ml/README.md`'s "Decode configuration" section
    for the full verification. Issue #178's own decode-parameter sweep (a
    small, n=80 sample) found `num_beams=8` may beat this default; issue
    #180 is what made overriding it here actually possible, to validate
    that finding at scale.

    Moves each batch's encoded tensors to `model.device` before calling
    `generate` -- `tokenizer(..., return_tensors="pt")` always returns
    CPU tensors regardless of where `model` lives, and `Seq2SeqTrainer`
    only handles device placement for its own training batches, not for
    a manually-written generation loop like this one. Missing this crashed
    the third real training run (issue #66) after ~3 hours of otherwise-
    successful training, on the very last step (validation-set generation
    for the model card): `RuntimeError: Expected all tensors to be on the
    same device, but got index is on cpu, different from other tensors on
    cuda:0`. This is a GPU-only failure mode -- on the CPU-only CI runners
    this repo's test suite runs on, `model.device` is always `cpu` too, so
    no mismatch is possible there regardless of whether `.to(model.device)`
    is called; the call itself is unit-tested directly instead (see
    `tests/unit/test_generate_translations.py`) so this can't silently
    regress even without real GPU hardware to reproduce the crash on.

    Strips a leftover leading direction-tag token from each decoded
    hypothesis via `training.direction.strip_leading_direction_tag` (issue
    #106): `skip_special_tokens=True` alone does not strip `__cak__` (an
    ordinary added-vocab token, unlike the real special token `__es__`),
    so every es->cak hypothesis previously came out as e.g. `"__cak__ Utz
    awäch?"` instead of `"Utz awäch?"`, corrupting the BLEU/chrF this
    function's output is used to compute for every real training run to
    date -- see `strip_leading_direction_tag`'s docstring and
    `ml/README.md` for the full history. This mirrors the fix already
    applied to `deployment/inference.py`'s real-time serving path (issue
    #8); both now share the one implementation.

    `logits_processor` (issue #193, default `None`) is passed straight
    through to `model.generate(...)` only when supplied -- omitted
    entirely otherwise, so every existing caller's decode behavior is
    completely unchanged. `evaluation.evaluate_checkpoint`'s
    `--debias-delta-multiplier` builds one via
    `evaluation.length_bias.build_length_bias_logits_processor` to apply
    the label-smoothing length-bias rectification (Liang, Wang & Cao,
    arXiv:2205.00659) at real decode time, without this shared generation
    function needing to know anything about that paper's math itself.
    """
    hypotheses: list[str | None] = [None] * len(examples)
    indices_by_target_lang: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        indices_by_target_lang.setdefault(example.target_lang, []).append(index)

    for target_lang, indices in indices_by_target_lang.items():
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(DIRECTION_TAGS[target_lang])
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_texts = [
                tag_source_text(examples[i].source_text, target_lang) for i in batch_indices
            ]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(model.device)
            generate_kwargs: dict[str, Any] = {
                "forced_bos_token_id": forced_bos_token_id,
                "max_length": max_length,
                "num_beams": num_beams,
            }
            if logits_processor is not None:
                generate_kwargs["logits_processor"] = logits_processor
            generated_ids = model.generate(**encoded, **generate_kwargs)
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for i, text in zip(batch_indices, decoded, strict=True):
                hypotheses[i] = strip_leading_direction_tag(text, target_lang)

    assert all(h is not None for h in hypotheses)
    return hypotheses  # type: ignore[return-value]


def save_model_and_tokenizer(model: Any, tokenizer: Any, model_dir: str) -> None:
    """Save the fine-tuned model + tokenizer to `model_dir` (`SM_MODEL_DIR`
    in a real Training Job), for SageMaker to upload as the job's model
    artifact. Trained weights are never published (see ADR 0002/CLAUDE.md)
    -- this only ever writes to the job's private model output directory.
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def run_training_job(
    args: argparse.Namespace,
    *,
    model_loader: Callable[[str], tuple[Any, Any]] = load_base_model_and_tokenizer,
    trainer: Callable[
        [Any, Any, list[TranslationExample], list[TranslationExample], argparse.Namespace], Any
    ] = fine_tune,
    translator: Callable[..., list[str]] = generate_translations,
) -> Path:
    """Run the full training job: load corpus -> extend vocab -> fine-tune
    -> save -> evaluate -> write model card. Returns the path to the
    written model card.

    Registration in SageMaker Model Registry is not this script's job (ADR
    0009): `submit_job.py` registers a completed, waited-for run's model
    client-side after this process exits, using its own already-fixed
    disk-streamed artifact download (issues #188/#189) -- see
    `submit_job.py`'s module docstring. `train.py` itself needs zero AWS
    credentials or SDK calls.

    `model_loader`/`trainer`/`translator` default to the real
    implementations above; tests inject fakes in their place (see this
    module's own docstring and `tests/integration/test_train_pipeline.py`).
    """
    run_id = args.run_id or datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")

    train_pairs = read_tsv_pairs(args.train)
    val_pairs = read_tsv_pairs(args.validation)

    train_examples = build_direction_examples(train_pairs, args.direction)
    val_examples = build_direction_examples(val_pairs, args.direction)

    model_source = resolve_model_source(args.init_model) if args.init_model else args.base_model
    # Computed before training touches anything -- see this function's own
    # docstring / `_resolve_vocab_extension_scoping_for_model_card` (issue
    # #143, PR #144 review) for why this can't just always be
    # `VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY`: a `--init-model` ancestor
    # whose own lineage wasn't fully Kaqchikel-only scoped means this
    # checkpoint's real vocabulary still carries pre-#125, unscoped tokens
    # inherited from it.
    vocab_extension_scoping_for_card = _resolve_vocab_extension_scoping_for_model_card(
        init_model=args.init_model, model_source=model_source
    )
    tokenizer, model = model_loader(model_source)
    apply_dropout_config(model, args.dropout)

    added_tokens = extend_vocabulary_for_examples(
        tokenizer,
        model,
        train_examples,
        seed=args.seed,
        subword_vocab_size=args.subword_vocab_size,
    )

    model = trainer(model, tokenizer, train_examples, val_examples, args)

    # Issue #175: `Seq2SeqTrainer.train()` leaves `model` in whatever mode
    # its own last internal training step used -- typically `.train()`
    # (dropout active) -- since it has no reason to switch back once its
    # internal train/eval loop finishes. Left uncorrected, the
    # `generate_translations` call below runs with dropout still active,
    # injecting real stochastic noise into generation and producing a
    # systematically *worse* (noisier) BLEU/chrF than the checkpoint's true
    # quality -- confirmed against the real checkpoint: forcing
    # `model.train()` before generation on a fixed slice measurably lowered
    # BLEU relative to `model.eval()` on the same weights/examples.
    #
    # Called before `save_model_and_tokenizer` too, though the order
    # relative to that call doesn't actually matter for the *saved*
    # artifact either way: `save_pretrained`/`from_pretrained` never
    # persists train/eval mode at all -- confirmed directly against the
    # real checkpoint, `PreTrainedModel.from_pretrained` unconditionally
    # calls `model.eval()` itself at the end of loading, regardless of
    # what mode the object being saved was in (see
    # tests/integration/test_save_reload_does_not_persist_train_mode.py).
    # What matters here is this function's own in-memory `model` object,
    # which is what `generate_translations` actually runs against.
    model.eval()

    save_model_and_tokenizer(model, tokenizer, args.model_dir)

    hypotheses = translator(model, tokenizer, val_examples, max_length=args.max_length)
    references = [example.target_text for example in val_examples]
    # Issue #178: recorded alongside predictions/references so BLEU/chrF can
    # be bucketed per direction (es->cak vs. cak->es) rather than only ever
    # reported as one combined number -- see evaluation.run.run_evaluation's
    # `directions_path` and evaluation.metrics.compute_metrics_by_direction.
    directions = [f"{example.source_lang}->{example.target_lang}" for example in val_examples]

    output_dir = Path(args.output_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.txt"
    references_path = output_dir / "references.txt"
    directions_path = output_dir / "directions.txt"
    predictions_path.write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
    references_path.write_text("\n".join(references) + "\n", encoding="utf-8")
    directions_path.write_text("\n".join(directions) + "\n", encoding="utf-8")

    notes_parts = []
    if args.direction == "both":
        notes_parts.append(
            "Single multilingual checkpoint trained with explicit direction "
            "tags for both es->cak and cak->es (see training/direction.py "
            "and ADR 0006) rather than two separate checkpoints."
        )
    if args.init_model:
        notes_parts.append(
            f"Continued training from a previous checkpoint ({args.init_model}) "
            "rather than starting from the base pretrained model -- "
            "train_sentence_count/hyperparameters below describe only this "
            "run's additional training, not the full cumulative history."
        )
    notes = " ".join(notes_parts) or None

    hyperparameters = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "subword_vocab_size": args.subword_vocab_size,
        "new_tokens_added": len(added_tokens),
        "resumed_from_checkpoint": bool(args.init_model),
    }
    # Issue #182: both recorded only when explicitly set (mirroring
    # vocab_extension_scoping's own "field absent = not applicable"
    # convention just below), so a run that never opts into either lever
    # renders an identical model card to one from before this ticket --
    # no visual diff on every ordinary run just because these flags exist.
    if args.dropout is not None:
        hyperparameters["dropout"] = args.dropout
    if args.bpe_dropout_alpha is not None:
        hyperparameters["bpe_dropout_alpha"] = args.bpe_dropout_alpha
    # Omitted entirely (rather than recorded as some "unscoped" sentinel)
    # when the lineage isn't fully Kaqchikel-only scoped -- matching
    # evaluation.evaluate_checkpoint's existing "field absent = legacy"
    # convention for every checkpoint that predates issue #125 outright
    # (issue #143, PR #144 review).
    if vocab_extension_scoping_for_card is not None:
        hyperparameters["vocab_extension_scoping"] = vocab_extension_scoping_for_card

    run_metadata = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "base_model": args.base_model,
        "direction": args.direction,
        "corpus_version": args.corpus_version,
        "train_sentence_count": len(train_pairs),
        "hyperparameters": hyperparameters,
        "notes": notes,
    }

    model_card_path = Path(args.model_dir) / "model_card.md"
    _metrics, model_card_path = run_evaluation(
        predictions_path,
        references_path,
        run_metadata,
        model_card_path,
        directions_path=directions_path,
    )

    return model_card_path


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_training_job(args)


if __name__ == "__main__":
    main()
