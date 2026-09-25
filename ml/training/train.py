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
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data.corpus_io import read_tsv_pairs
from evaluation.run import run_evaluation
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
            "Defaults to 0.0 (disabled): setting this above 0 crashes the "
            "real 5th real-hyperparameter training run with `ValueError: "
            "You cannot specify both decoder_input_ids and "
            "decoder_inputs_embeds at the same time`, a real incompatibility "
            "between the installed transformers version's label-smoothing "
            "loss path and M2M100's forward signature -- confirmed by "
            "reproducing it locally against the real checkpoint with a tiny "
            "dataset (see tests/integration/test_fine_tune_real_checkpoint.py, "
            "which guards against this specific regression), not assumed. "
            "Only override this if a transformers upgrade is confirmed to "
            "have fixed the interaction."
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
    """

    def __init__(self, examples: list[TranslationExample], tokenizer: Any, max_length: int):
        self._examples = examples
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self._examples[index]
        tagged_source = tag_source_text(example.source_text, example.target_lang)
        model_inputs = self._tokenizer(
            tagged_source, max_length=self._max_length, truncation=True
        )

        target_tag_id = self._tokenizer.convert_tokens_to_ids(
            DIRECTION_TAGS[example.target_lang]
        )
        target_budget = max(1, self._max_length - 2)  # room for the tag + eos below
        target_ids = self._tokenizer(
            example.target_text,
            add_special_tokens=False,
            max_length=target_budget,
            truncation=True,
        )["input_ids"]
        labels = [target_tag_id, *target_ids, self._tokenizer.eos_token_id]

        model_inputs["labels"] = labels
        return model_inputs


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
        output_dir=str(Path(args.model_dir) / "checkpoints"),
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
        save_strategy="no",
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
    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer

    train_dataset = TranslationDataset(train_examples, tokenizer, args.max_length)
    eval_dataset = (
        TranslationDataset(eval_examples, tokenizer, args.max_length) if eval_examples else None
    )

    training_args = build_training_arguments(args, len(train_examples))
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

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
) -> list[str]:
    """Generate hypothesis translations for `examples` using `model.generate`.

    Groups examples by target language so each batch uses the correct
    `forced_bos_token_id` (the direction tag token for that target
    language -- see `training.direction`), then generates greedily.

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
            generated_ids = model.generate(
                **encoded, forced_bos_token_id=forced_bos_token_id, max_length=max_length
            )
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
    tokenizer, model = model_loader(model_source)

    added_tokens = extend_vocabulary_for_examples(
        tokenizer,
        model,
        train_examples,
        seed=args.seed,
        subword_vocab_size=args.subword_vocab_size,
    )

    model = trainer(model, tokenizer, train_examples, val_examples, args)

    save_model_and_tokenizer(model, tokenizer, args.model_dir)

    hypotheses = translator(model, tokenizer, val_examples, max_length=args.max_length)
    references = [example.target_text for example in val_examples]

    output_dir = Path(args.output_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.txt"
    references_path = output_dir / "references.txt"
    predictions_path.write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
    references_path.write_text("\n".join(references) + "\n", encoding="utf-8")

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

    run_metadata = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "base_model": args.base_model,
        "direction": args.direction,
        "corpus_version": args.corpus_version,
        "train_sentence_count": len(train_pairs),
        "hyperparameters": {
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
        },
        "notes": notes,
    }

    model_card_path = Path(args.model_dir) / "model_card.md"
    _metrics, model_card_path = run_evaluation(
        predictions_path, references_path, run_metadata, model_card_path
    )
    return model_card_path


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_training_job(args)


if __name__ == "__main__":
    main()
