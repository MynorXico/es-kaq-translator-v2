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

This script recovers it: `--train`/`--train-direction` (both required)
must name the exact training corpus and direction the checkpoint's own
training history used (from that run's own model card/hyperparameters --
*not* necessarily the same as this script's own `--direction`, which only
controls what to *evaluate*, not what the checkpoint was *trained* with).
Given those, `training.tokenizer_extension.
patch_word_boundary_decoding_for_checkpoint` deterministically recomputes
the same whole-word boundary token set the checkpoint's training run(s)
actually added (no SentencePiece retraining, no randomness -- see that
function's docstring) and patches the reloaded tokenizer's decode behavior
in place, before any translation is generated. `--base-model` (already
recorded for provenance) is also used here to load the *pristine* base
tokenizer vocab this reconstruction is computed against. The number of
boundary tokens reconstructed/applied is recorded in the model card's
hyperparameters for traceability, and a warning is logged (not raised) if
any reconstructed token turns out to be missing from the checkpoint's own
vocabulary -- a sign `--train`/`--train-direction`/`--base-model` don't
actually match what the checkpoint was trained with.

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
"""

from __future__ import annotations

import argparse
import re
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data.corpus_io import read_tsv_pairs
from evaluation.run import run_evaluation
from training.direction import ALL_DIRECTION_TAG_TOKENS, DIRECTION_CHOICES, build_direction_examples
from training.tokenizer_extension import patch_word_boundary_decoding_for_checkpoint
from training.train import DEFAULT_BASE_MODEL, generate_translations, resolve_model_source

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
            "silently under-report or over-report issue #116's fix."
        ),
    )
    parser.add_argument(
        "--train-direction",
        choices=DIRECTION_CHOICES,
        default="both",
        help=(
            "The --direction value the checkpoint's own training run(s) "
            "used to build the sample text its vocabulary was extended "
            "against (training.train's own default is 'both'). This is "
            "*not* the same as this script's own --direction below, which "
            "only controls what to evaluate -- get this wrong and the "
            "word-boundary reconstruction below will be silently "
            "incorrect."
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


def load_checkpoint_tokenizer_and_model(source: str) -> tuple[Any, Any]:
    """Load the real `M2M100Tokenizer` + `M2M100ForConditionalGeneration`
    directly from `source` (a local checkpoint directory produced by a
    prior `training.train.save_model_and_tokenizer` call). Its vocabulary
    is already extended and its weights already fine-tuned -- nothing
    further to load or extend here. Lazily imports `transformers`, same
    laziness pattern as `training.train.load_base_model_and_tokenizer`.
    """
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    tokenizer = M2M100Tokenizer.from_pretrained(source)
    model = M2M100ForConditionalGeneration.from_pretrained(source)
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


def _warn_if_boundary_tokens_missing_from_checkpoint(
    boundary_tokens: list[str], tokenizer: Any
) -> None:
    """Log a warning (never raise) if any reconstructed boundary token
    isn't actually present in the checkpoint's own vocabulary -- a sign
    `--train`/`--train-direction`/`--base-model` don't actually match what
    the checkpoint was trained with (see `patch_word_boundary_decoding_
    for_checkpoint`'s docstring). Duck-typed fakes without `get_vocab`
    (not expected in real use, but used by some unit tests) are skipped
    silently rather than raising here.
    """
    if not boundary_tokens or not hasattr(tokenizer, "get_vocab"):
        return
    checkpoint_vocab = tokenizer.get_vocab()
    missing = [token for token in boundary_tokens if token not in checkpoint_vocab]
    if missing:
        import sys

        print(
            f"WARNING: {len(missing)} of {len(boundary_tokens)} reconstructed "
            "word-boundary tokens are missing from the checkpoint's own "
            "vocabulary -- --train/--train-direction/--base-model may not "
            f"match what this checkpoint was actually trained with. "
            f"Sample missing tokens: {missing[:10]!r}",
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
    # reloaded checkpoint's tokenizer can't just carry this itself.
    train_pairs = read_tsv_pairs(args.train)
    train_examples = build_direction_examples(train_pairs, args.train_direction)
    train_sample_texts = [ex.source_text for ex in train_examples] + [
        ex.target_text for ex in train_examples
    ]
    train_sample_texts.extend(ALL_DIRECTION_TAG_TOKENS)
    base_vocab = base_vocab_loader(args.base_model)
    boundary_tokens = patch_word_boundary_decoding_for_checkpoint(
        tokenizer, base_vocab, train_sample_texts
    )
    _warn_if_boundary_tokens_missing_from_checkpoint(boundary_tokens, tokenizer)

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
            "word_boundary_reconstruction_train_direction": args.train_direction,
            "word_boundary_tokens_reconstructed": len(boundary_tokens),
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
