"""Integration test for `training.train.fine_tune` against the real
`facebook/m2m100_418M` checkpoint (ADR 0003) -- the one function in
`train.py` that was, until this test, never exercised for real at all
(see `fine_tune`'s own docstring: `tests/integration/test_train_pipeline.py`
always injects a fake in its place).

This caught a real bug on the 3rd real training run with issue #79's new
hyperparameters: `label_smoothing_factor > 0` crashed with
`ValueError: You cannot specify both decoder_input_ids and
decoder_inputs_embeds at the same time`. No duck-typed fake could have
caught this -- it only reproduces against the real model. Issue #79 worked
around it by defaulting `--label-smoothing` to 0.0 (disabled), leaving the
crash itself unfixed.

**Issue #182 root-caused and fixed the real crash** (see
`training.train.attach_decoder_input_ids`'s docstring and
`tests/unit/test_decoder_input_ids.py`'s module docstring for the full
traced call stack): the installed `transformers` version's
`M2M100ForConditionalGeneration` no longer implements
`prepare_decoder_input_ids_from_labels`, so
`DataCollatorForSeq2Seq(tokenizer, model=model)`'s usual convenience
construction of `decoder_input_ids` silently never fires for this model --
invisible without label smoothing (the model's own forward pass builds
`decoder_input_ids` internally from `labels` in that case), but fatal with
it (`Trainer.compute_loss` pops `labels` out of the batch first when a
label smoother is active, so that internal fallback never runs either,
leaving both `decoder_input_ids` and `decoder_inputs_embeds` unset).
`training.train.build_data_collator` now computes `decoder_input_ids`
itself, unconditionally, closing the gap for every `--label-smoothing`
value. `test_fine_tune_runs_one_real_step_with_label_smoothing_enabled`
below is the real, non-zero `label_smoothing_factor` training step this
fix is required to make possible -- this is the actual regression test;
`test_fine_tune_runs_one_real_step_with_default_hyperparameters` above it
guards the unchanged default (0.0) path.

Runs one real, tiny training step on CPU (`fp16` is conditional on real
CUDA availability in `build_training_arguments`, so this works without a
GPU) -- slow-ish (~10-20s per test) but not download-heavy beyond the
cached tokenizer/model already fetched by
`tests/integration/test_tokenizer_extension_real_model.py`, and shares
that test's CI cache (see `.github/workflows/ci.yml`).
"""

from __future__ import annotations

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from training.direction import ALL_DIRECTION_TAG_TOKENS, TranslationExample
from training.tokenizer_extension import extend_tokenizer_vocab, resize_embeddings_for_new_tokens
from training.train import fine_tune, parse_args

MODEL_NAME = "facebook/m2m100_418M"


def _real_tokenizer_and_model():
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
    return tokenizer, model


def _tiny_examples() -> list[TranslationExample]:
    return [
        TranslationExample(
            source_text="Buenos días", target_text="Utz sq'ij", source_lang="es", target_lang="cak"
        ),
        TranslationExample(
            source_text="Gracias", target_text="Matyox", source_lang="es", target_lang="cak"
        ),
    ]


def test_fine_tune_runs_one_real_step_with_default_hyperparameters(tmp_path):
    tokenizer, model = _real_tokenizer_and_model()
    examples = _tiny_examples()
    sample_texts = (
        [ex.source_text for ex in examples]
        + [ex.target_text for ex in examples]
        + list(ALL_DIRECTION_TAG_TOKENS)
    )
    added = extend_tokenizer_vocab(tokenizer, sample_texts)
    resize_embeddings_for_new_tokens(model, len(added), seed=42)

    args = parse_args(
        [
            "--train",
            "unused",
            "--validation",
            "unused",
            "--corpus-version",
            "fixture-v0",
            "--model-dir",
            str(tmp_path / "model"),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--gradient-accumulation-steps",
            "1",
        ]
    )

    # Must not raise -- this is exactly what crashed for real with
    # --label-smoothing > 0 before the fix (issue #79).
    fine_tune(model, tokenizer, examples, [], args)


def test_fine_tune_runs_one_real_step_with_label_smoothing_enabled(tmp_path):
    """The actual regression test for issue #182's fix: a nonzero
    `--label-smoothing` must no longer crash. See this module's own
    docstring for the full root cause -- this reproduces the exact
    real-checkpoint scenario that raised `ValueError: You cannot specify
    both decoder_input_ids and decoder_inputs_embeds at the same time`
    before the fix.
    """
    tokenizer, model = _real_tokenizer_and_model()
    examples = _tiny_examples()
    sample_texts = (
        [ex.source_text for ex in examples]
        + [ex.target_text for ex in examples]
        + list(ALL_DIRECTION_TAG_TOKENS)
    )
    added = extend_tokenizer_vocab(tokenizer, sample_texts)
    resize_embeddings_for_new_tokens(model, len(added), seed=42)

    args = parse_args(
        [
            "--train",
            "unused",
            "--validation",
            "unused",
            "--corpus-version",
            "fixture-v0",
            "--model-dir",
            str(tmp_path / "model"),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--gradient-accumulation-steps",
            "1",
            "--label-smoothing",
            "0.2",
        ]
    )

    # Must not raise. Before issue #182's fix, this raised exactly the
    # decoder_input_ids/decoder_inputs_embeds ValueError described above.
    fine_tune(model, tokenizer, examples, [], args)
