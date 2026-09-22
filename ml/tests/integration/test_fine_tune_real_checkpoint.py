"""Integration test for `training.train.fine_tune` against the real
`facebook/m2m100_418M` checkpoint (ADR 0003) -- the one function in
`train.py` that was, until this test, never exercised for real at all
(see `fine_tune`'s own docstring: `tests/integration/test_train_pipeline.py`
always injects a fake in its place).

This caught a real bug on the 3rd real training run with issue #79's new
hyperparameters: `label_smoothing_factor > 0` crashes with
`ValueError: You cannot specify both decoder_input_ids and
decoder_inputs_embeds at the same time`, a real incompatibility between
the installed `transformers` version's label-smoothing loss path and
M2M100's forward signature. No duck-typed fake could have caught this --
it only reproduces against the real model. Fixed by defaulting
`--label-smoothing` to 0.0 (disabled); this test guards against that
regressing silently if the default is ever changed back without
confirming the interaction is fixed.

Runs one real, tiny training step on CPU (`fp16` is conditional on real
CUDA availability in `build_training_arguments`, so this works without a
GPU) -- slow-ish (~10-20s) but not download-heavy beyond the cached
tokenizer/model already fetched by
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
