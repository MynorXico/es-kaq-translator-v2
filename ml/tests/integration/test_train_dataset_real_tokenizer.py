"""Integration test for `training.train.TranslationDataset` against the real
`facebook/m2m100_418M` tokenizer (ADR 0003) -- the one piece of
`training/train.py` that a duck-typed fake tokenizer can't validate, since
the bug it guards against is specific to the real M2M100Tokenizer's
internal API (see `TranslationDataset`'s docstring).

This caught a real bug on the first actual (billable) SageMaker training
run: `TranslationDataset.__getitem__` called `tokenizer(text_target=...)`
to build labels, which crashes with `KeyError: None` because
`tokenizer.tgt_lang` is never set (this project's direction tags exist
specifically because Kaqchikel isn't in M2M100's own language table, so
`text_target=`'s internal `_switch_to_target_mode()` has nothing valid to
look up). `tests/integration/test_train_pipeline.py`'s fake tokenizer
duck-types only `get_vocab`/`add_tokens`/`convert_tokens_to_ids`/
`save_pretrained` and never calls the real `__call__`/`text_target=` code
path at all, so it had no way to catch this.
"""

from __future__ import annotations

from transformers import M2M100Tokenizer

from training.direction import ALL_DIRECTION_TAG_TOKENS, DIRECTION_TAGS, TranslationExample
from training.train import TranslationDataset

MODEL_NAME = "facebook/m2m100_418M"


def _real_tokenizer() -> M2M100Tokenizer:
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(list(ALL_DIRECTION_TAG_TOKENS))
    return tokenizer


def test_getitem_does_not_crash_on_text_target_lang_lookup():
    tokenizer = _real_tokenizer()
    examples = [
        TranslationExample(
            source_text="Buenos días", target_text="Utz sq'ij", source_lang="es", target_lang="cak"
        ),
    ]
    dataset = TranslationDataset(examples, tokenizer, max_length=32)

    item = dataset[0]  # must not raise KeyError: None

    assert "input_ids" in item
    assert "labels" in item


def test_labels_start_with_the_direction_tag_token_and_end_with_eos():
    tokenizer = _real_tokenizer()
    examples = [
        TranslationExample(
            source_text="Buenos días", target_text="Utz sq'ij", source_lang="es", target_lang="cak"
        ),
    ]
    dataset = TranslationDataset(examples, tokenizer, max_length=32)

    labels = dataset[0]["labels"]

    expected_tag_id = tokenizer.convert_tokens_to_ids(DIRECTION_TAGS["cak"])
    assert labels[0] == expected_tag_id
    assert labels[-1] == tokenizer.eos_token_id
    # Real subword content in between -- not just the tag and eos.
    assert len(labels) > 2


def test_labels_respect_max_length_budget():
    tokenizer = _real_tokenizer()
    long_target = "Utz sq'ij " * 50
    examples = [
        TranslationExample(
            source_text="hola", target_text=long_target, source_lang="es", target_lang="cak"
        ),
    ]
    dataset = TranslationDataset(examples, tokenizer, max_length=16)

    labels = dataset[0]["labels"]

    assert len(labels) <= 16
