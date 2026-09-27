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

# A source/target pair long and varied enough that SentencePiece's real
# subword sampling (nbest_size=-1) has many valid segmentations to choose
# between -- confirmed empirically (see this ticket's own investigation)
# to produce different segmentations across repeated real encode() calls
# at this alpha, unlike a short word/phrase with only one or two plausible
# splits. Plain Latin-alphabet text only -- this test is purely about
# confirming the sampling wiring against the real tokenizer, not about
# realistic Kaqchikel content.
_LONG_SOURCE_TEXT = "Buenos días, ¿cómo está usted el día de hoy en la comunidad?"
_LONG_TARGET_TEXT = "Buenos días, ¿cómo está usted el día de hoy en la comunidad?"


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


def test_subword_dropout_alpha_produces_varying_segmentations_across_calls():
    """Issue #182: `subword_dropout_alpha` must actually change what
    SentencePiece encodes to, against the real tokenizer -- a duck-typed
    fake (see `tests/unit/test_subword_dropout.py`) can confirm the
    *wiring* (the right kwargs reach `sp_model.encode`) but not that real
    SentencePiece sampling genuinely varies output, which is the whole
    point of BPE-dropout / subword regularization.
    """
    tokenizer = _real_tokenizer()
    examples = [
        TranslationExample(
            source_text=_LONG_SOURCE_TEXT,
            target_text=_LONG_TARGET_TEXT,
            source_lang="es",
            target_lang="cak",
        ),
    ]
    dataset = TranslationDataset(
        examples, tokenizer, max_length=64, subword_dropout_alpha=0.1
    )

    observed_input_id_sequences = {tuple(dataset[0]["input_ids"]) for _ in range(30)}

    assert len(observed_input_id_sequences) > 1


def test_subword_dropout_alpha_none_keeps_encoding_deterministic():
    """Control case: without opting in, the same example must always
    encode to the exact same ids (the pre-existing, deterministic
    behavior every past training run relied on).
    """
    tokenizer = _real_tokenizer()
    examples = [
        TranslationExample(
            source_text=_LONG_SOURCE_TEXT,
            target_text=_LONG_TARGET_TEXT,
            source_lang="es",
            target_lang="cak",
        ),
    ]
    dataset = TranslationDataset(examples, tokenizer, max_length=64)

    observed_input_id_sequences = {tuple(dataset[0]["input_ids"]) for _ in range(10)}

    assert len(observed_input_id_sequences) == 1


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
