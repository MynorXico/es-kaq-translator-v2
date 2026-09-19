"""Unit tests for translation-direction handling (single multilingual model
with direction tags vs. two separate checkpoints -- see the module
docstring in `training/direction.py` for the decision and rationale).
"""

import pytest

from training.direction import (
    ALL_DIRECTION_TAG_TOKENS,
    KAQCHIKEL,
    SPANISH,
    TranslationExample,
    build_direction_examples,
    tag_source_text,
)

PAIRS = [("Buenos días", "Utz awäch"), ("Gracias", "Matyox")]


def test_tag_source_text_prepends_the_target_language_tag():
    tagged = tag_source_text("Buenos días", KAQCHIKEL)
    assert tagged == "__cak__ Buenos días"


def test_tag_source_text_supports_both_directions():
    assert tag_source_text("Utz awäch", SPANISH) == "__es__ Utz awäch"


def test_all_direction_tag_tokens_cover_both_languages():
    assert set(ALL_DIRECTION_TAG_TOKENS) == {"__es__", "__cak__"}


def test_build_direction_examples_es_to_cak_only():
    examples = build_direction_examples(PAIRS, "es->cak")

    assert examples == [
        TranslationExample("Buenos días", "Utz awäch", SPANISH, KAQCHIKEL),
        TranslationExample("Gracias", "Matyox", SPANISH, KAQCHIKEL),
    ]


def test_build_direction_examples_cak_to_es_only():
    examples = build_direction_examples(PAIRS, "cak->es")

    assert examples == [
        TranslationExample("Utz awäch", "Buenos días", KAQCHIKEL, SPANISH),
        TranslationExample("Matyox", "Gracias", KAQCHIKEL, SPANISH),
    ]


def test_build_direction_examples_both_doubles_the_examples_in_each_direction():
    examples = build_direction_examples(PAIRS, "both")

    assert len(examples) == len(PAIRS) * 2
    directions = {(ex.source_lang, ex.target_lang) for ex in examples}
    assert directions == {(SPANISH, KAQCHIKEL), (KAQCHIKEL, SPANISH)}


def test_build_direction_examples_rejects_unknown_direction():
    with pytest.raises(ValueError):
        build_direction_examples(PAIRS, "en->fr")
