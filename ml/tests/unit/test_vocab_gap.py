"""Unit tests for identifying Kaqchikel characters/words missing from a base
tokenizer's vocabulary.

Fixture text (`tests/fixtures/sample_kaqchikel_text.txt`) is a handful of
hand-written Kaqchikel sentences -- never the real private ALMG corpus,
per ADR 0002 and docs/testing.md.
"""

from pathlib import Path

from training.vocab_gap import extract_characters, find_missing_characters, find_missing_words

FIXTURES = Path(__file__).parent.parent / "fixtures"
SAMPLE_TEXTS = (FIXTURES / "sample_kaqchikel_text.txt").read_text(encoding="utf-8").splitlines()


def test_extract_characters_returns_distinct_characters_across_all_texts():
    assert extract_characters(["ab", "ba", "c"]) == {"a", "b", "c"}


def test_find_missing_characters_flags_a_diacritic_not_in_base_vocab():
    # A base vocab covering plain Spanish (no "ä") is missing Kaqchikel's ä.
    base_vocab_chars = set("abcdefghijklmnopqrstuvwxyzáéíóúñ. ")
    missing = find_missing_characters(["utz awäch."], base_vocab_chars)
    assert missing == {"ä"}


def test_find_missing_characters_ignores_whitespace():
    base_vocab_chars = set("abc")
    assert find_missing_characters(["a b\tc"], base_vocab_chars) == set()


def test_find_missing_characters_returns_empty_when_fully_covered():
    base_vocab_chars = set("abc")
    assert find_missing_characters(["abc", "cab"], base_vocab_chars) == set()


def test_find_missing_characters_flags_apostrophe_for_glottalized_consonants():
    # Kaqchikel marks glottalized consonants (k', tz', ch', q') with a plain
    # apostrophe. A base vocabulary that only covers unaccented Latin
    # letters (no apostrophe) does not represent it -- this is the concrete
    # gap ADR 0003/#34 asks us to check for, not assume.
    base_vocab_chars = set("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZäöüïë.")
    missing = find_missing_characters(SAMPLE_TEXTS, base_vocab_chars)
    assert "'" in missing


def test_find_missing_words_flags_unseen_word_forms():
    base_vocab_tokens = {"Utz", "awäch."}
    missing = find_missing_words(["Utz awäch.", "Matyox."], base_vocab_tokens)
    assert missing == {"Matyox."}


def test_find_missing_words_returns_empty_when_all_words_known():
    base_vocab_tokens = {"hola", "mundo"}
    assert find_missing_words(["hola mundo"], base_vocab_tokens) == set()
