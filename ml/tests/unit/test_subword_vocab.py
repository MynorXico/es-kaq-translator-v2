"""Unit tests for training a Kaqchikel-only SentencePiece/Unigram subword
model and diffing it against a base tokenizer vocab (issue #82).

Uses the real `sentencepiece` library (already an `ml/` dependency,
lightweight and local -- no download, no GPU), the same way
`evaluation/metrics.py`'s unit tests use real `sacrebleu` directly. Fixture
text is the small hand-written Kaqchikel sample
(`tests/fixtures/sample_kaqchikel_text.txt`) -- never the real private ALMG
corpus (ADR 0002, docs/testing.md). It's repeated a few times before
training since SentencePiece's unigram trainer needs a handful of repeated
substrings to find genuine multi-character subword pieces worth keeping;
this mirrors how the real corpus (33k+ sentences) naturally has far more
repetition than 6 one-off sentences do.
"""

from pathlib import Path

import pytest

from training.subword_vocab import (
    WORD_BOUNDARY_MARKER,
    compute_new_subword_tokens,
    extract_vocab_pieces,
    select_high_value_subwords,
    train_subword_model,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
SAMPLE_TEXTS = (FIXTURES / "sample_kaqchikel_text.txt").read_text(encoding="utf-8").splitlines()
# Repeated so the unigram trainer has enough recurring substrings to learn
# genuine multi-character subwords, not just single characters.
REPEATED_SAMPLE_TEXTS = SAMPLE_TEXTS * 8


def test_train_subword_model_rejects_empty_texts():
    with pytest.raises(ValueError):
        train_subword_model([])


def test_train_subword_model_returns_nonempty_bytes():
    model_proto = train_subword_model(REPEATED_SAMPLE_TEXTS, vocab_size=60)
    assert isinstance(model_proto, bytes)
    assert len(model_proto) > 0


def test_train_subword_model_is_deterministic_for_the_same_input():
    first = train_subword_model(REPEATED_SAMPLE_TEXTS, vocab_size=60)
    second = train_subword_model(REPEATED_SAMPLE_TEXTS, vocab_size=60)
    assert first == second


def test_train_subword_model_does_not_hard_fail_on_an_unreachable_vocab_size():
    # A vocab_size far larger than a tiny corpus can support must not crash
    # the whole (potentially billable) run -- see module docstring on
    # hard_vocab_limit=False.
    model_proto = train_subword_model(SAMPLE_TEXTS, vocab_size=8000)
    assert len(extract_vocab_pieces(model_proto)) < 8000


def test_extract_vocab_pieces_excludes_sentencepiece_special_tokens():
    model_proto = train_subword_model(REPEATED_SAMPLE_TEXTS, vocab_size=60)
    pieces = extract_vocab_pieces(model_proto)
    assert "<unk>" not in pieces
    assert "<s>" not in pieces
    assert "</s>" not in pieces


def test_extract_vocab_pieces_includes_multi_character_subwords():
    model_proto = train_subword_model(REPEATED_SAMPLE_TEXTS, vocab_size=60)
    pieces = extract_vocab_pieces(model_proto)
    multi_char = [p.removeprefix(WORD_BOUNDARY_MARKER) for p in pieces]
    assert any(len(p) >= 2 for p in multi_char)


def test_select_high_value_subwords_filters_out_single_characters():
    pieces = ["a", "b", WORD_BOUNDARY_MARKER + "a", "tz", WORD_BOUNDARY_MARKER + "ri"]
    high_value = select_high_value_subwords(pieces)
    assert high_value == ["tz", WORD_BOUNDARY_MARKER + "ri"]


def test_select_high_value_subwords_keeps_the_word_boundary_marker_alone_out():
    # The bare marker by itself (SentencePiece always emits it as its own
    # piece) has zero content characters once stripped -- never "high value".
    pieces = [WORD_BOUNDARY_MARKER, "tz"]
    assert select_high_value_subwords(pieces) == ["tz"]


def test_select_high_value_subwords_respects_custom_min_length():
    pieces = ["ab", "abc", "abcd"]
    assert select_high_value_subwords(pieces, min_subword_length=3) == ["abc", "abcd"]


def test_compute_new_subword_tokens_excludes_tokens_already_in_base_vocab():
    model_proto = train_subword_model(REPEATED_SAMPLE_TEXTS, vocab_size=60)
    pieces = select_high_value_subwords(extract_vocab_pieces(model_proto))
    assert pieces  # sanity check on the fixture data itself
    base_vocab = {piece: i for i, piece in enumerate(pieces)}

    new_tokens = compute_new_subword_tokens(REPEATED_SAMPLE_TEXTS, base_vocab, vocab_size=60)

    assert new_tokens == []


def test_compute_new_subword_tokens_returns_new_high_value_pieces_not_in_base_vocab():
    base_vocab = {"a": 0, "e": 1}  # a minimal base vocab missing almost everything

    new_tokens = compute_new_subword_tokens(REPEATED_SAMPLE_TEXTS, base_vocab, vocab_size=60)

    assert new_tokens
    assert "a" not in new_tokens
    assert "e" not in new_tokens
    # Every returned token must be "high value" (>= 2 content characters).
    for token in new_tokens:
        content = token.removeprefix(WORD_BOUNDARY_MARKER)
        assert len(content) >= 2


def test_compute_new_subword_tokens_is_deterministic():
    base_vocab = {"a": 0}
    first = compute_new_subword_tokens(REPEATED_SAMPLE_TEXTS, base_vocab, vocab_size=60)
    second = compute_new_subword_tokens(REPEATED_SAMPLE_TEXTS, base_vocab, vocab_size=60)
    assert first == second
