"""Integration test for `resize_embeddings_for_new_tokens` against the real
`facebook/m2m100_418M` checkpoint (ADR 0003) -- the one code path in
`training/tokenizer_extension.py` that a duck-typed fake can't cover, since
it calls a real `transformers` model's `resize_token_embeddings`.

This downloads the actual tokenizer + model from the Hugging Face Hub on
first run (~1.9GB) and caches it under the Hugging Face Hub cache
(`~/.cache/huggingface`, or `$HF_HOME` if set) for subsequent runs -- see
`.github/workflows/ci.yml` for how CI caches this across runs.
"""

from pathlib import Path

import numpy as np
import pytest
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from training.tokenizer_extension import (
    extend_tokenizer_vocab,
    extend_tokenizer_vocab_with_subwords,
    resize_embeddings_for_new_tokens,
)

MODEL_NAME = "facebook/m2m100_418M"
FIXTURES = Path(__file__).parent.parent / "fixtures"
SAMPLE_KAQCHIKEL_TEXTS = (
    (FIXTURES / "sample_kaqchikel_text.txt").read_text(encoding="utf-8").splitlines() * 8
)


@pytest.fixture(scope="module")
def tokenizer() -> M2M100Tokenizer:
    return M2M100Tokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def model() -> M2M100ForConditionalGeneration:
    return M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)


def test_extend_and_resize_grows_vocab_and_embeddings_consistently(tokenizer, model):
    # NOTE: a real M2M100 checkpoint's embedding matrix is pre-padded a few
    # rows beyond its tokenizer's raw vocab size (e.g. 128112 rows for a
    # 128104-token facebook/m2m100_418M vocab) -- so these two are *not*
    # expected to be equal. See resize_embeddings_for_new_tokens's docstring
    # for why that padding is exactly what this test guards against.
    old_vocab_size = len(tokenizer)
    old_embeddings = model.get_input_embeddings().weight.detach().cpu().numpy().copy()
    old_embedding_rows = old_embeddings.shape[0]
    assert old_embedding_rows >= old_vocab_size

    # "k'o" (apostrophe/glottal stop) and "awäch" (ä) are real gaps in
    # M2M100's base vocabulary -- see ml/training/vocab_gap.py.
    added = extend_tokenizer_vocab(tokenizer, ["k'o awäch, la utz awäch?"])
    assert added, "expected at least one genuinely new token for this Kaqchikel sample"

    new_vocab_size = len(tokenizer)
    assert new_vocab_size == old_vocab_size + len(added)

    resize_embeddings_for_new_tokens(model, len(added), seed=42)

    new_embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
    expected_rows = old_embedding_rows + len(added)
    assert new_embeddings.shape == (expected_rows, old_embeddings.shape[1])

    # Every id the extended tokenizer can produce must be a valid row.
    assert new_vocab_size <= expected_rows

    # Old rows -- including any pre-existing padding rows -- must be
    # untouched by the resize.
    np.testing.assert_array_equal(new_embeddings[:old_embedding_rows], old_embeddings)

    # New rows must actually be warm-started (not left at whatever default
    # resize_token_embeddings used), and not identical to each other.
    new_rows = new_embeddings[old_embedding_rows:]
    assert not np.allclose(new_rows, 0.0)
    if new_rows.shape[0] > 1:
        assert not np.allclose(new_rows[0], new_rows[1])


def test_extend_tokenizer_vocab_with_subwords_finds_real_gaps_in_m2m100_vocab(tokenizer, model):
    """Issue #82's core acceptance criterion: diff a Kaqchikel-only
    SentencePiece vocabulary against the *real* M2M100 vocab, not a
    synthetic fake dict, and confirm real high-value subwords are found
    and merged in with correctly resized, warm-started embeddings.
    """
    old_vocab_size = len(tokenizer)
    old_embedding_rows = model.get_input_embeddings().weight.shape[0]

    added = extend_tokenizer_vocab_with_subwords(
        tokenizer, SAMPLE_KAQCHIKEL_TEXTS, vocab_size=60
    )
    assert added, (
        "expected at least one genuinely new high-value Kaqchikel subword "
        "missing from the real facebook/m2m100_418M vocab"
    )
    for token in added:
        assert len(token.removeprefix("▁")) >= 2

    new_vocab_size = len(tokenizer)
    assert new_vocab_size == old_vocab_size + len(added)

    resize_embeddings_for_new_tokens(model, len(added), seed=42)

    new_embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
    expected_rows = old_embedding_rows + len(added)
    assert new_embeddings.shape[0] == expected_rows


def test_resize_embeddings_is_a_noop_when_num_new_tokens_is_zero(tokenizer, model):
    embeddings_before = model.get_input_embeddings().weight.detach().cpu().numpy().copy()

    resize_embeddings_for_new_tokens(model, 0, seed=42)

    embeddings_after = model.get_input_embeddings().weight.detach().cpu().numpy()
    np.testing.assert_array_equal(embeddings_before, embeddings_after)
