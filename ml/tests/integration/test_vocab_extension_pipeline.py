"""Fast smoke test for the full tokenizer/vocab-extension pipeline wiring
(gap detection -> vocab extension -> embedding resize), against the tiny
hand-written Kaqchikel fixture text -- never the real corpus, never a real
M2M100 checkpoint. Confirms the pieces connect correctly; translation
quality is a separate concern (ml/evaluation), not tested here.
"""

from pathlib import Path

import numpy as np

from training.tokenizer_extension import compute_new_tokens_for_texts
from training.vocab_extension import extend_vocab, resize_embedding_matrix
from training.vocab_gap import find_missing_characters

FIXTURES = Path(__file__).parent.parent / "fixtures"
SAMPLE_TEXTS = (FIXTURES / "sample_kaqchikel_text.txt").read_text(encoding="utf-8").splitlines()

# A small synthetic base vocab standing in for a slice of M2M100's real
# vocabulary: plain unaccented Latin letters plus Spanish-only diacritics.
# It deliberately does not cover the glottal-stop apostrophe or the
# Kaqchikel-specific vowel "ä" used in the fixture text.
BASE_VOCAB = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz áéíóúñ.")}


def test_gap_extend_resize_pipeline_covers_the_sample_text():
    # 1. Detect what's missing.
    missing_chars = find_missing_characters(SAMPLE_TEXTS, BASE_VOCAB)
    assert "'" in missing_chars
    assert "ä" in missing_chars

    new_tokens = compute_new_tokens_for_texts(SAMPLE_TEXTS, BASE_VOCAB)
    assert new_tokens  # something needs to be added
    assert set(new_tokens) >= missing_chars

    # 2. Extend the vocab dict.
    extended_vocab = extend_vocab(BASE_VOCAB, new_tokens)
    assert len(extended_vocab) == len(BASE_VOCAB) + len(new_tokens)
    for token in new_tokens:
        assert token in extended_vocab

    # 3. Resize a fake embedding matrix to match the extended vocab.
    embedding_dim = 8
    base_embeddings = np.random.default_rng(0).normal(
        size=(len(BASE_VOCAB), embedding_dim)
    ).astype(np.float32)

    resized_embeddings = resize_embedding_matrix(
        base_embeddings, num_new_tokens=len(new_tokens), seed=0
    )

    assert resized_embeddings.shape == (len(extended_vocab), embedding_dim)
    np.testing.assert_array_equal(resized_embeddings[: len(BASE_VOCAB)], base_embeddings)

    # Every new token's id must land inside the resized embedding matrix's
    # new rows -- the vocab dict and the embedding matrix must stay aligned.
    for token in new_tokens:
        assert BASE_VOCAB and extended_vocab[token] < resized_embeddings.shape[0]
        assert extended_vocab[token] >= len(BASE_VOCAB)
