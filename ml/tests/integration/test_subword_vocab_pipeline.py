"""Fast smoke test for the full subword-vocabulary pipeline wiring (train a
Kaqchikel-only SentencePiece model -> diff against a base vocab -> extend
the vocab dict -> resize a fake embedding matrix), against the tiny
hand-written Kaqchikel fixture text -- never the real corpus, never a real
M2M100 checkpoint. Mirrors `test_vocab_extension_pipeline.py`'s approach for
the whole-word/character extension pipeline. Confirms the pieces connect;
translation quality is a separate concern (ml/evaluation), not tested here.
"""

from pathlib import Path

import numpy as np

from training.subword_vocab import compute_new_subword_tokens
from training.vocab_extension import extend_vocab, resize_embedding_matrix

FIXTURES = Path(__file__).parent.parent / "fixtures"
SAMPLE_TEXTS = (FIXTURES / "sample_kaqchikel_text.txt").read_text(encoding="utf-8").splitlines()
REPEATED_SAMPLE_TEXTS = SAMPLE_TEXTS * 8

# A small synthetic base vocab standing in for a slice of M2M100's real
# vocabulary -- covers plain Latin letters plus Spanish-only diacritics, but
# none of the Kaqchikel-specific multi-character subwords the fixture text
# actually needs (e.g. "tz", "▁ri", "▁jun").
BASE_VOCAB = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz áéíóúñ.'äöüïë")}


def test_subword_gap_extend_resize_pipeline_covers_the_sample_text():
    # 1. Train + diff: find high-value Kaqchikel subwords missing from the
    #    base vocab.
    new_tokens = compute_new_subword_tokens(REPEATED_SAMPLE_TEXTS, BASE_VOCAB, vocab_size=60)
    assert new_tokens  # the fixture text has real multi-character subwords
    for token in new_tokens:
        assert token not in BASE_VOCAB

    # 2. Extend the vocab dict -- exactly the same extension mechanism used
    #    for whole words/characters (training.vocab_extension.extend_vocab).
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

    for token in new_tokens:
        assert extended_vocab[token] < resized_embeddings.shape[0]
        assert extended_vocab[token] >= len(BASE_VOCAB)


def test_subword_gap_is_empty_when_base_vocab_already_covers_the_subwords():
    new_tokens = compute_new_subword_tokens(REPEATED_SAMPLE_TEXTS, BASE_VOCAB, vocab_size=60)
    already_covered_vocab = dict(BASE_VOCAB)
    already_covered_vocab.update({token: len(already_covered_vocab) + i for i, token in enumerate(new_tokens)})

    second_pass = compute_new_subword_tokens(
        REPEATED_SAMPLE_TEXTS, already_covered_vocab, vocab_size=60
    )
    assert second_pass == []
