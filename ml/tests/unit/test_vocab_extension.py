"""Unit tests for extending a base vocab dict and resizing an embedding matrix.

These operate on plain dicts/lists/numpy arrays only -- no real tokenizer or
model is involved, so this is fully testable without `transformers`/`torch`/
a real M2M100 checkpoint. See `training/tokenizer_extension.py` for the thin
wrapper that plugs this logic into the real Hugging Face API.
"""

import numpy as np
import pytest

from training.vocab_extension import extend_vocab, resize_embedding_matrix, select_new_tokens


def test_select_new_tokens_filters_out_existing_vocab_entries():
    existing_vocab = {"hola": 0, "mundo": 1}
    assert select_new_tokens(["hola", "matyox", "k'o"], existing_vocab) == ["k'o", "matyox"]


def test_select_new_tokens_deduplicates_candidates():
    assert select_new_tokens(["matyox", "matyox", "utz"], {}) == ["matyox", "utz"]


def test_select_new_tokens_returns_deterministic_sorted_order():
    result = select_new_tokens(["utz", "awäch", "matyox"], {})
    assert result == sorted(result)


def test_extend_vocab_assigns_contiguous_ids_after_base_vocab():
    base_vocab = {"a": 0, "b": 1, "c": 2}
    extended = extend_vocab(base_vocab, ["matyox", "utz"])
    assert extended["a"] == 0
    assert extended["b"] == 1
    assert extended["c"] == 2
    assert {extended["matyox"], extended["utz"]} == {3, 4}


def test_extend_vocab_never_mutates_base_vocab():
    base_vocab = {"a": 0}
    extend_vocab(base_vocab, ["b"])
    assert base_vocab == {"a": 0}


def test_extend_vocab_skips_tokens_already_present():
    base_vocab = {"a": 0, "b": 1}
    assert extend_vocab(base_vocab, ["a", "c"]) == {"a": 0, "b": 1, "c": 2}


def test_extend_vocab_handles_empty_new_tokens():
    base_vocab = {"a": 0}
    assert extend_vocab(base_vocab, []) == base_vocab


def test_resize_embedding_matrix_shape():
    base = np.zeros((10, 4), dtype=np.float32)
    resized = resize_embedding_matrix(base, num_new_tokens=3, seed=0)
    assert resized.shape == (13, 4)


def test_resize_embedding_matrix_preserves_original_rows():
    base = np.arange(20, dtype=np.float32).reshape(10, 2)
    resized = resize_embedding_matrix(base, num_new_tokens=2, seed=0)
    np.testing.assert_array_equal(resized[:10], base)


def test_resize_embedding_matrix_new_rows_are_near_the_mean_not_zero():
    base = np.ones((5, 3), dtype=np.float32) * 2.0
    resized = resize_embedding_matrix(base, num_new_tokens=1, seed=0)
    new_row = resized[5]
    assert not np.allclose(new_row, 0.0)
    np.testing.assert_allclose(new_row, np.full(3, 2.0), atol=0.5)


def test_resize_embedding_matrix_new_rows_are_not_identical_to_each_other():
    base = np.random.default_rng(1).normal(size=(20, 8)).astype(np.float32)
    resized = resize_embedding_matrix(base, num_new_tokens=4, seed=0)
    new_rows = resized[20:]
    # With real (non-degenerate) variance in the base matrix, added noise
    # should make the new rows distinct from one another.
    assert len({tuple(row.round(6)) for row in new_rows}) == 4


def test_resize_embedding_matrix_zero_new_tokens_returns_same_shape_copy():
    base = np.ones((4, 2), dtype=np.float32)
    resized = resize_embedding_matrix(base, num_new_tokens=0)
    np.testing.assert_array_equal(resized, base)
    assert resized is not base


def test_resize_embedding_matrix_preserves_dtype():
    base = np.ones((4, 2), dtype=np.float32)
    resized = resize_embedding_matrix(base, num_new_tokens=2, seed=0)
    assert resized.dtype == np.float32


def test_resize_embedding_matrix_rejects_negative_count():
    with pytest.raises(ValueError):
        resize_embedding_matrix(np.zeros((2, 2)), num_new_tokens=-1)


def test_resize_embedding_matrix_rejects_non_2d_input():
    with pytest.raises(ValueError):
        resize_embedding_matrix(np.zeros(4), num_new_tokens=1)
