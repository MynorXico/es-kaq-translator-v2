"""Unit tests for extending a base vocab dict and resizing an embedding matrix.

These operate on plain dicts/lists/numpy arrays only -- no real tokenizer or
model is involved, so this is fully testable without `transformers`/`torch`/
a real M2M100 checkpoint. See `training/tokenizer_extension.py` for the thin
wrapper that plugs this logic into the real Hugging Face API.
"""

import numpy as np
import pytest

from training.vocab_extension import (
    compositional_row_for_token,
    count_decomposable_tokens,
    decompose_token_into_known_pieces,
    extend_vocab,
    resize_embedding_matrix,
    resize_embedding_matrix_compositional,
    select_new_tokens,
)


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


# --- Compositional embedding initialization (issue #218) --------------------
#
# An alternative to resize_embedding_matrix's single global-mean-plus-noise
# warm start: decompose a new token into pieces already present in the base
# vocab, and initialize its row from the mean of those pieces' *own* existing
# embeddings -- a composition-aware starting point, rather than one uniform
# average applied identically to every new token regardless of content.


def test_decompose_token_into_known_pieces_prefers_the_longest_match():
    vocab = {"a": 0, "b": 1, "ab": 2, "c": 3}
    assert decompose_token_into_known_pieces("abc", vocab) == ["ab", "c"]


def test_decompose_token_into_known_pieces_falls_back_to_single_characters():
    vocab = {"a": 0, "b": 1}
    assert decompose_token_into_known_pieces("ab", vocab) == ["a", "b"]


def test_decompose_token_into_known_pieces_returns_none_when_unmatchable():
    vocab = {"a": 0}
    assert decompose_token_into_known_pieces("ab", vocab) is None


def test_decompose_token_into_known_pieces_handles_a_token_already_in_vocab():
    vocab = {"ab": 0}
    assert decompose_token_into_known_pieces("ab", vocab) == ["ab"]


def test_compositional_row_for_token_averages_piece_embeddings():
    vocab = {"a": 0, "b": 1}
    embeddings = np.array([[0.0, 0.0], [2.0, 2.0]])
    row = compositional_row_for_token("ab", vocab, embeddings)
    np.testing.assert_allclose(row, [1.0, 1.0])


def test_compositional_row_for_token_returns_none_when_undecomposable():
    vocab = {"a": 0}
    embeddings = np.array([[1.0, 1.0]])
    assert compositional_row_for_token("ab", vocab, embeddings) is None


def test_count_decomposable_tokens_counts_only_fully_covered_tokens():
    vocab = {"a": 0, "b": 1}
    assert count_decomposable_tokens(["ab", "ac", "ba"], vocab) == 2


def test_count_decomposable_tokens_counts_sibling_assisted_decompositions_too():
    """Must stay consistent with `resize_embedding_matrix_compositional`'s
    own sibling-aware processing order (see that function's docstring) --
    otherwise this coverage stat would understate how many tokens actually
    got a genuine composed row when that function is the one actually used.
    """
    vocab = {"a": 0}
    # "z" alone is undecomposable (not in vocab); "az" only becomes
    # decomposable once "z" (a sibling in the same batch) is resolved.
    assert count_decomposable_tokens(["z", "az"], vocab) == 1


def test_resize_embedding_matrix_compositional_uses_the_composed_mean():
    vocab = {"a": 0, "b": 1}
    # Global mean across both rows is [5, 5]; "aa" decomposes into ["a", "a"],
    # whose own composed mean is [0, 0] -- a correct implementation must land
    # near the latter, not the former.
    embeddings = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)

    resized = resize_embedding_matrix_compositional(embeddings, ["aa"], vocab, seed=0)

    new_row = resized[2]
    assert abs(new_row[0] - 0.0) < abs(new_row[0] - 5.0)


def test_resize_embedding_matrix_compositional_falls_back_to_global_mean_when_undecomposable():
    vocab = {"a": 0, "b": 1}
    # "zz" can't be decomposed from {"a", "b"} at all -- must fall back to
    # the same global-mean strategy resize_embedding_matrix itself uses,
    # never left at zero/uninitialized.
    embeddings = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)

    resized = resize_embedding_matrix_compositional(embeddings, ["zz"], vocab, seed=0)

    new_row = resized[2]
    assert abs(new_row[0] - 5.0) < abs(new_row[0] - 0.0)


def test_resize_embedding_matrix_compositional_preserves_original_rows():
    vocab = {"a": 0, "b": 1}
    embeddings = np.arange(4, dtype=np.float32).reshape(2, 2)
    resized = resize_embedding_matrix_compositional(embeddings, ["ab"], vocab, seed=0)
    np.testing.assert_array_equal(resized[:2], embeddings)


def test_resize_embedding_matrix_compositional_shape():
    vocab = {"a": 0, "b": 1}
    embeddings = np.zeros((2, 4), dtype=np.float32)
    resized = resize_embedding_matrix_compositional(embeddings, ["ab", "ba"], vocab, seed=0)
    assert resized.shape == (4, 4)


def test_resize_embedding_matrix_compositional_new_rows_are_not_identical_to_each_other():
    vocab = {"a": 0, "b": 1}
    embeddings = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]], dtype=np.float32)
    resized = resize_embedding_matrix_compositional(embeddings, ["ab", "ab", "ab"], vocab, seed=0)
    new_rows = resized[2:]
    assert len({tuple(row.round(6)) for row in new_rows}) == 3


def test_resize_embedding_matrix_compositional_zero_new_tokens_returns_same_shape_copy():
    vocab = {"a": 0}
    embeddings = np.ones((1, 2), dtype=np.float32)
    resized = resize_embedding_matrix_compositional(embeddings, [], vocab, seed=0)
    np.testing.assert_array_equal(resized, embeddings)
    assert resized is not embeddings


def test_resize_embedding_matrix_compositional_preserves_dtype():
    vocab = {"a": 0, "b": 1}
    embeddings = np.ones((2, 2), dtype=np.float32)
    resized = resize_embedding_matrix_compositional(embeddings, ["ab"], vocab, seed=0)
    assert resized.dtype == np.float32


def test_resize_embedding_matrix_compositional_rejects_non_2d_input():
    with pytest.raises(ValueError):
        resize_embedding_matrix_compositional(np.zeros(4), ["a"], {"a": 0})


def test_resize_embedding_matrix_compositional_lets_sibling_new_tokens_compose_from_each_other():
    """The realistic case this strategy exists for: a brand-new character
    (e.g. Kaqchikel "ä", never in the base vocab at all) is added in the
    *same* batch as longer new tokens that are built out of it. Those
    longer tokens must still get a genuine composed row -- using the new
    character's own (already-resolved) row as one of their pieces -- not
    fall all the way back to the global mean just because one of their
    pieces didn't pre-exist in the base vocab.
    """
    vocab = {"a": 0, "b": 1}
    # Global mean of the two existing rows is [5, 5].
    embeddings = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)

    # "z" is a brand-new single character (not in `vocab` at all) -- it has
    # no possible decomposition and must fall back to the global mean.
    # "az" is a brand-new *two*-character token built from "a" (an
    # existing base-vocab piece) and "z" (this same batch's own new
    # single-character token) -- it should compose from those two rows,
    # landing near their average, not near the unrelated global mean.
    resized = resize_embedding_matrix_compositional(embeddings, ["z", "az"], vocab, seed=0)

    z_row = resized[2]
    az_row = resized[3]
    np.testing.assert_allclose(z_row, [5.0, 5.0], atol=1.0)
    # mean("a"=[0,0], "z"≈[5,5]) ≈ [2.5, 2.5] -- closer to that than to the
    # unrelated global mean [5, 5] a non-incremental implementation would
    # have fallen back to (since "z" isn't in the original base vocab).
    assert abs(az_row[0] - 2.5) < abs(az_row[0] - 5.0)


def test_resize_embedding_matrix_compositional_sibling_composition_is_order_independent():
    """The composition above must hold regardless of `new_tokens`' input
    order -- processing happens shortest-first internally (so a token's
    own pieces are always resolved before it needs them), not in whatever
    order the caller happened to list tokens.
    """
    vocab = {"a": 0, "b": 1}
    embeddings = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)

    resized = resize_embedding_matrix_compositional(embeddings, ["az", "z"], vocab, seed=0)

    az_row = resized[2]
    assert abs(az_row[0] - 2.5) < abs(az_row[0] - 5.0)
