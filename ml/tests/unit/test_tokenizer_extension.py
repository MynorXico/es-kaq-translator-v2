"""Unit tests for the thin tokenizer-extension wrapper, against a fake
tokenizer that duck-types the two `transformers.M2M100Tokenizer` methods
this module actually calls (`get_vocab`, `add_tokens`).

This is how the wiring logic is exercised without a real `transformers`
install or a downloaded M2M100 checkpoint -- neither is available in this
sandbox. See the module docstring in `training/tokenizer_extension.py`.
"""

from pathlib import Path

from training.tokenizer_extension import (
    compute_new_tokens_for_texts,
    extend_tokenizer_vocab,
    extend_tokenizer_vocab_with_subwords,
    patch_word_boundary_decoding_for_checkpoint,
    reconstruct_whole_word_boundary_tokens,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
SAMPLE_KAQCHIKEL_TEXTS = (
    (FIXTURES / "sample_kaqchikel_text.txt").read_text(encoding="utf-8").splitlines() * 8
)


class FakeM2M100Tokenizer:
    """Minimal fake matching the two HF tokenizer methods we rely on."""

    def __init__(self, vocab: dict[str, int]):
        self._vocab = dict(vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def add_tokens(self, new_tokens: list[str]) -> int:
        added = 0
        next_id = max(self._vocab.values(), default=-1) + 1
        for token in new_tokens:
            if token in self._vocab:
                continue
            self._vocab[token] = next_id
            next_id += 1
            added += 1
        return added


class FakeDecodingTokenizer(FakeM2M100Tokenizer):
    """Extends `FakeM2M100Tokenizer` with a naive `convert_tokens_to_string`
    that just concatenates tokens with no separator -- standing in for a
    real `sentencepiece` `sp_model.decode()` call that has no idea how to
    space a token it never registered (issue #116's actual root cause).
    Unlike `FakeM2M100Tokenizer`, this fake *can* exercise
    `_mark_word_boundary_tokens`'s wrapper logic, since it implements the
    one method that logic patches.
    """

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)


def _base_vocab() -> dict[str, int]:
    vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyzáéíóúñ. ")}
    vocab["utz"] = len(vocab)
    return vocab


def test_compute_new_tokens_for_texts_includes_missing_apostrophe():
    new_tokens = compute_new_tokens_for_texts(["k'o", "utz"], _base_vocab())
    assert "'" in new_tokens


def test_compute_new_tokens_for_texts_excludes_already_known_words():
    new_tokens = compute_new_tokens_for_texts(["utz awäch"], _base_vocab())
    assert "utz" not in new_tokens


def test_extend_tokenizer_vocab_adds_tokens_to_fake_tokenizer():
    tokenizer = FakeM2M100Tokenizer(_base_vocab())

    added = extend_tokenizer_vocab(tokenizer, ["k'o awäch"])

    assert added
    updated_vocab = tokenizer.get_vocab()
    for token in added:
        assert token in updated_vocab


def test_extend_tokenizer_vocab_is_idempotent_on_second_call():
    tokenizer = FakeM2M100Tokenizer(_base_vocab())

    first = extend_tokenizer_vocab(tokenizer, ["k'o awäch"])
    second = extend_tokenizer_vocab(tokenizer, ["k'o awäch"])

    assert first
    assert second == []


def test_extend_tokenizer_vocab_returns_empty_list_when_fully_covered():
    tokenizer = FakeM2M100Tokenizer(_base_vocab())

    added = extend_tokenizer_vocab(tokenizer, ["utz"])

    assert added == []


def test_extend_tokenizer_vocab_with_subwords_adds_high_value_pieces():
    tokenizer = FakeM2M100Tokenizer(_base_vocab())

    added = extend_tokenizer_vocab_with_subwords(
        tokenizer, SAMPLE_KAQCHIKEL_TEXTS, vocab_size=60
    )

    assert added
    updated_vocab = tokenizer.get_vocab()
    for token in added:
        assert token in updated_vocab
        # Never re-adds a single character -- that's extend_tokenizer_vocab's
        # (character-gap) job already, not this subword-diff step's.
        assert len(token.removeprefix("▁")) >= 2


def test_extend_tokenizer_vocab_with_subwords_is_idempotent_on_second_call():
    tokenizer = FakeM2M100Tokenizer(_base_vocab())

    first = extend_tokenizer_vocab_with_subwords(tokenizer, SAMPLE_KAQCHIKEL_TEXTS, vocab_size=60)
    second = extend_tokenizer_vocab_with_subwords(
        tokenizer, SAMPLE_KAQCHIKEL_TEXTS, vocab_size=60
    )

    assert first
    assert second == []


def test_reconstruct_whole_word_boundary_tokens_matches_a_live_extension():
    """The whole point of `reconstruct_whole_word_boundary_tokens` (issue
    #116's re-evaluation gap): given the *same* pristine base vocab and
    the *same* sample texts `extend_tokenizer_vocab` originally saw, it
    must recover exactly the same set of multi-character (whole-word)
    tokens that a live `extend_tokenizer_vocab` call would mark as word
    boundaries -- without ever calling `add_tokens` itself.
    """
    base_vocab = _base_vocab()
    sample_texts = ["k'o awäch zqvbn ri nimalaj"]

    live_tokenizer = FakeM2M100Tokenizer(base_vocab)
    live_added = extend_tokenizer_vocab(live_tokenizer, sample_texts)
    live_whole_word_tokens = {token for token in live_added if len(token) > 1}

    reconstructed = reconstruct_whole_word_boundary_tokens(base_vocab, sample_texts)

    assert live_whole_word_tokens
    assert set(reconstructed) == live_whole_word_tokens


def test_reconstruct_whole_word_boundary_tokens_excludes_single_characters():
    base_vocab = _base_vocab()

    reconstructed = reconstruct_whole_word_boundary_tokens(base_vocab, ["k'o awäch"])

    assert all(len(token) > 1 for token in reconstructed)
    assert "'" not in reconstructed
    assert "ä" not in reconstructed


def test_reconstruct_whole_word_boundary_tokens_is_a_fixed_point_of_the_corpus():
    """Confirms the property this project's real re-evaluation relies on
    (see `training.tokenizer_extension`'s module docstring): re-running
    whole-word extension against a vocab that already covers the corpus
    (e.g. a continuation-training checkpoint re-extending against the same
    corpus its own earlier training pass already covered) is a no-op --
    so the reconstruction is unaffected by how many times the real
    checkpoint's training chain re-applied it.
    """
    base_vocab = _base_vocab()
    sample_texts = ["k'o awäch zqvbn"]

    first_pass = reconstruct_whole_word_boundary_tokens(base_vocab, sample_texts)
    already_extended_vocab = dict(base_vocab)
    next_id = max(already_extended_vocab.values()) + 1
    for offset, token in enumerate(first_pass):
        already_extended_vocab[token] = next_id + offset

    second_pass = reconstruct_whole_word_boundary_tokens(already_extended_vocab, sample_texts)

    assert first_pass
    assert second_pass == []


def test_patch_word_boundary_decoding_for_checkpoint_fixes_glued_words():
    """Simulates reloading a checkpoint's tokenizer that already has a
    whole-word token added (as if by a prior, now-forgotten
    `extend_tokenizer_vocab` call) but *without* its boundary-marking
    wrapper installed -- exactly what a real `save_pretrained()` /
    `from_pretrained()` round-trip loses (this module's docstring).
    """
    base_vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyzáéíóúñ. ")}
    base_vocab["▁de"] = len(base_vocab)
    base_vocab["▁dios"] = len(base_vocab)
    sample_texts = ["awach"]

    checkpoint_vocab = dict(base_vocab)
    checkpoint_vocab["awach"] = max(checkpoint_vocab.values()) + 1
    tokenizer = FakeDecodingTokenizer(checkpoint_vocab)

    # Before patching: the naive/real decode path glues the added token
    # directly onto its neighbor, exactly like the real bug.
    assert tokenizer.convert_tokens_to_string(["▁de", "▁dios", "awach"]) == "▁de▁diosawach"

    applied = patch_word_boundary_decoding_for_checkpoint(tokenizer, base_vocab, sample_texts)

    assert "awach" in applied
    decoded = tokenizer.convert_tokens_to_string(["▁de", "▁dios", "awach"])
    assert decoded == "▁de ▁dios awach"


def test_patch_word_boundary_decoding_for_checkpoint_is_a_noop_for_fakes_without_convert(
    tmp_path,
):
    """`FakeM2M100Tokenizer` doesn't implement `convert_tokens_to_string`
    (matching the other duck-typed fakes used across this project's
    eval-only wiring tests) -- patching must no-op cleanly rather than
    raise, exactly like `_mark_word_boundary_tokens` already documents.
    """
    base_vocab = _base_vocab()
    tokenizer = FakeM2M100Tokenizer(dict(base_vocab))

    applied = patch_word_boundary_decoding_for_checkpoint(tokenizer, base_vocab, ["k'o awäch"])

    assert applied  # still computed and returned for caller-side logging
    assert not hasattr(tokenizer, "convert_tokens_to_string")
