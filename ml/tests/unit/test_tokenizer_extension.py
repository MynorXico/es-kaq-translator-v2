"""Unit tests for the thin tokenizer-extension wrapper, against a fake
tokenizer that duck-types the two `transformers.M2M100Tokenizer` methods
this module actually calls (`get_vocab`, `add_tokens`).

This is how the wiring logic is exercised without a real `transformers`
install or a downloaded M2M100 checkpoint -- neither is available in this
sandbox. See the module docstring in `training/tokenizer_extension.py`.
"""

from training.tokenizer_extension import compute_new_tokens_for_texts, extend_tokenizer_vocab


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
