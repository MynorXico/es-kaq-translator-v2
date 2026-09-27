"""Unit tests for `training.train.enable_subword_sampling` (issue #182):
BPE-dropout / subword regularization support for training-time
tokenization, pluggable without touching `training/vocab_extension.py`,
`training/vocab_gap.py`, or `training/subword_vocab.py` at all (none of
those call `tokenizer.sp_model.encode` on the live fine-tuning tokenizer --
see this function's own docstring).

Uses a duck-typed fake tokenizer -- SentencePiece's real sampling behavior
(`enable_sampling=True, nbest_size=-1, alpha=...`) is instead confirmed
against the real `facebook/m2m100_418M` tokenizer in
`tests/integration/test_train_dataset_real_tokenizer.py`.
"""

from __future__ import annotations

import pytest

from training.train import enable_subword_sampling


class FakeSentencePieceModel:
    def __init__(self):
        self.encode_calls: list[dict] = []

    def encode(self, text, **kwargs):
        self.encode_calls.append({"text": text, **kwargs})
        return ["fake", "pieces"]


class FakeTokenizerWithSpModel:
    def __init__(self):
        self.sp_model = FakeSentencePieceModel()
        self.tokenize_calls: list[str] = []

    def _tokenize(self, text: str) -> list[str]:
        # The "deterministic" real-world implementation this context
        # manager temporarily replaces -- never calls sp_model with
        # sampling kwargs.
        self.tokenize_calls.append(text)
        return self.sp_model.encode(text, out_type=str)


class FakeTokenizerWithoutSpModel:
    """Duck-typed fake with no `sp_model` at all -- mirrors the fakes used
    elsewhere in this test suite (`training/tokenizer_extension.py`'s own
    unit tests) that only implement `get_vocab`/`add_tokens`.
    """


def test_enable_subword_sampling_is_a_noop_for_tokenizers_without_sp_model():
    tokenizer = FakeTokenizerWithoutSpModel()

    with enable_subword_sampling(tokenizer, alpha=0.1) as yielded:
        assert yielded is tokenizer  # usable inside the `with` block either way


def test_enable_subword_sampling_routes_tokenize_through_sampling_kwargs():
    tokenizer = FakeTokenizerWithSpModel()
    original_tokenize = tokenizer._tokenize

    with enable_subword_sampling(tokenizer, alpha=0.1, nbest_size=-1):
        tokenizer._tokenize("Utz awäch")

    assert tokenizer.sp_model.encode_calls == [
        {
            "text": "Utz awäch",
            "out_type": str,
            "enable_sampling": True,
            "nbest_size": -1,
            "alpha": 0.1,
        }
    ]
    # The fake's own tokenize_calls list was never appended to -- confirms
    # the original _tokenize was actually swapped out, not just wrapped.
    assert tokenizer.tokenize_calls == []
    # And it was restored afterward. Bound methods of the same underlying
    # function/instance compare equal (`==`) even though each attribute
    # access on an unpatched instance method creates a distinct wrapper
    # object (`is` would be a false negative here, not a meaningful check).
    assert tokenizer._tokenize == original_tokenize


def test_enable_subword_sampling_restores_original_tokenize_even_on_error():
    tokenizer = FakeTokenizerWithSpModel()
    original_tokenize = tokenizer._tokenize

    with pytest.raises(RuntimeError), enable_subword_sampling(tokenizer, alpha=0.1):
        raise RuntimeError("boom")

    assert tokenizer._tokenize == original_tokenize
