"""Thin integration layer wiring the pure vocab/embedding-extension logic in
`training.vocab_gap` / `training.vocab_extension` to a real Hugging Face
M2M100 tokenizer and model (`facebook/m2m100_418M`, per ADR 0003).

This module is intentionally thin: all the actual decision logic (which
characters/words are missing, how to dedupe against the existing vocab, how
to initialize new embedding rows) lives in `vocab_gap.py` /
`vocab_extension.py` and is fully unit-tested there against plain
dicts/lists/arrays -- never against a real tokenizer. `transformers` and
`torch` are deliberately *not* added as `ml/` dependencies (see
`ml/pyproject.toml`): this sandbox has no downloaded M2M100 checkpoint and
installing a GPU ML framework here just to leave it untested would be
misleading. `resize_embeddings_for_new_tokens` below lazily imports `torch`
inside its own body for that reason, and is not covered by the fast unit
test suite -- it can only be exercised against a real checkpoint in a real
training environment.

`extend_tokenizer_vocab`, in contrast, only relies on two methods any HF
tokenizer implements (`get_vocab()`, `add_tokens()`), so it *is* fully
testable here against a small fake object that duck-types those two
methods -- see `ml/tests/unit/test_tokenizer_extension.py`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from training.vocab_extension import resize_embedding_matrix, select_new_tokens
from training.vocab_gap import find_missing_characters, find_missing_words


class TokenizerLike(Protocol):
    """The subset of the HF tokenizer API `extend_tokenizer_vocab` needs."""

    def get_vocab(self) -> dict[str, int]: ...

    def add_tokens(self, new_tokens: list[str]) -> int: ...


def compute_new_tokens_for_texts(
    sample_texts: Iterable[str], base_vocab: dict[str, int]
) -> list[str]:
    """Compute the deduplicated list of new tokens needed to cover `sample_texts`.

    Combines missing single characters and missing whole word-forms (see
    `training.vocab_gap`), then filters/dedupes the combined candidates
    against `base_vocab` via `training.vocab_extension.select_new_tokens`,
    so nothing already representable gets added again.
    """
    sample_texts = list(sample_texts)
    vocab_tokens = set(base_vocab)
    candidates = find_missing_characters(sample_texts, vocab_tokens) | find_missing_words(
        sample_texts, vocab_tokens
    )
    return select_new_tokens(candidates, base_vocab)


def extend_tokenizer_vocab(tokenizer: TokenizerLike, sample_texts: Iterable[str]) -> list[str]:
    """Extend `tokenizer`'s vocabulary in place to cover `sample_texts`.

    Real usage: `tokenizer` is a `transformers.M2M100Tokenizer` loaded from
    `facebook/m2m100_418M`. After this call, the corresponding model's
    embeddings must be resized to match (see
    `resize_embeddings_for_new_tokens` below) before training -- adding
    tokens to the tokenizer alone does not touch the model.

    Returns the list of tokens this call attempted to add (empty if
    `sample_texts` is already fully covered by `tokenizer`'s vocabulary).
    """
    base_vocab = tokenizer.get_vocab()
    new_tokens = compute_new_tokens_for_texts(sample_texts, base_vocab)
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
    return new_tokens


def resize_embeddings_for_new_tokens(model, new_vocab_size: int, *, seed: int | None = None) -> None:
    """Resize `model`'s token embeddings and warm-start the newly added rows.

    Real usage: `model` is a `transformers.M2M100ForConditionalGeneration`
    loaded from the same checkpoint as the tokenizer passed to
    `extend_tokenizer_vocab`, and `new_vocab_size` is `len(tokenizer)`
    after that call. M2M100 ties its input and output embeddings by
    default, so resizing the input embedding matrix is sufficient.

    This calls HF's own `model.resize_token_embeddings` first (which
    allocates the new rows), then overwrites just the newly added rows
    using `training.vocab_extension.resize_embedding_matrix`'s
    mean-of-existing-rows-plus-noise initialization, rather than leaving
    them at whatever default `resize_token_embeddings` used.

    Not covered by the fast unit test suite: it requires `torch` and a
    real model checkpoint, neither available in this sandbox (see module
    docstring). The row-initialization math it delegates to is fully
    tested in isolation against a fake embedding matrix instead.
    """
    import torch

    embeddings = model.get_input_embeddings()
    old_weight = embeddings.weight.detach().cpu().numpy()
    old_vocab_size = old_weight.shape[0]
    num_new_tokens = new_vocab_size - old_vocab_size
    if num_new_tokens <= 0:
        return

    resized = resize_embedding_matrix(old_weight, num_new_tokens, seed=seed)

    model.resize_token_embeddings(new_vocab_size)
    with torch.no_grad():
        model.get_input_embeddings().weight[old_vocab_size:] = torch.from_numpy(
            resized[old_vocab_size:]
        )
