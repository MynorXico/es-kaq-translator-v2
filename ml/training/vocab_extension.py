"""Extend a base tokenizer vocabulary and its corresponding embedding matrix.

Two concerns are kept deliberately separate and independently testable:

- `select_new_tokens` / `extend_vocab`: pure dict/list operations -- decide
  which candidate tokens are actually new, and append them to a base vocab
  dict (token -> id) with contiguous new ids, without ever duplicating a
  token already present.
- `resize_embedding_matrix`: a pure NumPy operation -- given a base
  embedding matrix and a count of new tokens, return a new matrix with
  that many extra rows. New rows are *not* zero- or randomly-initialized
  from scratch (either would give the new tokens a meaningless, unstable
  starting point relative to the rest of the pretrained embedding space).
  Instead they start at the mean of the existing rows plus a small amount
  of noise -- a common "warm start" strategy for vocabulary extension that
  keeps new tokens in-distribution while still letting them diverge from
  each other during fine-tuning.

Neither function requires `transformers`/`torch`: they operate on plain
dicts/lists and NumPy arrays, so they're testable without a real M2M100
checkpoint. `training/tokenizer_extension.py` wires these into the actual
Hugging Face tokenizer/model API.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def select_new_tokens(candidate_tokens: Iterable[str], existing_vocab: dict[str, int]) -> list[str]:
    """Filter and dedupe candidate tokens against an existing vocab.

    Returns tokens from `candidate_tokens` not already present as a key in
    `existing_vocab`, deduplicated, in sorted order -- so vocabulary
    extension is deterministic and reproducible given the same inputs,
    which matters for traceability (ADR 0001: every run must be
    reproducible from its recorded config).
    """
    deduped = sorted(set(candidate_tokens))
    return [token for token in deduped if token not in existing_vocab]


def extend_vocab(base_vocab: dict[str, int], new_tokens: Iterable[str]) -> dict[str, int]:
    """Return a new vocab dict with `new_tokens` appended after `base_vocab`.

    New ids continue contiguously from `max(base_vocab.values()) + 1`. Any
    token in `new_tokens` already present in `base_vocab` is skipped --
    never duplicated, never reassigned a different id. `base_vocab` itself
    is never mutated.
    """
    extended = dict(base_vocab)
    to_add = select_new_tokens(new_tokens, base_vocab)
    next_id = (max(base_vocab.values()) + 1) if base_vocab else 0
    for offset, token in enumerate(to_add):
        extended[token] = next_id + offset
    return extended


def resize_embedding_matrix(
    embeddings: np.ndarray, num_new_tokens: int, *, seed: int | None = None
) -> np.ndarray:
    """Return `embeddings` resized to add `num_new_tokens` new rows.

    Given a `(vocab_size, dim)` matrix, returns a `(vocab_size +
    num_new_tokens, dim)` matrix. New rows are initialized to the mean of
    the existing rows, plus a small amount of Gaussian noise (scaled by
    each dimension's existing standard deviation) so the new tokens don't
    start out as exact duplicates of one another -- they still need
    distinct gradients to diverge from each other during fine-tuning.

    Pass `seed` for reproducible output (e.g. in tests); real training runs
    should fix a seed for the whole run regardless, per ADR 0001's
    traceability requirement.
    """
    if num_new_tokens < 0:
        raise ValueError("num_new_tokens must be >= 0")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D (vocab_size, dim) array")
    if num_new_tokens == 0:
        return embeddings.copy()

    mean_row = embeddings.mean(axis=0)
    std_row = embeddings.std(axis=0)

    rng = np.random.default_rng(seed)
    dim = embeddings.shape[1]
    noise = rng.normal(loc=0.0, scale=0.01, size=(num_new_tokens, dim)) * std_row
    new_rows = mean_row + noise

    return np.vstack([embeddings, new_rows]).astype(embeddings.dtype)
