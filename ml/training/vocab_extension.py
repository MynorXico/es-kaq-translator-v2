"""Extend a base tokenizer vocabulary and its corresponding embedding matrix.

Two concerns are kept deliberately separate and independently testable:

- `select_new_tokens` / `extend_vocab`: pure dict/list operations -- decide
  which candidate tokens are actually new, and append them to a base vocab
  dict (token -> id) with contiguous new ids, without ever duplicating a
  token already present.
- `resize_embedding_matrix` / `resize_embedding_matrix_compositional`: pure
  NumPy operations -- given a base embedding matrix and either a count of
  new tokens (`resize_embedding_matrix`) or the actual new token strings
  plus the vocab they're being added to (`resize_embedding_matrix_
  compositional`), return a new matrix with the corresponding extra rows.
  New rows are never zero- or randomly-initialized from scratch (either
  would give the new tokens a meaningless, unstable starting point
  relative to the rest of the pretrained embedding space). Two warm-start
  strategies are available (issue #218):

  - **`resize_embedding_matrix`** (the original, still the default):
    every new row starts at the mean of *all* existing rows, plus a small
    amount of noise -- a single global average applied uniformly to every
    new token, regardless of that token's own linguistic composition.
  - **`resize_embedding_matrix_compositional`**: decomposes each new
    token into smaller pieces already present in the base vocab
    (`decompose_token_into_known_pieces`), and initializes its row as the
    mean of *those pieces'* own existing, pretrained embeddings -- a
    composition-aware starting point. Falls back to the same global-mean
    strategy, per token, whenever a token can't be fully decomposed (e.g.
    it contains a character the base vocab has no representation for at
    all), so every new token still gets a real, in-distribution starting
    row regardless of which path produced it.

Neither function (nor any of the helpers around them) requires
`transformers`/`torch`: they operate on plain dicts/lists and NumPy
arrays, so they're testable without a real M2M100 checkpoint.
`training/tokenizer_extension.py` wires these into the actual Hugging Face
tokenizer/model API.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

# The two embedding-init strategies `training.tokenizer_extension.
# resize_embeddings_for_new_tokens` / `training.train` choose between
# (issue #218) -- named constants rather than bare string literals so a
# typo in a CLI choice list can't silently diverge from what this module
# actually checks for.
EMBEDDING_INIT_MEAN = "mean"
EMBEDDING_INIT_COMPOSITIONAL = "compositional"
EMBEDDING_INIT_STRATEGIES = (EMBEDDING_INIT_MEAN, EMBEDDING_INIT_COMPOSITIONAL)


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


# --- Compositional embedding initialization (issue #218) --------------------


def decompose_token_into_known_pieces(token: str, vocab: dict[str, int]) -> list[str] | None:
    """Greedily segment `token` into the longest possible run of substrings
    that are themselves keys of `vocab` ("maximum matching" -- a greedy,
    longest-match-first segmentation scanning left to right).

    Returns the ordered list of pieces if `token` can be fully covered this
    way, or `None` if some position has no matching piece at all -- not even
    a single character, meaning `token` contains a character `vocab` has no
    representation for whatsoever. `compositional_row_for_token` treats
    `None` as "give up, let the caller fall back to a different strategy for
    this token" rather than silently returning a partial/wrong decomposition.

    This is a simple greedy heuristic, not a true Viterbi/unigram-LM-optimal
    segmentation (SentencePiece's own decoding is more principled, and a
    greedy longest match can occasionally pick a locally-longest piece that
    blocks a better overall split) -- but it is deterministic, pure, and
    sufficient to recover a composition-aware signal for warm-starting a new
    token's embedding, which is this function's only purpose. Unlike
    `vocab_gap.find_missing_characters`/`find_missing_words` (which check
    for an exact, single-key match only), this actively tries to *build*
    `token` out of smaller existing pieces when no exact match exists.

    `token` is assumed, by convention, not to already be a key of `vocab`
    itself (callers only ever use this for genuinely new tokens -- see
    `training.vocab_extension.select_new_tokens`) -- if it is, the greedy
    match trivially returns `[token]` on its very first iteration (the
    longest possible candidate, `token` itself, is checked first), which is
    harmless but not the intended use.

    Returns `None` (not `[]`) for an empty `token`: the main loop below
    never executes at all when `length == 0`, so without this explicit
    guard an empty string would otherwise look like a *successful*, empty
    decomposition -- `compositional_row_for_token` would then average zero
    rows (`embeddings[[]].mean(axis=0)`), producing a NaN row instead of
    correctly falling back to a different strategy (code review on PR
    #219). Not currently reachable from the real pipeline
    (`training.vocab_gap.find_missing_characters`/`find_missing_words`
    never emit `""`), but guarded regardless, since nothing else in this
    function's contract rules it out for a caller passing one directly.
    """
    if not token:
        return None
    pieces: list[str] = []
    position = 0
    length = len(token)
    while position < length:
        match: str | None = None
        for end in range(length, position, -1):
            candidate = token[position:end]
            if candidate in vocab:
                match = candidate
                break
        if match is None:
            return None
        pieces.append(match)
        position += len(match)
    return pieces


def compositional_row_for_token(
    token: str, vocab: dict[str, int], embeddings: np.ndarray
) -> np.ndarray | None:
    """Return a new embedding row for `token`, computed as the mean of its
    decomposed pieces' *existing* rows in `embeddings` (per `vocab`'s id
    mapping) -- or `None` if `token` can't be fully decomposed into known
    pieces (`decompose_token_into_known_pieces` returned `None`).

    `vocab` and `embeddings` must agree: every id `vocab` maps a piece to
    must be a valid row index into `embeddings`.

    This is the single-token building block for composing a row against a
    *fixed* vocab/embeddings snapshot. `resize_embedding_matrix_
    compositional` (the real entry point `training.tokenizer_extension`
    uses) doesn't call this directly -- it needs a *growing* pool of
    pieces as it resolves a whole batch of new tokens (so a token can
    compose from another new token in the same batch, e.g. a brand-new
    single character), which this function's fixed-snapshot contract can't
    express. Kept as its own tested, directly-usable unit regardless, both
    for a plain "does this token decompose against a static vocab" use
    case and as a cross-check for that function's own per-token logic.
    """
    pieces = decompose_token_into_known_pieces(token, vocab)
    if pieces is None:
        return None
    ids = [vocab[piece] for piece in pieces]
    return embeddings[ids].mean(axis=0)


def count_decomposable_tokens(tokens: Iterable[str], vocab: dict[str, int]) -> int:
    """Count how many of `tokens` can be fully decomposed into existing
    `vocab` pieces *or* other, already-resolved tokens from the same batch
    (`decompose_token_into_known_pieces` returns non-`None`).

    Deliberately mirrors `resize_embedding_matrix_compositional`'s own
    shortest-first processing order and growing pool of available pieces
    (see that function's docstring) rather than checking each token against
    `vocab` alone -- otherwise this would *undercount* relative to what
    that function actually does: a token that only decomposes with a
    same-batch sibling's help (e.g. a whole word built from a brand-new
    character also being added in this run) would wrongly show up here as
    "fell back to the global mean" even though it got a genuine composed
    row.

    A cheap, pure coverage statistic -- independent of actually building any
    embedding rows -- that `training.train` records in a run's model card
    (`embedding_init_compositional_coverage`) whenever the compositional
    strategy is used, so a reader of a given run's model card can tell how
    much of that run's vocabulary extension actually got a
    composition-aware start versus falling back to the global mean,
    without having to re-derive it from the raw token list (ADR 0001's
    traceability requirement).

    **Caveat (code review on PR #219): "decomposable" here means "fully
    segmentable into known piece *keys*", not "built entirely from genuine
    pretrained signal".** A token that itself falls back to the global
    mean (e.g. a brand-new single character with no match anywhere in
    `vocab`) is still registered as an available *key* once resolved (see
    the loop below), so a longer sibling token built on top of it counts
    as "decomposable" here even though part of its own composed value is,
    transitively, the unrelated global mean. This matches
    `resize_embedding_matrix_compositional`'s own real behavior exactly
    (so the stat is never misleading about what that function actually
    did), but it does mean this count is an upper bound on "genuinely
    composed from real pretrained pieces", not a strict lower bound on it
    -- don't read a high `embedding_init_compositional_coverage` as proof
    every counted token avoided the global mean entirely.
    """
    tokens = list(tokens)
    available_pieces: dict[str, int] = dict(vocab)
    processing_order = sorted(range(len(tokens)), key=lambda i: (len(tokens[i]), i))

    decomposable = [False] * len(tokens)
    for index in processing_order:
        token = tokens[index]
        decomposable[index] = decompose_token_into_known_pieces(token, available_pieces) is not None
        if token not in available_pieces:
            # Value is never read for a new-token entry (only `in`
            # membership matters to decompose_token_into_known_pieces) --
            # any placeholder works.
            available_pieces[token] = -1

    return sum(decomposable)


def resize_embedding_matrix_compositional(
    embeddings: np.ndarray,
    new_tokens: Sequence[str],
    base_vocab: dict[str, int],
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Alternative to `resize_embedding_matrix`'s global-mean-plus-noise warm
    start (issue #218): initialize each new token's embedding row as the
    mean of its own decomposed pieces' embeddings
    (`compositional_row_for_token`), rather than a single uniform average
    applied identically to every new token regardless of content.

    `new_tokens` order matters: row `i` of the returned matrix's new rows
    corresponds to `new_tokens[i]`. `base_vocab` must be the tokenizer
    vocabulary as it existed *before* `new_tokens` were added -- mapping to
    row indices valid in `embeddings` as given -- and must never include
    any of `new_tokens` themselves (the real caller,
    `training.tokenizer_extension.resize_embeddings_for_new_tokens`,
    captures this from `tokenizer.get_vocab()` ahead of the
    `extend_tokenizer_vocab*` calls that produce `new_tokens`).

    ## Same-batch siblings can compose from each other

    `new_tokens` is processed in increasing-length order internally
    (ties broken by original position, for determinism), not the order
    given -- so a multi-character token can decompose using *other*
    `new_tokens` in this same batch that have already been resolved, not
    just `base_vocab`. This matters for the realistic case this strategy
    exists for: `training.vocab_gap.find_missing_characters` and
    `find_missing_words` (wired together by `training.tokenizer_extension.
    extend_tokenizer_vocab`) always add a genuinely new character (e.g.
    Kaqchikel "ä", entirely absent from M2M100's base vocab) in the exact
    same batch as the longer whole-word tokens built out of it -- without
    this, those words could never be composed at all (`base_vocab` alone
    has no row for "ä" yet), and every one of them would silently fall
    back to the global mean regardless of how much of the rest of the word
    genuinely is decomposable. A resolved single character still has no
    pieces of its own to compose from (it falls back to the global mean,
    like any other undecomposable token -- see below), but once resolved,
    it becomes a legitimate piece for any *other* new token in the same
    batch that contains it.

    Falls back to the same mean-of-all-existing-rows strategy
    `resize_embedding_matrix` uses, per token, whenever that token can't be
    fully decomposed even with sibling tokens available (some character in
    it has no representation anywhere in `base_vocab` or the rest of this
    batch) -- so every new token still gets a real, in-distribution
    starting point even when composition isn't possible, exactly matching
    the existing strategy's own guarantee (never left at zero/
    uninitialized). The same small amount of Gaussian noise (scaled by
    each dimension's existing standard deviation) that
    `resize_embedding_matrix` adds is added here too, to every row
    regardless of which path produced it, so new tokens that happen to
    decompose identically (or both fall back) still diverge from each
    other during fine-tuning -- and computed once, upfront, from a single
    `seed`-derived draw per original position (not per processing-order
    position), so this strategy's noise is reproducible independent of the
    internal shortest-first processing order.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D (vocab_size, dim) array")
    new_tokens = list(new_tokens)
    num_new_tokens = len(new_tokens)
    if num_new_tokens == 0:
        return embeddings.copy()

    mean_row = embeddings.mean(axis=0)
    std_row = embeddings.std(axis=0)
    dim = embeddings.shape[1]

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=0.01, size=(num_new_tokens, dim)) * std_row

    # Shortest-first so a token's potential pieces -- including other new
    # tokens from this same batch -- are always resolved before anything
    # that might need them (see docstring). Ties broken by original
    # position for determinism.
    processing_order = sorted(range(num_new_tokens), key=lambda i: (len(new_tokens[i]), i))

    # Keys available to decompose against: base_vocab's own pretrained
    # pieces, plus (as processing proceeds) this batch's own already-
    # resolved new tokens. Only used for `in` membership checks by
    # `decompose_token_into_known_pieces` -- values are real row ids for
    # base_vocab entries, and `None` placeholders for new-token entries
    # (whose actual center row lives in `resolved_centers` instead, since
    # it doesn't correspond to any real row in `embeddings`).
    available_pieces: dict[str, int | None] = dict(base_vocab)
    resolved_centers: dict[str, np.ndarray] = {}
    centers: list[np.ndarray | None] = [None] * num_new_tokens

    for index in processing_order:
        token = new_tokens[index]
        pieces = decompose_token_into_known_pieces(token, available_pieces)
        if pieces is None:
            center = mean_row
        else:
            piece_rows = [
                resolved_centers[piece]
                if piece in resolved_centers
                else embeddings[available_pieces[piece]]
                for piece in pieces
            ]
            center = np.mean(piece_rows, axis=0)
        centers[index] = center
        # Make this token itself available as a piece for any longer,
        # not-yet-processed sibling in this same batch -- but never
        # overwrite an earlier occurrence's already-resolved center if
        # `new_tokens` contains a literal duplicate string.
        if token not in available_pieces:
            available_pieces[token] = None
            resolved_centers[token] = center

    new_rows = np.empty((num_new_tokens, dim), dtype=embeddings.dtype)
    for index in range(num_new_tokens):
        new_rows[index] = centers[index] + noise[index]

    return np.vstack([embeddings, new_rows]).astype(embeddings.dtype)
