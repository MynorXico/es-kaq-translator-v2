"""Thin integration layer wiring the pure vocab/embedding-extension logic in
`training.vocab_gap` / `training.vocab_extension` / `training.subword_vocab`
to a real Hugging Face M2M100 tokenizer and model (`facebook/m2m100_418M`,
per ADR 0003).

This module is intentionally thin: all the actual decision logic (which
characters/words are missing, how to dedupe against the existing vocab, how
to initialize new embedding rows) lives in `vocab_gap.py` /
`vocab_extension.py` and is fully unit-tested there against plain
dicts/lists/arrays -- never against a real tokenizer.
`resize_embeddings_for_new_tokens` below is exercised against the real
`facebook/m2m100_418M` checkpoint in
`tests/integration/test_tokenizer_extension_real_model.py` (see ADR 0003);
it still lazily imports `torch` inside its own body so importing this
module never requires `torch` to be installed for callers that only need
`extend_tokenizer_vocab`.

`extend_tokenizer_vocab`, in contrast, only relies on two methods any HF
tokenizer implements (`get_vocab()`, `add_tokens()`), so it *is* fully
testable here against a small fake object that duck-types those two
methods -- see `ml/tests/unit/test_tokenizer_extension.py`.

Both `extend_tokenizer_vocab` and `extend_tokenizer_vocab_with_subwords`
additionally call `_mark_word_boundary_tokens` after `add_tokens()`, to
work around a real decode-time word-boundary-spacing bug in
`add_tokens()` for SentencePiece-backed tokenizers like M2M100's (issue
#116) -- see that function's docstring for the confirmed root cause. This
only patches the tokenizer's `convert_tokens_to_string` (a no-op for
duck-typed fakes that don't have one), so it doesn't affect the
`get_vocab()`/`add_tokens()`-only unit tests above; it's covered
separately, against the real checkpoint, in
`tests/integration/test_tokenizer_extension_word_boundary_spacing.py`.

## Re-evaluating an already-trained checkpoint (issue #116's re-evaluation gap)

`_mark_word_boundary_tokens`'s effect lives entirely on a live tokenizer
*instance* (a monkey-patched `convert_tokens_to_string` closure plus a
mutable set) -- it is never persisted by `save_pretrained()`, which only
writes a flat `added_tokens.json` list. Reloading a checkpoint trained
before this module called `_mark_word_boundary_tokens` (i.e. every
checkpoint trained before issue #116's fix, including ones already
deployed) therefore can't recover which added tokens were whole
words/characters (`extend_tokenizer_vocab`) versus subword pieces
(`extend_tokenizer_vocab_with_subwords`) from the saved files alone:
word-initial subword pieces are self-describing via their own
`WORD_BOUNDARY_MARKER` prefix, but whole-word tokens and subword
*continuation* pieces are both marker-less multi-character strings,
indistinguishable from each other by inspection.

`reconstruct_whole_word_boundary_tokens` / `patch_word_boundary_decoding_
for_checkpoint` below solve this without retraining: given the exact base
model vocab and training corpus text the checkpoint's own
`training.train.extend_vocabulary_for_examples` call used,
`compute_new_tokens_for_texts` is pure/deterministic (no SentencePiece
training, no seed dependency), so it can be re-run after the fact to
recover exactly the same whole-word token set -- see
`reconstruct_whole_word_boundary_tokens`'s own docstring for the full
argument, including why this still works even across a chain of
`--init-model` continuation runs. `evaluation.evaluate_checkpoint` is the
real caller of this.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from training.subword_vocab import (
    DEFAULT_MIN_SUBWORD_LENGTH,
    DEFAULT_MODEL_TYPE,
    DEFAULT_VOCAB_SIZE,
    WORD_BOUNDARY_MARKER,
    compute_new_subword_tokens,
)
from training.vocab_extension import resize_embedding_matrix, select_new_tokens
from training.vocab_gap import find_missing_characters, find_missing_words


class TokenizerLike(Protocol):
    """The subset of the HF tokenizer API `extend_tokenizer_vocab` needs."""

    def get_vocab(self) -> dict[str, int]: ...

    def add_tokens(self, new_tokens: list[str]) -> int: ...


# Attribute name used to stash the mutable set of "always decode with a
# leading space" tokens on a real tokenizer instance -- see
# `_mark_word_boundary_tokens`'s docstring (issue #116) for why this is
# necessary and why it isn't just another `add_tokens()` call.
_WORD_BOUNDARY_TOKENS_ATTR = "_translator_word_boundary_added_tokens"


def _mark_word_boundary_tokens(tokenizer: TokenizerLike, tokens: Iterable[str]) -> None:
    """Register `tokens` (already registered via `tokenizer.add_tokens()`)
    as tokens that must always decode with a leading space, working around
    a decode-time bug in `add_tokens()` for SentencePiece-backed tokenizers
    like M2M100's (issue #116).

    Root cause, confirmed against the real `facebook/m2m100_418M`
    tokenizer: `tokenizer.add_tokens()` only registers new tokens in HF's
    `added_tokens_encoder` overlay -- it never touches the real
    `sp_model` (`sentencepiece.SentencePieceProcessor`) that
    `M2M100Tokenizer.convert_tokens_to_string` delegates decoding to via
    `self.sp_model.decode(...)`. *Encoding* still works correctly (HF's
    generic added-token trie matches the literal added text directly
    against raw input text, entirely independent of `sp_model`), but on
    *decode*, `sp_model.decode()` has no way to apply its usual
    SentencePiece "word-boundary marker -> space" handling to a token
    string it has never seen: it silently concatenates an added token
    directly onto whatever piece preceded it, dropping the space that
    should have been there (e.g. "...de dios" + "awach" decodes as
    "...de diosawach", not "...de dios awach") -- this is exactly the
    "glued Spanish words" pattern documented in issue #116, not a rare
    edge case.

    This works around it at the decode layer, without touching the
    (correctly-working) encode path or `add_tokens()` itself: it wraps the
    tokenizer's own `convert_tokens_to_string` exactly once (subsequent
    calls just extend the same tracked set), grouping the token list into
    runs and joining those runs back together with an explicit space.

    A run boundary -- i.e. a real word boundary needing a space before it
    -- happens right before any token that *starts a new word*: either one
    of `tokens` (an added token we're told always starts a new word), or
    any other token that itself already carries SentencePiece's own
    word-boundary marker (a genuine native piece, or a word-initial added
    subword piece from `extend_tokenizer_vocab_with_subwords`). Everything
    else -- single missing characters from `find_missing_characters`,
    mid-word subword continuation pieces -- has no marker and is *not* in
    `tokens`, so it keeps accumulating into the current run and stays
    glued to whatever precedes it, exactly as before.

    Critically, this means an added *word-initial* token immediately
    followed by an added *continuation* piece (no marker) lands in the
    *same* run and gets glued together correctly -- e.g. `["▁zqvbn",
    "wkr"]` (two newly added subword pieces composing one Kaqchikel word,
    the exact scenario `extend_tokenizer_vocab_with_subwords`/#82 exists
    for) decodes as `"zqvbnwkr"`, not `"zqvbn wkr"`. An earlier version of
    this function always started a fresh run right after *any* token in
    `tokens`, which incorrectly inserted a space in exactly this case.

    No-ops for tokenizer-like objects that don't implement
    `convert_tokens_to_string` (e.g. the duck-typed fakes used in this
    module's unit tests, which only need `get_vocab`/`add_tokens`).
    """
    tokens = list(tokens)
    if not tokens or not hasattr(tokenizer, "convert_tokens_to_string"):
        return

    boundary_tokens: set[str] | None = getattr(tokenizer, _WORD_BOUNDARY_TOKENS_ATTR, None)
    if boundary_tokens is None:
        boundary_tokens = set()
        setattr(tokenizer, _WORD_BOUNDARY_TOKENS_ATTR, boundary_tokens)
        original_convert_tokens_to_string = tokenizer.convert_tokens_to_string

        def convert_tokens_to_string_with_word_boundaries(tokens_to_decode: list[str]) -> str:
            runs: list[list[str]] = []
            current_run: list[str] = []
            for token in tokens_to_decode:
                starts_new_word = token in boundary_tokens or token.startswith(
                    WORD_BOUNDARY_MARKER
                )
                if starts_new_word and current_run:
                    runs.append(current_run)
                    current_run = []
                current_run.append(token)
            if current_run:
                runs.append(current_run)

            decoded_runs = []
            for run in runs:
                # `sp_model.decode()` (what `original_convert_tokens_to_string`
                # delegates to) doesn't know about our own added tokens, so
                # it can't strip/convert a marker it never registered on
                # one -- do it ourselves before delegating, only for a
                # run's leading token if *we* are the ones vouching it
                # starts a new word. A genuinely native marker-carrying
                # leading token is left untouched: the real decode already
                # converts/strips its marker correctly on its own.
                if run[0] in boundary_tokens:
                    run = [run[0].removeprefix(WORD_BOUNDARY_MARKER), *run[1:]]
                decoded_runs.append(original_convert_tokens_to_string(run))
            return " ".join(part for part in decoded_runs if part).strip()

        tokenizer.convert_tokens_to_string = convert_tokens_to_string_with_word_boundaries

    boundary_tokens.update(tokens)


def _select_whole_word_tokens(tokens: Iterable[str]) -> list[str]:
    """Whole word-forms (anything longer than a single character) always
    start a new word wherever they occur and must decode with a leading
    space (issue #116); single characters are mid-word content and must
    stay glued to their neighbors. Shared by `extend_tokenizer_vocab` (a
    live extension) and `reconstruct_whole_word_boundary_tokens` (a
    from-scratch recomputation of what a live extension would have added,
    for a checkpoint that was never marked -- see this module's docstring)
    so the two can't drift on this filter.
    """
    return [token for token in tokens if len(token) > 1]


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
        _mark_word_boundary_tokens(tokenizer, _select_whole_word_tokens(new_tokens))
    return new_tokens


def reconstruct_whole_word_boundary_tokens(
    base_vocab: dict[str, int], sample_texts: Iterable[str]
) -> list[str]:
    """Recompute, from scratch, the whole-word/multi-character tokens that
    a live `extend_tokenizer_vocab` call would add for `sample_texts`
    against a *pristine* `base_vocab` -- e.g. a freshly loaded
    `facebook/m2m100_418M` tokenizer's own `get_vocab()`, never one with
    any prior extension already applied.

    ## Why this exists: a saved checkpoint can't tell you this on its own

    `_mark_word_boundary_tokens`'s effect (issue #116's fix) lives only on
    a live tokenizer *instance* -- a monkey-patched
    `convert_tokens_to_string` plus a mutable set -- and is never persisted
    by `save_pretrained()`, which only ever writes a flat `added_tokens.
    json` list. Reloading a checkpoint's tokenizer therefore can't tell,
    from the saved files alone, which marker-less multi-character added
    tokens are whole words from `extend_tokenizer_vocab` (must decode with
    a leading space) versus subword *continuation* pieces from
    `extend_tokenizer_vocab_with_subwords` (must stay glued to their
    neighbor) -- both are plain multi-character strings with no
    `WORD_BOUNDARY_MARKER` prefix. Word-*initial* subword pieces don't have
    this problem: they carry the marker themselves and are already handled
    correctly by `_mark_word_boundary_tokens`'s own marker check,
    regardless of whether they were ever explicitly registered in its
    `tokens` argument.

    ## Why recomputing this is safe

    `compute_new_tokens_for_texts` is a pure, deterministic function of
    `sample_texts` and `base_vocab` -- character/word set operations plus
    `training.vocab_extension.select_new_tokens`'s sorted dedupe, no
    SentencePiece training, no seed dependency anywhere in this step
    (unlike `extend_tokenizer_vocab_with_subwords`, which trains a fresh
    SentencePiece model and is *not* safe to reconstruct this way -- it
    doesn't need to be, since its output is self-describing via the
    marker). Given the exact base model vocab and the exact training
    corpus text a checkpoint's own `training.train.
    extend_vocabulary_for_examples` call used to build its `sample_texts`
    (every training example's source *and* target text, plus the
    direction tag tokens), re-running this recovers exactly the same
    whole-word/character token list the live call added.

    This still holds even across a chain of `--init-model` continuation
    runs that each re-ran `extend_tokenizer_vocab` against the *same*
    corpus: `select_new_tokens` only ever returns tokens not already in
    the vocab, so re-applying whole-word extension against a vocab that
    already covers the corpus is a no-op. The set of whole-word tokens a
    checkpoint's tokenizer actually has is therefore a fixed point of its
    training corpus, unaffected by how many training passes contributed to
    reaching it (confirmed against this project's real deployed checkpoint
    and its continuation chain, issue #116's re-evaluation follow-up).

    This is **not** safe to use if `sample_texts`/`base_vocab` don't
    actually match what the checkpoint's training history used (a
    different corpus version, a different `--direction`, or a different
    base model) -- callers must pass the exact training corpus/direction
    recorded in the checkpoint's own (or its continuation chain's
    earliest) training run metadata, not the evaluation run's own
    settings. See `evaluation.evaluate_checkpoint` for the real caller.
    """
    new_tokens = compute_new_tokens_for_texts(sample_texts, base_vocab)
    return _select_whole_word_tokens(new_tokens)


def patch_word_boundary_decoding_for_checkpoint(
    tokenizer: TokenizerLike, base_vocab: dict[str, int], sample_texts: Iterable[str]
) -> list[str]:
    """Patch an already-loaded checkpoint tokenizer's decode behavior with
    issue #116's word-boundary-spacing fix, for a checkpoint whose
    tokenizer was saved *without* `_mark_word_boundary_tokens` ever having
    been called on it (every checkpoint trained before this reconstruction
    path existed -- see `reconstruct_whole_word_boundary_tokens`'s
    docstring for why a saved checkpoint can't just carry this information
    directly).

    Returns the boundary token list actually applied, so callers (e.g.
    `evaluation.evaluate_checkpoint`) can sanity-check it against the
    checkpoint's own vocabulary -- e.g. warn if some reconstructed tokens
    are missing from the checkpoint's `get_vocab()`, a sign
    `sample_texts`/`base_vocab` don't actually match what the checkpoint
    was trained with.
    """
    boundary_tokens = reconstruct_whole_word_boundary_tokens(base_vocab, sample_texts)
    _mark_word_boundary_tokens(tokenizer, boundary_tokens)
    return boundary_tokens


def extend_tokenizer_vocab_with_subwords(
    tokenizer: TokenizerLike,
    kaqchikel_texts: Iterable[str],
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    model_type: str = DEFAULT_MODEL_TYPE,
    min_subword_length: int = DEFAULT_MIN_SUBWORD_LENGTH,
) -> list[str]:
    """Extend `tokenizer`'s vocabulary in place with high-value Kaqchikel
    subwords (issue #82), complementing `extend_tokenizer_vocab`'s
    character/whole-word coverage.

    Trains a fresh SentencePiece/Unigram model on `kaqchikel_texts` (the
    Kaqchikel-only side of the corpus -- never mixed with Spanish, see
    `training.subword_vocab`'s module docstring), diffs its vocabulary
    against `tokenizer.get_vocab()`, and adds whatever high-value subwords
    aren't already present. As with `extend_tokenizer_vocab`, the
    corresponding model's embeddings must still be resized afterward (see
    `resize_embeddings_for_new_tokens` below) -- this only touches the
    tokenizer.

    Returns the list of tokens this call attempted to add (empty if
    `kaqchikel_texts` is already fully covered).
    """
    base_vocab = tokenizer.get_vocab()
    new_tokens = compute_new_subword_tokens(
        kaqchikel_texts,
        base_vocab,
        vocab_size=vocab_size,
        model_type=model_type,
        min_subword_length=min_subword_length,
    )
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        # These pieces already carry SentencePiece's own word-boundary
        # marker convention from the trained subword model: a piece
        # starting with it begins a new word and must decode with a
        # leading space (issue #116); a piece without it is a mid-word
        # continuation and must stay glued -- already correct.
        _mark_word_boundary_tokens(
            tokenizer, (token for token in new_tokens if token.startswith(WORD_BOUNDARY_MARKER))
        )
    return new_tokens


def resize_embeddings_for_new_tokens(model, num_new_tokens: int, *, seed: int | None = None) -> None:
    """Grow `model`'s token embeddings by `num_new_tokens` rows and warm-start them.

    Real usage: `model` is a `transformers.M2M100ForConditionalGeneration`
    loaded from the same checkpoint as the tokenizer passed to
    `extend_tokenizer_vocab`, and `num_new_tokens` is `len(extend_tokenizer_vocab(...))`
    -- the count of tokens actually added, not the tokenizer's resulting
    `len(tokenizer)`. Those two are *not* interchangeable: a real M2M100
    checkpoint's embedding matrix is pre-padded a few rows beyond its
    tokenizer's raw vocab size (128112 rows for a 128104-token vocab, as
    of `facebook/m2m100_418M`), so comparing `len(tokenizer)` against the
    model's current embedding row count would silently no-op whenever the
    tokenizer's growth fit inside that existing padding -- leaving new
    token ids pointing at untrained padding rows instead of warm-started
    ones. Always growing by the actual new-token count sidesteps that
    entirely. M2M100 ties its input and output embeddings by default, so
    resizing the input embedding matrix is sufficient.

    This calls HF's own `model.resize_token_embeddings` first (which
    allocates the new rows), then overwrites just the newly added rows
    using `training.vocab_extension.resize_embedding_matrix`'s
    mean-of-existing-rows-plus-noise initialization, rather than leaving
    them at whatever default `resize_token_embeddings` used.

    Covered by `tests/integration/test_tokenizer_extension_real_model.py`
    against the real checkpoint (see ADR 0003); the row-initialization
    math it delegates to is additionally unit-tested in isolation against
    a fake embedding matrix.
    """
    if num_new_tokens <= 0:
        return

    import torch

    embeddings = model.get_input_embeddings()
    old_weight = embeddings.weight.detach().cpu().numpy()
    old_size = old_weight.shape[0]
    new_size = old_size + num_new_tokens

    resized = resize_embedding_matrix(old_weight, num_new_tokens, seed=seed)

    model.resize_token_embeddings(new_size)
    with torch.no_grad():
        model.get_input_embeddings().weight[old_size:new_size] = torch.from_numpy(
            resized[old_size:new_size]
        )
