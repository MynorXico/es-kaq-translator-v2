"""Train a Kaqchikel-only SentencePiece/Unigram subword vocabulary and diff
it against a base tokenizer's vocab to find high-value subwords it doesn't
already represent as single tokens.

## Why this exists (issue #82)

Run #66 already extended M2M100's tokenizer with ~61,901 whole word-forms
(`training.vocab_gap.find_missing_words`, wired into
`training.tokenizer_extension.extend_tokenizer_vocab`) -- at the time
assumed to be Kaqchikel, though issue #125's later investigation found a
large fraction of that run's added tokens were actually ordinary Spanish
words the whole-word step wasn't yet scoped away from (fixed in
`training.train.extend_vocabulary_for_examples` after #125; this
docstring's "Kaqchikel-only" framing below describes the current,
post-fix behavior, not what run #66 itself did). Whatever fraction of it
genuinely was Kaqchikel closes the "this exact word was never seen" gap,
but says nothing about the *subword* structure underneath: Kaqchikel is
agglutinative (ergative/absolutive
person marking, noun incorporation), so a word-form the model hasn't
memorized verbatim still gets fragmented by M2M100's generic multilingual
SentencePiece model into long, awkward multi-token chains it has no reason
to compose cleanly at inference. Three real training runs (#66, #76, and
its continuation) showed BLEU gains shrinking sharply while training loss
kept dropping -- the signature of a representational ceiling, not
undertraining (see issue #82 for the full numbers).

This module trains a fresh, small SentencePiece Unigram model *on the
Kaqchikel side of the corpus only* (never mixed with Spanish, which M2M100
already tokenizes natively), and diffs its resulting subword pieces against
a base tokenizer's existing vocab, so the ones missing from the base
tokenizer can be merged in -- reusing the same generic diff logic
(`training.vocab_extension.select_new_tokens`) already used for whole
words and characters, rather than building a second, fully separate
tokenizer. A separate tokenizer would sever M2M100's pretrained
multilingual embedding alignment, which is the model's main
transfer-learning advantage for a low-resource language like Kaqchikel.

## Design notes

- `hard_vocab_limit=False` is passed to `SentencePieceTrainer` deliberately:
  the requested `vocab_size` is a target, not a hard requirement.
  SentencePiece's unigram trainer otherwise raises a hard `RuntimeError`
  when the requested size can't be reached from the given text -- this
  isn't a rare corner case, it hits every small/fixture-sized input this
  repo's fast test suite uses, and would otherwise force every caller to
  precompute a "safe" vocab_size by hand. With this flag, a too-large
  request is silently capped to whatever the trainer can actually produce,
  instead of crashing outright -- also safer for a real (billable)
  SageMaker run, since a slightly-too-optimistic vocab_size shouldn't blow
  up an entire multi-hour job at the vocab-training step.
- Subword pieces of length 1 (once SentencePiece's own leading word-
  boundary marker is stripped) are filtered out by
  `select_high_value_subwords`: single-character coverage is already
  `training.vocab_gap.find_missing_characters`'s job, so re-adding single
  characters here would be redundant, not "high value".
- Training happens fully in memory (`io.BytesIO`, via SentencePiece's
  `sentence_iterator`/`model_writer` kwargs) -- no temp files, nothing
  written to disk by default. Callers who do persist the trained model
  proto must treat it under the same privacy rules as whatever corpus
  text it was trained from (ADR 0002): a subword vocabulary trained on the
  private ALMG corpus is derived from that corpus and must stay private
  too, even though it doesn't reproduce corpus sentences verbatim.

Neither this module nor `training.vocab_extension` requires a real M2M100
checkpoint -- both operate on plain strings/dicts, `sentencepiece`'s own
(lightweight, local, no-download) trainer, and the base tokenizer's
`get_vocab()` dict. `training/tokenizer_extension.py` wires this to a real
Hugging Face tokenizer, same pattern as the whole-word/character extension
already there.
"""

from __future__ import annotations

import io
from collections.abc import Iterable

from training.vocab_extension import select_new_tokens

DEFAULT_VOCAB_SIZE = 8000
DEFAULT_MODEL_TYPE = "unigram"
DEFAULT_MIN_SUBWORD_LENGTH = 2

# SentencePiece's own word-boundary marker (U+2581 LOWER ONE EIGHTH BLOCK,
# rendered "▁"), prepended to the first piece of each whitespace-delimited
# word. Not a real character of Kaqchikel or Spanish text.
WORD_BOUNDARY_MARKER = "▁"

# Pieces every SentencePiece model reserves for its own control tokens --
# never real subword content, and already present (or equivalent) in any
# base tokenizer's vocab, so these are dropped before diffing.
_SPECIAL_PIECES = frozenset({"<unk>", "<s>", "</s>", "<pad>"})


def train_subword_model(
    texts: Iterable[str],
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    model_type: str = DEFAULT_MODEL_TYPE,
) -> bytes:
    """Train a SentencePiece model on `texts` and return its serialized
    model proto bytes (never written to disk -- see module docstring).

    `texts` should be Kaqchikel-only sentences (see module docstring) --
    this function itself doesn't know or care what language it's given;
    it's the caller's (`training.train`'s) responsibility to pass only the
    Kaqchikel side of the corpus.

    Raises `ValueError` if `texts` is empty -- SentencePiece itself fails
    on an empty input with a much less obvious low-level error.
    """
    import sentencepiece as spm

    texts = list(texts)
    if not texts:
        raise ValueError("texts must be non-empty to train a subword model")

    model_writer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(texts),
        model_writer=model_writer,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        # See module docstring: don't hard-fail when vocab_size can't be
        # reached from the given text (always true for this repo's tiny
        # test fixtures, and a real risk on a real corpus too).
        hard_vocab_limit=False,
    )
    return model_writer.getvalue()


def extract_vocab_pieces(model_proto: bytes) -> list[str]:
    """Return every subword piece a trained SentencePiece model proto knows,
    excluding its own reserved special/control tokens (`_SPECIAL_PIECES`).
    """
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_proto=model_proto)
    return [
        processor.id_to_piece(i)
        for i in range(processor.get_piece_size())
        if processor.id_to_piece(i) not in _SPECIAL_PIECES
    ]


def select_high_value_subwords(
    pieces: Iterable[str], *, min_subword_length: int = DEFAULT_MIN_SUBWORD_LENGTH
) -> list[str]:
    """Filter `pieces` down to genuine multi-character subword chunks.

    Strips SentencePiece's leading word-boundary marker (if present) before
    measuring length, so e.g. "▁ri" (content "ri", length 2) passes the
    default threshold, but "▁a" (content "a", length 1) does not --
    single characters are `training.vocab_gap.find_missing_characters`'s
    job already, not a "new" signal this module should re-surface.
    """
    high_value = []
    for piece in pieces:
        content = piece.removeprefix(WORD_BOUNDARY_MARKER)
        if len(content) >= min_subword_length:
            high_value.append(piece)
    return high_value


def compute_new_subword_tokens(
    texts: Iterable[str],
    base_vocab: dict[str, int],
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    model_type: str = DEFAULT_MODEL_TYPE,
    min_subword_length: int = DEFAULT_MIN_SUBWORD_LENGTH,
) -> list[str]:
    """Train a subword model on `texts`, then return the deduplicated,
    high-value subwords it discovers that `base_vocab` doesn't already
    have -- ready to hand to a tokenizer's `add_tokens` call exactly like
    any other new-token list in this pipeline
    (`training.vocab_extension.select_new_tokens` does the actual diffing,
    the same function used for whole-word/character extension).
    """
    model_proto = train_subword_model(texts, vocab_size=vocab_size, model_type=model_type)
    pieces = extract_vocab_pieces(model_proto)
    high_value_pieces = select_high_value_subwords(pieces, min_subword_length=min_subword_length)
    return select_new_tokens(high_value_pieces, base_vocab)
