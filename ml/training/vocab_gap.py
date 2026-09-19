"""Identify Kaqchikel characters/word-forms not covered by a base tokenizer
vocabulary.

Kaqchikel is written in standard Latin script -- the same alphabet Spanish
uses, plus a handful of extra vowels (ä, ë, ï, ö, ü) and a plain apostrophe
marking glottalized consonants (k', tz', ch', q', etc.). Because a
multilingual tokenizer like M2M100's already covers Spanish, it's tempting
to assume Kaqchikel "just works" too -- but that assumption needs checking,
not taking on faith (per #34/ADR 0003): the extra vowels and the glottal
apostrophe are exactly the characters a Spanish-only vocabulary would be
missing. This module only checks what's actually missing; it doesn't
assume any particular gap up front.
"""

from __future__ import annotations

from collections.abc import Iterable


def extract_characters(texts: Iterable[str]) -> set[str]:
    """Return the set of distinct characters appearing across `texts`."""
    characters: set[str] = set()
    for text in texts:
        characters.update(text)
    return characters


def find_missing_characters(texts: Iterable[str], vocab_characters: Iterable[str]) -> set[str]:
    """Return characters used in `texts` that are absent from `vocab_characters`.

    Whitespace is never reported as missing -- every tokenizer handles a
    plain space, so surfacing it here would just be noise obscuring the
    characters that actually matter.
    """
    known = set(vocab_characters)
    return {c for c in extract_characters(texts) if c not in known and not c.isspace()}


def find_missing_words(texts: Iterable[str], vocab_tokens: Iterable[str]) -> set[str]:
    """Return whole word-forms from `texts` absent from `vocab_tokens`.

    This is a coarse, whole-word signal, not a real subword-segmentation
    simulation: a word missing here may still be representable by the base
    tokenizer as a sequence of several known subword/character pieces
    (inefficient, but not information-losing). It's useful for flagging
    common Kaqchikel word-forms -- especially ones carrying the glottal
    apostrophe -- worth adding as single whole tokens, rather than assuming
    the model will learn them well as fine-grained fragments.
    """
    known = set(vocab_tokens)
    words: set[str] = set()
    for text in texts:
        words.update(text.split())
    return {w for w in words if w not in known}
