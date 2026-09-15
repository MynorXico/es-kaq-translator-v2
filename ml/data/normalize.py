"""Whitespace/encoding normalization for Spanish-Kaqchikel sentence pairs.

Note on casing: normalization deliberately never lowercases or otherwise
changes case. Both Spanish and Kaqchikel use capitalization meaningfully
(sentence-initial letters, proper nouns), and Kaqchikel additionally
distinguishes some digraphs/letters where case-folding could be lossy.
Lowercasing here would destroy signal the model should learn from, so it
is left to the tokenizer/training config to decide (e.g. via a
case-sensitive vocabulary) rather than being baked into the corpus.
"""

from __future__ import annotations

import re
import unicodedata

SentencePair = tuple[str, str]

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize a single sentence: Unicode NFC + whitespace cleanup.

    - Applies Unicode NFC normalization so visually-identical characters
      (e.g. a precomposed "á" vs. "a" + combining acute accent) collapse to
      the same codepoint sequence. This matters here because source texts
      digitized from different tools can encode accented Spanish letters
      and the Kaqchikel saltillo/apostrophe inconsistently.
    - Collapses any run of whitespace (spaces, tabs, newlines) to a single
      space, and strips leading/trailing whitespace.
    - Does not change case (see module docstring).
    """
    normalized = unicodedata.normalize("NFC", text)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def normalize_pair(es: str, cak: str) -> SentencePair:
    """Apply `normalize_text` to both sides of a sentence pair."""
    return normalize_text(es), normalize_text(cak)
