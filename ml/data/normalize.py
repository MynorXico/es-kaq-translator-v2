"""Whitespace/encoding normalization for Spanish-Kaqchikel sentence pairs.

Note on casing: normalization deliberately never lowercases or otherwise
changes case. Both Spanish and Kaqchikel use capitalization meaningfully
(sentence-initial letters, proper nouns), and Kaqchikel additionally
distinguishes some digraphs/letters where case-folding could be lossy.
Lowercasing here would destroy signal the model should learn from, so it
is left to the tokenizer/training config to decide (e.g. via a
case-sensitive vocabulary) rather than being baked into the corpus.

Glottal-stop/apostrophe look-alike normalization (issue #90): the
Kaqchikel glottal stop and glottalized consonants (k', tz', ch', q', ...)
are marked in ALMG orthography with a plain ASCII apostrophe (U+0027).
Aggregate measurement of the real ALMG training corpus (32,906 sentences,
described only via counts/frequencies here -- never raw content, per ADR
0002) found a substantial minority of sentences instead use U+02C8
(MODIFIER LETTER VERTICAL LINE, "ˈ") for the same phoneme -- almost
certainly a font/OCR/typesetting substitution artifact from whichever
source document/digitization pipeline those sentences came from, not an
intentional orthographic choice. Left alone, this fragments what should be
the same grapheme (e.g. "k'" vs. "kˈ") into different tokenizer output,
adding avoidable sparsity for an already vocabulary-constrained
low-resource model (see `ml/README.md`).

U+02BC (MODIFIER LETTER APOSTROPHE) and U+2019 (RIGHT SINGLE QUOTATION
MARK) were also candidates raised in issue #90 as common smart-quote/
typesetting substitutes for a plain apostrophe, but a real-corpus scan
found zero occurrences of either -- so, per the issue's explicit
"verify before assuming" guidance, `_GLOTTAL_LOOKALIKE_TRANSLATION` maps
only the confirmed codepoint. Extend the mapping if a future corpus
version is found to contain other look-alikes, rather than guessing ahead
of evidence.
"""

from __future__ import annotations

import re
import unicodedata

SentencePair = tuple[str, str]

_WHITESPACE_RE = re.compile(r"\s+")

_ASCII_APOSTROPHE = "'"

# Codepoints confirmed (via aggregate scan of the real corpus, issue #90)
# to be used as glottal-stop/apostrophe look-alikes, mapped to the ASCII
# apostrophe -- ALMG's standard orthography and the corpus's majority
# convention. See the module docstring for which candidates were checked
# and ruled out.
_GLOTTAL_LOOKALIKE_TRANSLATION = str.maketrans(
    {
        "ˈ": _ASCII_APOSTROPHE,  # MODIFIER LETTER VERTICAL LINE
    }
)


def normalize_glottal_marks(text: str) -> str:
    """Map confirmed glottal-stop/apostrophe look-alike codepoints to `'`.

    Only touches codepoints confirmed present in the real corpus (see
    module docstring) -- currently just U+02C8. Leaves the ASCII apostrophe
    and any unconfirmed look-alike (e.g. U+02BC, U+2019) unchanged.
    """
    return text.translate(_GLOTTAL_LOOKALIKE_TRANSLATION)


def normalize_text(text: str) -> str:
    """Normalize a single sentence: Unicode NFC, glottal-mark, and whitespace cleanup.

    - Applies Unicode NFC normalization so visually-identical characters
      (e.g. a precomposed "á" vs. "a" + combining acute accent) collapse to
      the same codepoint sequence. This matters here because source texts
      digitized from different tools can encode accented Spanish letters
      and the Kaqchikel saltillo/apostrophe inconsistently.
    - Maps confirmed glottal-stop/apostrophe look-alike codepoints (see
      `normalize_glottal_marks`) to the ASCII apostrophe.
    - Collapses any run of whitespace (spaces, tabs, newlines) to a single
      space, and strips leading/trailing whitespace.
    - Does not change case (see module docstring).
    """
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalize_glottal_marks(normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def normalize_pair(es: str, cak: str) -> SentencePair:
    """Apply `normalize_text` to both sides of a sentence pair."""
    return normalize_text(es), normalize_text(cak)
