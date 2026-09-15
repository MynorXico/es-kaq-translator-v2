"""Deduplication of Spanish-Kaqchikel sentence pairs."""

from __future__ import annotations

from collections.abc import Iterable

SentencePair = tuple[str, str]


def deduplicate_pairs(pairs: Iterable[SentencePair]) -> list[SentencePair]:
    """Remove exact-duplicate (source, target) pairs, keeping first occurrence.

    A pair is only considered a duplicate if both the Spanish source and the
    Kaqchikel target match exactly. A repeated Spanish sentence with a
    different Kaqchikel translation (e.g. two valid ways to translate the
    same greeting) is not a duplicate.
    """
    seen: set[SentencePair] = set()
    result: list[SentencePair] = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        result.append(pair)
    return result
