"""Sentence-length filtering for Spanish-Kaqchikel sentence pairs.

Drops pairs that are empty, too short, too long, or whose two sides have a
character-length ratio suggesting misalignment rather than a genuinely
long/short translation of the same content.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

SentencePair = tuple[str, str]


@dataclass(frozen=True)
class LengthFilterConfig:
    """Configurable thresholds for length-based filtering."""

    min_length: int = 1
    max_length: int = 400
    max_ratio: float = 4.0


def is_valid_length(es: str, cak: str, config: LengthFilterConfig | None = None) -> bool:
    """Return whether a sentence pair passes the length/ratio thresholds."""
    config = config or LengthFilterConfig()

    es_len = len(es.strip())
    cak_len = len(cak.strip())

    if es_len == 0 or cak_len == 0:
        return False
    if es_len < config.min_length or cak_len < config.min_length:
        return False
    if es_len > config.max_length or cak_len > config.max_length:
        return False

    longer, shorter = max(es_len, cak_len), min(es_len, cak_len)
    return not longer / shorter > config.max_ratio


def filter_pairs_by_length(
    pairs: Iterable[SentencePair], config: LengthFilterConfig | None = None
) -> list[SentencePair]:
    """Keep only pairs that pass `is_valid_length`."""
    config = config or LengthFilterConfig()
    return [(es, cak) for es, cak in pairs if is_valid_length(es, cak, config=config)]
