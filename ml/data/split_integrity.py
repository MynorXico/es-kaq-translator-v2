"""Train/val split integrity validation for Spanish-Kaqchikel sentence pairs.

The main risk this guards against is evaluation leakage: if a sentence
pair (or even just one side of a pair) appears in both the training and
validation sets, validation metrics (BLEU/chrF) become optimistic and no
longer reflect generalization.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

SentencePair = tuple[str, str]


@dataclass(frozen=True)
class SplitIntegrityReport:
    """Overlap found between a train split and a val split."""

    exact_pair_overlap: list[SentencePair] = field(default_factory=list)
    source_overlap: list[str] = field(default_factory=list)
    target_overlap: list[str] = field(default_factory=list)

    @property
    def has_exact_overlap(self) -> bool:
        return len(self.exact_pair_overlap) > 0


class SplitIntegrityError(ValueError):
    """Raised when train/val splits overlap on exact sentence pairs."""

    def __init__(self, report: SplitIntegrityReport):
        self.report = report
        super().__init__(
            f"Found {len(report.exact_pair_overlap)} exact sentence pair(s) "
            "shared between train and val splits -- this is data leakage."
        )


def validate_split_integrity(
    train_pairs: Iterable[SentencePair],
    val_pairs: Iterable[SentencePair],
    raise_on_overlap: bool = True,
) -> SplitIntegrityReport:
    """Check a train/val split for overlapping sentence pairs.

    Always computes exact-pair overlap (both sides match -- the strict
    leakage case) as well as one-sided overlap on the Spanish source and
    the Kaqchikel target alone (weaker signal, reported but not fatal).

    By default, raises `SplitIntegrityError` if any exact-pair overlap is
    found. Pass `raise_on_overlap=False` to only get the report back.
    """
    train_pairs = list(train_pairs)
    val_pairs = list(val_pairs)

    train_pair_set = set(train_pairs)
    exact_pair_overlap = [pair for pair in val_pairs if pair in train_pair_set]

    train_sources = {es for es, _ in train_pairs}
    train_targets = {cak for _, cak in train_pairs}

    seen_sources: set[str] = set()
    source_overlap: list[str] = []
    seen_targets: set[str] = set()
    target_overlap: list[str] = []
    for es, cak in val_pairs:
        if es in train_sources and es not in seen_sources:
            source_overlap.append(es)
            seen_sources.add(es)
        if cak in train_targets and cak not in seen_targets:
            target_overlap.append(cak)
            seen_targets.add(cak)

    # De-duplicate exact_pair_overlap while preserving first-seen order.
    seen_exact: set[SentencePair] = set()
    deduped_exact_overlap = []
    for pair in exact_pair_overlap:
        if pair not in seen_exact:
            deduped_exact_overlap.append(pair)
            seen_exact.add(pair)

    report = SplitIntegrityReport(
        exact_pair_overlap=deduped_exact_overlap,
        source_overlap=source_overlap,
        target_overlap=target_overlap,
    )

    if raise_on_overlap and report.has_exact_overlap:
        raise SplitIntegrityError(report)

    return report
