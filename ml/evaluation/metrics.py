"""BLEU and chrF computation, wrapping sacrebleu.

We deliberately use the standard `sacrebleu` implementation rather than
hand-rolling BLEU/chrF (these metrics have enough subtle edge cases --
tokenization, smoothing, n-gram order -- that a hand-rolled version would
be a maintenance and correctness liability). This module is a thin,
well-tested wrapper: its job is to call sacrebleu correctly and return a
predictable structure, not to reimplement the metrics themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import sacrebleu


@dataclass(frozen=True)
class MetricsResult:
    """BLEU and chrF scores, both on their native 0-100 scale."""

    bleu: float
    chrf: float
    num_sentences: int


def compute_metrics(hypotheses: list[str], references: list[str]) -> MetricsResult:
    """Compute corpus-level BLEU and chrF for a set of hypothesis/reference pairs.

    Args:
        hypotheses: model output, one string per sentence.
        references: gold translations, one string per sentence, aligned
            1:1 with `hypotheses` by index.

    Returns:
        A `MetricsResult` with `bleu` and `chrf`, each on sacrebleu's
        native 0-100 scale (higher is better).

    Raises:
        ValueError: if `hypotheses` and `references` have different
            lengths (a misalignment bug elsewhere should fail loudly
            here rather than silently score the wrong pairs).
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"hypotheses and references must be the same length "
            f"(got {len(hypotheses)} hypotheses, {len(references)} references)"
        )

    # sacrebleu's corpus_* functions accept multiple reference streams (for
    # multi-reference evaluation); we only ever have one reference per
    # sentence, so we wrap it in a single-element list of reference streams.
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])

    return MetricsResult(
        bleu=bleu.score,
        chrf=chrf.score,
        num_sentences=len(hypotheses),
    )
