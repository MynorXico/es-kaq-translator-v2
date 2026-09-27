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


def compute_metrics_by_direction(
    hypotheses: list[str], references: list[str], directions: list[str]
) -> dict[str, MetricsResult]:
    """Compute BLEU/chrF separately for each distinct translation direction.

    Every reported number so far (see `ml/README.md`'s history) mixed
    es->cak and cak->es into one combined score -- there was no per-
    direction breakdown anywhere. This buckets `hypotheses`/`references` by
    their aligned `directions` label (e.g. `"es->cak"`/`"cak->es"`, matching
    `training.direction`'s `f"{source_lang}->{target_lang}"` convention)
    before delegating to `compute_metrics` for each bucket, so a strong
    direction can't mask a weak one (or vice versa) in the combined number.
    This is also the concrete evidence ADR 0006 needs to decide
    one-model-vs-two.

    Args:
        hypotheses: model output, one string per sentence.
        references: gold translations, one string per sentence, aligned
            1:1 with `hypotheses` by index.
        directions: direction label per sentence, aligned 1:1 with
            `hypotheses`/`references` by index.

    Returns:
        A dict mapping each distinct direction label to its own
        `MetricsResult`, in first-seen order (so, for this project's own
        callers, `"es->cak"` before `"cak->es"` -- see
        `training.direction.build_direction_examples`).

    Raises:
        ValueError: if `hypotheses`, `references`, and `directions` don't
            all have the same length -- the same "fail loudly on
            misalignment" contract as `compute_metrics`.
    """
    if not (len(hypotheses) == len(references) == len(directions)):
        raise ValueError(
            f"hypotheses, references, and directions must all be the same "
            f"length (got {len(hypotheses)} hypotheses, {len(references)} "
            f"references, {len(directions)} directions)"
        )

    buckets: dict[str, tuple[list[str], list[str]]] = {}
    for hypothesis, reference, direction in zip(hypotheses, references, directions):
        bucket_hypotheses, bucket_references = buckets.setdefault(direction, ([], []))
        bucket_hypotheses.append(hypothesis)
        bucket_references.append(reference)

    return {
        direction: compute_metrics(hypotheses=bucket_hypotheses, references=bucket_references)
        for direction, (bucket_hypotheses, bucket_references) in buckets.items()
    }
