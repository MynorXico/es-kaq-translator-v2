"""Model card generation for a single training run.

A model card records *identifiers and aggregate metadata* about a run --
corpus version tag, sentence counts, hyperparameters, base model, and
computed metrics -- never raw corpus content. This matters because a
model card is a public-facing artifact (it may accompany a public
release announcement or documentation) while the private ALMG corpus and
the trained weights themselves must never be published (ADR 0002). Keep
this module strictly limited to counts/identifiers, not sentence text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from evaluation.metrics import MetricsResult


@dataclass(frozen=True)
class ModelCardData:
    """Metadata describing one training run, sufficient to reproduce/trace it."""

    run_id: str
    timestamp: str
    base_model: str
    direction: str
    corpus_version: str
    train_sentence_count: int
    validation_sentence_count: int
    hyperparameters: dict[str, Any]
    metrics: MetricsResult
    notes: str | None = field(default=None)
    # Issue #178: per-direction BLEU/chrF (e.g. "es->cak"/"cak->es" ->
    # MetricsResult), in the order they should be rendered. `None` when the
    # caller didn't supply direction labels (e.g. a very old model card
    # predating this field, or a caller that genuinely has none) -- omitted
    # from the rendered card entirely rather than rendered as an empty
    # section, so a card without direction data reads no differently than
    # before this field existed.
    per_direction_metrics: dict[str, MetricsResult] | None = field(default=None)


def render_model_card(data: ModelCardData) -> str:
    """Render a `ModelCardData` as a human-readable Markdown document."""
    lines = [
        f"# Model card: {data.run_id}",
        "",
        f"- **Timestamp**: {data.timestamp}",
        f"- **Base model**: {data.base_model}",
        f"- **Direction**: {data.direction}",
        f"- **Corpus version**: {data.corpus_version}",
        f"- **Training sentences**: {data.train_sentence_count:,}",
        f"- **Validation sentences**: {data.validation_sentence_count:,}",
        "",
        "## Hyperparameters",
        "",
    ]
    for key, value in data.hyperparameters.items():
        lines.append(f"- **{key}**: {value}")

    lines += [
        "",
        "## Metrics (validation set)",
        "",
        f"- **BLEU**: {data.metrics.bleu:.1f}",
        f"- **chrF**: {data.metrics.chrf:.1f}",
        f"- **Sentences evaluated**: {data.metrics.num_sentences:,}",
    ]

    if data.per_direction_metrics:
        lines += [
            "",
            "## Metrics by direction",
            "",
        ]
        for direction, direction_metrics in data.per_direction_metrics.items():
            lines += [
                f"### {direction}",
                "",
                f"- **BLEU**: {direction_metrics.bleu:.1f}",
                f"- **chrF**: {direction_metrics.chrf:.1f}",
                f"- **Sentences evaluated**: {direction_metrics.num_sentences:,}",
                "",
            ]
        lines.pop()  # drop the trailing blank line from the last direction block

    if data.notes:
        lines += [
            "",
            "## Notes",
            "",
            data.notes,
        ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Parsing metrics back out of a rendered model card (issue #190)
# ---------------------------------------------------------------------------
#
# Moved here from `training.submit_job` (where it originated) so both
# `training.train`'s self-registration (reads the model card it just wrote,
# from local disk -- no S3 download at all) and `training.submit_job`'s
# existing-artifact registration path (reads a model card fetched from a
# completed job's artifact) can share one implementation, instead of two
# copies of the same regex silently drifting apart. `render_model_card`
# above is the only producer of this format; keep the two in sync.

_METRIC_LINE_RE = {
    "bleu": re.compile(r"\*\*BLEU\*\*:\s*([0-9.]+)"),
    "chrf": re.compile(r"\*\*chrF\*\*:\s*([0-9.]+)"),
}


def parse_model_card_metrics(model_card_text: str) -> dict[str, str]:
    """Extract BLEU/chrF from a rendered model card (`render_model_card`'s
    output format). Returns string values, ready to use as SageMaker
    `CustomerMetadataProperties` (which only accepts strings).
    """
    metrics: dict[str, str] = {}
    for name, pattern in _METRIC_LINE_RE.items():
        match = pattern.search(model_card_text)
        if match:
            metrics[name] = match.group(1)
    return metrics
