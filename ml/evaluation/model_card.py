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

    if data.notes:
        lines += [
            "",
            "## Notes",
            "",
            data.notes,
        ]

    return "\n".join(lines) + "\n"
