"""Entrypoint tying BLEU/chrF computation and model card generation together.

This is deliberately parameterized by paths passed in by the caller (e.g.
a SageMaker post-training step) -- it never hardcodes a path to the real
private validation set. Callers are responsible for pointing it at
whatever predictions/references files exist for a given run (fixture
data in tests, real S3-sourced files in a training pipeline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.metrics import MetricsResult, compute_metrics
from evaluation.model_card import ModelCardData, render_model_card


def _read_lines(path: str | Path) -> list[str]:
    """Read a text file into a list of stripped, non-empty lines."""
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def run_evaluation(
    predictions_path: str | Path,
    references_path: str | Path,
    run_metadata: dict[str, Any],
    output_path: str | Path,
) -> tuple[MetricsResult, Path]:
    """Compute metrics for one run and write its model card to `output_path`.

    Args:
        predictions_path: file with one model hypothesis per line.
        references_path: file with one gold reference per line, aligned
            1:1 with `predictions_path` by line order.
        run_metadata: everything about the run other than the metrics
            themselves -- run_id, timestamp, base_model, direction,
            corpus_version, train_sentence_count, hyperparameters, and
            an optional notes string. `validation_sentence_count` is
            derived from `references_path`, not taken from this dict, so
            it can't drift out of sync with the actual file evaluated.
        output_path: where to write the rendered Markdown model card.

    Returns:
        A tuple of the computed `MetricsResult` and the `Path` the model
        card was written to (== `output_path`, returned for convenience).
    """
    hypotheses = _read_lines(predictions_path)
    references = _read_lines(references_path)

    metrics = compute_metrics(hypotheses=hypotheses, references=references)

    card_data = ModelCardData(
        run_id=run_metadata["run_id"],
        timestamp=run_metadata["timestamp"],
        base_model=run_metadata["base_model"],
        direction=run_metadata["direction"],
        corpus_version=run_metadata["corpus_version"],
        train_sentence_count=run_metadata["train_sentence_count"],
        validation_sentence_count=len(references),
        hyperparameters=run_metadata["hyperparameters"],
        metrics=metrics,
        notes=run_metadata.get("notes"),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_model_card(card_data), encoding="utf-8")

    return metrics, output_path
