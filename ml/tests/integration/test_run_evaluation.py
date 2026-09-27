"""Integration smoke test for the evaluation+model-card entrypoint.

Per docs/testing.md, this confirms the pipeline wiring connects (reads
prediction/reference files, computes metrics, renders and writes a model
card) against a tiny fixture dataset -- never the real private validation
set, and this is not a model-quality assertion.
"""

from pathlib import Path

import pytest

from evaluation.run import run_evaluation

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_run_evaluation_writes_model_card_and_returns_metrics(tmp_path):
    output_path = tmp_path / "model-card.md"

    metrics, card_path = run_evaluation(
        predictions_path=FIXTURES / "eval_predictions.txt",
        references_path=FIXTURES / "eval_references.txt",
        run_metadata={
            "run_id": "smoke-test-run",
            "timestamp": "2026-09-14T00:00:00Z",
            "base_model": "facebook/m2m100_418M",
            "direction": "cak-to-es",
            "corpus_version": "fixture-v0",
            "train_sentence_count": 100,
            "hyperparameters": {"learning_rate": 3e-5, "epochs": 1},
        },
        output_path=output_path,
    )

    # Wiring check: metrics are the actual computed scores, not placeholders.
    assert 0.0 <= metrics.bleu <= 100.0
    assert 0.0 <= metrics.chrf <= 100.0
    assert metrics.num_sentences == 4

    assert card_path == output_path
    assert output_path.exists()

    card_text = output_path.read_text()
    assert "smoke-test-run" in card_text
    assert "facebook/m2m100_418M" in card_text
    assert "fixture-v0" in card_text
    # Validation sentence count comes from the actual references file, not
    # a caller-supplied number that could drift out of sync with it.
    assert "4" in card_text
    assert f"{metrics.bleu:.1f}" in card_text
    assert f"{metrics.chrf:.1f}" in card_text


def test_run_evaluation_keeps_a_blank_hypothesis_line_aligned(tmp_path):
    # Reproduces the real bug found on the 5th real training run (#66):
    # a blank hypothesis line (the model generated nothing for one
    # example -- plausible given this corpus's many single-word/bare-
    # number entries) must stay in place, not be silently dropped and
    # desync every hypothesis after it from its reference.
    predictions = tmp_path / "predictions.txt"
    references = tmp_path / "references.txt"
    predictions.write_text("hola\n\nadios\ngracias\n", encoding="utf-8")
    references.write_text("utz\nla utz awäch\nchabe'\nmatyox\n", encoding="utf-8")

    metrics, _ = run_evaluation(
        predictions_path=predictions,
        references_path=references,
        run_metadata={
            "run_id": "smoke-test-run",
            "timestamp": "2026-09-14T00:00:00Z",
            "base_model": "facebook/m2m100_418M",
            "direction": "cak-to-es",
            "corpus_version": "fixture-v0",
            "train_sentence_count": 100,
            "hyperparameters": {},
        },
        output_path=tmp_path / "model-card.md",
    )

    assert metrics.num_sentences == 4


def test_run_evaluation_computes_per_direction_metrics_when_directions_path_given(tmp_path):
    # Issue #178: bucket predictions/references by direction before scoring,
    # and surface both directions' numbers on the model card, not just the
    # combined score.
    predictions = tmp_path / "predictions.txt"
    references = tmp_path / "references.txt"
    directions = tmp_path / "directions.txt"
    predictions.write_text("Utz awäch.\nHola.\n", encoding="utf-8")
    references.write_text("Utz awäch.\nAdios.\n", encoding="utf-8")
    directions.write_text("es->cak\ncak->es\n", encoding="utf-8")

    _metrics, card_path = run_evaluation(
        predictions_path=predictions,
        references_path=references,
        run_metadata={
            "run_id": "smoke-test-run",
            "timestamp": "2026-09-14T00:00:00Z",
            "base_model": "facebook/m2m100_418M",
            "direction": "both",
            "corpus_version": "fixture-v0",
            "train_sentence_count": 100,
            "hyperparameters": {},
        },
        output_path=tmp_path / "model-card.md",
        directions_path=directions,
    )

    card_text = card_path.read_text(encoding="utf-8")
    assert "## Metrics by direction" in card_text
    assert "es->cak" in card_text
    assert "cak->es" in card_text


def test_run_evaluation_omits_per_direction_metrics_without_directions_path(tmp_path):
    output_path = tmp_path / "model-card.md"

    run_evaluation(
        predictions_path=FIXTURES / "eval_predictions.txt",
        references_path=FIXTURES / "eval_references.txt",
        run_metadata={
            "run_id": "smoke-test-run",
            "timestamp": "2026-09-14T00:00:00Z",
            "base_model": "facebook/m2m100_418M",
            "direction": "cak-to-es",
            "corpus_version": "fixture-v0",
            "train_sentence_count": 100,
            "hyperparameters": {"learning_rate": 3e-5, "epochs": 1},
        },
        output_path=output_path,
    )

    card_text = output_path.read_text(encoding="utf-8")
    assert "Metrics by direction" not in card_text


def test_run_evaluation_rejects_mismatched_line_counts(tmp_path):
    bad_predictions = tmp_path / "predictions.txt"
    bad_predictions.write_text("Only one line.\n")

    with pytest.raises(ValueError):
        run_evaluation(
            predictions_path=bad_predictions,
            references_path=FIXTURES / "eval_references.txt",
            run_metadata={
                "run_id": "smoke-test-run",
                "timestamp": "2026-09-14T00:00:00Z",
                "base_model": "facebook/m2m100_418M",
                "direction": "cak-to-es",
                "corpus_version": "fixture-v0",
                "train_sentence_count": 100,
                "hyperparameters": {},
            },
            output_path=tmp_path / "model-card.md",
        )
