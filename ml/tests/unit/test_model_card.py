"""Unit tests for model card rendering (evaluation.model_card).

A model card only ever records identifiers/aggregate metadata about a
training run (corpus version tag, sentence counts, hyperparameters,
metrics) -- never raw corpus content, since a model card is a
public-facing artifact and the private ALMG corpus must never leak into
one (ADR 0002).
"""

from evaluation.metrics import MetricsResult
from evaluation.model_card import ModelCardData, render_model_card


def make_card_data(**overrides) -> ModelCardData:
    defaults = {
        "run_id": "run-2026-09-14-001",
        "timestamp": "2026-09-14T12:00:00Z",
        "base_model": "facebook/m2m100_418M",
        "direction": "es-to-cak",
        "corpus_version": "almg-v1+community-v0",
        "train_sentence_count": 33616,
        "validation_sentence_count": 3735,
        "hyperparameters": {"learning_rate": 3e-5, "epochs": 10, "batch_size": 16},
        "metrics": MetricsResult(bleu=24.7, chrf=48.2, num_sentences=3735),
        "notes": None,
    }
    defaults.update(overrides)
    return ModelCardData(**defaults)


def test_render_model_card_includes_all_core_fields():
    card = render_model_card(make_card_data())

    assert "run-2026-09-14-001" in card
    assert "facebook/m2m100_418M" in card
    assert "es-to-cak" in card
    assert "almg-v1+community-v0" in card
    assert "33616" in card or "33,616" in card
    assert "3735" in card or "3,735" in card
    assert "learning_rate" in card
    assert "3e-05" in card or "3e-5" in card or "0.00003" in card
    assert "24.7" in card
    assert "48.2" in card


def test_render_model_card_includes_notes_when_present():
    card = render_model_card(
        make_card_data(notes="Validation set has 12 exact-duplicate pairs; see issue #X.")
    )

    assert "exact-duplicate" in card


def test_render_model_card_omits_notes_section_when_absent():
    card = render_model_card(make_card_data(notes=None))

    assert "## Notes" not in card


def test_render_model_card_is_markdown_with_a_title():
    card = render_model_card(make_card_data())

    assert card.startswith("# ")
