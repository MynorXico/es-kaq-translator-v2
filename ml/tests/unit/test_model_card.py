"""Unit tests for model card rendering (evaluation.model_card).

A model card only ever records identifiers/aggregate metadata about a
training run (corpus version tag, sentence counts, hyperparameters,
metrics) -- never raw corpus content, since a model card is a
public-facing artifact and the private ALMG corpus must never leak into
one (ADR 0002).
"""

from evaluation.metrics import MetricsResult
from evaluation.model_card import ModelCardData, parse_model_card_metrics, render_model_card


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


def test_render_model_card_includes_per_direction_metrics_when_present():
    # Issue #178: every combined-direction number mixes es->cak and cak->es
    # -- the model card must be able to report both separately, which is
    # also the concrete evidence ADR 0006 needs for one-model-vs-two.
    card = render_model_card(
        make_card_data(
            per_direction_metrics={
                "es->cak": MetricsResult(bleu=10.5, chrf=30.5, num_sentences=1867),
                "cak->es": MetricsResult(bleu=20.5, chrf=40.5, num_sentences=1868),
            }
        )
    )

    assert "## Metrics by direction" in card
    assert "es->cak" in card
    assert "cak->es" in card
    assert "10.5" in card
    assert "30.5" in card
    assert "20.5" in card
    assert "40.5" in card
    assert "1,867" in card or "1867" in card
    assert "1,868" in card or "1868" in card


def test_render_model_card_omits_per_direction_section_when_absent():
    card = render_model_card(make_card_data(per_direction_metrics=None))

    assert "Metrics by direction" not in card


# ---------------------------------------------------------------------------
# parse_model_card_metrics (issue #190: moved here from training.submit_job
# so both training.train's self-registration and training.submit_job's
# existing-artifact registration can share one implementation instead of
# each maintaining its own copy of the same regex).
# ---------------------------------------------------------------------------


def test_parse_model_card_metrics_extracts_bleu_and_chrf_from_a_rendered_card():
    card = render_model_card(make_card_data())

    metrics = parse_model_card_metrics(card)

    assert metrics["bleu"] == "24.7"
    assert metrics["chrf"] == "48.2"


def test_parse_model_card_metrics_returns_empty_dict_when_metrics_absent():
    metrics = parse_model_card_metrics("# Model card: run-x\n\nNo metrics here.\n")

    assert metrics == {}
