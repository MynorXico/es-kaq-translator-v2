"""Unit tests for the BLEU/chrF wrapper in evaluation.metrics.

These sanity-check that our wrapper calls sacrebleu correctly and
interprets its results correctly (per docs/testing.md: don't just trust
the library's output blindly). All fixtures are small, hand-written
sentence pairs we can reason about directly -- never the real private
validation set.
"""

import pytest

from evaluation.metrics import compute_metrics

REFERENCES = [
    "Xa jun q'ij xub'än ri samaj.",
    "Ri tijonel nrajo' nutzijoj jun k'ak'a' tzij.",
    "Ronojel qtinamit e utziläj winaqi'.",
]


def test_identical_hypothesis_and_reference_scores_near_perfect():
    result = compute_metrics(hypotheses=list(REFERENCES), references=REFERENCES)

    assert result.bleu == pytest.approx(100.0, abs=1e-6)
    assert result.chrf == pytest.approx(100.0, abs=1e-6)


def test_completely_wrong_hypothesis_scores_low():
    # Unrelated hypotheses sharing essentially no n-grams with the references.
    hypotheses = [
        "El perro corre en el parque.",
        "Mañana lloverá mucho en la costa.",
        "Compré tres kilos de manzanas verdes.",
    ]

    result = compute_metrics(hypotheses=hypotheses, references=REFERENCES)

    assert result.bleu < 10.0
    assert result.chrf < 30.0


def test_compute_metrics_returns_bleu_and_chrf_fields():
    result = compute_metrics(hypotheses=["Utz awäch."], references=["Utz awäch."])

    assert isinstance(result.bleu, float)
    assert isinstance(result.chrf, float)


def test_compute_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_metrics(hypotheses=["one", "two"], references=["only one reference"])
