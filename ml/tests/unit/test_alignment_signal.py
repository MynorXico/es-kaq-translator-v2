"""Unit tests for `data.alignment_signal` (issue #199).

Pure-logic tests only -- no real embedding model is loaded here (that
would require `torch`/`transformers` and real network access, and is
exercised separately as a maintainer-run diagnostic against the real
corpus, never in this repo's automated test suite; see `ml/README.md`).
All vectors below are hand-constructed NumPy arrays, never real corpus
content (ADR 0002).
"""

from __future__ import annotations

import numpy as np
import pytest

from data.alignment_signal import (
    PairSimilarityReport,
    analyze_alignment_signal,
    auc_separation,
    build_shuffled_control_indices,
    cosine_similarities,
)


def test_cosine_similarities_is_one_for_identical_vectors():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])

    sims = cosine_similarities(a, a)

    assert sims == pytest.approx([1.0, 1.0])


def test_cosine_similarities_is_zero_for_orthogonal_vectors():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])

    sims = cosine_similarities(a, b)

    assert sims == pytest.approx([0.0])


def test_cosine_similarities_handles_zero_vectors_without_dividing_by_zero():
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 0.0]])

    sims = cosine_similarities(a, b)

    assert sims == pytest.approx([0.0])


def test_auc_separation_is_one_when_perfectly_separated():
    positive = np.array([0.8, 0.9, 0.95])
    negative = np.array([0.1, 0.2, 0.3])

    auc = auc_separation(positive, negative)

    assert auc == pytest.approx(1.0)


def test_auc_separation_is_half_when_indistinguishable():
    positive = np.array([0.5, 0.5, 0.5, 0.5])
    negative = np.array([0.5, 0.5, 0.5, 0.5])

    auc = auc_separation(positive, negative)

    assert auc == pytest.approx(0.5)


def test_build_shuffled_control_indices_is_a_derangement():
    indices = build_shuffled_control_indices(10, seed=42)

    assert sorted(indices.tolist()) == list(range(10))
    assert not any(i == v for i, v in enumerate(indices))


def test_build_shuffled_control_indices_is_deterministic():
    first = build_shuffled_control_indices(20, seed=7)
    second = build_shuffled_control_indices(20, seed=7)

    assert first.tolist() == second.tolist()


def test_build_shuffled_control_indices_rejects_too_small_input():
    with pytest.raises(ValueError):
        build_shuffled_control_indices(1, seed=1)


def test_analyze_alignment_signal_reports_aggregate_stats_only():
    # Two "real" pairs that are identical (perfect similarity) and two more
    # that are near-orthogonal (low similarity) -- a designed signal whose
    # aggregate stats we know the shape of.
    es_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    cak_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    report = analyze_alignment_signal(es_embeddings, cak_embeddings, seed=1)

    assert isinstance(report, PairSimilarityReport)
    assert report.num_pairs == 4
    assert report.real_pair_mean_similarity == pytest.approx(1.0)
    assert 0.0 <= report.auc <= 1.0


def test_analyze_alignment_signal_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        analyze_alignment_signal(np.zeros((3, 2)), np.zeros((2, 2)), seed=1)
