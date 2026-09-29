"""Unit tests for `evaluation.length_bias.rectify_probabilities`: the pure,
framework-free (NumPy-only) implementation of the decode-time debiasing
operation from Liang, Wang & Cao, "The Implicit Length Bias of Label
Smoothing on Beam Search Decoding" (arXiv:2205.00659), Eq. 4 -- issue #193.

These operate on plain NumPy arrays only -- no real tokenizer/model/torch is
involved, so this is fully testable without a real checkpoint, mirroring
`training/vocab_extension.py`'s "pure NumPy logic, separately wired to the
real Hugging Face API" split (see `evaluation/length_bias.py`'s module
docstring and `tests/unit/test_length_bias_logits_processor.py` for the
torch-based wiring layer this pure logic is cross-checked against).
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.length_bias import rectify_probabilities


def test_rectify_probabilities_is_a_no_op_when_delta_is_zero():
    # delta=0 is this project's existing (pre-#193) behavior: ReLU(p - 0)
    # == p for any valid probability distribution, and p already sums to
    # 1, so rectification must be a true no-op -- never a different
    # numerical path than "no debiasing at all" (issue #193's own decode
    # config must stay identical when a caller doesn't opt in).
    probs = np.array([0.7, 0.2, 0.1])

    rectified = rectify_probabilities(probs, delta=0.0)

    np.testing.assert_allclose(rectified, probs)


def test_rectify_probabilities_exactly_inverts_label_smoothings_interpolation():
    # Eq. 2 of the paper: the label-smoothed model's own (approximately)
    # optimal prediction is p_hat = (1 - alpha) * q + alpha / V, for the
    # *true* (hard, one-hot here for simplicity) distribution q. Eq. 3 is
    # the exact algebraic inverse: p_db = (p_hat - alpha/V) / (1 - alpha).
    # This is the theoretically exact case of Eq. 4 (delta == alpha/V) --
    # rectifying a perfectly-smoothed p_hat must recover the original q
    # exactly, not just approximately.
    alpha = 0.1
    vocab_size = 5
    q = np.array([0.0, 0.0, 1.0, 0.0, 0.0])  # one-hot ground truth
    p_hat = (1 - alpha) * q + alpha / vocab_size
    delta = alpha / vocab_size

    rectified = rectify_probabilities(p_hat, delta=delta)

    np.testing.assert_allclose(rectified, q, atol=1e-10)


def test_rectify_probabilities_clips_negative_values_and_renormalizes():
    # A real (imperfectly optimized) model can assign some tokens a
    # probability below delta -- Eq. 4's ReLU must floor these at 0 rather
    # than let them go negative, then renormalize the remaining mass to
    # sum to 1 (not left at whatever partial sum ReLU leaves behind).
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    delta = 0.1

    rectified = rectify_probabilities(probs, delta=delta)

    # ReLU(probs - 0.1) = [0.4, 0.2, 0.05, 0.0], sum = 0.65
    expected = np.array([0.4, 0.2, 0.05, 0.0]) / 0.65
    np.testing.assert_allclose(rectified, expected)
    assert rectified.sum() == pytest.approx(1.0)
    assert (rectified >= 0).all()


def test_rectify_probabilities_falls_back_to_the_original_distribution_when_delta_is_too_large():
    # If delta exceeds every token's probability, Eq. 4's denominator is
    # zero -- naively dividing would produce NaNs. This must never happen:
    # falling back to the (renormalized) original distribution is the
    # safest available behavior, and is explicitly documented, not a
    # silent NaN propagating into beam search's sequence scores.
    probs = np.array([0.4, 0.35, 0.25])
    delta = 0.9  # larger than every entry

    rectified = rectify_probabilities(probs, delta=delta)

    assert not np.isnan(rectified).any()
    np.testing.assert_allclose(rectified, probs)


def test_rectify_probabilities_operates_row_wise_on_a_batch():
    # model.generate's real usage is a (batch_size * num_beams, vocab_size)
    # 2D tensor -- each row (one decode candidate) must be rectified and
    # renormalized independently of every other row.
    probs = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.34, 0.33, 0.33],
        ]
    )
    delta = 0.15

    rectified = rectify_probabilities(probs, delta=delta)

    assert rectified.shape == probs.shape
    np.testing.assert_allclose(rectified.sum(axis=-1), [1.0, 1.0])
    # Row 0: ReLU([0.55, 0.05, -0.05]) = [0.55, 0.05, 0.0], sum=0.6
    np.testing.assert_allclose(rectified[0], np.array([0.55, 0.05, 0.0]) / 0.6)
