"""Unit tests for `evaluation.length_bias.build_length_bias_logits_processor`
-- the real `torch`/`transformers`-native implementation of Eq. 4's
debiasing operation used at actual decode time (issue #193).

Uses real (small, CPU-only) `torch` tensors directly, no model download --
the same convention already established by e.g.
`tests/unit/test_checkpoint_averaging.py`/`tests/unit/test_decoder_input_ids.py`
for testing tensor-consuming pure functions without a real checkpoint.

The core claim under test: this `torch`-native implementation must be
numerically identical to `evaluation.length_bias.rectify_probabilities`'s
independent NumPy reference implementation of the same formula (Eq. 4) --
see `evaluation/length_bias.py`'s module docstring for why two
implementations exist at all, and why they must never be allowed to
silently drift apart.
"""

from __future__ import annotations

import numpy as np
import torch

from evaluation.length_bias import build_length_bias_logits_processor, rectify_probabilities


def _apply_processor(delta: float, logits: torch.Tensor) -> torch.Tensor:
    processor_list = build_length_bias_logits_processor(delta)
    assert len(processor_list) == 1
    # input_ids is unused by this processor (Eq. 4 only depends on the
    # current step's distribution) -- an empty placeholder is fine.
    input_ids = torch.zeros((logits.shape[0], 0), dtype=torch.long)
    result_logits = logits
    for processor in processor_list:
        result_logits = processor(input_ids, result_logits)
    return result_logits


def test_delta_zero_is_a_no_op_up_to_log_softmax_renormalization():
    logits = torch.tensor([[2.0, 1.0, 0.1, -1.0]])

    result = _apply_processor(0.0, logits)

    # delta=0 rectification is a true no-op on the probability distribution
    # (see test_length_bias.py), so log_softmax of the processor's output
    # must exactly match log_softmax of the original, untouched logits.
    torch.testing.assert_close(
        torch.log_softmax(result, dim=-1), torch.log_softmax(logits, dim=-1)
    )


def test_matches_the_numpy_reference_implementation_for_a_real_delta():
    logits = torch.tensor([[3.0, 1.0, 0.5, -2.0, -3.0]])
    delta = 0.05

    result = _apply_processor(delta, logits)

    expected_probs = rectify_probabilities(
        torch.softmax(logits, dim=-1).numpy(), delta=delta
    )
    # The processor returns log-probabilities (see module docstring);
    # softmax-ing them back must recover the same rectified distribution
    # the pure NumPy reference implementation computes directly.
    actual_probs = torch.softmax(result, dim=-1).numpy()
    np.testing.assert_allclose(actual_probs, expected_probs, atol=1e-6)


def test_matches_the_numpy_reference_implementation_across_a_batch_of_rows():
    logits = torch.tensor(
        [
            [3.0, 1.0, 0.5, -2.0, -3.0],
            [0.1, 0.1, 0.1, 5.0, -1.0],
        ]
    )
    delta = 0.02

    result = _apply_processor(delta, logits)

    expected_probs = rectify_probabilities(
        torch.softmax(logits, dim=-1).numpy(), delta=delta
    )
    actual_probs = torch.softmax(result, dim=-1).numpy()
    np.testing.assert_allclose(actual_probs, expected_probs, atol=1e-6)


def test_excluded_tokens_score_negative_infinity_not_a_tiny_finite_value():
    # A large enough delta should completely zero out low-probability
    # tokens (ReLU floor) -- those tokens must score -inf (mathematically
    # impossible), not a tiny-but-finite log-probability that could still
    # let beam search pick them under a sufficiently long sequence.
    logits = torch.tensor([[5.0, -5.0, -5.0]])
    delta = 0.4  # softmax([5,-5,-5]) ~= [0.9999, 0.00005, 0.00005] -- both small entries excluded

    result = _apply_processor(delta, logits)

    assert result[0, 1].item() == float("-inf")
    assert result[0, 2].item() == float("-inf")
    assert result[0, 0].item() > float("-inf")
