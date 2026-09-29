"""Decode-time length-bias rectification for label-smoothed checkpoints
(issue #193).

## Background

Issue #182's cheap-tier label-smoothing experiment (Model Package v9, 13
epochs, `--label-smoothing 0.1`, otherwise identical to the v7 baseline)
scored BLEU 13.2 / chrF 36.4 against v7's baseline BLEU 14.0 / chrF 36.5 --
rejected in Model Registry alongside the other two cheap-tier levers (see
`ml/README.md`, "Cheap-tier quality-experiment prep"). That comparison used
this project's ordinary decode procedure (beam search, `num_beams=5`,
`training.train.generate_translations`), which never accounts for a real,
documented interaction between label smoothing and beam search: Liang, Wang
& Cao, "The Implicit Length Bias of Label Smoothing on Beam Search Decoding"
(arXiv:2205.00659), show that label smoothing implicitly biases beam search
toward *shorter* outputs, and that a decode-time rectification recovers real
BLEU that a raw comparison masks -- up to +2.8 BLEU at beam size 200 in
their experiments, smaller but nonzero at their beam size 4 (closest to this
project's own beam=5).

## The paper's actual method (not a generic length-penalty heuristic)

This module implements exactly the paper's own proposed correction, not an
independently-invented length-penalty term:

- Eq. 2: for a model trained with label smoothing factor `alpha` over a
  vocabulary of size `V`, the (approximately) optimal per-token prediction
  the model learns to output is
  `p_hat = (1 - alpha) * q + alpha / V`,
  where `q` is the *true* target distribution label smoothing was trying to
  approximate ("ground truth" in the paper's terms).
- Sec. 3.1: because beam search scores a candidate sequence by summing
  `log(p_hat)` over its tokens rather than the true `log(q)`, it implicitly
  applies a `log(1 - alpha)` penalty to *every* generated token -- a bias
  toward shorter sequences that grows with sequence length.
- Eq. 3: rearranging Eq. 2 gives the *exact* inverse transform,
  `p_db = (p_hat - alpha / V) / (1 - alpha)`, recovering the unbiased `q`
  from the model's actual (label-smoothing-biased) output distribution.
- Eq. 4: since a real, imperfectly-optimized model can assign some tokens a
  probability below `alpha / V` (which Eq. 3 would turn negative), the
  paper generalizes Eq. 3 to a tunable debiasing parameter `delta` with a
  ReLU floor and renormalization:

      p_db_i = ReLU(p_hat_i - delta) / sum_j(ReLU(p_hat_j - delta))

  `delta = alpha / V` is the theoretically exact value (Eq. 3, restated as
  a special case of Eq. 4); the paper's own experiments (Table 1) found
  that a *larger* `delta = 1/V` gives near-peak BLEU across every language
  pair they tested at beam size 4 (their smallest beam, closest to this
  project's beam=5) -- their explicit finding (Sec. 1, item 3) is that
  "noise introduced by LS alone does not fully explain the length bias
  baked into the model", i.e. stronger-than-theoretically-justified
  debiasing is empirically beneficial. This module therefore does not
  hardcode one specific `delta`; `evaluation.evaluate_checkpoint`'s
  `--debias-delta-multiplier` flag lets a caller try either value (or any
  other multiple of `1 / vocab_size`) -- see that module for the real,
  at-scale re-evaluation this was built for (issue #193).

## Two implementations, cross-checked against each other

`rectify_probabilities` below is a pure NumPy implementation of Eq. 4,
operating on plain arrays with no `torch`/`transformers` dependency --
fully unit-testable (`tests/unit/test_length_bias.py`) without a real
checkpoint, mirroring `training/vocab_extension.py`'s "pure logic, wired to
the real API separately" split.

`build_length_bias_logits_processor` builds the actual
`transformers.LogitsProcessor` used at real decode time -- it has to operate
on live `torch.Tensor`s already resident on whatever device `model.generate`
is running on (CPU or GPU), so it reimplements the same Eq. 4 arithmetic
directly in `torch` rather than round-tripping every decode step's
`(batch_size * num_beams, vocab_size)` tensor through NumPy (a real,
per-step CPU<->GPU transfer cost this project can't afford at real
validation-set scale -- see `ml/README.md`'s own GPU-thermal-throttling
notes for how expensive this project's real evaluation runs already are).
This is therefore a second, independent implementation of the same
formula -- `tests/unit/test_length_bias_logits_processor.py` asserts its
output is numerically identical (after `log_softmax`, since a
`LogitsProcessor`'s contract is to return updated *logits*, not
probabilities) to `rectify_probabilities`'s reference implementation for the
same inputs, so the two can't silently drift apart the way
`training.train.shift_tokens_right`'s own docstring warns against for a
similar "two implementations of one formula" risk.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def rectify_probabilities(probs: np.ndarray, *, delta: float) -> np.ndarray:
    """Apply Eq. 4's debiasing operation to a probability distribution (or a
    batch of them, rectified independently row-wise along the last axis):

        p_db_i = ReLU(p_i - delta) / sum_j(ReLU(p_j - delta))

    `delta=0.0` is an exact no-op (`ReLU(p - 0) == p` for any valid
    probability distribution, which already sums to 1) -- this project's
    existing, pre-#193 decode behavior when rectification isn't opted into.

    Falls back to the original (renormalized) distribution, rather than
    dividing by zero, for any row where `delta` meets or exceeds every
    entry -- an edge case only reachable with a `delta` far larger than
    this project's own real, small experiments use (see module docstring),
    but one the paper's own Eq. 4 doesn't explicitly guard against, so this
    explicitly does, favoring the safest available fallback over a NaN
    silently corrupting a decode step's sequence scores.
    """
    probs = np.asarray(probs, dtype=np.float64)
    rectified = np.clip(probs - delta, a_min=0.0, a_max=None)
    row_sums = rectified.sum(axis=-1, keepdims=True)
    zero_rows = row_sums <= 0.0
    safe_row_sums = np.where(zero_rows, 1.0, row_sums)
    normalized = rectified / safe_row_sums
    fallback = probs / probs.sum(axis=-1, keepdims=True)
    return np.where(zero_rows, fallback, normalized)


def build_length_bias_logits_processor(delta: float) -> Any:
    """Build a real `transformers.LogitsProcessorList` (ready to pass as
    `model.generate(..., logits_processor=...)`) applying Eq. 4's debiasing
    operation to the model's own predicted next-token distribution at every
    decode step -- see this module's docstring for the full derivation and
    why this is a second, `torch`-native implementation of the same formula
    `rectify_probabilities` implements in NumPy.

    Lazily imports `torch`/`transformers`, the same laziness convention
    `training.train`/`training.tokenizer_extension` use for every
    torch/transformers-dependent function in this project, so importing
    this module itself never requires either.

    A `LogitsProcessor`'s contract (per `transformers`) is to accept and
    return raw *logits* (pre-softmax), for a `(batch_size * num_beams,
    vocab_size)` tensor of whatever decode candidates are still active at
    that step -- not the post-softmax probabilities Eq. 4 is written in
    terms of. This processor converts to probabilities via `softmax`,
    applies Eq. 4, then converts back via `log`: `transformers`' own
    `log_softmax` normalization (applied by beam search *after* every
    registered `LogitsProcessor` runs, including this one) is invariant
    under this round-trip, since the rectified probabilities already sum to
    1 per row (`log_softmax(log(p))  ==  log(p)` whenever `p` sums to 1) --
    so this needs no separate "undo the earlier softmax" bookkeeping.
    """
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    class LengthBiasRectifyingLogitsProcessor(LogitsProcessor):
        """Applies Eq. 4's debiasing operation with a fixed `delta`,
        computed once by the caller as `debias_delta_multiplier /
        vocab_size` (see `evaluation.evaluate_checkpoint`).
        """

        def __init__(self, delta: float):
            self.delta = delta

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            probs = torch.softmax(scores, dim=-1)
            rectified = torch.clamp(probs - self.delta, min=0.0)
            row_sums = rectified.sum(dim=-1, keepdim=True)
            zero_rows = row_sums <= 0.0
            safe_row_sums = torch.where(zero_rows, torch.ones_like(row_sums), row_sums)
            normalized = rectified / safe_row_sums
            fallback = probs / probs.sum(dim=-1, keepdim=True)
            result = torch.where(zero_rows, fallback, normalized)
            # log(0) correctly maps to -inf (a token ReLU excluded must
            # score -inf, i.e. impossible) -- only clamp the strictly
            # positive side against underflow, never floor the zeros
            # themselves, or an excluded token would wrongly regain a
            # tiny-but-finite score.
            tiny = torch.finfo(result.dtype).tiny
            return torch.log(torch.where(result > 0, result.clamp(min=tiny), result))

    return LogitsProcessorList([LengthBiasRectifyingLogitsProcessor(delta)])
