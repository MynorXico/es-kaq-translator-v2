"""Unit tests for `training.train.shift_tokens_right` /
`attach_decoder_input_ids` (issue #182).

## Root cause of the label-smoothing crash (confirmed against the real
installed `transformers` version, not assumed)

`ValueError: You cannot specify both decoder_input_ids and
decoder_inputs_embeds at the same time` only ever fires when
`--label-smoothing` is > 0. Reproducing it locally and tracing the actual
call stack (not just reading the misleading error message) showed the
real defect is the *opposite* of what that message implies: neither
`decoder_input_ids` nor `decoder_inputs_embeds` is being passed to
`M2M100Decoder.forward` at all in that case.

The chain: `DataCollatorForSeq2Seq(tokenizer, model=model)` only computes
`decoder_input_ids` itself when
`hasattr(model, "prepare_decoder_input_ids_from_labels")` -- and the
installed `transformers` version's `M2M100ForConditionalGeneration` no
longer defines that method at all (confirmed directly:
`hasattr(M2M100ForConditionalGeneration, "prepare_decoder_input_ids_from_labels")`
is `False`; grepping the installed package shows only a handful of other
model families -- bart, t5, mbart, pegasus, ... -- still define it).
Without label smoothing, this doesn't matter: `Trainer.compute_loss`
leaves `"labels"` in the batch dict it passes to `model(**inputs)`, and
`M2M100ForConditionalGeneration.forward`'s own `if labels is not None:
decoder_input_ids = shift_tokens_right(...)` branch builds
`decoder_input_ids` internally. But `label_smoothing_factor > 0` makes
`Trainer` set a `label_smoother`, and `Trainer.compute_loss` then *pops*
`"labels"` out of the batch before calling the model (so it can apply
smoothing itself against the raw logits) -- so that internal branch never
fires, `decoder_input_ids` and `decoder_inputs_embeds` are both `None`,
and `M2M100Decoder.forward`'s own `(input_ids is None) ^ (inputs_embeds is
not None)` guard (intended to catch "neither given" and "both given"
alike) trips on the "neither given" case, which the exception's fixed
wording just doesn't distinguish from "both given".

The fix: compute `decoder_input_ids` ourselves, unconditionally, in the
data collator -- *before* `Trainer.compute_loss` ever gets a chance to pop
`"labels"` -- rather than depending on a model-specific convenience hook
that this transformers version no longer provides for M2M100.
`shift_tokens_right` below is a pure reimplementation of the exact
transform `M2M100ForConditionalGeneration.forward`'s own (dead, for our
purposes) `labels is not None` branch performs, so precomputing it changes
nothing about the actual decoder inputs the model would have used anyway
-- confirmed with a real training step in
`tests/integration/test_fine_tune_real_checkpoint.py`.
"""

from __future__ import annotations

import torch
import transformers.models.m2m_100.modeling_m2m_100 as real_m2m_100_module

from training.train import attach_decoder_input_ids, shift_tokens_right

PAD_TOKEN_ID = 1
DECODER_START_TOKEN_ID = 2  # matches facebook/m2m100_418M's own eos_token_id


def test_shift_tokens_right_delegates_to_the_real_transformers_implementation(monkeypatch):
    """Code review on PR #183: `shift_tokens_right` must delegate to the
    real `transformers.models.m2m_100.modeling_m2m_100.shift_tokens_right`
    (the same function `M2M100ForConditionalGeneration.forward` already
    calls internally -- see this function's own docstring) rather than
    maintain an independent, from-scratch reimplementation that could
    silently drift from upstream if a future transformers version changes
    its semantics. Monkeypatches the real function with a spy and asserts
    our wrapper's return value *is* the spy's return value (identity, not
    just an equal-looking tensor computed separately), proving delegation
    actually happened.
    """
    sentinel = object()
    calls = []

    def fake_shift_tokens_right(labels, pad_token_id, decoder_start_token_id):
        calls.append((labels, pad_token_id, decoder_start_token_id))
        return sentinel

    monkeypatch.setattr(real_m2m_100_module, "shift_tokens_right", fake_shift_tokens_right)

    labels = torch.tensor([[10, 11, 12]])
    result = shift_tokens_right(labels, PAD_TOKEN_ID, DECODER_START_TOKEN_ID)

    assert result is sentinel
    assert calls == [(labels, PAD_TOKEN_ID, DECODER_START_TOKEN_ID)]


def test_shift_tokens_right_prepends_start_token_and_drops_last_column():
    labels = torch.tensor([[10, 11, 12], [20, 21, 22]])

    shifted = shift_tokens_right(labels, PAD_TOKEN_ID, DECODER_START_TOKEN_ID)

    assert shifted.tolist() == [[2, 10, 11], [2, 20, 21]]


def test_shift_tokens_right_replaces_minus_100_padding_with_pad_token_id():
    # -100 (the loss-ignore index DataCollatorForSeq2Seq pads labels with)
    # must never reach an embedding lookup as a literal id -- it has to be
    # replaced with the real pad token id after the shift places it inside
    # the decoder_input_ids row (matches the installed transformers
    # version's own m2m_100 shift_tokens_right exactly).
    labels = torch.tensor([[10, -100, 12]])

    shifted = shift_tokens_right(labels, PAD_TOKEN_ID, DECODER_START_TOKEN_ID)

    assert shifted.tolist() == [[2, 10, 1]]


def test_attach_decoder_input_ids_adds_shifted_field_computed_from_labels():
    batch = {
        "input_ids": torch.tensor([[5, 6]]),
        "labels": torch.tensor([[10, 11, -100]]),
    }

    result = attach_decoder_input_ids(batch, PAD_TOKEN_ID, DECODER_START_TOKEN_ID)

    assert result is batch  # mutates in place, returned for convenience
    assert result["decoder_input_ids"].tolist() == [[2, 10, 11]]
    # Untouched keys stay untouched.
    assert result["input_ids"].tolist() == [[5, 6]]
    assert result["labels"].tolist() == [[10, 11, -100]]


def test_attach_decoder_input_ids_is_a_noop_when_labels_absent():
    batch = {"input_ids": torch.tensor([[5, 6]])}

    result = attach_decoder_input_ids(batch, PAD_TOKEN_ID, DECODER_START_TOKEN_ID)

    assert "decoder_input_ids" not in result


def test_attach_decoder_input_ids_is_a_noop_when_labels_is_none():
    # DataCollatorForSeq2Seq always sets batch["labels"] = None (rather than
    # omitting the key) when no example in the batch had labels -- guard
    # against that shape too, not just a missing key.
    batch = {"input_ids": torch.tensor([[5, 6]]), "labels": None}

    result = attach_decoder_input_ids(batch, PAD_TOKEN_ID, DECODER_START_TOKEN_ID)

    assert "decoder_input_ids" not in result
