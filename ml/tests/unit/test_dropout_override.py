"""Unit tests for `training.train.apply_dropout_config` (issue #182).

`facebook/m2m100_418M` is fine-tuned with whatever dropout probability its
own pretrained config happens to specify, never deliberately set for this
project's fine-tuning regime. This tests the wiring logic against mostly
duck-typed fake modules (no real model download/forward pass), except for
the attention-skip check itself: code review on PR #183 switched
`apply_dropout_config` from a fragile `type(module).__name__ ==
"M2M100Attention"` string match to a real `isinstance(module,
M2M100Attention)` check (robust to a future transformers version using a
subclass or alternate attention implementation, e.g. SDPA/FlashAttention
variants) -- which means a plain duck-typed fake can no longer stand in
for "an attention module" at all; `FakeAttentionModule` below is a real
(if minimally constructed) subclass of the actual
`transformers.models.m2m_100.modeling_m2m_100.M2M100Attention` so
`isinstance` genuinely holds, the same way a real subclass introduced by a
future transformers version would. The real checkpoint's actual submodule
shape (which classes really own a `.dropout` float attribute set from
`config.dropout` vs. `config.attention_dropout`) is additionally confirmed
end-to-end in `tests/integration/test_dropout_real_checkpoint.py`.
"""

from __future__ import annotations

from transformers.models.m2m_100.modeling_m2m_100 import M2M100Attention

from training.train import apply_dropout_config


class FakeConfig:
    def __init__(self, dropout: float, attention_dropout: float):
        self.dropout = dropout
        self.attention_dropout = attention_dropout


class FakeGeneralDropoutModule:
    """Duck-typed stand-in for a module whose `.dropout` attribute mirrors
    `config.dropout` (e.g. real M2M100Encoder/M2M100Decoder/
    M2M100EncoderLayer/M2M100DecoderLayer instances) -- a plain float
    attribute, not an `nn.Dropout` submodule (confirmed by reading the
    installed transformers version's own modeling_m2m_100.py: these call
    `nn.functional.dropout(hidden_states, p=self.dropout, ...)` directly).
    """

    def __init__(self, dropout: float):
        self.dropout = dropout


class FakeAttentionModule(M2M100Attention):
    """A real, if minimally constructed, subclass of the actual
    `M2M100Attention` -- deliberately skips calling the real `__init__`
    (which needs a full `M2M100Config`/embed_dim/head count) since all
    `apply_dropout_config` (or `isinstance`) needs is the real class in the
    MRO, not a fully-initialized attention module. Python's `isinstance`
    is based on `type`/MRO, not on which `__init__` ran, so this is a
    genuine `M2M100Attention` instance as far as production code can tell.
    """

    def __init__(self, dropout: float):
        self.dropout = dropout


class FakeModel:
    def __init__(self, dropout: float, attention_dropout: float):
        self.config = FakeConfig(dropout, attention_dropout)
        self.encoder_layer = FakeGeneralDropoutModule(dropout)
        self.decoder_layer = FakeGeneralDropoutModule(dropout)
        self.attention = FakeAttentionModule(attention_dropout)

    def modules(self):
        return [self.encoder_layer, self.decoder_layer, self.attention]


def test_apply_dropout_config_overrides_config_and_general_dropout_submodules():
    model = FakeModel(dropout=0.1, attention_dropout=0.1)

    apply_dropout_config(model, 0.3)

    assert model.config.dropout == 0.3
    assert model.encoder_layer.dropout == 0.3
    assert model.decoder_layer.dropout == 0.3


def test_apply_dropout_config_never_touches_attention_dropout():
    model = FakeModel(dropout=0.1, attention_dropout=0.1)

    apply_dropout_config(model, 0.3)

    assert model.attention.dropout == 0.1
    assert model.config.attention_dropout == 0.1


def test_apply_dropout_config_is_a_noop_when_dropout_is_none():
    model = FakeModel(dropout=0.1, attention_dropout=0.1)

    apply_dropout_config(model, None)

    assert model.config.dropout == 0.1
    assert model.encoder_layer.dropout == 0.1
