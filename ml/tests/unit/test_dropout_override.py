"""Unit tests for `training.train.apply_dropout_config` (issue #182).

`facebook/m2m100_418M` is fine-tuned with whatever dropout probability its
own pretrained config happens to specify, never deliberately set for this
project's fine-tuning regime. This tests the pure wiring logic against
duck-typed fake modules (no real model download); the real checkpoint's
actual submodule shape (which classes really own a `.dropout` float
attribute set from `config.dropout` vs. `config.attention_dropout`) is
additionally confirmed in
`tests/integration/test_tokenizer_extension_real_model.py`.
"""

from __future__ import annotations

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


def _make_fake_attention_class(dropout: float):
    """`apply_dropout_config` identifies attention modules to skip by
    `type(module).__name__ == "M2M100Attention"` (see its own docstring) --
    build a fake class with that exact name so the skip-list check matches
    production code exactly, without needing the real transformers import.
    """

    def __init__(self, dropout: float):
        self.dropout = dropout

    return type("M2M100Attention", (), {"__init__": __init__})(dropout)


class FakeModel:
    def __init__(self, dropout: float, attention_dropout: float):
        self.config = FakeConfig(dropout, attention_dropout)
        self.encoder_layer = FakeGeneralDropoutModule(dropout)
        self.decoder_layer = FakeGeneralDropoutModule(dropout)
        self.attention = _make_fake_attention_class(attention_dropout)

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
