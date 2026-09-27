"""Integration test for `training.train.apply_dropout_config` against the
real `facebook/m2m100_418M` checkpoint (ADR 0003) -- confirms the actual
submodule shape the unit tests in `tests/unit/test_dropout_override.py`
assume (duck-typed fakes) against real transformers classes, not just a
model this project made up: real `M2M100Encoder`/`M2M100Decoder`/
`M2M100EncoderLayer`/`M2M100DecoderLayer` instances really do each own a
plain-float `.dropout` attribute set from `config.dropout`, and real
`M2M100Attention` instances really do own a *different* `.dropout`
attribute set from `config.attention_dropout` that must be left alone.
"""

from __future__ import annotations

from transformers import M2M100ForConditionalGeneration
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Attention

from training.train import apply_dropout_config

MODEL_NAME = "facebook/m2m100_418M"


def test_apply_dropout_config_changes_real_general_dropout_submodules():
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
    original_attention_dropouts = [
        module.dropout for module in model.modules() if isinstance(module, M2M100Attention)
    ]
    assert original_attention_dropouts  # sanity: the real model has attention modules

    apply_dropout_config(model, 0.3)

    assert model.config.dropout == 0.3
    general_dropout_values = [
        module.dropout
        for module in model.modules()
        if not isinstance(module, M2M100Attention) and isinstance(getattr(module, "dropout", None), float)
    ]
    assert general_dropout_values  # sanity: at least one real module was touched
    assert all(value == 0.3 for value in general_dropout_values)

    # Attention dropout (a different config value, config.attention_dropout)
    # must be completely unaffected by --dropout.
    new_attention_dropouts = [
        module.dropout for module in model.modules() if isinstance(module, M2M100Attention)
    ]
    assert new_attention_dropouts == original_attention_dropouts
