"""Unit tests for `evaluation.evaluate_checkpoint`'s GPU device placement
(issue #170).

`training.train.generate_translations` moves every encoded batch to
`model.device` before calling `model.generate` (see
`tests/unit/test_generate_translations.py`), but until this fix the model
loaded by `load_checkpoint_tokenizer_and_model` was never actually moved
onto a GPU device even when one was available --
`M2M100ForConditionalGeneration.from_pretrained` places a model on CPU by
default, so `model.device` stayed `"cpu"` and the entire generation pass
ran there regardless of the environment's real hardware. A real eval run
(7,218 validation examples, beam=5) took ~12 hours on a CPU-only
environment because of this.

This mirrors `training.train.build_training_arguments`'s existing
`fp16=torch.cuda.is_available()` pattern (see
`tests/unit/test_build_training_arguments.py`), but for device placement
rather than mixed-precision.

The device-selection logic is pulled into its own small, pure-ish helper
(`_move_model_to_cuda_if_available`), mirroring this module's existing
convention of testing small underscore-prefixed helpers directly against
duck-typed fakes (see `_build_train_sample_texts`,
`_diagnose_word_boundary_reconstruction`) rather than the thin
`from_pretrained(...)`-calling wrapper functions themselves
(`load_checkpoint_tokenizer_and_model`, `load_base_tokenizer_vocab`),
which -- like `training.train.load_base_model_and_tokenizer` -- are never
directly unit tested: `transformers`' top-level package is a lazy-loading
module (`transformers.utils.import_utils._LazyModule`), and monkeypatching
its `M2M100Tokenizer`/`M2M100ForConditionalGeneration` attributes does not
reliably intercept `load_checkpoint_tokenizer_and_model`'s own
`from transformers import ...` statement (confirmed directly: the patched
class is visible via `transformers.M2M100Tokenizer` attribute access
immediately after patching, yet a subsequent `from transformers import
M2M100Tokenizer` in the same scope still resolves to the real class) --
so a fake real load would require actually reaching out to Hugging Face
Hub for a nonexistent repo. This helper isolates exactly the one line of
new logic this issue is about, without any of that.
"""

from __future__ import annotations

import torch

from evaluation.evaluate_checkpoint import _move_model_to_cuda_if_available


class FakeModelWithTo:
    """Duck-typed fake standing in for a real `M2M100ForConditionalGeneration`
    checkpoint: only `.to(device)`, recording every call so tests can assert
    on it without a real forward pass, real weights, or actual GPU hardware.
    """

    def __init__(self) -> None:
        self.to_calls: list[str] = []

    def to(self, device: str) -> FakeModelWithTo:
        self.to_calls.append(device)
        return self


def test_move_model_to_cuda_if_available_moves_the_model_when_cuda_is_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = FakeModelWithTo()

    result = _move_model_to_cuda_if_available(model)

    assert model.to_calls == ["cuda"]
    assert result is model


def test_move_model_to_cuda_if_available_leaves_the_model_untouched_without_cuda(monkeypatch):
    """CI/CPU-only environments (this repo's own CI runners included) must
    see no behavior change at all: `.to()` must never be called when no
    CUDA device is available -- matches this issue's explicit "no
    behavior/performance change on CPU-only environments" acceptance
    criterion.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = FakeModelWithTo()

    result = _move_model_to_cuda_if_available(model)

    assert model.to_calls == []
    assert result is model
