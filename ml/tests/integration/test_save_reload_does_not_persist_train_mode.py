"""Integration test confirming a claim `training.train.run_training_job`'s
own `model.eval()` fix (issue #175) relies on: train/eval mode is never
persisted through `save_pretrained`/`from_pretrained` -- `from_pretrained`
always resets a freshly loaded model to eval mode itself, regardless of
what mode the object being saved was in.

This matters for `run_training_job`'s call ordering: it calls `model.eval()`
once, before *both* `save_model_and_tokenizer` and the post-training
`generate_translations` call, rather than needing a second call in between.
That's only correct if the saved-then-reloaded checkpoint (e.g. what
`evaluation.evaluate_checkpoint.load_checkpoint_tokenizer_and_model` or
`deployment.inference.model_fn` load later) doesn't inherit whatever mode
`run_training_job`'s own in-memory model happened to be in at save time.
Confirmed directly here rather than assumed from documentation, matching
this project's existing "verify real API behavior, don't assume" convention
(see e.g. `training.train`'s `--label-smoothing` docstring and
`tests/integration/test_fine_tune_real_checkpoint.py`).

Uses the real `facebook/m2m100_418M` checkpoint (ADR 0003), already cached
by `tests/integration/test_tokenizer_extension_real_model.py` -- see that
module's docstring and `.github/workflows/ci.yml` for the Hugging Face Hub
cache this shares.
"""

from __future__ import annotations

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from training.train import save_model_and_tokenizer

MODEL_NAME = "facebook/m2m100_418M"


def test_save_and_reload_does_not_persist_train_mode(tmp_path):
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Deliberately save while the model is in *train* mode -- exactly the
    # state `run_training_job` would be in right after `trainer.train()`,
    # before its own `model.eval()` fix (issue #175).
    model.train()
    assert model.training is True

    save_model_and_tokenizer(model, tokenizer, str(tmp_path))

    reloaded = M2M100ForConditionalGeneration.from_pretrained(str(tmp_path))

    # `from_pretrained` unconditionally calls `model.eval()` itself at the
    # end of loading (confirmed against the installed transformers
    # version's `PreTrainedModel.from_pretrained` source) -- so the
    # reloaded checkpoint ends up in eval mode regardless of the mode the
    # saved model was in. Train/eval mode is simply never written to disk
    # by `save_pretrained` in the first place.
    assert reloaded.training is False
