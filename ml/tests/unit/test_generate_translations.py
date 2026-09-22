"""Unit test for `training.train.generate_translations`'s device placement.

This specifically guards against a real bug found on the 3rd real training
run (issue #66): the encoded batch tensors were never moved to
`model.device` before `model.generate(...)`, which crashes with
`RuntimeError: ... but got index is on cpu, different from other tensors
on cuda:0` whenever `model` lives on a GPU -- which is every real training
run, since this only ever runs on `ml.g4dn.xlarge` (per
`training/submit_job.py`).

This is a GPU-only failure mode: on the CPU-only CI runners this repo's
test suite runs on, `model.device` is always `"cpu"` too, so a real
`transformers` model would never reproduce the crash here regardless of
whether the fix is present. Instead of a real (GPU-only-reproducible)
integration test, this directly asserts the specific call this bug was
missing -- `.to(model.device)` on the encoded batch -- via a spy, so a
regression is caught even without real GPU hardware.
"""

from __future__ import annotations

from training.direction import TranslationExample
from training.train import generate_translations

SENTINEL_DEVICE = "sentinel-cuda:0"


class SpyEncoding(dict):
    """Stands in for a real `transformers.BatchEncoding` (what
    `tokenizer(..., return_tensors="pt")` actually returns): a dict with a
    `.to(device)` method. Records what device it was moved to.
    """

    def __init__(self, data: dict):
        super().__init__(data)
        self.moved_to: list[str] = []

    def to(self, device: str) -> SpyEncoding:
        self.moved_to.append(device)
        return self


class FakeTokenizerForGeneration:
    def __init__(self, decoded_texts: list[str] | None = None):
        self.last_encoding: SpyEncoding | None = None
        self._decoded_texts = decoded_texts

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"__es__": 100, "__cak__": 101}[token]

    def __call__(self, texts, **kwargs) -> SpyEncoding:
        encoding = SpyEncoding({"input_ids": [[1, 2, 3]] * len(texts)})
        self.last_encoding = encoding
        return encoding

    def batch_decode(self, generated_ids, **kwargs) -> list[str]:
        if self._decoded_texts is not None:
            return self._decoded_texts
        return [f"decoded-{i}" for i in range(len(generated_ids))]


class FakeModelWithDevice:
    def __init__(self, device: str):
        self.device = device
        self.generate_called_with: dict | None = None

    def generate(self, **kwargs) -> list[int]:
        self.generate_called_with = kwargs
        return [0]  # one "generated" id per call, matches batch_decode's fake


def test_generate_translations_moves_encoded_batch_to_model_device():
    tokenizer = FakeTokenizerForGeneration()
    model = FakeModelWithDevice(device=SENTINEL_DEVICE)
    examples = [
        TranslationExample(
            source_text="hola", target_text="la", source_lang="es", target_lang="cak"
        )
    ]

    generate_translations(model, tokenizer, examples)

    assert tokenizer.last_encoding is not None
    assert tokenizer.last_encoding.moved_to == [SENTINEL_DEVICE]


def test_generate_translations_passes_the_moved_encoding_to_generate():
    tokenizer = FakeTokenizerForGeneration()
    model = FakeModelWithDevice(device=SENTINEL_DEVICE)
    examples = [
        TranslationExample(
            source_text="hola", target_text="la", source_lang="es", target_lang="cak"
        )
    ]

    generate_translations(model, tokenizer, examples)

    assert model.generate_called_with is not None
    assert model.generate_called_with["forced_bos_token_id"] == 101


# ---------------------------------------------------------------------------
# Direction-tag leak (issue #106): __cak__ is an ordinary added-vocab token
# (not a real special token, unlike __es__ -- one of M2M100's own pretrained
# language codes), so tokenizer.batch_decode(..., skip_special_tokens=True)
# does not strip it, and it survived verbatim as the literal first word of
# every es->cak hypothesis before this fix -- corrupting every real
# training run's BLEU/chrF for that direction (see
# training.direction.strip_leading_direction_tag's docstring, and
# deployment/inference.py's test_inference.py, which caught the same bug
# for the real-time serving path in issue #8).
# ---------------------------------------------------------------------------


def test_generate_translations_strips_a_literal_leading_cak_tag_from_decoded_output():
    tokenizer = FakeTokenizerForGeneration(decoded_texts=["__cak__ Utz awäch?"])
    model = FakeModelWithDevice(device=SENTINEL_DEVICE)
    examples = [
        TranslationExample(
            source_text="Buenos días",
            target_text="Utz awäch?",
            source_lang="es",
            target_lang="cak",
        )
    ]

    hypotheses = generate_translations(model, tokenizer, examples)

    assert hypotheses == ["Utz awäch?"]


def test_generate_translations_leaves_es_target_output_without_a_leading_tag_untouched():
    # __es__-target output already has the tag stripped by
    # skip_special_tokens=True (it's a pretrained special token), so this is
    # a control case: there's nothing left for the defensive strip to do.
    tokenizer = FakeTokenizerForGeneration(decoded_texts=["Buenos días"])
    model = FakeModelWithDevice(device=SENTINEL_DEVICE)
    examples = [
        TranslationExample(
            source_text="Utz awäch?",
            target_text="Buenos días",
            source_lang="cak",
            target_lang="es",
        )
    ]

    hypotheses = generate_translations(model, tokenizer, examples)

    assert hypotheses == ["Buenos días"]
