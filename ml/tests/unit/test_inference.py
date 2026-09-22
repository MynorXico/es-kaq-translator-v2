"""Unit tests for `deployment.inference` -- the custom SageMaker real-time/
serverless inference handler for issue #8.

Everything here is pure logic or duck-typed fakes: request parsing/
validation, the request/response JSON contract, and the direction-tag
generation logic (mirroring `training.train.generate_translations`, see
that module's own test for the same duck-typing approach). `model_fn`
(loading real weights) is deliberately not covered here -- it needs a real
checkpoint and is exercised only by the maintainer's real deployment
verification (see `ml/README.md`'s "Serving (`deployment/`)" section).
"""

from __future__ import annotations

import json

import pytest

from deployment.inference import (
    CONTENT_TYPE_JSON,
    InferenceRequest,
    input_fn,
    output_fn,
    parse_request,
    predict_fn,
    translate,
)

# ---------------------------------------------------------------------------
# parse_request
# ---------------------------------------------------------------------------


def test_parse_request_accepts_a_valid_es_to_cak_payload():
    request = parse_request({"source_lang": "es", "target_lang": "cak", "text": "Buenos días"})

    assert request == InferenceRequest(source_lang="es", target_lang="cak", text="Buenos días")


def test_parse_request_accepts_a_valid_cak_to_es_payload():
    request = parse_request({"source_lang": "cak", "target_lang": "es", "text": "La utz awäch?"})

    assert request.source_lang == "cak"
    assert request.target_lang == "es"


@pytest.mark.parametrize("missing_key", ["source_lang", "target_lang", "text"])
def test_parse_request_raises_on_missing_required_field(missing_key):
    payload = {"source_lang": "es", "target_lang": "cak", "text": "hola"}
    del payload[missing_key]

    with pytest.raises(ValueError, match=missing_key):
        parse_request(payload)


def test_parse_request_raises_on_unsupported_language_code():
    with pytest.raises(ValueError, match="source_lang"):
        parse_request({"source_lang": "en", "target_lang": "cak", "text": "hello"})


def test_parse_request_raises_when_source_and_target_are_the_same():
    with pytest.raises(ValueError, match="same"):
        parse_request({"source_lang": "es", "target_lang": "es", "text": "hola"})


def test_parse_request_raises_on_empty_text():
    with pytest.raises(ValueError, match="text"):
        parse_request({"source_lang": "es", "target_lang": "cak", "text": "   "})


def test_parse_request_raises_when_text_is_not_a_string():
    with pytest.raises(ValueError, match="text"):
        parse_request({"source_lang": "es", "target_lang": "cak", "text": 123})


def test_parse_request_strips_surrounding_whitespace_from_text():
    request = parse_request({"source_lang": "es", "target_lang": "cak", "text": "  hola  "})

    assert request.text == "hola"


# ---------------------------------------------------------------------------
# input_fn / output_fn -- the SageMaker content-type contract
# ---------------------------------------------------------------------------


def test_input_fn_parses_a_json_body_with_the_expected_content_type():
    body = json.dumps({"source_lang": "es", "target_lang": "cak", "text": "hola"})

    request = input_fn(body, CONTENT_TYPE_JSON)

    assert request == InferenceRequest(source_lang="es", target_lang="cak", text="hola")


def test_input_fn_rejects_an_unsupported_content_type():
    body = json.dumps({"source_lang": "es", "target_lang": "cak", "text": "hola"})

    with pytest.raises(ValueError, match="content type"):
        input_fn(body, "text/csv")


def test_input_fn_raises_a_clear_error_on_malformed_json():
    with pytest.raises(ValueError, match="JSON"):
        input_fn("not json", CONTENT_TYPE_JSON)


def test_output_fn_serializes_translated_text_as_json():
    # output_fn returns just the serialized body (a str), not a
    # (body, content_type) tuple -- see output_fn's own docstring for the
    # real-endpoint bug this guards against.
    body = output_fn({"translated_text": "La utz awäch?"}, CONTENT_TYPE_JSON)

    assert isinstance(body, str)
    assert json.loads(body) == {"translated_text": "La utz awäch?"}


def test_output_fn_rejects_an_unsupported_accept_type():
    with pytest.raises(ValueError, match="accept"):
        output_fn({"translated_text": "x"}, "text/csv")


# ---------------------------------------------------------------------------
# translate -- the direction-tag generation logic, against duck-typed fakes
# (same approach as tests/unit/test_generate_translations.py)
# ---------------------------------------------------------------------------


class SpyEncoding(dict):
    def __init__(self, data: dict):
        super().__init__(data)
        self.moved_to: list[str] = []

    def to(self, device: str) -> SpyEncoding:
        self.moved_to.append(device)
        return self


class FakeTokenizer:
    def __init__(self, decoded_texts: list[str] | None = None):
        self.last_texts: list[str] | None = None
        self.last_encoding: SpyEncoding | None = None
        self._decoded_texts = decoded_texts

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"__es__": 100, "__cak__": 101}[token]

    def __call__(self, texts, **kwargs) -> SpyEncoding:
        self.last_texts = list(texts)
        encoding = SpyEncoding({"input_ids": [[1, 2, 3]] * len(texts)})
        self.last_encoding = encoding
        return encoding

    def batch_decode(self, generated_ids, **kwargs) -> list[str]:
        if self._decoded_texts is not None:
            return self._decoded_texts
        return [f"decoded-{i}" for i in range(len(generated_ids))]


class FakeModel:
    def __init__(self, device: str = "sentinel-cpu"):
        self.device = device
        self.generate_called_with: dict | None = None

    def generate(self, **kwargs):
        self.generate_called_with = kwargs
        return [0]


def test_translate_prepends_the_target_direction_tag_to_the_source_text():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    request = InferenceRequest(source_lang="es", target_lang="cak", text="hola")

    translate(model, tokenizer, request)

    assert tokenizer.last_texts == ["__cak__ hola"]


def test_translate_uses_the_target_langs_direction_tag_as_forced_bos_token_id():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    request = InferenceRequest(source_lang="cak", target_lang="es", text="La utz awäch?")

    translate(model, tokenizer, request)

    assert model.generate_called_with["forced_bos_token_id"] == 100  # __es__


def test_translate_moves_the_encoded_batch_to_model_device():
    tokenizer = FakeTokenizer()
    model = FakeModel(device="sentinel-cuda:0")
    request = InferenceRequest(source_lang="es", target_lang="cak", text="hola")

    translate(model, tokenizer, request)

    assert tokenizer.last_encoding.moved_to == ["sentinel-cuda:0"]


def test_translate_returns_the_decoded_hypothesis():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    request = InferenceRequest(source_lang="es", target_lang="cak", text="hola")

    result = translate(model, tokenizer, request)

    assert result == "decoded-0"


def test_translate_strips_a_literal_leading_direction_tag_from_the_output():
    # Regression test for a real bug found via a real endpoint invocation
    # (issue #8): __cak__ is an ordinary added vocab token (unlike __es__,
    # one of M2M100's own pretrained *special* tokens), so
    # skip_special_tokens=True does not strip it, and it survives verbatim
    # as the literal first word of cak-target output -- see
    # training.direction.strip_leading_direction_tag's own docstring for
    # the full story.
    tokenizer = FakeTokenizer(decoded_texts=["__cak__ q'ij"])
    model = FakeModel()
    request = InferenceRequest(source_lang="es", target_lang="cak", text="Buenos días")

    result = translate(model, tokenizer, request)

    assert result == "q'ij"


def test_translate_leaves_output_without_a_leading_tag_untouched():
    # __es__-target output already has the tag stripped by
    # skip_special_tokens=True (it's a pretrained special token), so
    # there's nothing left for the defensive strip to do here.
    tokenizer = FakeTokenizer(decoded_texts=["bueno lado"])
    model = FakeModel()
    request = InferenceRequest(source_lang="cak", target_lang="es", text="La utz awäch?")

    result = translate(model, tokenizer, request)

    assert result == "bueno lado"


# ---------------------------------------------------------------------------
# predict_fn -- wiring input_fn's output through translate()
# ---------------------------------------------------------------------------


def test_predict_fn_returns_a_translated_text_dict():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    request = InferenceRequest(source_lang="es", target_lang="cak", text="hola")

    result = predict_fn(request, (model, tokenizer))

    assert result == {"translated_text": "decoded-0"}
