"""Custom SageMaker inference handler for the fine-tuned Spanish<->Kaqchikel
M2M100 checkpoint (issue #8, ADR 0001).

## Why a custom handler is needed

This model is not a standard `text2text-generation` pipeline: translation
direction is controlled by an explicit direction-tag token (`__es__`/
`__cak__`) prepended to the source text and used as `forced_bos_token_id`
at generation time (see `training.direction` and
`training.train.generate_translations`, which this module's `translate()`
mirrors for a single real-time request instead of a batch of evaluation
examples). The SageMaker HuggingFace Inference Toolkit's default
`text2text-generation` pipeline has no notion of this tagging scheme, so a
default deployment (no `entry_point`) would silently ignore direction
entirely. This module is packaged as a SageMaker Inference Toolkit
`code/inference.py` (see `deployment/package_model.py`) implementing the
four hooks the toolkit calls, in order, for every request:
`model_fn` -> `input_fn` -> `predict_fn` -> `output_fn`.

## Request/response contract (documented for issue #9's `apps/api` client)

Request body (`Content-Type: application/json`):

```json
{"source_lang": "es", "target_lang": "cak", "text": "Buenos días"}
```

- `source_lang`/`target_lang`: one of `"es"` (Spanish) or `"cak"`
  (Kaqchikel) -- the same two-letter codes `training.direction` uses
  (`SPANISH`/`KAQCHIKEL`). Must be different from each other (there is no
  same-language "translation").
- `text`: non-empty string (after stripping leading/trailing whitespace)
  to translate, in the language named by `source_lang`.

Response body (`Accept: application/json`):

```json
{"translated_text": "Utz sq'ij"}
```

Malformed/invalid requests (missing field, unsupported language code,
empty text, wrong content type) raise `ValueError` from `input_fn`, which
the SageMaker Hugging Face Inference Toolkit surfaces to the caller as a
4xx client error with the message in the response body -- confirmed
against the real deployed endpoint (not assumed), see `ml/README.md`.

## What's testable without a real model, and what isn't

`parse_request`, `input_fn`, `output_fn`, and `translate`'s surrounding
logic (direction-tag text construction, `forced_bos_token_id` lookup,
`.to(model.device)` placement) are pure/duck-typable and unit-tested in
`tests/unit/test_inference.py`, following the exact same duck-typed-fake
approach as `training.train.generate_translations`'s own test. `model_fn`
(loading real `M2M100Tokenizer`/`M2M100ForConditionalGeneration` weights)
cannot be unit-tested without a real checkpoint and is only exercised by a
real deployment (see `ml/README.md`'s "Serving" section for how that was
verified end-to-end for issue #8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from training.direction import (
    DIRECTION_TAGS,
    KAQCHIKEL,
    SPANISH,
    strip_leading_direction_tag,
    tag_source_text,
)

CONTENT_TYPE_JSON = "application/json"

SUPPORTED_LANGS = (SPANISH, KAQCHIKEL)

# Matches training/train.py's own default -- the model was fine-tuned with
# this max sequence length, so generation should match it too.
DEFAULT_MAX_LENGTH = 128


@dataclass(frozen=True)
class InferenceRequest:
    """One parsed, validated translation request."""

    source_lang: str
    target_lang: str
    text: str


def parse_request(payload: dict[str, Any]) -> InferenceRequest:
    """Validate and parse a request body dict into an `InferenceRequest`.

    Raises `ValueError` (with a message naming the offending field) for
    any contract violation: a missing field, an unsupported language
    code, matching source/target languages, or empty/non-string text.
    """
    for required_key in ("source_lang", "target_lang", "text"):
        if required_key not in payload:
            raise ValueError(f"Missing required field: {required_key!r}")

    source_lang = payload["source_lang"]
    target_lang = payload["target_lang"]
    text = payload["text"]

    if source_lang not in SUPPORTED_LANGS:
        raise ValueError(
            f"Unsupported source_lang {source_lang!r}; expected one of {SUPPORTED_LANGS}"
        )
    if target_lang not in SUPPORTED_LANGS:
        raise ValueError(
            f"Unsupported target_lang {target_lang!r}; expected one of {SUPPORTED_LANGS}"
        )
    if source_lang == target_lang:
        raise ValueError(
            f"source_lang and target_lang must differ (both were {source_lang!r}); "
            "there is no same-language translation."
        )
    if not isinstance(text, str):
        # Deliberately ValueError, not TypeError (ruff's TRY004 default
        # preference) -- every contract violation in this function raises
        # ValueError uniformly, so `input_fn` (and any caller) can catch a
        # single exception type for "this request is invalid" regardless
        # of which field/reason.
        raise ValueError(f"text must be a string, got {type(text).__name__}")  # noqa: TRY004

    stripped_text = text.strip()
    if not stripped_text:
        raise ValueError("text must not be empty")

    return InferenceRequest(source_lang=source_lang, target_lang=target_lang, text=stripped_text)


def input_fn(request_body: str, content_type: str) -> InferenceRequest:
    """SageMaker Inference Toolkit hook: parse the raw request body.

    Only `application/json` is supported -- see this module's docstring
    for the full contract.
    """
    if content_type != CONTENT_TYPE_JSON:
        raise ValueError(
            f"Unsupported content type {content_type!r}; expected {CONTENT_TYPE_JSON!r}"
        )
    try:
        payload = json.loads(request_body)
    except json.JSONDecodeError as error:
        raise ValueError(f"Request body is not valid JSON: {error}") from error

    return parse_request(payload)


def translate(
    model: Any,
    tokenizer: Any,
    request: InferenceRequest,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> str:
    """Generate a single translation for `request`, mirroring
    `training.train.generate_translations`'s direction-tag mechanism for a
    single real-time request instead of a batch of evaluation examples:
    prepend the target language's direction tag to the source text, use it
    as `forced_bos_token_id`, and greedily generate.

    Moves the encoded batch to `model.device` before calling `generate`,
    same reasoning as `generate_translations` -- `tokenizer(...,
    return_tensors="pt")` always returns CPU tensors regardless of where
    `model` lives.
    """
    tagged_text = tag_source_text(request.text, request.target_lang)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(DIRECTION_TAGS[request.target_lang])

    encoded = tokenizer(
        [tagged_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    generated_ids = model.generate(
        **encoded, forced_bos_token_id=forced_bos_token_id, max_length=max_length
    )
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return strip_leading_direction_tag(decoded[0], request.target_lang)


def predict_fn(request: InferenceRequest, model_and_tokenizer: tuple[Any, Any]) -> dict[str, str]:
    """SageMaker Inference Toolkit hook: run inference on the already-parsed
    request, returning the response body's data (before JSON serialization,
    which `output_fn` handles).
    """
    model, tokenizer = model_and_tokenizer
    translated_text = translate(model, tokenizer, request)
    return {"translated_text": translated_text}


def output_fn(prediction: dict[str, str], accept: str) -> str:
    """SageMaker Inference Toolkit hook: serialize `predict_fn`'s output.

    Only `application/json` is supported -- see this module's docstring.

    Returns just the serialized body (a `str`), *not* a `(body,
    content_type)` tuple -- confirmed against the real deployed endpoint
    for issue #8: the `sagemaker-huggingface-inference-toolkit`'s
    `handler_service.postprocess` calls `output_fn` and uses its return
    value directly as the response body, setting the response's actual
    content type separately via `context.set_response_content_type(...)`.
    An earlier version of this function returned a tuple, which the
    toolkit then serialized *as the response body itself* -- the real
    endpoint returned `["{\\"translated_text\\": ...}", "application/json"]`
    instead of the documented `{"translated_text": ...}` contract, only
    caught by testing a real invocation end-to-end, not by the (correctly
    passing, but wrongly-scoped) unit tests, which asserted the tuple
    shape as if it were the actual contract.
    """
    if accept != CONTENT_TYPE_JSON:
        raise ValueError(f"Unsupported accept type {accept!r}; expected {CONTENT_TYPE_JSON!r}")
    return json.dumps(prediction)


def model_fn(model_dir: str) -> tuple[Any, Any]:
    """SageMaker Inference Toolkit hook: load the fine-tuned checkpoint.

    Loads the real `M2M100Tokenizer`/`M2M100ForConditionalGeneration` saved
    by `training.train.save_model_and_tokenizer` (`model.save_pretrained`/
    `tokenizer.save_pretrained` to the same directory), matching how
    `training.train`'s own `load_base_model_and_tokenizer` loads a
    checkpoint. Not unit-tested here -- it needs real weights on disk; see
    this module's docstring.
    """
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    tokenizer = M2M100Tokenizer.from_pretrained(model_dir)
    model = M2M100ForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer
