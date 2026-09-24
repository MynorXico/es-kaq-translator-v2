"""Calls the deployed SageMaker endpoint hosting the fine-tuned Spanish<->
Kaqchikel model (issue #9), translating between this API's own `Direction`
enum and the endpoint's `source_lang`/`target_lang` contract.

See `ml/deployment/inference.py`'s module docstring for the endpoint's full
request/response contract -- this module is the client side of it:

- Request: `{"source_lang": "es"|"cak", "target_lang": "cak"|"es", "text": str}`
- Response: `{"translated_text": str}`
- Invalid input (bad language code, empty text, etc.) is surfaced by the
  real endpoint as a `ModelError` (a `ClientError` subtype boto3 raises for
  the model container's own 4xx response) -- confirmed against the real
  deployed endpoint, not assumed (see `ml/deployment/inference.py`).
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.models import Direction, TranslateRequest, TranslateResponse

# Maps this API's own Direction enum to the endpoint's (source_lang,
# target_lang) two-letter-code pair -- see ml/deployment/inference.py's
# SUPPORTED_LANGS (SPANISH="es", KAQCHIKEL="cak").
_ENDPOINT_LANGS: dict[Direction, tuple[str, str]] = {
    Direction.ES_TO_CAK: ("es", "cak"),
    Direction.CAK_TO_ES: ("cak", "es"),
}

_CONTENT_TYPE_JSON = "application/json"

_INVALID_INPUT_MESSAGE = (
    "No se pudo traducir el texto enviado. Verifica el contenido e inténtalo de nuevo."
)
_UNAVAILABLE_MESSAGE = (
    "El servicio de traducción no está disponible en este momento. Inténtalo más tarde."
)
_TIMEOUT_MESSAGE = "El servicio de traducción tardó demasiado en responder. Inténtalo de nuevo."


class TranslationServiceError(Exception):
    """Raised when the SageMaker endpoint can't produce a translation.

    Carries an HTTP status code and a Spanish, user-facing message,
    mirroring `app.validation`'s error-message convention -- `app.main`
    maps this to a JSON body of the shape `{"error": "<message>"}`, the
    same shape used for request-validation errors.
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _endpoint_name() -> str:
    endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
    if not endpoint_name:
        raise RuntimeError("SAGEMAKER_ENDPOINT_NAME environment variable is not set")
    return endpoint_name


def translate_via_sagemaker(
    request: TranslateRequest, *, runtime_client: Any = None
) -> TranslateResponse:
    """Translate `request.text` by invoking the deployed SageMaker endpoint.

    `runtime_client` is injectable for tests (mirroring
    `ml/training/submit_job.py`'s `resolve_stack_outputs` client-injection
    convention) -- defaults to a real `boto3.client("sagemaker-runtime")`.
    """
    client = runtime_client if runtime_client is not None else boto3.client("sagemaker-runtime")

    source_lang, target_lang = _ENDPOINT_LANGS[request.direction]
    payload = {"source_lang": source_lang, "target_lang": target_lang, "text": request.text}

    try:
        response = client.invoke_endpoint(
            EndpointName=_endpoint_name(),
            ContentType=_CONTENT_TYPE_JSON,
            Accept=_CONTENT_TYPE_JSON,
            Body=json.dumps(payload),
        )
    except ClientError as error:
        error_code = error.response.get("Error", {}).get("Code")
        if error_code == "ModelError":
            # The model container's own 4xx (bad input) -- see
            # ml/deployment/inference.py's input_fn/parse_request.
            raise TranslationServiceError(400, _INVALID_INPUT_MESSAGE) from error
        raise TranslationServiceError(502, _UNAVAILABLE_MESSAGE) from error
    except BotoCoreError as error:
        # Network/timeout/connection failures reaching the endpoint at all
        # (distinct from ClientError, which means the endpoint responded
        # with an error) -- e.g. ConnectTimeoutError, ReadTimeoutError,
        # EndpointConnectionError.
        raise TranslationServiceError(504, _TIMEOUT_MESSAGE) from error

    body = json.loads(response["Body"].read())
    return TranslateResponse(translation=body["translated_text"])
