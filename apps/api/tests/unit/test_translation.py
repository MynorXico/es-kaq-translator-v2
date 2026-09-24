"""Unit tests for `app.translation`'s direction<->lang mapping and error
handling, with the `boto3` `sagemaker-runtime` client mocked at the
boundary (see `ml/tests/unit/test_submit_job.py`'s `resolve_stack_outputs`
tests for the same injected-client convention used here).
"""

import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError, ConnectTimeoutError

from app.models import Direction, TranslateRequest
from app.translation import TranslationServiceError, translate_via_sagemaker


def _fake_response(translated_text: str) -> dict:
    body = MagicMock()
    body.read.return_value = json.dumps({"translated_text": translated_text}).encode("utf-8")
    return {"Body": body}


@pytest.fixture(autouse=True)
def _endpoint_name_env(monkeypatch):
    monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "traductor-kaqchikel-es-cak-test")


def test_translates_es_to_cak_via_the_endpoint():
    fake_client = MagicMock()
    fake_client.invoke_endpoint.return_value = _fake_response("Utz sq'ij")

    result = translate_via_sagemaker(
        TranslateRequest(text="Buenos días", direction=Direction.ES_TO_CAK),
        runtime_client=fake_client,
    )

    assert result.translation == "Utz sq'ij"
    _, kwargs = fake_client.invoke_endpoint.call_args
    assert kwargs["EndpointName"] == "traductor-kaqchikel-es-cak-test"
    assert kwargs["ContentType"] == "application/json"
    assert kwargs["Accept"] == "application/json"
    sent_body = json.loads(kwargs["Body"])
    assert sent_body == {"source_lang": "es", "target_lang": "cak", "text": "Buenos días"}


def test_translates_cak_to_es_via_the_endpoint():
    fake_client = MagicMock()
    fake_client.invoke_endpoint.return_value = _fake_response("Buenos días")

    result = translate_via_sagemaker(
        TranslateRequest(text="Utz sq'ij", direction=Direction.CAK_TO_ES),
        runtime_client=fake_client,
    )

    assert result.translation == "Buenos días"
    _, kwargs = fake_client.invoke_endpoint.call_args
    sent_body = json.loads(kwargs["Body"])
    assert sent_body == {"source_lang": "cak", "target_lang": "es", "text": "Utz sq'ij"}


def test_maps_model_error_to_a_400_translation_service_error():
    fake_client = MagicMock()
    fake_client.invoke_endpoint.side_effect = ClientError(
        {
            "Error": {"Code": "ModelError", "Message": "Received client error (400)"},
            "OriginalStatusCode": 400,
            "OriginalMessage": "text must not be empty",
        },
        "InvokeEndpoint",
    )

    with pytest.raises(TranslationServiceError) as excinfo:
        translate_via_sagemaker(
            TranslateRequest(text="Hola", direction=Direction.ES_TO_CAK),
            runtime_client=fake_client,
        )

    assert excinfo.value.status_code == 400


def test_maps_other_client_errors_to_a_502_translation_service_error():
    fake_client = MagicMock()
    fake_client.invoke_endpoint.side_effect = ClientError(
        {"Error": {"Code": "ValidationException", "Message": "Endpoint not found"}},
        "InvokeEndpoint",
    )

    with pytest.raises(TranslationServiceError) as excinfo:
        translate_via_sagemaker(
            TranslateRequest(text="Hola", direction=Direction.ES_TO_CAK),
            runtime_client=fake_client,
        )

    assert excinfo.value.status_code == 502


def test_maps_a_timeout_to_a_504_translation_service_error():
    fake_client = MagicMock()
    fake_client.invoke_endpoint.side_effect = ConnectTimeoutError(endpoint_url="https://example.test")

    with pytest.raises(TranslationServiceError) as excinfo:
        translate_via_sagemaker(
            TranslateRequest(text="Hola", direction=Direction.ES_TO_CAK),
            runtime_client=fake_client,
        )

    assert excinfo.value.status_code == 504
