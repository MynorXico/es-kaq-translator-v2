"""Unit tests for `app.worker_handler` -- the Worker Lambda entry point
(ADR 0008), invoked directly with a `{"job_id": ...}` payload (never
through FastAPI/Mangum). `boto3` is mocked at both boundaries it touches:
`app.jobs`'s DynamoDB resource (see `tests/_fake_dynamodb.py`) and
`app.translation`'s `sagemaker-runtime` client (same convention as
`tests/unit/test_translation.py`).
"""

from __future__ import annotations

import json
import logging
import time
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from app import translation as translation_module
from app.worker_handler import handler
from tests._fake_dynamodb import FakeDynamoDBResource


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("TRANSLATE_JOBS_TABLE_NAME", "traductor-kaqchikel-translate-jobs-test")
    monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "traductor-kaqchikel-es-cak-test")


@pytest.fixture
def fake_resource(monkeypatch) -> FakeDynamoDBResource:
    resource = FakeDynamoDBResource()
    monkeypatch.setattr("app.jobs.boto3.resource", lambda *args, **kwargs: resource)
    return resource


def _mock_sagemaker(monkeypatch, *, translated_text: str | None = None, error: Exception | None = None):
    fake_client = MagicMock()
    if error is not None:
        fake_client.invoke_endpoint.side_effect = error
    else:
        body = MagicMock()
        body.read.return_value = json.dumps({"translated_text": translated_text}).encode("utf-8")
        fake_client.invoke_endpoint.return_value = {"Body": body}
    monkeypatch.setattr(translation_module.boto3, "client", lambda *args, **kwargs: fake_client)


def test_handler_translates_and_marks_the_job_succeeded(monkeypatch, fake_resource, caplog):
    job_id = "11111111-1111-4111-8111-111111111111"
    fake_resource.table.seed(
        {
            "job_id": job_id,
            "status": "pending",
            "text": "Buenos días",
            "direction": "es-to-cak",
            "ttl": int(time.time()) + 3600,
        }
    )
    _mock_sagemaker(monkeypatch, translated_text="Utz sq'ij")

    with caplog.at_level(logging.INFO, logger="app.request"):
        handler({"job_id": job_id}, None)

    item = fake_resource.table.items[job_id]
    assert item["status"] == "succeeded"
    assert item["translation"] == "Utz sq'ij"

    # Issue #47's structured logging, revived job_id-keyed for ADR 0008
    # (app.observability.log_job_resolution) -- the only production
    # observability for a SageMaker-side failure/slowness now that the
    # actual SageMaker call has moved off the Request Lambda API Gateway's
    # own CloudWatch metrics can see.
    records = [r for r in caplog.records if r.name == "app.request"]
    assert len(records) == 1
    payload = json.loads(records[0].message)
    assert payload["job_id"] == job_id
    assert payload["direction"] == "es-to-cak"
    assert payload["input_length"] == len("Buenos días")
    assert payload["status_code"] == 200
    assert payload["error_type"] is None
    assert isinstance(payload["latency_ms"], (int, float))


def test_handler_catches_a_translation_service_error_and_marks_the_job_failed(
    monkeypatch, fake_resource, caplog
):
    job_id = "22222222-2222-4222-8222-222222222222"
    fake_resource.table.seed(
        {
            "job_id": job_id,
            "status": "pending",
            "text": "Hola",
            "direction": "es-to-cak",
            "ttl": int(time.time()) + 3600,
        }
    )
    _mock_sagemaker(
        monkeypatch,
        error=ClientError(
            {"Error": {"Code": "ModelError", "Message": "Received client error (400)"}},
            "InvokeEndpoint",
        ),
    )

    with caplog.at_level(logging.INFO, logger="app.request"):
        handler({"job_id": job_id}, None)

    item = fake_resource.table.items[job_id]
    assert item["status"] == "failed"
    assert item["status_code"] == 400
    assert item["error"] == (
        "No se pudo traducir el texto enviado. Verifica el contenido e inténtalo de nuevo."
    )

    records = [r for r in caplog.records if r.name == "app.request"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    payload = json.loads(records[0].message)
    assert payload["job_id"] == job_id
    assert payload["status_code"] == 400
    assert payload["error_type"] == "TranslationServiceError"


def test_handler_is_a_noop_when_the_job_record_is_already_gone(monkeypatch, fake_resource):
    # e.g. the job already expired by the time this (deliberately slow,
    # cold-started) Worker invocation actually runs -- nothing to translate
    # or write back to; must not raise.
    _mock_sagemaker(monkeypatch, translated_text="should never be used")

    handler({"job_id": "33333333-3333-4333-8333-333333333333"}, None)

    assert fake_resource.table.items == {}
