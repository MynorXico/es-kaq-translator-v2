"""Integration tests for the `/v1/translate-jobs` routes (ADR 0008),
through the real FastAPI app via `TestClient`. `boto3` is mocked at the
same boundaries as `tests/unit/test_jobs.py` (DynamoDB via
`tests/_fake_dynamodb.py`) and `tests/integration/test_main.py` (the
`sagemaker-runtime` client) -- the whole point of this ADR is that
`POST /v1/translate-jobs` never reaches the SageMaker mock at all, since
that call only ever happens inside `app.worker_handler`, a separate Lambda
this route only fires an async invoke towards (itself mocked here, since
the Request Lambda never waits on or sees the Worker's result).
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app import translation as translation_module
from app.main import app
from tests._fake_dynamodb import FakeDynamoDBResource

client = TestClient(app)


def _env(monkeypatch) -> None:
    monkeypatch.setenv("TRANSLATE_JOBS_TABLE_NAME", "traductor-kaqchikel-translate-jobs-test")
    monkeypatch.setenv("WORKER_FUNCTION_NAME", "traductor-kaqchikel-translate-worker-test")


def _fake_dynamodb(monkeypatch) -> FakeDynamoDBResource:
    resource = FakeDynamoDBResource()
    monkeypatch.setattr("app.jobs.boto3.resource", lambda *args, **kwargs: resource)
    return resource


def _fail_if_sagemaker_called(monkeypatch) -> MagicMock:
    """Asserts the SageMaker boundary is never touched by this route --
    the entire point of the enqueue/poll split (ADR 0008).
    """
    fake_client = MagicMock()
    fake_client.invoke_endpoint.side_effect = AssertionError(
        "POST /v1/translate-jobs must never call the SageMaker mock directly"
    )
    monkeypatch.setattr(translation_module.boto3, "client", lambda *args, **kwargs: fake_client)
    return fake_client


def _fake_lambda_invoke(monkeypatch) -> MagicMock:
    fake_client = MagicMock()
    monkeypatch.setattr("app.jobs.boto3.client", lambda *args, **kwargs: fake_client)
    return fake_client


def test_post_translate_jobs_never_calls_sagemaker_directly(monkeypatch):
    _env(monkeypatch)
    _fake_dynamodb(monkeypatch)
    _fail_if_sagemaker_called(monkeypatch)
    _fake_lambda_invoke(monkeypatch)

    response = client.post(
        "/v1/translate-jobs",
        json={"text": "Buenos días", "direction": "es-to-cak"},
    )

    assert response.status_code == 202


def test_post_translate_jobs_returns_a_job_id_and_fires_one_async_worker_invoke(monkeypatch):
    _env(monkeypatch)
    _fake_dynamodb(monkeypatch)
    _fail_if_sagemaker_called(monkeypatch)
    fake_lambda = _fake_lambda_invoke(monkeypatch)

    response = client.post(
        "/v1/translate-jobs",
        json={"text": "Buenos días", "direction": "es-to-cak"},
    )

    body = response.json()
    job_id = body["job_id"]
    assert uuid.UUID(job_id).version == 4

    fake_lambda.invoke.assert_called_once()
    _, kwargs = fake_lambda.invoke.call_args
    assert kwargs["FunctionName"] == "traductor-kaqchikel-translate-worker-test"
    assert kwargs["InvocationType"] == "Event"


def test_post_translate_jobs_rejects_empty_text(monkeypatch):
    _env(monkeypatch)
    _fake_dynamodb(monkeypatch)
    _fail_if_sagemaker_called(monkeypatch)
    _fake_lambda_invoke(monkeypatch)

    response = client.post(
        "/v1/translate-jobs",
        json={"text": "", "direction": "es-to-cak"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": "El texto no puede estar vacío."}


def test_get_translate_job_returns_pending_before_the_worker_finishes(monkeypatch):
    _env(monkeypatch)
    resource = _fake_dynamodb(monkeypatch)
    job_id = "11111111-1111-4111-8111-111111111111"
    resource.table.seed(
        {"job_id": job_id, "status": "pending", "text": "Hola", "direction": "es-to-cak", "ttl": int(time.time()) + 3600}
    )

    response = client.get(f"/v1/translate-jobs/{job_id}")

    assert response.status_code == 200
    assert response.json() == {"status": "pending"}


def test_get_translate_job_returns_the_translation_once_succeeded(monkeypatch):
    _env(monkeypatch)
    resource = _fake_dynamodb(monkeypatch)
    job_id = "22222222-2222-4222-8222-222222222222"
    resource.table.seed(
        {"job_id": job_id, "status": "succeeded", "translation": "Utz sq'ij", "ttl": int(time.time()) + 3600}
    )

    response = client.get(f"/v1/translate-jobs/{job_id}")

    assert response.status_code == 200
    assert response.json() == {"status": "succeeded", "translation": "Utz sq'ij"}


def test_get_translate_job_returns_the_failure_status_code_and_message(monkeypatch):
    _env(monkeypatch)
    resource = _fake_dynamodb(monkeypatch)
    job_id = "33333333-3333-4333-8333-333333333333"
    resource.table.seed(
        {
            "job_id": job_id,
            "status": "failed",
            "status_code": 502,
            "error": "El servicio de traducción no está disponible en este momento. Inténtalo más tarde.",
            "ttl": int(time.time()) + 3600,
        }
    )

    response = client.get(f"/v1/translate-jobs/{job_id}")

    assert response.status_code == 200
    assert response.json() == {
        "status": "failed",
        "status_code": 502,
        "error": "El servicio de traducción no está disponible en este momento. Inténtalo más tarde.",
    }


def test_get_translate_job_returns_404_for_an_unknown_job_id(monkeypatch):
    _env(monkeypatch)
    _fake_dynamodb(monkeypatch)

    response = client.get(f"/v1/translate-jobs/{uuid.uuid4()}")

    assert response.status_code == 404
    assert "error" in response.json()


def test_get_translate_job_returns_404_for_an_expired_but_not_yet_swept_job(monkeypatch):
    _env(monkeypatch)
    resource = _fake_dynamodb(monkeypatch)
    job_id = "44444444-4444-4444-8444-444444444444"
    # Physically still present (DynamoDB's TTL sweep is best-effort, can lag
    # up to 48h per ADR 0008), but its own ttl is already in the past.
    resource.table.seed(
        {"job_id": job_id, "status": "succeeded", "translation": "Utz sq'ij", "ttl": int(time.time()) - 60}
    )

    response = client.get(f"/v1/translate-jobs/{job_id}")

    assert response.status_code == 404
    assert "error" in response.json()


def test_get_translate_job_returns_400_for_a_malformed_job_id(monkeypatch):
    _env(monkeypatch)
    _fake_dynamodb(monkeypatch)

    response = client.get("/v1/translate-jobs/not-a-uuid")

    assert response.status_code == 400
    assert "error" in response.json()
