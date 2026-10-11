"""Unit tests for `app.jobs` -- the DynamoDB-backed `TranslateJobsTable`
read/write layer underlying ADR 0008's async translate-job pattern, with
`boto3` mocked at the boundary (see `tests/_fake_dynamodb.py`).
"""

from __future__ import annotations

import time
import uuid

import pytest

from app.jobs import create_job, get_job, get_job_input, invoke_worker, mark_failed, mark_succeeded
from app.models import Direction, JobStatus, TranslateRequest
from tests._fake_dynamodb import FakeDynamoDBTable


@pytest.fixture(autouse=True)
def _table_name_env(monkeypatch):
    monkeypatch.setenv("TRANSLATE_JOBS_TABLE_NAME", "traductor-kaqchikel-translate-jobs-test")


@pytest.fixture
def fake_table() -> FakeDynamoDBTable:
    return FakeDynamoDBTable()


class _FakeResource:
    def __init__(self, table: FakeDynamoDBTable) -> None:
        self._table = table

    def Table(self, _name: str) -> FakeDynamoDBTable:
        return self._table


def test_create_job_returns_an_unguessable_uuid4_job_id(fake_table):
    job_id = create_job(
        TranslateRequest(text="Buenos días", direction=Direction.ES_TO_CAK),
        dynamodb_resource=_FakeResource(fake_table),
    )

    # Must be a real UUID4, not sequential/predictable -- GET is
    # unauthenticated, so job_id is a de facto bearer credential (ADR 0008).
    parsed = uuid.UUID(job_id)
    assert parsed.version == 4


def test_create_job_writes_a_pending_item_with_text_direction_and_a_ttl(fake_table):
    before = int(time.time())
    job_id = create_job(
        TranslateRequest(text="Buenos días", direction=Direction.ES_TO_CAK),
        dynamodb_resource=_FakeResource(fake_table),
    )

    item = fake_table.items[job_id]
    assert item["status"] == "pending"
    assert item["text"] == "Buenos días"
    assert item["direction"] == "es-to-cak"
    # ~1 hour TTL (ADR 0008), checked loosely rather than pinned to an exact
    # second to avoid a flaky test.
    assert before + 3500 <= item["ttl"] <= before + 3700


def test_get_job_returns_the_stored_item_for_a_pending_job(fake_table):
    fake_table.seed(
        {
            "job_id": "11111111-1111-4111-8111-111111111111",
            "status": "pending",
            "text": "Hola",
            "direction": "es-to-cak",
            "ttl": int(time.time()) + 3600,
        }
    )

    job = get_job("11111111-1111-4111-8111-111111111111", dynamodb_resource=_FakeResource(fake_table))

    assert job is not None
    assert job["status"] == "pending"


def test_get_job_returns_none_for_an_unknown_job_id(fake_table):
    job = get_job("22222222-2222-4222-8222-222222222222", dynamodb_resource=_FakeResource(fake_table))

    assert job is None


def test_get_job_returns_none_for_an_expired_but_not_yet_swept_item(fake_table):
    # DynamoDB's TTL sweep is best-effort and can lag up to 48h behind the
    # nominal expiry (ADR 0008) -- the item can still be physically present
    # with a ttl in the past, and get_job must treat that as "not found"
    # itself, not rely on the sweep alone.
    fake_table.seed(
        {
            "job_id": "33333333-3333-4333-8333-333333333333",
            "status": "succeeded",
            "translation": "Utz sq'ij",
            "ttl": int(time.time()) - 60,
        }
    )

    job = get_job("33333333-3333-4333-8333-333333333333", dynamodb_resource=_FakeResource(fake_table))

    assert job is None


def test_get_job_input_reconstructs_the_submitted_translate_request(fake_table):
    fake_table.seed(
        {
            "job_id": "44444444-4444-4444-8444-444444444444",
            "status": "pending",
            "text": "Utz sq'ij",
            "direction": "cak-to-es",
            "ttl": int(time.time()) + 3600,
        }
    )

    request = get_job_input(
        "44444444-4444-4444-8444-444444444444", dynamodb_resource=_FakeResource(fake_table)
    )

    assert request is not None
    assert request.text == "Utz sq'ij"
    assert request.direction == Direction.CAK_TO_ES


def test_get_job_input_returns_none_for_an_unknown_job_id(fake_table):
    request = get_job_input(
        "55555555-5555-4555-8555-555555555555", dynamodb_resource=_FakeResource(fake_table)
    )

    assert request is None


def test_mark_succeeded_writes_the_translation_on_a_pending_job(fake_table):
    job_id = "66666666-6666-4666-8666-666666666666"
    fake_table.seed({"job_id": job_id, "status": "pending", "ttl": int(time.time()) + 3600})

    mark_succeeded(job_id, "Utz sq'ij", dynamodb_resource=_FakeResource(fake_table))

    item = fake_table.items[job_id]
    assert item["status"] == JobStatus.SUCCEEDED.value
    assert item["translation"] == "Utz sq'ij"


def test_mark_failed_writes_the_status_code_and_error_on_a_pending_job(fake_table):
    job_id = "77777777-7777-4777-8777-777777777777"
    fake_table.seed({"job_id": job_id, "status": "pending", "ttl": int(time.time()) + 3600})

    mark_failed(job_id, 502, "mensaje de error", dynamodb_resource=_FakeResource(fake_table))

    item = fake_table.items[job_id]
    assert item["status"] == JobStatus.FAILED.value
    assert item["status_code"] == 502
    assert item["error"] == "mensaje de error"


def test_mark_succeeded_does_not_clobber_an_already_failed_job(fake_table):
    # The conditional-write guard (ADR 0008): whichever terminal write lands
    # first wins, the second is a silent no-op, not a race/clobber. Here the
    # onFailure path (simulated by pre-seeding a terminal "failed" item) won
    # first; a late Worker success write must not overwrite it.
    job_id = "88888888-8888-4888-8888-888888888888"
    fake_table.seed(
        {
            "job_id": job_id,
            "status": "failed",
            "status_code": 504,
            "error": "El servicio de traducción tardó demasiado en responder. Inténtalo de nuevo.",
            "ttl": int(time.time()) + 3600,
        }
    )

    mark_succeeded(job_id, "Utz sq'ij", dynamodb_resource=_FakeResource(fake_table))

    item = fake_table.items[job_id]
    assert item["status"] == "failed"
    assert "translation" not in item


def test_mark_failed_does_not_clobber_an_already_succeeded_job(fake_table):
    # The reverse race: the Worker's own normal success write landed first;
    # a late onFailure write (e.g. from a destination delivery racing right
    # at the timeout boundary) must not overwrite it either.
    job_id = "99999999-9999-4999-8999-999999999999"
    fake_table.seed(
        {
            "job_id": job_id,
            "status": "succeeded",
            "translation": "Utz sq'ij",
            "ttl": int(time.time()) + 3600,
        }
    )

    mark_failed(job_id, 504, "mensaje de error", dynamodb_resource=_FakeResource(fake_table))

    item = fake_table.items[job_id]
    assert item["status"] == "succeeded"
    assert "error" not in item


def test_invoke_worker_fires_an_async_event_invoke_with_the_job_id(monkeypatch):
    monkeypatch.setenv("WORKER_FUNCTION_NAME", "traductor-kaqchikel-translate-worker-test")
    from unittest.mock import MagicMock

    fake_lambda_client = MagicMock()

    invoke_worker("job-123", lambda_client=fake_lambda_client)

    fake_lambda_client.invoke.assert_called_once()
    _, kwargs = fake_lambda_client.invoke.call_args
    assert kwargs["FunctionName"] == "traductor-kaqchikel-translate-worker-test"
    assert kwargs["InvocationType"] == "Event"
    import json

    assert json.loads(kwargs["Payload"]) == {"job_id": "job-123"}
