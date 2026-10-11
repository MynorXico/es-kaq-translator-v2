"""Unit tests for `app.failure_handler` -- the Worker Lambda's `onFailure`
Lambda Destination entry point (ADR 0008). Invoked directly here with a
synthetic Lambda async-invocation failure envelope, the shape AWS itself
documents for destination targets
(https://docs.aws.amazon.com/lambda/latest/dg/invocation-async.html#invocation-async-destinations)
-- no real AWS needed, this is a platform-defined payload shape, not
something `app.worker_handler` constructs.
"""

from __future__ import annotations

import time

import pytest

from app.failure_handler import handler
from tests._fake_dynamodb import FakeDynamoDBResource


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("TRANSLATE_JOBS_TABLE_NAME", "traductor-kaqchikel-translate-jobs-test")


@pytest.fixture
def fake_resource(monkeypatch) -> FakeDynamoDBResource:
    resource = FakeDynamoDBResource()
    monkeypatch.setattr("app.jobs.boto3.resource", lambda *args, **kwargs: resource)
    return resource


def _failure_envelope(job_id: str) -> dict:
    """A realistic Lambda async-invocation `onFailure` destination event --
    see this module's docstring for AWS's documented shape. `requestPayload`
    is exactly what the Request Lambda originally invoked the Worker with
    (`app.jobs.invoke_worker`'s `{"job_id": ...}`).
    """
    return {
        "version": "1.0",
        "timestamp": "2026-10-11T00:00:00.000Z",
        "requestContext": {
            "requestId": "deadbeef-0000-0000-0000-000000000000",
            "functionArn": "arn:aws:lambda:us-east-1:111111111111:function:worker:$LATEST",
            "condition": "Timeout",
            "approximateInvokeCount": 1,
        },
        "requestPayload": {"job_id": job_id},
        "responseContext": {
            "statusCode": 200,
            "executedVersion": "$LATEST",
            "functionError": "Unhandled",
        },
        "responsePayload": {
            "errorMessage": "2026-10-11T00:00:00.000Z deadbeef Task timed out after 90.00 seconds",
            "errorType": "Sandbox.Timedout",
        },
    }


def test_marks_the_job_failed_on_a_worker_timeout_destination_event(fake_resource):
    job_id = "11111111-1111-4111-8111-111111111111"
    fake_resource.table.seed({"job_id": job_id, "status": "pending", "ttl": int(time.time()) + 3600})

    handler(_failure_envelope(job_id), None)

    item = fake_resource.table.items[job_id]
    assert item["status"] == "failed"
    assert isinstance(item["status_code"], int)
    assert isinstance(item["error"], str) and item["error"]


def test_does_not_clobber_a_job_the_worker_already_marked_succeeded(fake_resource):
    # The (rare) near-simultaneous race (ADR 0008): the Worker finished and
    # wrote "succeeded" right at its own timeout boundary, but the
    # destination still fires. The conditional-write guard means whichever
    # landed first wins -- here, the Worker's own write.
    job_id = "22222222-2222-4222-8222-222222222222"
    fake_resource.table.seed(
        {"job_id": job_id, "status": "succeeded", "translation": "Utz sq'ij", "ttl": int(time.time()) + 3600}
    )

    handler(_failure_envelope(job_id), None)

    item = fake_resource.table.items[job_id]
    assert item["status"] == "succeeded"
    assert "error" not in item
