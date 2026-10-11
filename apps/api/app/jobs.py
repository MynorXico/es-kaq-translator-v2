"""DynamoDB-backed storage for async translate jobs (ADR 0008).

`infra/cdk/lib/api-stack.ts` provisions one `TranslateJobsTable` per
environment: on-demand billing, partition key `job_id`, a `ttl` attribute
(~1 hour) so jobs self-expire. This module is the only place in `apps/api`
that reads/writes it, shared by the Request Lambda (`app.main`'s
`/v1/translate-jobs` routes), the Worker Lambda (`app.worker_handler`), and
the Worker's `onFailure` Lambda Destination (`app.failure_handler`).

Every DynamoDB/Lambda client call is injectable (`dynamodb_resource`/
`lambda_client`), mirroring `app.translation.translate_via_sagemaker`'s
`runtime_client` convention -- defaults to a real `boto3` client/resource,
replaced with a fake at the boundary in tests (see `tests/_fake_dynamodb.py`).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import boto3
from botocore.exceptions import ClientError

from app.models import Direction, JobStatus, TranslateRequest

# ~1 hour (ADR 0008) -- short enough to bound how long any submitted text is
# retained, long enough to comfortably outlast a real poll loop.
_JOB_TTL_SECONDS = 60 * 60

# DynamoDB's TTL attribute is a best-effort background sweep that can lag
# up to 48h behind an item's nominal expiry (AWS's documented behavior) --
# this module always re-checks ttl itself (see get_job) rather than relying
# on the sweep alone to make an expired item disappear.


def _table_name() -> str:
    table_name = os.environ.get("TRANSLATE_JOBS_TABLE_NAME")
    if not table_name:
        raise RuntimeError("TRANSLATE_JOBS_TABLE_NAME environment variable is not set")
    return table_name


def _worker_function_name() -> str:
    function_name = os.environ.get("WORKER_FUNCTION_NAME")
    if not function_name:
        raise RuntimeError("WORKER_FUNCTION_NAME environment variable is not set")
    return function_name


def _table(dynamodb_resource: Any = None) -> Any:
    resource = dynamodb_resource if dynamodb_resource is not None else boto3.resource("dynamodb")
    return resource.Table(_table_name())


def create_job(request: TranslateRequest, *, dynamodb_resource: Any = None) -> str:
    """Create a pending job record, returning its `job_id`.

    Generated via `uuid.uuid4()`, not a sequential/predictable value: since
    `GET /v1/translate-jobs/{job_id}` is unauthenticated, `job_id` is a de
    facto bearer credential for the submitted text and its translation
    (ADR 0008). `request.text`/`request.direction` are stored on the item
    too -- this is the first place `apps/api` persists submitted text at
    all (previously: Lambda -> SageMaker -> response, nothing written to
    disk/DB), transiently, until `ttl` expires.
    """
    job_id = str(uuid.uuid4())
    now = int(time.time())
    _table(dynamodb_resource).put_item(
        Item={
            "job_id": job_id,
            "status": JobStatus.PENDING.value,
            "text": request.text,
            "direction": request.direction.value,
            "ttl": now + _JOB_TTL_SECONDS,
        }
    )
    return job_id


def get_job(job_id: str, *, dynamodb_resource: Any = None) -> dict[str, Any] | None:
    """Request-Lambda-side read (the `GET` poll route): the job's current
    record, or `None` if unknown or expired-but-not-yet-swept.
    """
    item = _table(dynamodb_resource).get_item(Key={"job_id": job_id}).get("Item")
    if item is None:
        return None
    if int(item["ttl"]) < int(time.time()):
        return None
    return item


def get_job_input(job_id: str, *, dynamodb_resource: Any = None) -> TranslateRequest | None:
    """Worker-side read: the submitted `text`/`direction` for a job, or
    `None` if the job record is missing (e.g. already expired).
    """
    item = _table(dynamodb_resource).get_item(Key={"job_id": job_id}).get("Item")
    if item is None:
        return None
    return TranslateRequest(text=item["text"], direction=Direction(item["direction"]))


def mark_succeeded(job_id: str, translation: str, *, dynamodb_resource: Any = None) -> None:
    """Writes a `succeeded` terminal status + the translated text."""
    _conditional_update(
        job_id,
        {"status": JobStatus.SUCCEEDED.value, "translation": translation},
        dynamodb_resource=dynamodb_resource,
    )


def mark_failed(
    job_id: str, status_code: int, error: str, *, dynamodb_resource: Any = None
) -> None:
    """Writes a `failed` terminal status + the Spanish error message and
    the numeric HTTP status apps/api would have returned synchronously.
    """
    _conditional_update(
        job_id,
        {"status": JobStatus.FAILED.value, "status_code": status_code, "error": error},
        dynamodb_resource=dynamodb_resource,
    )


def _conditional_update(
    job_id: str, fields: dict[str, Any], *, dynamodb_resource: Any = None
) -> None:
    """Writes a terminal job status, guarded by a DynamoDB conditional
    write (ADR 0008): the Worker Lambda's own normal write (`mark_succeeded`/
    `mark_failed`) and the `onFailure` Lambda Destination's write
    (`app.failure_handler`, also via `mark_failed`) race to write a
    terminal status for the same job -- whichever lands first wins, and the
    second write is silently a no-op rather than clobbering the first.

    Guarded by `status` not yet existing, or still `pending` -- i.e. this
    never overwrites an item that's already `succeeded`/`failed`.
    """
    table = _table(dynamodb_resource)
    names = {f"#{key}": key for key in fields}
    values = {f":{key}": value for key, value in fields.items()}
    update_expression = "SET " + ", ".join(f"#{key} = :{key}" for key in fields)

    # The condition needs its own name/value placeholders, distinct from
    # the SET clause's own #status/:status (whose :status value is the new
    # terminal status, not "pending").
    names["#status"] = "status"
    values[":pending_status"] = JobStatus.PENDING.value

    try:
        table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
            ConditionExpression="attribute_not_exists(#status) OR #status = :pending_status",
        )
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") != "ConditionalCheckFailedException":
            raise


def invoke_worker(job_id: str, *, lambda_client: Any = None) -> None:
    """Fires one async (`InvocationType=Event`) invoke of the Worker
    Lambda for `job_id` (ADR 0008) -- fire-and-forget: the Request Lambda
    doesn't wait for (or see) the Worker's result, only starts it.
    """
    client = lambda_client if lambda_client is not None else boto3.client("lambda")
    client.invoke(
        FunctionName=_worker_function_name(),
        InvocationType="Event",
        Payload=json.dumps({"job_id": job_id}).encode("utf-8"),
    )
