"""`onFailure` Lambda Destination entry point for the Worker Lambda
(ADR 0008).

A third handler entry point on the same shared Docker image as
`app.lambda_handler` (Request Lambda) and `app.worker_handler` (Worker
Lambda), wired in `infra/cdk/lib/api-stack.ts` as the Worker Lambda's
async-invoke `onFailure` destination. Fires when the Worker Lambda is
killed by its own timeout or crashes before its own code gets a chance to
write a terminal status -- Lambda's async-invocation failure destinations
fire on a timeout too, not just an unhandled exception, so a client can
never be left polling a `pending` job forever.

Invoked by Lambda itself with its own documented async-invocation failure
envelope (see
https://docs.aws.amazon.com/lambda/latest/dg/invocation-async.html#invocation-async-destinations)
-- a platform-defined shape, not one this project controls.
`requestPayload` is the original event the Worker Lambda was invoked with
(`app.jobs.invoke_worker`'s `{"job_id": ...}`).

Marks the job `failed` via the same conditional-write-guarded
`app.jobs.mark_failed` the Worker Lambda's own normal failure path uses
(ADR 0008): whichever write lands first wins, this is a no-op if the
Worker already wrote a terminal status itself.

Also logs the resolution via `app.observability.log_job_resolution`
(issue #47) -- without this, a Worker that's reliably timing out (e.g. a
SageMaker endpoint stuck cold) would be invisible to every log-based
signal, since this is the *only* path that ever runs when the Worker's
own code never gets a chance to log anything itself. `direction`/
`input_length` are a best-effort read of the job's own record (it was
already written by `app.main.create_translate_job`) -- `None` if the
record is somehow missing those fields, never a crash; `latency_ms` is
always `None` here, since this handler has no visibility into how long
the Worker actually ran before being killed.
"""

from __future__ import annotations

from typing import Any

from app.jobs import get_job, mark_failed
from app.observability import log_job_resolution

# Covers both a genuine timeout and an unhandled crash -- from the client's
# perspective both just mean "the translation never came back in time",
# the same framing as app.translation's own 504 case for a slow upstream.
_WORKER_FAILURE_STATUS_CODE = 504
_WORKER_FAILURE_MESSAGE = (
    "El servicio de traducción tardó demasiado en responder. Inténtalo de nuevo."
)
_WORKER_FAILURE_ERROR_TYPE = "WorkerTimeoutOrCrash"


def handler(event: dict[str, Any], context: Any) -> None:
    job_id = event["requestPayload"]["job_id"]
    mark_failed(job_id, _WORKER_FAILURE_STATUS_CODE, _WORKER_FAILURE_MESSAGE)

    job = get_job(job_id)
    direction = job.get("direction") if job else None
    input_length = len(job["text"]) if job and "text" in job else None
    log_job_resolution(
        job_id=job_id,
        direction=direction,
        input_length=input_length,
        latency_ms=None,
        status_code=_WORKER_FAILURE_STATUS_CODE,
        error_type=_WORKER_FAILURE_ERROR_TYPE,
    )
