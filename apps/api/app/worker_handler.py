"""Worker Lambda entry point (ADR 0008).

Invoked only via an async (`InvocationType=Event`) `lambda:InvokeFunction`
call fired by the Request Lambda (`app.jobs.invoke_worker`, called from
`app.main.create_translate_job`) -- never fronted by API Gateway, and not
a FastAPI/Mangum app like `app.lambda_handler`: the event shape here is
this project's own (`{"job_id": "<uuid>"}`), not an API Gateway proxy
event, so there's a plain `try`/`except` around `translate_via_sagemaker`
instead of FastAPI's exception-handler machinery.

Reads the job's submitted `text`/`direction` back from DynamoDB
(`app.jobs.get_job_input`) -- the invoke payload carries only `job_id`,
since the submitted text is already held in the job record itself (ADR
0008) -- runs it through the existing `translate_via_sagemaker` call
against the unchanged SageMaker endpoint, and writes a terminal
`succeeded`/`failed` status back onto the job record. Configured with a
longer `Duration` than the Request Lambda (`infra/cdk/lib/api-stack.ts`,
e.g. 90s) since it isn't constrained by API Gateway's 29s integration
timeout at all; if this handler is killed by that longer timeout before
returning, `app.failure_handler`'s `onFailure` Lambda Destination is what
marks the job `failed` instead.

Every terminal resolution (succeeded or the caught `TranslationServiceError`
case) is logged via `app.observability.log_job_resolution` -- this is now
the only place a SageMaker-side failure/slowness is observable at all,
since the actual SageMaker call moved off the Request Lambda API Gateway's
own CloudWatch metrics can see (issue #47, ADR 0008's "Observability
trade-off").
"""

from __future__ import annotations

import time
from typing import Any

from app.jobs import get_job_input, mark_failed, mark_succeeded
from app.observability import log_job_resolution
from app.translation import TranslationServiceError, translate_via_sagemaker


def handler(event: dict[str, Any], context: Any) -> None:
    job_id = event["job_id"]

    request = get_job_input(job_id)
    if request is None:
        # The job record is already gone (e.g. expired) by the time this
        # invocation actually runs -- nothing to translate or write back to,
        # and nothing worth logging as a job resolution either.
        return

    direction = request.direction.value
    input_length = len(request.text)
    start = time.perf_counter()

    try:
        result = translate_via_sagemaker(request)
    except TranslationServiceError as error:
        mark_failed(job_id, error.status_code, error.message)
        log_job_resolution(
            job_id=job_id,
            direction=direction,
            input_length=input_length,
            latency_ms=(time.perf_counter() - start) * 1000,
            status_code=error.status_code,
            error_type=type(error).__name__,
        )
        return

    mark_succeeded(job_id, result.translation)
    log_job_resolution(
        job_id=job_id,
        direction=direction,
        input_length=input_length,
        latency_ms=(time.perf_counter() - start) * 1000,
        status_code=200,
        error_type=None,
    )
