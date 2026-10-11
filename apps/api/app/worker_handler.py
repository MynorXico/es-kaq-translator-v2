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
"""

from __future__ import annotations

from typing import Any

from app.jobs import get_job_input, mark_failed, mark_succeeded
from app.translation import TranslationServiceError, translate_via_sagemaker


def handler(event: dict[str, Any], context: Any) -> None:
    job_id = event["job_id"]

    request = get_job_input(job_id)
    if request is None:
        # The job record is already gone (e.g. expired) by the time this
        # invocation actually runs -- nothing to translate or write back to.
        return

    try:
        result = translate_via_sagemaker(request)
    except TranslationServiceError as error:
        mark_failed(job_id, error.status_code, error.message)
        return

    mark_succeeded(job_id, result.translation)
