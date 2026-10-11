"""Structured logging for translate-job resolution (ADR 0008, issue #47).

Emits one JSON log line each time a translate job (ADR 0008) resolves to a
terminal status -- `succeeded` or `failed` -- with operational metadata
only: `job_id`, direction, input length, latency, and status/error type,
never the raw request/response translation text. That's a light
privacy-hygiene measure (this isn't the private ALMG corpus -- see
`docs/data-governance.md` -- but there's still no reason to persist
user-submitted text in application logs) and it keeps log volume/size
predictable regardless of how long a submitted text is.

This replaces the old `log_translate_request` (the synchronous
`/v1/translate` contract's per-request log line), which became dead code
once that route was retired -- nothing else is logged for a translate job
otherwise, which would have silently dropped issue #47's only production
observability for SageMaker-side failures/latency the moment ADR 0008's
job split moved the actual SageMaker call off the Request Lambda API
Gateway's own CloudWatch metrics can see.

Called from two places, both after a job's terminal status is decided,
never before:
- `app.worker_handler`, the common case: `translate_via_sagemaker`
  succeeds or raises a `TranslationServiceError`. `latency_ms` there is
  the actual time spent on that call.
- `app.failure_handler` (the Worker Lambda's `onFailure` Lambda
  Destination): fires when the Worker is killed by its own timeout or
  crashes before its own code gets a chance to log anything. This handler
  has no visibility into how long the Worker actually ran before being
  killed, nor always the original `direction`/`text` (a best-effort
  DynamoDB read that can itself come back empty) -- `latency_ms`,
  `direction`, and `input_length` are therefore all optional here, `None`
  rather than a fabricated number.

`job_id` (unlike the old per-request log line, which had no concept of
one) lets a CloudWatch Logs Insights query correlate a request's
enqueue-time Request Lambda log line with its resolution-time Worker/
onFailure Lambda log line -- addressing the "Observability trade-off" ADR
0008 flags explicitly: splitting one logical translate request across two
Lambdas means its two halves land in separate CloudWatch log groups,
correlated only by `job_id`.
"""

from __future__ import annotations

import json
import logging

# A dedicated logger (rather than `app.main`'s module logger or the root
# logger) so this can be configured/filtered independently -- e.g. tests
# capture exactly this logger's records via `caplog.at_level(...,
# logger="app.request")` without picking up unrelated log output.
logger = logging.getLogger("app.request")
logger.setLevel(logging.INFO)


def log_job_resolution(
    *,
    job_id: str,
    direction: str | None,
    input_length: int | None,
    latency_ms: float | None,
    status_code: int,
    error_type: str | None,
) -> None:
    """Log one structured JSON line describing a translate job's terminal
    resolution.

    Successful jobs (`error_type is None`) log at INFO; jobs that ended in
    an error log at WARNING, so the two are easy to tell apart or filter
    on independently (e.g. for a log-based CloudWatch metric filter, or
    just `grep`ing local logs) -- the same convention the old
    `log_translate_request` used.
    """
    payload = {
        "event": "translate_job_resolved",
        "job_id": job_id,
        "direction": direction,
        "input_length": input_length,
        "latency_ms": None if latency_ms is None else round(latency_ms, 2),
        "status_code": status_code,
        "error_type": error_type,
    }
    level = logging.INFO if error_type is None else logging.WARNING
    logger.log(level, json.dumps(payload))
