"""Structured request logging for `/v1/translate` (issue #47).

Emits one JSON log line per request with operational metadata only --
direction, input length, latency, and status/error type -- never the raw
request/response translation text. That's a light privacy-hygiene
measure (this isn't the private ALMG corpus -- see
`docs/data-governance.md` -- but there's still no reason to persist
user-submitted text in application logs) and it keeps log volume/size
predictable regardless of how long a submitted text is.

This deliberately covers only the current synchronous `/v1/translate`
request/response cycle. ADR 0008's future async job pattern will need its
own `job_id`-keyed logging when that's actually built (see that ADR's
"Consequences" section) -- not preemptively designed for here.
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


def log_translate_request(
    *,
    direction: str,
    input_length: int,
    latency_ms: float,
    status_code: int,
    error_type: str | None,
) -> None:
    """Log one structured JSON line describing a `/v1/translate` request.

    Successful requests (`error_type is None`) log at INFO; requests that
    ended in an error log at WARNING, so the two are easy to tell apart or
    filter on independently (e.g. for a future log-based CloudWatch metric
    filter, or just `grep`ing local logs).
    """
    payload = {
        "event": "translate_request",
        "direction": direction,
        "input_length": input_length,
        "latency_ms": round(latency_ms, 2),
        "status_code": status_code,
        "error_type": error_type,
    }
    level = logging.INFO if error_type is None else logging.WARNING
    logger.log(level, json.dumps(payload))
