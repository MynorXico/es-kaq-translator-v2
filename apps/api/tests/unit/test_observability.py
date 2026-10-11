"""Unit tests for `app.observability`'s structured job-resolution logging
(ADR 0008, issue #47).

Pure logic, no ASGI app/HTTP layer involved -- exercises
`log_job_resolution` directly, using `caplog` to inspect the emitted log
record the same way pytest's own docs recommend.
"""

import json
import logging

from app.observability import log_job_resolution


def test_logs_a_single_json_line_with_the_required_metadata_fields(caplog):
    with caplog.at_level(logging.INFO, logger="app.request"):
        log_job_resolution(
            job_id="11111111-1111-4111-8111-111111111111",
            direction="es-to-cak",
            input_length=42,
            latency_ms=123.456,
            status_code=200,
            error_type=None,
        )

    records = [r for r in caplog.records if r.name == "app.request"]
    assert len(records) == 1

    payload = json.loads(records[0].message)
    assert payload["job_id"] == "11111111-1111-4111-8111-111111111111"
    assert payload["direction"] == "es-to-cak"
    assert payload["input_length"] == 42
    assert payload["status_code"] == 200
    assert payload["error_type"] is None
    assert isinstance(payload["latency_ms"], (int, float))


def test_logs_at_warning_level_and_includes_the_error_type_on_failure(caplog):
    with caplog.at_level(logging.INFO, logger="app.request"):
        log_job_resolution(
            job_id="22222222-2222-4222-8222-222222222222",
            direction="cak-to-es",
            input_length=5,
            latency_ms=10.0,
            status_code=502,
            error_type="TranslationServiceError",
        )

    records = [r for r in caplog.records if r.name == "app.request"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING

    payload = json.loads(records[0].message)
    assert payload["status_code"] == 502
    assert payload["error_type"] == "TranslationServiceError"


def test_supports_an_unknown_latency_for_the_onfailure_destination_path(caplog):
    # app.failure_handler has no visibility into how long the Worker Lambda
    # actually ran before being killed -- latency_ms must be allowed to be
    # None there, not force a fabricated number.
    with caplog.at_level(logging.INFO, logger="app.request"):
        log_job_resolution(
            job_id="33333333-3333-4333-8333-333333333333",
            direction=None,
            input_length=None,
            latency_ms=None,
            status_code=504,
            error_type="WorkerTimeoutOrCrash",
        )

    payload = json.loads(caplog.records[0].message)
    assert payload["latency_ms"] is None
    assert payload["direction"] is None
    assert payload["input_length"] is None


def test_never_logs_the_raw_request_or_translation_text(caplog):
    secret_input = "un texto que nunca debe aparecer en los logs"
    secret_output = "jun tzij ri man tikirel ta yalöx pa log"

    with caplog.at_level(logging.INFO, logger="app.request"):
        log_job_resolution(
            job_id="44444444-4444-4444-8444-444444444444",
            direction="es-to-cak",
            input_length=len(secret_input),
            latency_ms=1.0,
            status_code=200,
            error_type=None,
        )

    assert secret_input not in caplog.text
    assert secret_output not in caplog.text
    # Only metadata keys should ever be present -- guards against someone
    # later adding a `text`/`translation` field to the payload by mistake.
    payload = json.loads(caplog.records[0].message)
    assert set(payload.keys()) == {
        "event",
        "job_id",
        "direction",
        "input_length",
        "latency_ms",
        "status_code",
        "error_type",
    }
