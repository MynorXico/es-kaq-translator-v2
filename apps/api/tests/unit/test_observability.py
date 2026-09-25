"""Unit tests for `app.observability`'s structured request logging (#47).

Pure logic, no ASGI app/HTTP layer involved -- exercises
`log_translate_request` directly, using `caplog` to inspect the emitted
log record the same way pytest's own docs recommend.
"""

import json
import logging

from app.observability import log_translate_request


def test_logs_a_single_json_line_with_the_required_metadata_fields(caplog):
    with caplog.at_level(logging.INFO, logger="app.request"):
        log_translate_request(
            direction="es-to-cak",
            input_length=42,
            latency_ms=123.456,
            status_code=200,
            error_type=None,
        )

    records = [r for r in caplog.records if r.name == "app.request"]
    assert len(records) == 1

    payload = json.loads(records[0].message)
    assert payload["direction"] == "es-to-cak"
    assert payload["input_length"] == 42
    assert payload["status_code"] == 200
    assert payload["error_type"] is None
    assert isinstance(payload["latency_ms"], (int, float))


def test_logs_at_warning_level_and_includes_the_error_type_on_failure(caplog):
    with caplog.at_level(logging.INFO, logger="app.request"):
        log_translate_request(
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


def test_never_logs_the_raw_request_or_translation_text(caplog):
    secret_input = "un texto que nunca debe aparecer en los logs"
    secret_output = "jun tzij ri man tikirel ta yalöx pa log"

    with caplog.at_level(logging.INFO, logger="app.request"):
        log_translate_request(
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
        "direction",
        "input_length",
        "latency_ms",
        "status_code",
        "error_type",
    }
