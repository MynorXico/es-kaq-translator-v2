"""Integration tests for the AWS Lambda entry point (`app.lambda_handler`).

`apps/api` is deployed as a Lambda container image behind an API Gateway
HTTP API (payload format 2.0), per ADR 0001 and issue #96. `Mangum`
adapts API Gateway proxy events into ASGI calls against the same
`app.main.app` instance the rest of the test suite already exercises via
`TestClient` -- these tests instead drive it through synthetic API
Gateway v2 events, the actual shape Lambda receives in production, to
confirm the adapter wiring itself works end-to-end.
"""

import json
from typing import Any

from app.lambda_handler import handler


def _http_api_v2_event(
    method: str, path: str, body: str | None = None, headers: dict[str, str] | None = None
) -> dict[str, Any]:
    """A minimal, realistic API Gateway HTTP API v2 proxy event."""
    return {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": path,
        "rawQueryString": "",
        "headers": headers or ({"content-type": "application/json"} if body else {}),
        "requestContext": {
            "http": {
                "method": method,
                "path": path,
                "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1",
            },
            "domainName": "example.execute-api.us-east-1.amazonaws.com",
            "stage": "$default",
        },
        "isBase64Encoded": False,
        "body": body,
    }


def test_health_via_lambda_handler():
    event = _http_api_v2_event("GET", "/health")

    response = handler(event, None)

    assert response["statusCode"] == 200
    assert json.loads(response["body"]) == {"status": "ok"}


def test_translate_stub_via_lambda_handler():
    body = json.dumps({"text": "Hola", "direction": "es-to-cak"})
    event = _http_api_v2_event("POST", "/v1/translate", body=body)

    response = handler(event, None)

    assert response["statusCode"] == 200
    assert json.loads(response["body"])["translation"]


def test_translate_validation_error_via_lambda_handler():
    body = json.dumps({"text": "", "direction": "es-to-cak"})
    event = _http_api_v2_event("POST", "/v1/translate", body=body)

    response = handler(event, None)

    assert response["statusCode"] == 422
    assert json.loads(response["body"]) == {"error": "El texto no puede estar vacío."}
