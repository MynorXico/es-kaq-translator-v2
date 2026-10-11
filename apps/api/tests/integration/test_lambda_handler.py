"""Integration tests for the AWS Lambda entry point (`app.lambda_handler`).

`apps/api` is deployed as a Lambda container image behind a regional API
Gateway REST API (v1), per ADR 0001/ADR 0005 and issue #151 (previously an
HTTP API, v2, payload format 2.0 -- see issue #96 and `lambda_handler`'s
own docstring for why that changed). `Mangum` adapts API Gateway proxy
events into ASGI calls against the same `app.main.app` instance the rest
of the test suite already exercises via `TestClient` -- these tests
instead drive it through synthetic API Gateway events, the actual shapes
Lambda receives in production, to confirm the adapter wiring itself works
end-to-end.

Both the current REST API v1 event shape (`_rest_api_v1_event`, what the
real deployed stack sends since issue #151) and the prior HTTP API v2
shape (`_http_api_v2_event`) are covered here: `Mangum` infers which
handler to use per-event (see `mangum.handlers.api_gateway.APIGateway.infer`
and `mangum.handlers.http_gateway.HTTPGateway.infer`), so verifying only
one shape wouldn't prove the other still works if `Mangum` or its
inference ever changed -- and keeping the v2 coverage also guards against
a silent regression for `LambdaAtEdge`/`ALB`-style deployments this app
doesn't currently use but that share the same `handler` entry point.
"""

import json
from typing import Any
from unittest.mock import MagicMock

from app.lambda_handler import handler
from tests._fake_dynamodb import FakeDynamoDBResource


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


def _rest_api_v1_event(
    method: str, path: str, body: str | None = None, headers: dict[str, str] | None = None
) -> dict[str, Any]:
    """A minimal, realistic API Gateway REST API (v1) proxy event.

    Shaped after a `{proxy+}` greedy-resource, `AWS_PROXY`-integration
    request -- what `infra/cdk/lib/api-stack.ts`'s `LambdaRestApi` (issue
    #151) actually sends. `resource` + `requestContext` (no top-level
    `version` key) is exactly what `mangum.handlers.api_gateway.APIGateway
    .infer` checks for to select this handler over the HTTP API v2 one.
    """
    return {
        "resource": "/{proxy+}",
        "path": path,
        "httpMethod": method,
        "headers": headers or ({"content-type": "application/json"} if body else None),
        "multiValueHeaders": None,
        "queryStringParameters": None,
        "multiValueQueryStringParameters": None,
        "pathParameters": {"proxy": path.lstrip("/")},
        "stageVariables": None,
        "requestContext": {
            "resourcePath": "/{proxy+}",
            "httpMethod": method,
            "path": f"/test{path}",
            "stage": "test",
            "identity": {"sourceIp": "127.0.0.1"},
            "domainName": "example.execute-api.us-east-1.amazonaws.com",
        },
        "isBase64Encoded": False,
        "body": body,
    }


def test_health_via_lambda_handler():
    event = _http_api_v2_event("GET", "/health")

    response = handler(event, None)

    assert response["statusCode"] == 200
    assert json.loads(response["body"]) == {"status": "ok"}


def test_health_via_lambda_handler_rest_api_v1_event():
    # Issue #151/ADR 0005: the real, currently-deployed event shape, once
    # ApiStack migrated off HttpApi to a WAF-associable RestApi.
    event = _rest_api_v1_event("GET", "/health")

    response = handler(event, None)

    assert response["statusCode"] == 200
    assert json.loads(response["body"]) == {"status": "ok"}


def _mock_jobs_boundary(monkeypatch) -> FakeDynamoDBResource:
    # Mocked at the same boundary as tests/integration/test_translate_jobs.py
    # -- app.jobs's own DynamoDB resource / Lambda client, never real AWS.
    monkeypatch.setenv("TRANSLATE_JOBS_TABLE_NAME", "traductor-kaqchikel-translate-jobs-test")
    monkeypatch.setenv("WORKER_FUNCTION_NAME", "traductor-kaqchikel-translate-worker-test")
    resource = FakeDynamoDBResource()
    monkeypatch.setattr("app.jobs.boto3.resource", lambda *args, **kwargs: resource)
    monkeypatch.setattr("app.jobs.boto3.client", lambda *args, **kwargs: MagicMock())
    return resource


def test_create_translate_job_via_lambda_handler(monkeypatch):
    _mock_jobs_boundary(monkeypatch)

    body = json.dumps({"text": "Hola", "direction": "es-to-cak"})
    event = _http_api_v2_event("POST", "/v1/translate-jobs", body=body)

    response = handler(event, None)

    assert response["statusCode"] == 202
    assert "job_id" in json.loads(response["body"])


def test_create_translate_job_via_lambda_handler_rest_api_v1_event(monkeypatch):
    # Same as test_create_translate_job_via_lambda_handler, but through the
    # real (post-#151) REST API v1 event shape -- the actual JSON body, path
    # routing, and response Mangum produces for a POST all need to keep
    # working, not just GET /health's simpler, bodyless request.
    _mock_jobs_boundary(monkeypatch)

    body = json.dumps({"text": "Hola", "direction": "es-to-cak"})
    event = _rest_api_v1_event("POST", "/v1/translate-jobs", body=body)

    response = handler(event, None)

    assert response["statusCode"] == 202
    assert "job_id" in json.loads(response["body"])


def test_create_translate_job_validation_error_via_lambda_handler(monkeypatch):
    _mock_jobs_boundary(monkeypatch)

    body = json.dumps({"text": "", "direction": "es-to-cak"})
    event = _http_api_v2_event("POST", "/v1/translate-jobs", body=body)

    response = handler(event, None)

    assert response["statusCode"] == 422
    assert json.loads(response["body"]) == {"error": "El texto no puede estar vacío."}


def test_create_translate_job_validation_error_via_lambda_handler_rest_api_v1_event(monkeypatch):
    _mock_jobs_boundary(monkeypatch)

    body = json.dumps({"text": "", "direction": "es-to-cak"})
    event = _rest_api_v1_event("POST", "/v1/translate-jobs", body=body)

    response = handler(event, None)

    assert response["statusCode"] == 422
    assert json.loads(response["body"]) == {"error": "El texto no puede estar vacío."}


def test_get_translate_job_via_lambda_handler_rest_api_v1_event(monkeypatch):
    # GET (ADR 0008) must also route correctly through Mangum's REST API v1
    # path-parameter handling, not just POST -- the real stack serves both
    # /v1/translate-jobs routes off the same {proxy+} resource.
    resource = _mock_jobs_boundary(monkeypatch)
    job_id = "11111111-1111-4111-8111-111111111111"
    resource.table.seed({"job_id": job_id, "status": "pending", "ttl": 9999999999})

    event = _rest_api_v1_event("GET", f"/v1/translate-jobs/{job_id}")

    response = handler(event, None)

    assert response["statusCode"] == 200
    assert json.loads(response["body"]) == {"status": "pending"}
