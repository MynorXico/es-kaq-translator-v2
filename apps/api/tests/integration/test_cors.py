"""Integration tests for CORS configuration (see issue #46 / ADR 0001).

`apps/web` (S3 + CloudFront) and `apps/api` (API Gateway + Lambda) are
deployed as separate origins, so `apps/api` must send the right
`Access-Control-Allow-*` headers for the `apps/web` origins to be able to
call it from a browser. Allowed origins are read from the `ALLOWED_ORIGINS`
env var (comma-separated), defaulting to Vite's local dev server origin
when unset.
"""

import importlib

from fastapi.testclient import TestClient

from app import main as main_module


def _client_with_reloaded_app() -> TestClient:
    # The allowed origins are read once, at import time, so re-import the
    # module after changing the env var to pick up the new configuration.
    importlib.reload(main_module)
    return TestClient(main_module.app)


def test_default_dev_origin_receives_cors_headers(monkeypatch):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    client = _client_with_reloaded_app()

    response = client.get("/health", headers={"Origin": "http://localhost:5173"})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_disallowed_origin_receives_no_cors_headers(monkeypatch):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    client = _client_with_reloaded_app()

    response = client.get("/health", headers={"Origin": "https://not-allowed.example"})

    assert "access-control-allow-origin" not in response.headers


def test_allowed_origins_are_configurable_via_env_var(monkeypatch):
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        "https://configured-one.example,https://configured-two.example",
    )
    client = _client_with_reloaded_app()

    response = client.get("/health", headers={"Origin": "https://configured-two.example"})
    assert response.headers["access-control-allow-origin"] == "https://configured-two.example"

    # The default localhost origin is no longer allowed once ALLOWED_ORIGINS is set.
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert "access-control-allow-origin" not in response.headers


def test_translate_jobs_post_preflight_allows_configured_origin(monkeypatch):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    client = _client_with_reloaded_app()

    response = client.options(
        "/v1/translate-jobs",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_translate_jobs_post_preflight_rejects_disallowed_origin(monkeypatch):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    client = _client_with_reloaded_app()

    response = client.options(
        "/v1/translate-jobs",
        headers={
            "Origin": "https://not-allowed.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert "access-control-allow-origin" not in response.headers


def test_translate_jobs_get_preflight_allows_configured_origin(monkeypatch):
    # ADR 0008: GET /v1/translate-jobs/{job_id} (the poll route) must be
    # explicitly allowed too, not just POST -- main.py's CORSMiddleware
    # allow_methods lists "GET" defensively for this reason.
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    client = _client_with_reloaded_app()

    response = client.options(
        "/v1/translate-jobs/11111111-1111-4111-8111-111111111111",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "GET" in response.headers["access-control-allow-methods"]
