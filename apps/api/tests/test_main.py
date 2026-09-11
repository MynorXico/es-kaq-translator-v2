from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_translate_stub_returns_placeholder():
    response = client.post(
        "/v1/translate",
        json={"text": "Hola", "direction": "es-to-cak"},
    )
    assert response.status_code == 200
    assert response.json()["translation"]


def test_translate_rejects_empty_text():
    response = client.post(
        "/v1/translate",
        json={"text": "", "direction": "es-to-cak"},
    )
    assert response.status_code == 422
