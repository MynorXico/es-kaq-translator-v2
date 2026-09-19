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
    assert response.json() == {"error": "El texto no puede estar vacío."}


def test_translate_rejects_text_over_length_limit():
    response = client.post(
        "/v1/translate",
        json={"text": "a" * 2001, "direction": "es-to-cak"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": "El texto no puede tener más de 2000 caracteres."}


def test_translate_rejects_invalid_direction():
    response = client.post(
        "/v1/translate",
        json={"text": "Hola", "direction": "en-to-fr"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": "La dirección debe ser 'es-to-cak' o 'cak-to-es'."}
