import json
from unittest.mock import MagicMock

from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

from app import translation as translation_module
from app.main import app

client = TestClient(app)


def _mock_invoke_endpoint(monkeypatch, *, translated_text: str | None = None, error: Exception | None = None):
    """Mocks `boto3`'s `sagemaker-runtime` client at the boundary `app.translation`
    calls it through, per docs/testing.md's "mocked at the boundary" policy for
    apps/api integration tests -- never hits real AWS.
    """
    monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "traductor-kaqchikel-es-cak-test")
    fake_client = MagicMock()
    if error is not None:
        fake_client.invoke_endpoint.side_effect = error
    else:
        body = MagicMock()
        body.read.return_value = json.dumps({"translated_text": translated_text}).encode("utf-8")
        fake_client.invoke_endpoint.return_value = {"Body": body}
    monkeypatch.setattr(translation_module.boto3, "client", lambda *args, **kwargs: fake_client)
    return fake_client


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_translate_returns_the_sagemaker_endpoints_translation(monkeypatch):
    _mock_invoke_endpoint(monkeypatch, translated_text="Utz sq'ij")

    response = client.post(
        "/v1/translate",
        json={"text": "Buenos días", "direction": "es-to-cak"},
    )

    assert response.status_code == 200
    assert response.json() == {"translation": "Utz sq'ij"}


def test_translate_maps_a_model_error_to_a_400_with_a_spanish_message(monkeypatch):
    _mock_invoke_endpoint(
        monkeypatch,
        error=ClientError(
            {"Error": {"Code": "ModelError", "Message": "Received client error (400)"}},
            "InvokeEndpoint",
        ),
    )

    response = client.post(
        "/v1/translate",
        json={"text": "Hola", "direction": "es-to-cak"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "No se pudo traducir el texto enviado. Verifica el contenido e inténtalo de nuevo."
    }


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
