"""Verifies `/v1/translate-jobs`'s published OpenAPI schema (what `/docs`
renders from) documents realistic examples and the actual contract (ADR
0008). `app.main.app.openapi()` is the same schema FastAPI serves at
`/openapi.json` and renders at `/docs`.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _create_job_operation() -> dict:
    schema = client.get("/openapi.json").json()
    return schema["paths"]["/v1/translate-jobs"]["post"]


def _get_job_operation() -> dict:
    schema = client.get("/openapi.json").json()
    return schema["paths"]["/v1/translate-jobs/{job_id}"]["get"]


def _schema_examples(schema_or_ref: dict, components: dict) -> list:
    """Resolves a (possibly $ref'd) schema's `examples` list."""
    if "$ref" in schema_or_ref:
        name = schema_or_ref["$ref"].rsplit("/", 1)[-1]
        schema_or_ref = components["schemas"][name]
    return schema_or_ref.get("examples", [])


def _media_type_examples(media_type_object: dict, components: dict) -> list:
    """Resolves the effective example value(s) for an OpenAPI media type
    object, checking the per-response `example`/`examples` override before
    falling back to the referenced schema's own `examples`.
    """
    if "example" in media_type_object:
        return [media_type_object["example"]]
    if "examples" in media_type_object:
        return [named["value"] for named in media_type_object["examples"].values()]
    return _schema_examples(media_type_object["schema"], components)


def test_create_translate_job_request_body_has_a_realistic_example():
    operation = _create_job_operation()
    components = client.get("/openapi.json").json()["components"]
    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]

    examples = _schema_examples(request_schema, components)

    assert examples, "expected the TranslateRequest schema to document an example"
    assert examples[0] == {"text": "Buenos días", "direction": "es-to-cak"}


def test_create_translate_job_responds_202_with_a_job_id_example():
    operation = _create_job_operation()
    components = client.get("/openapi.json").json()["components"]

    assert "202" in operation["responses"], "expected a documented 202 response"
    response_schema = operation["responses"]["202"]["content"]["application/json"]["schema"]

    examples = _schema_examples(response_schema, components)

    assert examples, "expected the JobCreatedResponse schema to document an example"
    assert "job_id" in examples[0]


def test_create_translate_job_documents_the_validation_error_response():
    operation = _create_job_operation()
    components = client.get("/openapi.json").json()["components"]

    response = operation["responses"]["422"]
    schema = response["content"]["application/json"]["schema"]
    resolved = schema
    if "$ref" in schema:
        resolved = components["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
    assert "error" in resolved["properties"]

    media_type_object = response["content"]["application/json"]
    examples = _media_type_examples(media_type_object, components)
    assert examples, "expected the 422 response to document an example"
    assert examples[0] == {"error": "El texto no puede estar vacío."}


def test_get_translate_job_response_documents_all_three_status_shapes():
    operation = _get_job_operation()
    components = client.get("/openapi.json").json()["components"]
    response_schema = operation["responses"]["200"]["content"]["application/json"]["schema"]

    examples = _schema_examples(response_schema, components)

    assert examples, "expected the JobStatusResponse schema to document examples"
    statuses = {example["status"] for example in examples}
    assert statuses == {"pending", "succeeded", "failed"}

    succeeded_example = next(e for e in examples if e["status"] == "succeeded")
    assert "translation" in succeeded_example

    failed_example = next(e for e in examples if e["status"] == "failed")
    assert "status_code" in failed_example
    assert "error" in failed_example


def test_get_translate_job_documents_the_400_and_404_responses():
    operation = _get_job_operation()
    components = client.get("/openapi.json").json()["components"]
    responses = operation["responses"]

    for status_code in ("400", "404"):
        assert status_code in responses, f"expected a documented {status_code} response"
        response = responses[status_code]
        schema = response["content"]["application/json"]["schema"]
        resolved = schema
        if "$ref" in schema:
            resolved = components["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
        assert "error" in resolved["properties"]

        media_type_object = response["content"]["application/json"]
        examples = _media_type_examples(media_type_object, components)
        assert examples, f"expected the {status_code} response to document an example"
        assert "error" in examples[0]


def test_get_translate_job_operation_description_mentions_what_each_error_status_means():
    operation = _get_job_operation()
    description = operation.get("description", "")

    for status_code in ("400", "404"):
        assert status_code in description, (
            f"expected the endpoint description to explain the {status_code} response"
        )


def test_old_synchronous_translate_route_is_gone():
    # ADR 0008: the old POST /v1/translate synchronous contract is retired
    # entirely, not left documented alongside the new routes.
    schema = client.get("/openapi.json").json()
    assert "/v1/translate" not in schema["paths"]
