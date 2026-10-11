from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Direction(str, Enum):
    ES_TO_CAK = "es-to-cak"
    CAK_TO_ES = "cak-to-es"


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    direction: Direction

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"text": "Buenos días", "direction": "es-to-cak"}]}
    )


class TranslateResponse(BaseModel):
    translation: str

    model_config = ConfigDict(json_schema_extra={"examples": [{"translation": "Utz sq'ij"}]})


class ErrorResponse(BaseModel):
    """The shape every error response uses, whether it comes from request
    validation (`app.validation`), a malformed/unknown job lookup
    (`app.main.get_translate_job`), or (historically) a failure talking to
    the SageMaker endpoint directly (`app.translation.TranslationServiceError`,
    now only ever raised inside `app.worker_handler`, not a live FastAPI route).
    """

    error: str = Field(description="Spanish, user-facing description of what went wrong.")


class JobStatus(str, Enum):
    """A `TranslateJobsTable` item's lifecycle state (ADR 0008)."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobCreatedResponse(BaseModel):
    """`POST /v1/translate-jobs`'s `202 Accepted` body (ADR 0008)."""

    job_id: str = Field(description="Opaque job identifier to poll via GET /v1/translate-jobs/{job_id}.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"}],
        }
    )


class JobStatusResponse(BaseModel):
    """`GET /v1/translate-jobs/{job_id}`'s body (ADR 0008).

    Always a `200`; `status` discriminates which of `translation` (only set
    when `succeeded`) or `status_code`/`error` (only set when `failed`) is
    present. Served with `response_model_exclude_none=True` so a `pending`
    response is exactly `{"status": "pending"}`, not padded with explicit
    `null` fields for the other two shapes.
    """

    status: JobStatus
    translation: str | None = Field(
        default=None, description="Only present when status is 'succeeded'."
    )
    status_code: int | None = Field(
        default=None,
        description=(
            "Only present when status is 'failed' -- the numeric HTTP status "
            "apps/api would have returned synchronously for this failure."
        ),
    )
    error: str | None = Field(
        default=None, description="Only present when status is 'failed'."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"status": "pending"},
                {"status": "succeeded", "translation": "Utz sq'ij"},
                {
                    "status": "failed",
                    "status_code": 502,
                    "error": (
                        "El servicio de traducción no está disponible en este "
                        "momento. Inténtalo más tarde."
                    ),
                },
            ],
        }
    )
