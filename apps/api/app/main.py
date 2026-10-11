import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_allowed_origins
from app.jobs import create_job, get_job, invoke_worker
from app.models import (
    ErrorResponse,
    JobCreatedResponse,
    JobStatus,
    JobStatusResponse,
    TranslateRequest,
)
from app.validation import translate_validation_error_message

app = FastAPI(title="Traductor Kaqchikel API", version="0.0.1")

# Allow apps/web (a separate CloudFront/S3 origin, per ADR 0001) to call
# /v1/translate-jobs. GET is explicitly listed (ADR 0008) even though a
# plain cross-origin GET with no custom headers is a CORS "simple request"
# that wouldn't actually be blocked by this on its own (no preflight is
# triggered, and allow_methods only governs the preflight response) --
# still correct to list it defensively rather than rely on that nuance.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": translate_validation_error_message(exc.errors())},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Anything not already handled by a more specific handler above (e.g.
    # a misconfigured environment, or a malformed upstream response) --
    # still logged by the route itself (see /v1/translate), this just
    # gives it a real 500 response instead of an unhandled-exception
    # traceback reaching the client.
    return JSONResponse(status_code=500, content={"error": "Ocurrió un error inesperado."})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Spanish, user-facing messages for the two GET-only failure cases this
# route introduces (ADR 0008) -- distinct from each other and from
# app.translation's messages, which only ever apply inside app.worker_handler
# now, not on a live FastAPI route's own exception path.
_INVALID_JOB_ID_MESSAGE = "El identificador del trabajo no es válido."
_JOB_NOT_FOUND_MESSAGE = (
    "No se encontró esa traducción. Puede que el enlace sea incorrecto o que haya expirado."
)


@app.post(
    "/v1/translate-jobs",
    status_code=202,
    response_model=JobCreatedResponse,
    description=(
        "Enqueues a translation job between Spanish and Kaqchikel, per the "
        "`direction` requested (ADR 0008). Returns immediately with a "
        "`job_id` to poll via `GET /v1/translate-jobs/{job_id}` -- this "
        "route never calls the SageMaker endpoint itself (that happens "
        "asynchronously, in a separate Worker Lambda), so it isn't subject "
        "to API Gateway's 29-second integration timeout the way a cold "
        "SageMaker Serverless Inference start could be.\n\n"
        "- **422**: `text` or `direction` failed request validation (e.g. "
        "empty/over-length text, or an unrecognized `direction`)."
    ),
    responses={
        422: {
            "model": ErrorResponse,
            "description": "`text` or `direction` failed request validation.",
            "content": {
                "application/json": {"example": {"error": "El texto no puede estar vacío."}}
            },
        },
    },
)
def create_translate_job(request: TranslateRequest) -> JobCreatedResponse:
    job_id = create_job(request)
    invoke_worker(job_id)
    return JobCreatedResponse(job_id=job_id)


@app.get(
    "/v1/translate-jobs/{job_id}",
    response_model=JobStatusResponse,
    response_model_exclude_none=True,
    description=(
        "Polls a translation job enqueued via `POST /v1/translate-jobs` "
        "(ADR 0008). Always responds `200`; `status` discriminates between "
        "`pending`, `succeeded` (with `translation`), and `failed` (with "
        "`status_code`/`error` -- the numeric HTTP status and Spanish "
        "message this API would have returned synchronously for that "
        "failure, carried explicitly since the outer HTTP response is "
        "always `200` here).\n\n"
        "- **400**: `job_id` is not a well-formed UUID.\n"
        "- **404**: `job_id` is well-formed but unknown, or has expired."
    ),
    responses={
        400: {
            "model": ErrorResponse,
            "description": "`job_id` is not a well-formed UUID.",
            "content": {"application/json": {"example": {"error": _INVALID_JOB_ID_MESSAGE}}},
        },
        404: {
            "model": ErrorResponse,
            "description": "`job_id` is well-formed but unknown or has expired.",
            "content": {"application/json": {"example": {"error": _JOB_NOT_FOUND_MESSAGE}}},
        },
    },
)
def get_translate_job(job_id: str) -> JobStatusResponse | JSONResponse:
    try:
        uuid.UUID(job_id)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": _INVALID_JOB_ID_MESSAGE})

    job = get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": _JOB_NOT_FOUND_MESSAGE})

    status = job["status"]
    if status == JobStatus.SUCCEEDED.value:
        return JobStatusResponse(status=JobStatus.SUCCEEDED, translation=job["translation"])
    if status == JobStatus.FAILED.value:
        return JobStatusResponse(
            status=JobStatus.FAILED,
            status_code=int(job["status_code"]),
            error=job["error"],
        )
    return JobStatusResponse(status=JobStatus.PENDING)
