import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_allowed_origins
from app.models import ErrorResponse, TranslateRequest, TranslateResponse
from app.observability import log_translate_request
from app.translation import TranslationServiceError, translate_via_sagemaker
from app.validation import translate_validation_error_message

app = FastAPI(title="Traductor Kaqchikel API", version="0.0.1")

# Allow apps/web (a separate CloudFront/S3 origin, per ADR 0001) to call
# /v1/translate. Restricted to the method/header it actually uses.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_methods=["POST"],
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


@app.exception_handler(TranslationServiceError)
async def translation_service_error_handler(
    request: Request, exc: TranslationServiceError
) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


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


@app.post(
    "/v1/translate",
    response_model=TranslateResponse,
    description=(
        "Translates `text` between Spanish and Kaqchikel by invoking the "
        "deployed SageMaker endpoint, per the `direction` requested.\n\n"
        "Every error response (whichever status code) uses the same "
        "`{\"error\": \"<Spanish message>\"}` shape:\n\n"
        "- **400**: the model itself rejected the input as untranslatable "
        "(e.g. unsupported content) -- not a request-validation failure.\n"
        "- **422**: `text` or `direction` failed request validation (e.g. "
        "empty/over-length text, or an unrecognized `direction`) before "
        "the request ever reached the model.\n"
        "- **502**: the SageMaker endpoint is unavailable, or returned a "
        "response that violates its own documented contract.\n"
        "- **504**: the SageMaker endpoint took too long to respond."
    ),
    responses={
        400: {
            "model": ErrorResponse,
            "description": "The model rejected the submitted text as untranslatable.",
            "content": {
                "application/json": {
                    "example": {
                        "error": (
                            "No se pudo traducir el texto enviado. "
                            "Verifica el contenido e inténtalo de nuevo."
                        )
                    }
                }
            },
        },
        422: {
            "model": ErrorResponse,
            "description": "`text` or `direction` failed request validation.",
            "content": {
                "application/json": {
                    "example": {"error": "El texto no puede estar vacío."}
                }
            },
        },
        502: {
            "model": ErrorResponse,
            "description": "The translation service is unavailable or returned a malformed response.",
            "content": {
                "application/json": {
                    "example": {
                        "error": (
                            "El servicio de traducción no está disponible en este "
                            "momento. Inténtalo más tarde."
                        )
                    }
                }
            },
        },
        504: {
            "model": ErrorResponse,
            "description": "The translation service took too long to respond.",
            "content": {
                "application/json": {
                    "example": {
                        "error": (
                            "El servicio de traducción tardó demasiado en "
                            "responder. Inténtalo de nuevo."
                        )
                    }
                }
            },
        },
    },
)
def translate(request: TranslateRequest) -> TranslateResponse:
    # Structured logging (issue #47) starts here, after request validation
    # (`TranslateRequest`'s own field constraints) has already passed --
    # a malformed request never reaches this function body at all, so it
    # never has a `direction`/`text` worth logging metadata about in the
    # first place. This covers the actual translate call: the thing whose
    # error rate/latency the CloudWatch alarms in infra/cdk care about.
    direction = request.direction.value
    input_length = len(request.text)
    start = time.perf_counter()
    try:
        result = translate_via_sagemaker(request)
    except TranslationServiceError as error:
        log_translate_request(
            direction=direction,
            input_length=input_length,
            latency_ms=(time.perf_counter() - start) * 1000,
            status_code=error.status_code,
            error_type=type(error).__name__,
        )
        raise
    except Exception as error:
        # Anything that isn't a TranslationServiceError -- e.g. a
        # RuntimeError from a misconfigured environment, or a pydantic
        # ValidationError from a malformed upstream SageMaker response --
        # still surfaces as a 500 and still needs to be logged, or the
        # structured logs silently miss exactly the failure modes the
        # CloudWatch alarms exist to catch.
        log_translate_request(
            direction=direction,
            input_length=input_length,
            latency_ms=(time.perf_counter() - start) * 1000,
            status_code=500,
            error_type=type(error).__name__,
        )
        raise

    log_translate_request(
        direction=direction,
        input_length=input_length,
        latency_ms=(time.perf_counter() - start) * 1000,
        status_code=200,
        error_type=None,
    )
    return result
