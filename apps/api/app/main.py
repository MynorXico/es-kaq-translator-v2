from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_allowed_origins
from app.models import ErrorResponse, TranslateRequest, TranslateResponse
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
    return translate_via_sagemaker(request)
