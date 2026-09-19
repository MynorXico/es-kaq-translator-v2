from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_allowed_origins
from app.models import TranslateRequest, TranslateResponse
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest) -> TranslateResponse:
    # Stub: will call the SageMaker Serverless Inference endpoint once trained (ADR 0001).
    return TranslateResponse(translation="Esta función estará disponible próximamente.")
