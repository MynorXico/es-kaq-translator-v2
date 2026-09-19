from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.models import TranslateRequest, TranslateResponse
from app.validation import translate_validation_error_message

app = FastAPI(title="Traductor Kaqchikel API", version="0.0.1")


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
