from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_allowed_origins
from app.models import TranslateRequest, TranslateResponse

app = FastAPI(title="Traductor Kaqchikel API", version="0.0.1")

# Allow apps/web (a separate CloudFront/S3 origin, per ADR 0001) to call
# /v1/translate. Restricted to the method/header it actually uses.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest) -> TranslateResponse:
    # Stub: will call the SageMaker Serverless Inference endpoint once trained (ADR 0001).
    return TranslateResponse(translation="Esta función estará disponible próximamente.")
