from fastapi import FastAPI

from app.models import TranslateRequest, TranslateResponse

app = FastAPI(title="Traductor Kaqchikel API", version="0.0.1")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest) -> TranslateResponse:
    # Stub: will call the SageMaker Serverless Inference endpoint once trained (ADR 0001).
    return TranslateResponse(translation="Esta función estará disponible próximamente.")
