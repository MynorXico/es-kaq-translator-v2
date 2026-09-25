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
    """The shape every error response from `/v1/translate` uses, whether it
    comes from request validation (`app.validation`) or from a failure
    talking to the SageMaker endpoint (`app.translation.TranslationServiceError`).
    """

    error: str = Field(description="Spanish, user-facing description of what went wrong.")
