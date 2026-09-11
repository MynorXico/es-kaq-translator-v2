from enum import Enum

from pydantic import BaseModel, Field


class Direction(str, Enum):
    ES_TO_CAK = "es-to-cak"
    CAK_TO_ES = "cak-to-es"


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    direction: Direction


class TranslateResponse(BaseModel):
    translation: str
