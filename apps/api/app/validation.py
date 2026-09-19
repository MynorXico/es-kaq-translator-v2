"""Maps pydantic/FastAPI request validation errors to Spanish, user-facing messages."""

from collections.abc import Iterable, Mapping
from typing import Any

_FIELD_MESSAGES: dict[tuple[str, str], str] = {
    ("text", "string_too_short"): "El texto no puede estar vacío.",
    ("text", "string_too_long"): "El texto no puede tener más de 2000 caracteres.",
    ("direction", "enum"): "La dirección debe ser 'es-to-cak' o 'cak-to-es'.",
}

_FALLBACK_MESSAGE = "Los datos enviados no son válidos."


def translate_validation_error_message(errors: Iterable[Mapping[str, Any]]) -> str:
    """Return a single Spanish message for the first recognized validation error.

    Falls back to a generic Spanish message when the error doesn't match one
    of the known cases (empty text, over-length text, invalid direction).
    """
    for error in errors:
        loc = error.get("loc", ())
        field = loc[-1] if loc else None
        message = _FIELD_MESSAGES.get((field, error.get("type")))
        if message:
            return message
    return _FALLBACK_MESSAGE
