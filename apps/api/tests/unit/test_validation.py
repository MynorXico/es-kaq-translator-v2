from app.validation import translate_validation_error_message


def test_maps_empty_text_error_to_spanish_message():
    errors = [
        {
            "type": "string_too_short",
            "loc": ("body", "text"),
            "msg": "String should have at least 1 character",
        }
    ]
    assert translate_validation_error_message(errors) == "El texto no puede estar vacío."


def test_maps_over_length_text_error_to_spanish_message():
    errors = [
        {
            "type": "string_too_long",
            "loc": ("body", "text"),
            "msg": "String should have at most 2000 characters",
        }
    ]
    assert (
        translate_validation_error_message(errors)
        == "El texto no puede tener más de 2000 caracteres."
    )


def test_maps_invalid_direction_error_to_spanish_message():
    errors = [
        {
            "type": "enum",
            "loc": ("body", "direction"),
            "msg": "Input should be 'es-to-cak' or 'cak-to-es'",
        }
    ]
    assert (
        translate_validation_error_message(errors)
        == "La dirección debe ser 'es-to-cak' o 'cak-to-es'."
    )


def test_falls_back_to_generic_message_for_unrecognized_errors():
    errors = [{"type": "some_other_error", "loc": ("body", "text"), "msg": "..."}]
    assert translate_validation_error_message(errors) == "Los datos enviados no son válidos."
