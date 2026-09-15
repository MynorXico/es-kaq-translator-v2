from data.normalize import normalize_pair, normalize_text


def test_collapses_internal_whitespace():
    assert normalize_text("Utz   awäch") == "Utz awäch"


def test_strips_leading_and_trailing_whitespace():
    assert normalize_text("  Matyox \n") == "Matyox"


def test_normalizes_tabs_and_newlines_to_single_space():
    assert normalize_text("Buenos\tdías\n") == "Buenos días"


def test_applies_unicode_nfc_normalization():
    # "á" as a decomposed sequence (a + combining acute accent, NFD) must
    # become the single precomposed codepoint (NFC) so that visually
    # identical strings compare equal and tokenize consistently.
    decomposed = "Buenos días"  # "d" + "i" + combining acute
    assert normalize_text(decomposed) == "Buenos días"


def test_does_not_change_case():
    # Kaqchikel and Spanish both use capitalization meaningfully (proper
    # nouns, sentence-initial letters); lowercasing would destroy that
    # signal, so normalization must never change case.
    assert normalize_text("Rijaʼ Ixim Achi") == "Rijaʼ Ixim Achi"


def test_normalize_pair_applies_to_both_sides():
    es, cak = normalize_pair("  Hola   mundo  ", "Utz\tawäch\n")
    assert es == "Hola mundo"
    assert cak == "Utz awäch"


def test_normalize_text_empty_string_stays_empty():
    assert normalize_text("   ") == ""
