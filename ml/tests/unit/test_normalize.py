from data.normalize import normalize_glottal_marks, normalize_pair, normalize_text


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


# Issue #90: aggregate measurement of the real ALMG corpus (32,906
# sentences, described only via counts/frequencies -- ADR 0002) confirmed
# U+02C8 (MODIFIER LETTER VERTICAL LINE, "ˈ") is used as a glottal-stop/
# apostrophe look-alike in a substantial minority of the corpus, standing
# in for the ASCII apostrophe (U+0027) that ALMG orthography and the
# corpus's majority convention actually use. U+02BC (MODIFIER LETTER
# APOSTROPHE) and U+2019 (RIGHT SINGLE QUOTATION MARK) were also candidates
# per the issue, but a real-corpus scan found zero occurrences of either --
# so, per the issue's explicit "verify before assuming" guidance, neither
# is normalized here. Only fixtures below use these codepoints; the real
# corpus is never read by this test suite (docs/testing.md).


def test_normalize_glottal_marks_maps_modifier_letter_vertical_line_to_ascii_apostrophe():
    assert normalize_glottal_marks("kˈo") == "k'o"


def test_normalize_glottal_marks_leaves_ascii_apostrophe_unchanged():
    assert normalize_glottal_marks("k'o") == "k'o"


def test_normalize_glottal_marks_leaves_unconfirmed_lookalikes_unchanged():
    # U+02BC and U+2019: candidates the issue named but confirmed *absent*
    # from the real corpus -- left untouched rather than guessed at.
    assert normalize_glottal_marks("kʼo") == "kʼo"
    assert normalize_glottal_marks("k’o") == "k’o"


def test_normalize_glottal_marks_handles_multiple_occurrences():
    assert normalize_glottal_marks("chˈabäl, tzˈi7, qˈij") == "ch'abäl, tz'i7, q'ij"


def test_normalize_text_normalizes_glottal_lookalikes():
    # normalize_text (used by the corpus-cleaning pipeline) must apply this
    # mapping too, not just the standalone helper.
    assert normalize_text("Ütz  kˈo  ri  jun") == "Ütz kˈo ri jun".replace("ˈ", "'")


def test_normalize_pair_normalizes_glottal_lookalikes_on_both_sides():
    es, cak = normalize_pair("kˈo (es)", "kˈo (cak)")
    assert es == "k'o (es)"
    assert cak == "k'o (cak)"
