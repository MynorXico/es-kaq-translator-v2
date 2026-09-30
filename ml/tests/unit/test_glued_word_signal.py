"""Unit tests for `data.glued_word_signal` (issue #199).

All fixture text here is hand-written/synthetic -- never the real private
ALMG corpus (ADR 0002 / docs/data-governance.md). Word examples mirror the
*shape* of the concrete OCR/digitization glued-word artifacts issue
#116/#125 found in generated model output ("dedios", "queestá",
"deguatemala", ...), without reusing any real corpus content.
"""

from __future__ import annotations

from data.glued_word_signal import (
    DEFAULT_GLUE_PREFIXES,
    DEFAULT_KNOWN_GLUED_PATTERNS,
    GluedWordReport,
    build_word_frequencies,
    find_glued_word_candidate,
    scan_corpus_for_glued_words,
    scan_known_patterns,
)


def test_build_word_frequencies_counts_case_insensitively():
    freqs = build_word_frequencies(["Dios es bueno.", "dios dios."])

    assert freqs["dios"] == 3


def test_scan_known_patterns_counts_zero_when_absent():
    sentences = ["Buenos días.", "La utz awäch?"]

    counts = scan_known_patterns(sentences, patterns=("dedios", "queestá"))

    assert counts == {"dedios": 0, "queestá": 0}


def test_scan_known_patterns_counts_occurrences_case_insensitively():
    sentences = ["Esto es DeDios.", "Otra frase sin el patrón."]

    counts = scan_known_patterns(sentences, patterns=("dedios",))

    assert counts["dedios"] == 1


def test_default_known_glued_patterns_matches_issue_116_examples():
    # These are the exact strings issue #116 found in generated output --
    # kept as the default sweep target, not reinvented.
    assert "dedios" in DEFAULT_KNOWN_GLUED_PATTERNS
    assert "deguatemala" in DEFAULT_KNOWN_GLUED_PATTERNS


def test_find_glued_word_candidate_flags_a_rare_word_splitting_into_two_frequent_words():
    # "unpalabra" ("un" + "palabra") never otherwise appears, while both
    # "un" (as a prefix/clitic) and "palabra" are common in the fixture
    # vocabulary -- the designed positive case.
    frequencies = build_word_frequencies(
        ["palabra palabra palabra palabra palabra", "unpalabra"]
    )

    candidate = find_glued_word_candidate(
        "unpalabra",
        frequencies,
        prefixes=("un",),
        max_word_frequency=1,
        min_suffix_frequency=3,
        min_suffix_length=3,
    )

    assert candidate == "un"


def test_find_glued_word_candidate_ignores_frequent_whole_words():
    # "deporte" is an ordinary, frequent Spanish word in this fixture --
    # must not be flagged just because it starts with the clitic "de".
    frequencies = build_word_frequencies(
        ["deporte deporte deporte", "porte porte porte porte"]
    )

    candidate = find_glued_word_candidate(
        "deporte",
        frequencies,
        prefixes=("de",),
        max_word_frequency=1,
        min_suffix_frequency=3,
        min_suffix_length=3,
    )

    assert candidate is None


def test_find_glued_word_candidate_ignores_short_suffixes():
    frequencies = build_word_frequencies(["dea dea dea dea", "dea"])

    candidate = find_glued_word_candidate(
        "dea",
        frequencies,
        prefixes=("de",),
        max_word_frequency=5,
        min_suffix_frequency=1,
        min_suffix_length=3,
    )

    assert candidate is None


def test_scan_corpus_for_glued_words_returns_aggregate_counts_only():
    pairs = [
        ("Esto es dedios verdad.", "Ri tzij."),
        ("Una frase normal.", "Jun tzij chik."),
    ]

    report = scan_corpus_for_glued_words(pairs, known_patterns=("dedios",))

    assert isinstance(report, GluedWordReport)
    assert report.num_sentences_scanned == 2
    assert report.known_pattern_counts == {"dedios": 1}
    assert isinstance(report.candidate_counts_by_prefix, dict)


def test_default_glue_prefixes_are_common_spanish_clitics():
    assert "de" in DEFAULT_GLUE_PREFIXES
    assert "que" in DEFAULT_GLUE_PREFIXES
