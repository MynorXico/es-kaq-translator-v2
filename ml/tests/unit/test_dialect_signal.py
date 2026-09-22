"""Unit tests for `data.dialect_signal`.

All fixture sentences here are hand-written/synthetic, engineered to have
an obvious, designed orthographic split between two groups -- never the
real private ALMG corpus (ADR 0002 / docs/data-governance.md). They are
not claimed to be linguistically authentic examples of real Kaqchikel
dialect variants; they exist only to exercise the clustering/feature
logic against a signal we know the answer to.
"""

from __future__ import annotations

import unicodedata

import pytest

from data.dialect_signal import (
    DialectSignalReport,
    analyze_dialect_signal,
    build_ngram_vectors,
    character_ngrams,
    feature_frequencies,
    kmeans_two_clusters,
    render_report,
)

# Two small, fully disjoint vocabularies (no shared spelling of any word),
# each turned into several sentences by rotating word order. This isolates
# the thing being tested -- a systematic *orthographic* difference -- from
# semantic/content differences, which would otherwise dominate a character
# n-gram signal over such a tiny corpus. Group A uses apostrophe-marked
# glottalized consonants, central vowel "ä", and doubled ("tense") vowels;
# Group B respells the same made-up word roots without those features and
# with "h" where group A uses "j".
_GROUP_A_WORDS = ["k'o", "chuwäch", "ronojel", "q'ij", "na'oj", "winaqi'"]
_GROUP_B_WORDS = ["ko", "chuwach", "ronohel", "qih", "naoh", "winaqih"]


def _rotations(words: list[str]) -> list[str]:
    return [
        " ".join(words[i:] + words[:i]) + "."
        for i in range(len(words))
    ]


_GROUP_A = _rotations(_GROUP_A_WORDS)
_GROUP_B = _rotations(_GROUP_B_WORDS)


def test_character_ngrams_returns_expected_trigrams():
    result = character_ngrams("k'o", n=3)

    assert result == ["k'o"]


def test_character_ngrams_handles_text_shorter_than_n():
    assert character_ngrams("ab", n=3) == []


def test_feature_frequencies_counts_tense_vowel_doubling():
    freqs = feature_frequencies("Ri qaaw xub'ee ri b'eey.")

    assert freqs["tense_vowel_doubling"] > 0


def test_feature_frequencies_counts_central_vowel_marks():
    freqs = feature_frequencies("Ri qawäch xuk'ïx ri k'öx.")

    assert freqs["central_vowel_marks"] > 0


def test_feature_frequencies_counts_glottalized_apostrophe_consonants():
    freqs = feature_frequencies("Ri k'o' xtz'et jun ch'utin q'apoj.")

    assert freqs["glottal_apostrophe"] > 0


def test_feature_frequencies_counts_word_initial_h():
    freqs = feature_frequencies("Hun hay xukul le hix.")

    assert freqs["word_initial_h"] > 0


def test_feature_frequencies_is_zero_for_text_without_the_pattern():
    freqs = feature_frequencies("xyz xyz xyz")

    assert freqs["tense_vowel_doubling"] == 0
    assert freqs["central_vowel_marks"] == 0
    assert freqs["glottal_apostrophe"] == 0
    assert freqs["word_initial_h"] == 0


def test_feature_frequencies_normalizes_by_length_not_just_raw_count():
    short_text = "k'o "
    long_text = "k'o " * 100

    short_freqs = feature_frequencies(short_text)
    long_freqs = feature_frequencies(long_text)

    # Same density of the pattern -> roughly the same normalized frequency,
    # not a raw count that would scale with repetition count.
    assert short_freqs["glottal_apostrophe"] == pytest.approx(
        long_freqs["glottal_apostrophe"], rel=1e-6
    )


def test_build_ngram_vectors_returns_one_row_per_sentence():
    sentences = _GROUP_A + _GROUP_B

    vocab, vectors = build_ngram_vectors(sentences, n=3, top_k=50)

    assert vectors.shape[0] == len(sentences)
    assert vectors.shape[1] == len(vocab)
    assert len(vocab) <= 50


def test_kmeans_two_clusters_separates_a_designed_two_group_signal():
    sentences = _GROUP_A + _GROUP_B
    _vocab, vectors = build_ngram_vectors(sentences, n=3, top_k=100)

    labels = kmeans_two_clusters(vectors)

    assert len(labels) == len(sentences)
    group_a_labels = set(labels[: len(_GROUP_A)])
    group_b_labels = set(labels[len(_GROUP_A) :])
    # Each engineered group should land entirely in one cluster; which
    # cluster index (0 or 1) is arbitrary, so just check they don't mix.
    assert len(group_a_labels) == 1
    assert len(group_b_labels) == 1
    assert group_a_labels != group_b_labels


def test_analyze_dialect_signal_reports_aggregate_stats_only():
    pairs = [(f"es sentence {i}", cak) for i, cak in enumerate(_GROUP_A + _GROUP_B)]

    report = analyze_dialect_signal(pairs, ngram_n=3, top_k_ngrams=100)

    assert isinstance(report, DialectSignalReport)
    assert report.num_sentences == len(_GROUP_A) + len(_GROUP_B)
    assert sum(report.cluster_sizes) == report.num_sentences
    assert set(report.feature_frequencies_by_cluster.keys()) == {0, 1}
    for freqs in report.feature_frequencies_by_cluster.values():
        assert "tense_vowel_doubling" in freqs
        assert "central_vowel_marks" in freqs
        assert "glottal_apostrophe" in freqs
        assert "word_initial_h" in freqs


def test_analyze_dialect_signal_requires_at_least_two_sentences():
    with pytest.raises(ValueError):
        analyze_dialect_signal([("hola", "utz")], ngram_n=3, top_k_ngrams=10)


# Issue #90: `analyze_dialect_signal` must run `data.normalize.normalize_text`
# over the Kaqchikel side before computing character-level features.
# Without this, precomposed-vs-decomposed Unicode variants of central
# vowel marks (a + combining diaeresis vs. a single precomposed "ä") and
# glottal-stop look-alike codepoints (U+02C8 vs. the ASCII apostrophe)
# would silently confound the `central_vowel_marks`/`glottal_apostrophe`
# feature counts, exactly the same root-cause bug class `data.normalize`
# already exists to fix everywhere else in the pipeline.
def test_analyze_dialect_signal_normalizes_decomposed_central_vowel_marks():
    precomposed = "Ri qawäch xuk'ïx ri k'öx."
    decomposed = unicodedata.normalize("NFD", precomposed)
    assert decomposed != precomposed  # sanity: fixture actually differs pre-normalization

    pairs_precomposed = [("es a", precomposed), ("es b", "xyz xyz xyz.")]
    pairs_decomposed = [("es a", decomposed), ("es b", "xyz xyz xyz.")]

    report_precomposed = analyze_dialect_signal(pairs_precomposed, ngram_n=3, top_k_ngrams=50)
    report_decomposed = analyze_dialect_signal(pairs_decomposed, ngram_n=3, top_k_ngrams=50)

    assert (
        report_precomposed.feature_frequencies_by_cluster
        == report_decomposed.feature_frequencies_by_cluster
    )


def test_analyze_dialect_signal_normalizes_glottal_stop_lookalikes():
    ascii_apostrophe = "Ri k'o' xtz'et jun ch'utin q'apoj."
    lookalike = ascii_apostrophe.replace("'", "ˈ")
    assert lookalike != ascii_apostrophe  # sanity: fixture actually differs pre-normalization

    pairs_ascii = [("es a", ascii_apostrophe), ("es b", "xyz xyz xyz.")]
    pairs_lookalike = [("es a", lookalike), ("es b", "xyz xyz xyz.")]

    report_ascii = analyze_dialect_signal(pairs_ascii, ngram_n=3, top_k_ngrams=50)
    report_lookalike = analyze_dialect_signal(pairs_lookalike, ngram_n=3, top_k_ngrams=50)

    assert (
        report_ascii.feature_frequencies_by_cluster
        == report_lookalike.feature_frequencies_by_cluster
    )


def test_render_report_never_includes_raw_sentence_text():
    pairs = [(f"es sentence {i}", cak) for i, cak in enumerate(_GROUP_A + _GROUP_B)]
    report = analyze_dialect_signal(pairs, ngram_n=3, top_k_ngrams=100)

    rendered = render_report(report)

    assert isinstance(rendered, str)
    assert str(report.num_sentences) in rendered
    for sentence in _GROUP_A + _GROUP_B:
        assert sentence not in rendered
