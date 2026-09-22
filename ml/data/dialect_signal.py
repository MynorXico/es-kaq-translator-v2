"""Cheap, unsupervised dialect-variant signal detection for the Kaqchikel
side of the corpus (issue #51).

**What this is**: a diagnostic tool, not a labeling authority. It computes
two kinds of purely computational signal over the Kaqchikel side of a
corpus of sentence pairs:

1. An unsupervised 2-cluster split over character n-gram frequencies
   (`build_ngram_vectors` + `kmeans_two_clusters`) -- a cheap way to check
   whether the corpus's Kaqchikel text separates into two orthographically
   distinct groups at all, without assuming anything about what those
   groups mean.
2. Frequency counts for a handful of orthographic features known from
   Kaqchikel dialectology/orthography to vary across dialects or across
   pre-ALMG vs. ALMG-standard spelling conventions (`feature_frequencies`):
   tense/long vowel doubling (e.g. "aa", "ee"), central vowel marks ("ä",
   "ë", "ï", "ö", "ü"), apostrophe-marked glottalized consonants (e.g.
   "k'", "tz'"), and word-initial "h" (a historical proxy for older
   Mayan-language orthographies that used "h" where the modern ALMG
   standard uses "j" for the same sound). See the `ml/README.md` section
   on issue #51 for citations and caveats.

**What this is not**: authoritative dialect labeling. An unsupervised
n-gram cluster split can just as easily reflect topic, register, or
source-document differences as genuine dialect variation, and the feature
list above is a hypothesis grounded in general Kaqchikel dialectology
literature, not a validated model of *this specific* corpus. Any split
this module reports should be treated as a starting hypothesis for human/
expert review (ideally an ALMG-affiliated linguist), not a ground-truth
labeling to act on directly.

**Privacy**: every public function here returns only aggregate
counts/frequencies/cluster sizes -- never raw sentence text (ADR 0002 /
docs/data-governance.md). `render_report` is safe to paste into a public
GitHub issue comment as-is.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from data.corpus_io import SentencePair, read_tsv_pairs

# Regex patterns for orthographic features documented in the Kaqchikel
# dialectology/orthography literature (see module docstring). Each is a
# *proxy* signal, not a certain marker of any specific dialect.
_FEATURE_PATTERNS: dict[str, re.Pattern[str]] = {
    "tense_vowel_doubling": re.compile(r"(?i)(aa|ee|ii|oo|uu)"),
    "central_vowel_marks": re.compile(r"(?i)[äëïöü]"),
    "glottal_apostrophe": re.compile(r"[a-zA-Z]'"),
    "word_initial_h": re.compile(r"(?i)\bh[aeiouäëïöü]"),
}


def character_ngrams(text: str, n: int = 3) -> list[str]:
    """Return all overlapping character n-grams of `text`.

    Returns an empty list if `text` is shorter than `n` characters, rather
    than raising -- short entries (this corpus has many single-word/short
    entries, see `ml/README.md`) are common enough to handle gracefully.
    """
    if len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def feature_frequencies(text: str) -> dict[str, float]:
    """Count known orthographic features in `text`, normalized by length.

    Normalized as matches per 1,000 characters, so frequency is comparable
    across texts of different lengths (a raw count would just track text
    length, not spelling density).
    """
    length = max(len(text), 1)
    return {
        name: (len(pattern.findall(text)) / length) * 1000
        for name, pattern in _FEATURE_PATTERNS.items()
    }


def build_ngram_vectors(
    sentences: list[str], n: int = 3, top_k: int = 200
) -> tuple[list[str], np.ndarray]:
    """Build a per-sentence character-n-gram frequency matrix.

    The n-gram vocabulary is the `top_k` most frequent n-grams across the
    whole `sentences` corpus (fewer if the corpus doesn't have that many
    distinct n-grams). Each row is L2-normalized (unit vector), so
    sentence length doesn't dominate the resulting vector and Euclidean
    distance between rows behaves like cosine distance -- the standard
    similarity notion for this kind of sparse text-frequency clustering.

    Returns:
        `(vocab, vectors)`: the ordered list of n-gram vocabulary entries
        (most frequent first) and a `(len(sentences), len(vocab))` float
        matrix.
    """
    per_sentence_ngrams = [character_ngrams(s, n=n) for s in sentences]

    corpus_counts: Counter[str] = Counter()
    for ngrams in per_sentence_ngrams:
        corpus_counts.update(ngrams)

    vocab = [ngram for ngram, _ in corpus_counts.most_common(top_k)]
    vocab_index = {ngram: i for i, ngram in enumerate(vocab)}

    vectors = np.zeros((len(sentences), len(vocab)), dtype=float)
    for row, ngrams in enumerate(per_sentence_ngrams):
        if not ngrams:
            continue
        row_counts = Counter(ngrams)
        for ngram, count in row_counts.items():
            col = vocab_index.get(ngram)
            if col is not None:
                vectors[row, col] = count
        norm = np.linalg.norm(vectors[row])
        if norm > 0:
            vectors[row] /= norm

    return vocab, vectors


def kmeans_two_clusters(vectors: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """Deterministically partition `vectors` into exactly two clusters.

    Uses Lloyd's algorithm (k-means) with a deterministic "farthest pair"
    initialization -- the first row, and the row farthest from it -- rather
    than random initialization, so results are reproducible given the same
    input (important for a diagnostic tool: re-running against the same
    corpus should always report the same split).
    """
    n = vectors.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    centroid_0 = vectors[0]
    distances_from_0 = np.sum((vectors - centroid_0) ** 2, axis=1)
    centroid_1 = vectors[int(np.argmax(distances_from_0))]
    centroids = np.stack([centroid_0, centroid_1])

    labels: np.ndarray | None = None
    for _ in range(max_iterations):
        d0 = np.sum((vectors - centroids[0]) ** 2, axis=1)
        d1 = np.sum((vectors - centroids[1]) ** 2, axis=1)
        new_labels = (d1 < d0).astype(int)

        if labels is not None and np.array_equal(new_labels, labels):
            labels = new_labels
            break

        labels = new_labels
        for cluster in (0, 1):
            mask = labels == cluster
            if mask.any():
                centroids[cluster] = vectors[mask].mean(axis=0)

    assert labels is not None
    return labels


@dataclass(frozen=True)
class DialectSignalReport:
    """Aggregate-only findings from `analyze_dialect_signal`.

    Deliberately holds no sentence text, so it's always safe to log, print,
    or paste into a public GitHub issue (ADR 0002).
    """

    num_sentences: int
    cluster_sizes: tuple[int, int]
    feature_frequencies_by_cluster: dict[int, dict[str, float]]
    ngram_n: int
    top_k_ngrams: int


def analyze_dialect_signal(
    pairs: list[SentencePair], ngram_n: int = 3, top_k_ngrams: int = 200
) -> DialectSignalReport:
    """Run the full dialect-signal analysis over a corpus of sentence pairs.

    Args:
        pairs: `(spanish, kaqchikel)` sentence pairs, e.g. from
            `data.corpus_io.read_tsv_pairs`. Only the Kaqchikel side is
            analyzed. Blank Kaqchikel entries are skipped.
        ngram_n: character n-gram length for the unsupervised clustering.
        top_k_ngrams: size of the n-gram vocabulary used for clustering.

    Raises:
        ValueError: if fewer than 2 non-blank Kaqchikel sentences are
            present -- a 2-way cluster split is meaningless below that.
    """
    kaqchikel_sentences = [cak for _, cak in pairs if cak.strip()]
    if len(kaqchikel_sentences) < 2:
        raise ValueError(
            "analyze_dialect_signal requires at least 2 non-blank Kaqchikel "
            f"sentences, got {len(kaqchikel_sentences)}"
        )

    _vocab, vectors = build_ngram_vectors(kaqchikel_sentences, n=ngram_n, top_k=top_k_ngrams)
    labels = kmeans_two_clusters(vectors)

    cluster_sizes = (
        int(np.sum(labels == 0)),
        int(np.sum(labels == 1)),
    )

    feature_frequencies_by_cluster = {}
    for cluster in (0, 1):
        cluster_text = " ".join(
            sentence
            for sentence, label in zip(kaqchikel_sentences, labels, strict=True)
            if label == cluster
        )
        feature_frequencies_by_cluster[cluster] = feature_frequencies(cluster_text)

    return DialectSignalReport(
        num_sentences=len(kaqchikel_sentences),
        cluster_sizes=cluster_sizes,
        feature_frequencies_by_cluster=feature_frequencies_by_cluster,
        ngram_n=ngram_n,
        top_k_ngrams=top_k_ngrams,
    )


def render_report(report: DialectSignalReport) -> str:
    """Render a `DialectSignalReport` as human-readable, publish-safe text."""
    lines = [
        "Dialect signal report",
        "======================",
        "",
        f"Kaqchikel sentences analyzed: {report.num_sentences:,}",
        f"Character n-gram length: {report.ngram_n} (vocabulary cap: {report.top_k_ngrams})",
        "",
        "Cluster sizes (unsupervised 2-way split, char n-gram frequencies):",
        f"  - Cluster 0: {report.cluster_sizes[0]:,} sentences",
        f"  - Cluster 1: {report.cluster_sizes[1]:,} sentences",
        "",
        "Orthographic feature frequency per cluster (matches per 1,000 characters):",
    ]
    feature_names = sorted(next(iter(report.feature_frequencies_by_cluster.values())).keys())
    for cluster in (0, 1):
        freqs = report.feature_frequencies_by_cluster[cluster]
        lines.append(f"  Cluster {cluster}:")
        for name in feature_names:
            lines.append(f"    - {name}: {freqs[name]:.2f}")
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml.data.dialect_signal",
        description=(
            "Report aggregate dialect-variant signal statistics for a Spanish-Kaqchikel "
            "corpus TSV file. Prints only cluster sizes and feature frequencies -- never "
            "sentence content -- so its output is safe to share (see module docstring)."
        ),
    )
    parser.add_argument("corpus_path", help="Local path or s3:// URI of a corpus TSV file.")
    parser.add_argument("--ngram-n", type=int, default=3)
    parser.add_argument("--top-k-ngrams", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    pairs = read_tsv_pairs(args.corpus_path)
    report = analyze_dialect_signal(pairs, ngram_n=args.ngram_n, top_k_ngrams=args.top_k_ngrams)
    print(render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
