"""Full-corpus OCR/digitization glued-word artifact sweep (issue #199).

**Background.** Issues #116/#125 found systematic word-boundary space loss
in *generated model output* (e.g. "dedios", "queestá", "deguatemala",
"parachoch", "queestaban" -- concatenations of a Spanish clitic/stopword
and the following word), root-caused to a `tokenizer.add_tokens()`
spacing bug (#116) and fixed. That finding was against decoded output, on
a sample -- never against the raw corpus text itself, at full scale. This
module does that full-corpus sweep, to answer a real, separate question:
does the *source corpus text* (not just past buggy decode output) also
contain this kind of glued-word artifact, e.g. from the original
OCR/digitization of source documents?

Two independent signals, both aggregate-only (ADR 0002 -- see
`GluedWordReport`, never raw sentence text):

1. **Known-pattern sweep** (`scan_known_patterns`): exact, case-insensitive
   substring counts for the specific strings issue #116 already confirmed
   as glued-word artifacts. Cheap, precise, but only catches patterns
   already known -- a zero-occurrence result here means "the previously
   observed decode-time artifacts are not present as raw corpus text",
   not "no glued words exist at all".
2. **General heuristic** (`find_glued_word_candidate` /
   `scan_corpus_for_glued_words`): flags a word as a *candidate* glued
   artifact only if (a) the word itself is rare in the corpus (a genuine
   OCR-glued artifact should be a one-off accident, not a recurring
   spelling), and (b) it starts with a common Spanish clitic/stopword
   whose *remainder*, after stripping that prefix, is itself independently
   frequent in the corpus (i.e. plausibly "clitic" + "a word the corpus
   already knows well", glued together). This is a deliberately
   conservative heuristic with a known false-positive mode: many ordinary
   Spanish words happen to start with a common clitic and have a
   coincidentally-frequent remainder (e.g. a word starting with "de-"
   whose tail, by coincidence, is also a common standalone word) without
   being glued artifacts at all. Real-corpus results (see `ml/README.md`)
   found this heuristic's candidate list dominated by exactly this kind of
   false positive -- treat its output as a weak hypothesis generator, the
   same caution `data.dialect_signal` documents for its own clustering
   signal, not a labeling authority.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from data.corpus_io import SentencePair, read_tsv_pairs

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")

# The exact strings issue #116 found glued together in generated output.
# Kept as the default sweep target rather than reinvented -- see module
# docstring.
DEFAULT_KNOWN_GLUED_PATTERNS: tuple[str, ...] = (
    "dedios",
    "queestá",
    "queesta",
    "deguatemala",
    "parachoch",
    "queestaban",
)

# Common Spanish clitics/stopwords short enough, and frequent enough, that
# an OCR/digitization pipeline dropping a space after one of them is a
# plausible failure mode. Not exhaustive -- extend if a future corpus scan
# finds evidence of another one, rather than guessing ahead of evidence
# (same "verify before assuming" discipline as `data.normalize`).
DEFAULT_GLUE_PREFIXES: tuple[str, ...] = (
    "de",
    "que",
    "la",
    "el",
    "los",
    "las",
    "en",
    "un",
    "una",
    "y",
    "es",
    "su",
    "sus",
    "lo",
    "al",
    "del",
    "por",
    "con",
    "no",
    "se",
    "le",
)


def build_word_frequencies(sentences: Iterable[str]) -> Counter[str]:
    """Count word-form frequencies (lowercased) across `sentences`."""
    counts: Counter[str] = Counter()
    for sentence in sentences:
        for word in _WORD_RE.findall(sentence):
            counts[word.lower()] += 1
    return counts


def scan_known_patterns(
    sentences: Iterable[str], patterns: tuple[str, ...] = DEFAULT_KNOWN_GLUED_PATTERNS
) -> dict[str, int]:
    """Count case-insensitive occurrences of each known glued-word pattern.

    Returns a mapping of pattern -> total occurrence count across every
    sentence in `sentences`. A pattern with a zero count is still present
    in the result (not omitted), so a caller can distinguish "checked, zero
    found" from "never checked".
    """
    counts = dict.fromkeys(patterns, 0)
    for sentence in sentences:
        lowered = sentence.lower()
        for pattern in patterns:
            counts[pattern] += lowered.count(pattern.lower())
    return counts


def find_glued_word_candidate(
    word: str,
    frequencies: Counter[str],
    prefixes: tuple[str, ...] = DEFAULT_GLUE_PREFIXES,
    max_word_frequency: int = 1,
    min_suffix_frequency: int = 20,
    min_suffix_length: int = 4,
) -> str | None:
    """Return the matched glue prefix if `word` looks like a glued artifact.

    A word is flagged only if it's itself rare (`frequency <=
    max_word_frequency` -- a real OCR-glue accident shouldn't recur
    verbatim many times) *and* removing a candidate prefix leaves a
    sufficiently long (`min_suffix_length`), independently frequent
    (`min_suffix_frequency`) remainder. Returns `None` (no candidate) if no
    prefix satisfies both conditions. When multiple prefixes match, the
    longest is preferred (fewer, more specific false positives).
    """
    lowered = word.lower()
    word_frequency = frequencies.get(lowered, 0)
    if word_frequency > max_word_frequency:
        return None

    best_prefix: str | None = None
    for prefix in prefixes:
        if not lowered.startswith(prefix):
            continue
        suffix = lowered[len(prefix) :]
        if len(suffix) < min_suffix_length:
            continue
        if frequencies.get(suffix, 0) < min_suffix_frequency:
            continue
        if best_prefix is None or len(prefix) > len(best_prefix):
            best_prefix = prefix

    return best_prefix


@dataclass(frozen=True)
class GluedWordReport:
    """Aggregate-only findings from `scan_corpus_for_glued_words`.

    Holds only counts -- never sentence text or the actual candidate word
    forms -- so it's always safe to log, print, or paste into a public
    GitHub issue (ADR 0002), matching `data.dialect_signal.DialectSignalReport`.
    """

    num_sentences_scanned: int
    known_pattern_counts: dict[str, int]
    candidate_counts_by_prefix: dict[str, int] = field(default_factory=dict)
    num_candidate_words: int = 0


def scan_corpus_for_glued_words(
    pairs: list[SentencePair],
    known_patterns: tuple[str, ...] = DEFAULT_KNOWN_GLUED_PATTERNS,
    prefixes: tuple[str, ...] = DEFAULT_GLUE_PREFIXES,
    max_word_frequency: int = 1,
    min_suffix_frequency: int = 20,
    min_suffix_length: int = 4,
) -> GluedWordReport:
    """Run both glued-word signals over the Spanish side of a corpus.

    Only the Spanish side is scanned: every known concrete example
    (#116/#125) is a Spanish clitic-prefix pattern, and `prefixes` is a
    Spanish-specific stopword list -- scanning Kaqchikel text against a
    Spanish clitic list would be meaningless.
    """
    es_sentences = [es for es, _ in pairs]
    known_pattern_counts = scan_known_patterns(es_sentences, patterns=known_patterns)

    frequencies = build_word_frequencies(es_sentences)
    candidate_counts_by_prefix: Counter[str] = Counter()
    num_candidate_words = 0
    seen_words: set[str] = set()
    for sentence in es_sentences:
        for word in _WORD_RE.findall(sentence):
            lowered = word.lower()
            if lowered in seen_words:
                continue
            seen_words.add(lowered)
            candidate = find_glued_word_candidate(
                lowered,
                frequencies,
                prefixes=prefixes,
                max_word_frequency=max_word_frequency,
                min_suffix_frequency=min_suffix_frequency,
                min_suffix_length=min_suffix_length,
            )
            if candidate is not None:
                candidate_counts_by_prefix[candidate] += 1
                num_candidate_words += 1

    return GluedWordReport(
        num_sentences_scanned=len(pairs),
        known_pattern_counts=known_pattern_counts,
        candidate_counts_by_prefix=dict(candidate_counts_by_prefix),
        num_candidate_words=num_candidate_words,
    )


def render_report(report: GluedWordReport) -> str:
    """Render a `GluedWordReport` as human-readable, publish-safe text."""
    lines = [
        "Glued-word signal report",
        "=========================",
        "",
        f"Sentences scanned: {report.num_sentences_scanned:,}",
        "",
        "Known-pattern sweep (issue #116/#125 confirmed decode-time artifacts):",
    ]
    for pattern, count in report.known_pattern_counts.items():
        lines.append(f"  - {pattern!r}: {count:,} occurrence(s)")
    lines.extend(
        [
            "",
            (
                "General heuristic candidates (rare word + frequent clitic-prefixed "
                f"remainder): {report.num_candidate_words:,} distinct word form(s)"
            ),
        ]
    )
    for prefix, count in sorted(report.candidate_counts_by_prefix.items()):
        lines.append(f"  - prefix {prefix!r}: {count:,} candidate(s)")
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml.data.glued_word_signal",
        description=(
            "Report aggregate OCR/digitization glued-word signal statistics for a "
            "Spanish-Kaqchikel corpus TSV file. Prints only counts -- never sentence "
            "content -- so its output is safe to share (see module docstring)."
        ),
    )
    parser.add_argument("corpus_path", help="Local path or s3:// URI of a corpus TSV file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    pairs = read_tsv_pairs(args.corpus_path)
    report = scan_corpus_for_glued_words(pairs)
    print(render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
