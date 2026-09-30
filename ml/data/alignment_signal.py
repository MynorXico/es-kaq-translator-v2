"""Embedding-based source/target alignment-quality signal (issue #199).

**What this is.** Per the AmericasNLP 2024 shared task's BSC team (the
strongest data-preprocessing precedent cited in issue #199): score each
parallel pair's source/target semantic similarity using a multilingual
sentence embedding model, as a signal for misaligned/noisy pairs (a pair
whose two sides aren't really translations of each other should embed as
*less* similar than a real translation pair, if the embedding model has
genuine cross-lingual understanding of both languages).

**The open question this module exists to answer, empirically rather than
by assumption**: does *any* available multilingual sentence embedding
model have real Kaqchikel-language signal, or does a source/target
similarity score for a Kaqchikel sentence just reflect noise (shared
surface tokens, sentence length, punctuation) rather than genuine
semantic alignment? Research for issue #199 checked two major model
families directly against their own published language coverage:

- **LaBSE** (Feng et al. 2022, 109 languages) -- Kaqchikel is not in its
  language list.
- **NLLB-200 / FLORES-200** (Meta AI, 200 languages) -- Kaqchikel and
  every other Mayan language are absent from the FLORES-200 language list
  NLLB-200 is trained/evaluated against (confirmed against the
  `facebookresearch/flores` `flores200` README directly, not assumed);
  the closest indigenous-language coverage it has is Quechua and
  Guaraní.

Neither model has ever seen Kaqchikel during training. This module's
pure, testable logic (`cosine_similarities`, `auc_separation`,
`build_shuffled_control_indices`, `analyze_alignment_signal`) computes a
concrete, real diagnostic anyway rather than stopping at "no coverage
confirmed, so skip it": a real/matched-pair similarity distribution is
compared against a deliberately shuffled/misaligned control (`auc_separation`
-- 0.5 means the model's similarity score cannot distinguish a real
translation pair from a random one at all; 1.0 means perfect separation),
which turns "does this model have any usable signal for this language
pair" into an empirical question answerable from the corpus itself,
without needing the model to have formal training coverage.

**Real-corpus result (issue #199, see `ml/README.md` for the full
writeup)**: run against a real sample of the ALMG training corpus with
`sentence-transformers/LaBSE`, `AUC ~= 0.66` -- weak but non-chance
separation, far below what a model with genuine coverage of both languages
would be expected to show (well above 0.9 in typical LaBSE alignment-
filtering use, e.g. the BSC team's own reported results on languages
LaBSE was actually trained on). **Conclusion: no viable embedding model
with adequate Kaqchikel coverage exists today** -- the weak signal found is
too unreliable to use as a hard corpus filter (it would likely drop
legitimately well-aligned but unusually-phrased pairs about as often as it
catches real misalignment). This project relies on corpus-internal
consistency checks instead (`data.length_filter`'s existing source/target
length-ratio filter), per the fallback issue #199 anticipated.

**Privacy**: every public function here operates on/returns embeddings or
aggregate statistics -- never raw sentence text (ADR 0002). The real-model
embedding step (loading a HuggingFace checkpoint and encoding real corpus
text) is a separate, maintainer-run diagnostic, not exercised in this
repo's automated test suite (same convention as
`data.dialect_signal`/`evaluation.evaluate_checkpoint`'s heavy,
real-checkpoint steps).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from data.corpus_io import read_tsv_pairs


def cosine_similarities(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two aligned `(n, dim)` matrices.

    A zero vector on either side yields a similarity of `0.0` for that row
    rather than raising or producing `nan` from a zero-division -- a
    degenerate embedding shouldn't crash a corpus-wide sweep.
    """
    a_norms = np.linalg.norm(a, axis=1)
    b_norms = np.linalg.norm(b, axis=1)
    denom = a_norms * b_norms
    dot = np.sum(a * b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sims = np.where(denom > 0, dot / denom, 0.0)
    return sims


def auc_separation(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """Rank-based AUC: how well `positive_scores` separate above `negative_scores`.

    Equivalent to the Mann-Whitney U statistic normalized to `[0, 1]`. `0.5`
    means the two score distributions are indistinguishable (no signal);
    `1.0` means every positive score exceeds every negative score (perfect
    separation); `0.0` is the reverse. Implemented via rank statistics
    (no `scipy` dependency) so it stays consistent with this project's
    "framework-free data/logic code" convention (`ml/README.md`).
    """
    n_pos = len(positive_scores)
    n_neg = len(negative_scores)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auc_separation requires at least one score in each group")

    combined = np.concatenate([positive_scores, negative_scores])
    ranks = _average_ranks(combined)
    positive_rank_sum = ranks[:n_pos].sum()
    return float((positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank `values` ascending (1-based), averaging ranks within tied groups.

    Standard tie-handling for a Mann-Whitney U / rank-based AUC statistic --
    without this, `np.argsort`'s arbitrary but consistent tie-breaking would
    silently bias the result whenever scores repeat (e.g. every score being
    identical), which real cosine-similarity scores can do at low precision
    or in fully-degenerate test fixtures.
    """
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks_for_sorted = np.empty(len(values), dtype=float)

    i = 0
    while i < len(sorted_values):
        j = i
        while j < len(sorted_values) and sorted_values[j] == sorted_values[i]:
            j += 1
        # Positions i..j-1 (0-based) are tied; their 1-based rank range is
        # (i+1)..j, so the average rank is the midpoint of that range.
        average_rank = (i + 1 + j) / 2
        ranks_for_sorted[i:j] = average_rank
        i = j

    ranks = np.empty(len(values), dtype=float)
    ranks[order] = ranks_for_sorted
    return ranks


def build_shuffled_control_indices(n: int, seed: int) -> np.ndarray:
    """Return a deterministic derangement (no fixed points) of `range(n)`.

    Used to build a "definitely misaligned" control set by pairing each
    source embedding with a *different* sentence's target embedding,
    instead of its real counterpart. Deterministic given `seed`, so a
    diagnostic run is reproducible.

    Raises:
        ValueError: if `n < 2` -- a derangement with no fixed points is
            impossible for `n < 2`.
    """
    if n < 2:
        raise ValueError(f"build_shuffled_control_indices requires n >= 2, got {n}")

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    while True:
        candidate = rng.permutation(indices)
        if not np.any(candidate == indices):
            return candidate


@dataclass(frozen=True)
class PairSimilarityReport:
    """Aggregate-only findings from `analyze_alignment_signal`.

    Deliberately holds no per-sentence scores or text -- only summary
    statistics -- so it's always safe to log, print, or paste into a
    public GitHub issue (ADR 0002), matching
    `data.dialect_signal.DialectSignalReport` /
    `data.glued_word_signal.GluedWordReport`.
    """

    num_pairs: int
    real_pair_mean_similarity: float
    real_pair_median_similarity: float
    shuffled_control_mean_similarity: float
    shuffled_control_median_similarity: float
    auc: float


def analyze_alignment_signal(
    source_embeddings: np.ndarray, target_embeddings: np.ndarray, seed: int = 42
) -> PairSimilarityReport:
    """Compare real-pair similarity against a shuffled-control baseline.

    Args:
        source_embeddings: `(n, dim)` embeddings for the source side of
            each pair (e.g. Spanish).
        target_embeddings: `(n, dim)` embeddings for the target side of
            each pair (e.g. Kaqchikel), aligned by row with
            `source_embeddings`.
        seed: seed for the deterministic shuffled-control derangement.

    Raises:
        ValueError: if the two embedding arrays don't have the same number
            of rows.
    """
    if source_embeddings.shape[0] != target_embeddings.shape[0]:
        raise ValueError(
            "analyze_alignment_signal requires source_embeddings and target_embeddings "
            f"to have the same number of rows, got {source_embeddings.shape[0]} "
            f"and {target_embeddings.shape[0]}"
        )

    real_sims = cosine_similarities(source_embeddings, target_embeddings)
    shuffled_indices = build_shuffled_control_indices(len(target_embeddings), seed=seed)
    shuffled_sims = cosine_similarities(source_embeddings, target_embeddings[shuffled_indices])

    return PairSimilarityReport(
        num_pairs=source_embeddings.shape[0],
        real_pair_mean_similarity=float(np.mean(real_sims)),
        real_pair_median_similarity=float(np.median(real_sims)),
        shuffled_control_mean_similarity=float(np.mean(shuffled_sims)),
        shuffled_control_median_similarity=float(np.median(shuffled_sims)),
        auc=auc_separation(real_sims, shuffled_sims),
    )


def render_report(report: PairSimilarityReport, model_name: str) -> str:
    """Render a `PairSimilarityReport` as human-readable, publish-safe text."""
    return (
        "Alignment-quality signal report\n"
        "================================\n\n"
        f"Embedding model: {model_name}\n"
        f"Pairs analyzed: {report.num_pairs:,}\n\n"
        f"Real-pair similarity:      mean={report.real_pair_mean_similarity:.4f}  "
        f"median={report.real_pair_median_similarity:.4f}\n"
        f"Shuffled-control similarity: mean={report.shuffled_control_mean_similarity:.4f}  "
        f"median={report.shuffled_control_median_similarity:.4f}\n\n"
        f"AUC (real pair similarity > shuffled control): {report.auc:.4f}\n"
        "(0.5 = no separation/no usable signal, 1.0 = perfect separation)\n"
    )


def _embed_sentences(texts: list[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """Encode `texts` with a real HuggingFace sentence-embedding checkpoint.

    Lazily imports `torch`/`transformers` so this module's pure logic
    above stays importable/testable without those (heavy) dependencies
    ever loading in a fast unit-test run. Never called from this repo's
    automated test suite -- see module docstring.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            output = model(**encoded).pooler_output
        normalized = torch.nn.functional.normalize(output, p=2, dim=1)
        all_embeddings.append(normalized.numpy())
    return np.concatenate(all_embeddings, axis=0)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml.data.alignment_signal",
        description=(
            "Report aggregate embedding-based source/target alignment-quality "
            "statistics for a Spanish-Kaqchikel corpus TSV file. Requires torch/"
            "transformers and downloads a real embedding checkpoint -- a "
            "maintainer-run diagnostic, never run in CI/this repo's test suite "
            "(see module docstring). Prints only aggregate stats, never sentence "
            "content."
        ),
    )
    parser.add_argument("corpus_path", help="Local path or s3:// URI of a corpus TSV file.")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/LaBSE",
        help="HuggingFace model id for a sentence-embedding checkpoint.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Number of pairs to randomly sample from the corpus (default: 300).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    import random

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    pairs = read_tsv_pairs(args.corpus_path)
    rng = random.Random(args.seed)
    sample = rng.sample(pairs, min(args.sample_size, len(pairs)))

    source_embeddings = _embed_sentences([es for es, _ in sample], args.model_name)
    target_embeddings = _embed_sentences([cak for _, cak in sample], args.model_name)

    report = analyze_alignment_signal(source_embeddings, target_embeddings, seed=args.seed)
    print(render_report(report, model_name=args.model_name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
