"""Unit tests for issue #143: `evaluate_checkpoint`'s provenance-aware
selection between the old (pre-#125, unscoped) and new (post-#125,
Kaqchikel-only) `train_sample_texts` construction used to reconstruct
issue #116's word-boundary-spacing fix.

Pure functions only (no torch/transformers import required), mirroring
`tests/unit/test_evaluate_checkpoint_source.py`'s reasoning for why that
matters for fast tests.
"""

from __future__ import annotations

from evaluation.evaluate_checkpoint import (
    _build_train_sample_texts,
    _read_checkpoint_vocab_extension_scoping,
)
from training.direction import ALL_DIRECTION_TAG_TOKENS
from training.train import VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY

TRAIN_PAIRS = [("Buenos días", "Utz awäch"), ("Gracias", "Matyox")]


def test_read_checkpoint_vocab_extension_scoping_returns_none_without_a_model_card(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    assert _read_checkpoint_vocab_extension_scoping(str(checkpoint_dir)) is None


def test_read_checkpoint_vocab_extension_scoping_returns_none_for_a_pre_125_model_card(tmp_path):
    """A model card written before issue #125's fix never recorded this
    field at all -- its absence must be treated as implicitly "old/
    unscoped", matching the currently-deployed checkpoint
    (`run-20260922T141956Z`, Model Package v4).
    """
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_card.md").write_text(
        "# Model card: run-x\n\n## Hyperparameters\n\n- **new_tokens_added**: 5\n",
        encoding="utf-8",
    )

    assert _read_checkpoint_vocab_extension_scoping(str(checkpoint_dir)) is None


def test_read_checkpoint_vocab_extension_scoping_reads_a_post_125_model_card(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_card.md").write_text(
        "# Model card: run-y\n\n## Hyperparameters\n\n"
        "- **new_tokens_added**: 5\n"
        "- **vocab_extension_scoping**: kaqchikel_only\n",
        encoding="utf-8",
    )

    assert (
        _read_checkpoint_vocab_extension_scoping(str(checkpoint_dir))
        == VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY
    )


def test_read_checkpoint_vocab_extension_scoping_is_never_fatal_for_a_non_local_source(tmp_path):
    """Mirrors `_read_checkpoint_new_tokens_added`'s "never raises" contract
    -- a resolved checkpoint source that isn't a real local directory (or
    has no readable model card) must not crash an otherwise-valid run.
    """
    assert _read_checkpoint_vocab_extension_scoping("not-a-real-path") is None


def test_build_train_sample_texts_uses_both_columns_for_legacy_unscoped_provenance():
    """`None` provenance (no recorded field) must reproduce the exact
    pre-#125 "every pair's both columns" construction this script always
    used, unchanged -- correctness for every checkpoint trained so far
    depends on this staying put.
    """
    texts = _build_train_sample_texts(TRAIN_PAIRS, None)

    assert texts == [
        "Buenos días",
        "Gracias",
        "Utz awäch",
        "Matyox",
        *ALL_DIRECTION_TAG_TOKENS,
    ]


def test_build_train_sample_texts_uses_kaqchikel_only_column_for_post_125_provenance():
    """Post-#125 provenance must reproduce `extend_vocabulary_for_examples`'s
    new Kaqchikel-only construction (`training.direction.
    collect_texts_for_language`) -- the Spanish column must never appear.
    """
    texts = _build_train_sample_texts(TRAIN_PAIRS, VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY)

    assert texts == ["Utz awäch", "Matyox", *ALL_DIRECTION_TAG_TOKENS]
    assert "Buenos días" not in texts
    assert "Gracias" not in texts
