"""Unit tests for `training.checkpoint_averaging`'s pure logic (issue #182):
finding/selecting `checkpoint-<step>` directories and elementwise-averaging
state dicts. No real model/tokenizer needed -- filesystem-touching
`average_checkpoints` itself (fixture-scale safetensors files, not a real
checkpoint) is covered separately in
`tests/integration/test_checkpoint_averaging_pipeline.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from training.checkpoint_averaging import (
    DEFAULT_AVERAGE_N,
    average_state_dicts,
    find_checkpoint_dirs,
    select_last_n_checkpoints,
)


def test_default_average_n_matches_cli_default():
    # Code review on PR #183: training.train.DEFAULT_SAVE_TOTAL_LIMIT is
    # derived from this constant (with a margin) rather than a separately
    # hardcoded number, specifically so the two can't drift apart --
    # pin the constant's own value here too.
    assert DEFAULT_AVERAGE_N == 3


def test_find_checkpoint_dirs_sorts_numerically_not_lexicographically(tmp_path: Path):
    for name in ("checkpoint-9", "checkpoint-10", "checkpoint-2"):
        (tmp_path / name).mkdir()
    (tmp_path / "not-a-checkpoint").mkdir()
    (tmp_path / "checkpoint-not-numeric").mkdir()
    (tmp_path / "some-file.txt").write_text("x", encoding="utf-8")

    dirs = find_checkpoint_dirs(tmp_path)

    assert [path.name for path in dirs] == ["checkpoint-2", "checkpoint-9", "checkpoint-10"]


def test_find_checkpoint_dirs_returns_empty_list_when_none_exist(tmp_path: Path):
    (tmp_path / "unrelated").mkdir()

    assert find_checkpoint_dirs(tmp_path) == []


def test_select_last_n_checkpoints_returns_the_most_recent_n():
    dirs = [Path(f"checkpoint-{i}") for i in (1, 2, 3, 4, 5)]

    assert select_last_n_checkpoints(dirs, 2) == [Path("checkpoint-4"), Path("checkpoint-5")]


def test_select_last_n_checkpoints_clamps_when_fewer_are_available():
    dirs = [Path("checkpoint-1"), Path("checkpoint-2")]

    assert select_last_n_checkpoints(dirs, 5) == dirs


def test_select_last_n_checkpoints_rejects_non_positive_n():
    with pytest.raises(ValueError):
        select_last_n_checkpoints([Path("checkpoint-1")], 0)


def test_average_state_dicts_accepts_a_lazy_generator_not_just_a_list():
    """Code review on PR #183: `average_checkpoints` must be able to pass a
    lazy generator (one checkpoint's state dict loaded at a time) rather
    than a pre-materialized list, so it never holds every checkpoint's full
    weights in memory simultaneously. A real generator supports neither
    indexing nor `len()`/truthiness-based emptiness checks, so this fails
    loudly (`TypeError`) if `average_state_dicts` ever assumes list-like
    random access again.
    """

    def _generate():
        yield {"weight": torch.tensor([1.0, 2.0])}
        yield {"weight": torch.tensor([3.0, 4.0])}

    averaged = average_state_dicts(_generate())

    assert torch.allclose(averaged["weight"], torch.tensor([2.0, 3.0]))


def test_average_state_dicts_rejects_an_empty_generator():
    # A generator is always truthy (no __len__/__bool__), unlike a list --
    # emptiness must be detected by actually iterating, not `if not ...`.
    with pytest.raises(ValueError):
        average_state_dicts(iter([]))


def test_average_state_dicts_computes_elementwise_mean():
    state_dicts = [
        {"weight": torch.tensor([1.0, 2.0, 3.0])},
        {"weight": torch.tensor([3.0, 4.0, 5.0])},
    ]

    averaged = average_state_dicts(state_dicts)

    assert torch.allclose(averaged["weight"], torch.tensor([2.0, 3.0, 4.0]))


def test_average_state_dicts_preserves_original_dtype():
    state_dicts = [
        {"weight": torch.tensor([1.0, 2.0], dtype=torch.float16)},
        {"weight": torch.tensor([3.0, 4.0], dtype=torch.float16)},
    ]

    averaged = average_state_dicts(state_dicts)

    assert averaged["weight"].dtype == torch.float16
    assert torch.allclose(averaged["weight"].float(), torch.tensor([2.0, 3.0]))


def test_average_state_dicts_rejects_mismatched_keys():
    state_dicts = [
        {"weight_a": torch.tensor([1.0])},
        {"weight_b": torch.tensor([1.0])},
    ]

    with pytest.raises(ValueError):
        average_state_dicts(state_dicts)


def test_average_state_dicts_rejects_mismatched_shapes():
    state_dicts = [
        {"weight": torch.tensor([1.0, 2.0])},
        {"weight": torch.tensor([1.0, 2.0, 3.0])},
    ]

    with pytest.raises(ValueError):
        average_state_dicts(state_dicts)


def test_average_state_dicts_rejects_empty_input():
    with pytest.raises(ValueError):
        average_state_dicts([])
