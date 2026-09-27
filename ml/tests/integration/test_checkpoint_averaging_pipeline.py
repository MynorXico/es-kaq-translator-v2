"""Fast smoke test for `training.checkpoint_averaging.average_checkpoints`'
filesystem wiring, against tiny fixture "checkpoint" directories -- real
`safetensors` files, but never a real model/tokenizer download (per
docs/testing.md: pipeline wiring gets a fast integration smoke test
against a tiny fixture, not the real checkpoint).
"""

from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from training import checkpoint_averaging
from training.checkpoint_averaging import (
    WEIGHTS_FILENAME,
    average_checkpoints,
    find_checkpoint_dirs,
    select_last_n_checkpoints,
)


def _make_fixture_checkpoint(
    root: Path, step: int, weight_value: float, *, config_marker: str
) -> Path:
    checkpoint_dir = root / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True)
    save_file(
        {"embedding.weight": torch.full((2, 3), weight_value)},
        str(checkpoint_dir / WEIGHTS_FILENAME),
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"marker": config_marker}), encoding="utf-8"
    )
    (checkpoint_dir / "generation_config.json").write_text("{}", encoding="utf-8")
    # Training-progress files that must NOT be copied into the averaged
    # output directory (see _EXCLUDED_FILENAMES's own docstring).
    (checkpoint_dir / "optimizer.pt").write_text("fake-optimizer-state", encoding="utf-8")
    (checkpoint_dir / "scheduler.pt").write_text("fake-scheduler-state", encoding="utf-8")
    (checkpoint_dir / "rng_state.pth").write_text("fake-rng-state", encoding="utf-8")
    (checkpoint_dir / "trainer_state.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "training_args.bin").write_text("fake-training-args", encoding="utf-8")
    return checkpoint_dir


def test_average_checkpoints_writes_elementwise_mean_weights(tmp_path: Path):
    checkpoints_root = tmp_path / "checkpoints"
    _make_fixture_checkpoint(checkpoints_root, 1, weight_value=1.0, config_marker="run")
    _make_fixture_checkpoint(checkpoints_root, 2, weight_value=3.0, config_marker="run")

    checkpoint_dirs = find_checkpoint_dirs(checkpoints_root)
    output_dir = average_checkpoints(checkpoint_dirs, tmp_path / "averaged")

    averaged_weights = load_file(str(output_dir / WEIGHTS_FILENAME))
    assert torch.allclose(averaged_weights["embedding.weight"], torch.full((2, 3), 2.0))


def test_average_checkpoints_copies_non_weight_files_from_the_last_checkpoint(tmp_path: Path):
    checkpoints_root = tmp_path / "checkpoints"
    _make_fixture_checkpoint(checkpoints_root, 1, weight_value=1.0, config_marker="from-epoch-1")
    _make_fixture_checkpoint(checkpoints_root, 2, weight_value=3.0, config_marker="from-epoch-2")

    checkpoint_dirs = find_checkpoint_dirs(checkpoints_root)
    output_dir = average_checkpoints(checkpoint_dirs, tmp_path / "averaged")

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config == {"marker": "from-epoch-2"}
    assert (output_dir / "generation_config.json").exists()


def test_average_checkpoints_never_copies_training_progress_files(tmp_path: Path):
    checkpoints_root = tmp_path / "checkpoints"
    _make_fixture_checkpoint(checkpoints_root, 1, weight_value=1.0, config_marker="run")
    _make_fixture_checkpoint(checkpoints_root, 2, weight_value=3.0, config_marker="run")

    checkpoint_dirs = find_checkpoint_dirs(checkpoints_root)
    output_dir = average_checkpoints(checkpoint_dirs, tmp_path / "averaged")

    for excluded in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
                     "training_args.bin"):
        assert not (output_dir / excluded).exists()


def test_average_checkpoints_averages_only_the_selected_last_n(tmp_path: Path):
    """End-to-end: find -> select last N -> average, matching the CLI's own
    wiring (training.checkpoint_averaging.main) -- averaging only the last
    2 of 3 fixture checkpoints must exclude the oldest one's weight value
    from the result.
    """
    checkpoints_root = tmp_path / "checkpoints"
    _make_fixture_checkpoint(checkpoints_root, 1, weight_value=100.0, config_marker="run")
    _make_fixture_checkpoint(checkpoints_root, 2, weight_value=2.0, config_marker="run")
    _make_fixture_checkpoint(checkpoints_root, 3, weight_value=4.0, config_marker="run")

    checkpoint_dirs = find_checkpoint_dirs(checkpoints_root)
    selected = select_last_n_checkpoints(checkpoint_dirs, 2)
    output_dir = average_checkpoints(selected, tmp_path / "averaged")

    averaged_weights = load_file(str(output_dir / WEIGHTS_FILENAME))
    # (2.0 + 4.0) / 2 = 3.0 -- not influenced by the oldest checkpoint's 100.0.
    assert torch.allclose(averaged_weights["embedding.weight"], torch.full((2, 3), 3.0))


def test_average_checkpoints_releases_earlier_checkpoints_before_loading_later_ones(
    tmp_path: Path, monkeypatch
):
    """Code review on PR #183: `average_checkpoints` previously loaded every
    checkpoint's full state dict into a list *before* averaging any of them
    (`state_dicts = [_load_state_dict(cd) for cd in checkpoint_dirs]`),
    so peak memory scaled with N instead of ~2 (current + running-sum
    accumulator) -- a real risk for a utility meant to be run casually
    (e.g. on a laptop), especially once real checkpoints are ~2GB each.

    Proves the fix processes one checkpoint at a time: checkpoint 1's raw
    state dict must already be garbage-collected by the time checkpoint 3
    is loaded. A "load all N upfront" implementation would still be
    holding checkpoint 1 alive (referenced by its own materialized list)
    at that point, so this fails against the old implementation and passes
    against the new one.
    """
    checkpoints_root = tmp_path / "checkpoints"
    _make_fixture_checkpoint(checkpoints_root, 1, weight_value=1.0, config_marker="run")
    _make_fixture_checkpoint(checkpoints_root, 2, weight_value=3.0, config_marker="run")
    _make_fixture_checkpoint(checkpoints_root, 3, weight_value=5.0, config_marker="run")
    checkpoint_dirs = find_checkpoint_dirs(checkpoints_root)

    real_load = checkpoint_averaging._load_state_dict
    live_refs: list[weakref.ReferenceType] = []

    def tracking_load(checkpoint_dir):
        if len(live_refs) == 2:
            gc.collect()
            assert live_refs[0]() is None, (
                "checkpoint 1's weight tensor is still alive while loading "
                "checkpoint 3 -- average_checkpoints is holding every "
                "checkpoint's weights in memory at once instead of "
                "accumulating incrementally."
            )
        state_dict = real_load(checkpoint_dir)
        # Weakref the tensor value itself -- a plain dict doesn't support
        # weakref, but a real torch.Tensor does.
        live_refs.append(weakref.ref(state_dict["embedding.weight"]))
        return state_dict

    monkeypatch.setattr(checkpoint_averaging, "_load_state_dict", tracking_load)

    average_checkpoints(checkpoint_dirs, tmp_path / "averaged")

    assert len(live_refs) == 3  # sanity: all 3 checkpoints were actually loaded


def test_average_checkpoints_rejects_empty_checkpoint_list(tmp_path: Path):
    with pytest.raises(ValueError):
        average_checkpoints([], tmp_path / "averaged")


def test_average_checkpoints_raises_on_missing_weights_file(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints" / "checkpoint-1"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        average_checkpoints([checkpoint_dir], tmp_path / "averaged")
