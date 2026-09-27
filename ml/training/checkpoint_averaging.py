"""Post-hoc checkpoint-averaging utility (issue #182).

`training.train.build_training_arguments` now defaults to
`save_strategy="epoch"` (previously `"no"` -- no intermediate checkpoint
ever existed for any past training run), so a real training run's
`--model-dir/checkpoints/` now contains one `checkpoint-<step>` directory
per epoch (the shape `transformers.Seq2SeqTrainer`'s own checkpointing
writes: `model.safetensors`, `config.json`, `generation_config.json`,
`trainer_state.json`, `optimizer.pt`, `scheduler.pt`, `rng_state.pth`,
`training_args.bin` -- confirmed directly against a real
`facebook/m2m100_418M` fine-tuning run, not assumed).

Averaging the weights of the final few epochs' checkpoints ("checkpoint
averaging"/"weight averaging") is a well-known, cheap way to reduce
end-of-training noise in NMT fine-tuning, without any additional training
cost -- it only requires checkpoints to already exist, which is exactly
what `save_strategy="epoch"` now provides. This module is the small,
post-hoc utility that does the averaging; it does **not** run any
training itself, and this ticket does not evaluate whether averaging
actually improves BLEU/chrF for this project -- that is a separate,
maintainer-approved follow-up experiment (see `ml/README.md`).

Kept deliberately independent of `training/train.py`: it only ever reads
already-saved checkpoint directories from disk and writes a new one back
out, the same "operates on data/artifacts outside the repo, never wired
into a live training run automatically" shape as
`evaluation/evaluate_checkpoint.py`.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

WEIGHTS_FILENAME = "model.safetensors"

_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")

# Files a real Seq2SeqTrainer checkpoint directory contains that describe
# *training progress* (optimizer/scheduler/RNG state, the trainer's own
# step/epoch bookkeeping) or *this specific run's* CLI config -- none of
# which are meaningful facts about the averaged weights themselves, so
# they're never copied into an averaged checkpoint's output directory.
# WEIGHTS_FILENAME is excluded too since it's written separately, from the
# averaged tensors, not copied from any one source checkpoint.
_EXCLUDED_FILENAMES = frozenset(
    {
        WEIGHTS_FILENAME,
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    }
)


def find_checkpoint_dirs(checkpoints_root: Path) -> list[Path]:
    """Return every immediate `checkpoint-<step>` subdirectory of
    `checkpoints_root` (the shape `Seq2SeqTrainer`'s own `save_strategy`
    writes -- see this module's own docstring), sorted ascending by the
    integer step number in each name -- not lexicographically, since
    `"checkpoint-10"` must sort after `"checkpoint-9"`, not before it as a
    plain string sort would.

    Returns an empty list if `checkpoints_root` has no matching
    subdirectories (e.g. a run that predates issue #182's
    `save_strategy="epoch"` default).
    """
    numbered_dirs = []
    for entry in checkpoints_root.iterdir():
        if not entry.is_dir():
            continue
        match = _CHECKPOINT_DIR_RE.match(entry.name)
        if match:
            numbered_dirs.append((int(match.group(1)), entry))
    numbered_dirs.sort(key=lambda pair: pair[0])
    return [path for _, path in numbered_dirs]


def select_last_n_checkpoints(checkpoint_dirs: list[Path], n: int) -> list[Path]:
    """Return the last `n` entries of `checkpoint_dirs` (already sorted
    ascending by step -- see `find_checkpoint_dirs`), i.e. the N most
    recent epochs to average.

    Clamped to however many are actually available rather than raising --
    a short run with fewer epochs than `n` still averages whatever
    checkpoints it has, rather than failing outright.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return checkpoint_dirs[-n:]


def average_state_dicts(state_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Elementwise-average a list of same-shaped state dicts (checkpoint
    weight tensors).

    Every state dict must have exactly the same set of keys and matching
    tensor shapes for each key -- raises `ValueError` rather than silently
    averaging a subset or broadcasting mismatched shapes, since that would
    otherwise produce a checkpoint that looks superficially valid but is
    actually meaningless (e.g. checkpoints accidentally taken from two
    different runs/architectures).

    Averages in `float32` regardless of the input dtype (so `float16`
    checkpoints, real for this project's GPU training runs -- `fp16=True`
    whenever CUDA is available, see `training.train.build_training_arguments`
    -- don't lose precision mid-average), then casts each result back to
    its original per-key dtype.
    """
    import torch

    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")

    reference_keys = set(state_dicts[0])
    for index, state_dict in enumerate(state_dicts[1:], start=1):
        if set(state_dict) != reference_keys:
            raise ValueError(
                f"state_dicts[{index}] has different keys than state_dicts[0] -- "
                "checkpoints must be from the same training run/architecture."
            )

    averaged: dict[str, Any] = {}
    for key in reference_keys:
        tensors = [state_dict[key] for state_dict in state_dicts]
        first_shape = tensors[0].shape
        for other_index, tensor in enumerate(tensors[1:], start=1):
            if tensor.shape != first_shape:
                raise ValueError(
                    f"Mismatched shape for key {key!r}: state_dicts[0] has "
                    f"{first_shape}, state_dicts[{other_index}] has {tensor.shape}."
                )
        stacked = torch.stack([tensor.float() for tensor in tensors])
        averaged[key] = stacked.mean(dim=0).to(tensors[0].dtype)
    return averaged


def _load_state_dict(checkpoint_dir: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    weights_path = checkpoint_dir / WEIGHTS_FILENAME
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"{weights_path} not found -- average_checkpoints only supports "
            "single-file (non-sharded) model.safetensors checkpoints (every "
            "real training run's checkpoint fits comfortably under the "
            "sharding threshold for facebook/m2m100_418M's size)."
        )
    return load_file(str(weights_path))


def average_checkpoints(checkpoint_dirs: list[Path], output_dir: Path) -> Path:
    """Average the weights of `checkpoint_dirs` and write a new, standalone
    checkpoint directory to `output_dir`. Returns `output_dir`.

    Loads each checkpoint's `model.safetensors`, averages them via
    `average_state_dicts`, and writes the result as `output_dir/
    model.safetensors`. Also copies every other, non-training-state file
    (`config.json`, `generation_config.json`, ...) from the *last* (highest
    step number) entry of `checkpoint_dirs` -- these are checkpoint-
    invariant within one training run, so any one of them would do, and the
    last is the most representative choice.

    **Does not copy tokenizer files.** `Seq2SeqTrainer`'s own per-checkpoint
    saves never include them -- only `training.train.save_model_and_tokenizer`'s
    one final save, to the run's own `--model-dir` (not
    `--model-dir/checkpoints/<checkpoint>`), does. Copy the tokenizer files
    from that directory into `output_dir` separately before loading the
    averaged checkpoint with `from_pretrained`.

    Raises `ValueError` if `checkpoint_dirs` is empty, or if the
    checkpoints' weights don't actually match (see `average_state_dicts`).
    """
    if not checkpoint_dirs:
        raise ValueError("checkpoint_dirs must be non-empty")

    from safetensors.torch import save_file

    state_dicts = [_load_state_dict(checkpoint_dir) for checkpoint_dir in checkpoint_dirs]
    averaged = average_state_dicts(state_dicts)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(averaged, str(output_dir / WEIGHTS_FILENAME))

    last_checkpoint_dir = checkpoint_dirs[-1]
    for entry in last_checkpoint_dir.iterdir():
        if entry.is_file() and entry.name not in _EXCLUDED_FILENAMES:
            shutil.copy2(entry, output_dir / entry.name)

    return output_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml.training.checkpoint_averaging",
        description=(
            "Average the final N epochs' checkpoint weights from a training "
            "run's --model-dir/checkpoints/ directory (issue #182)."
        ),
    )
    parser.add_argument(
        "checkpoints_dir",
        help="Path to a training run's checkpoints directory (e.g. <model-dir>/checkpoints).",
    )
    parser.add_argument("output_dir", help="Directory to write the averaged checkpoint to.")
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of final (most recent) epoch checkpoints to average (default: 3).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    checkpoints_root = Path(args.checkpoints_dir)
    checkpoint_dirs = find_checkpoint_dirs(checkpoints_root)
    if not checkpoint_dirs:
        print(f"No checkpoint-<step> directories found under {checkpoints_root}")
        return 1

    selected = select_last_n_checkpoints(checkpoint_dirs, args.n)
    output_dir = average_checkpoints(selected, Path(args.output_dir))
    print(
        f"Averaged {len(selected)} checkpoint(s) "
        f"({', '.join(path.name for path in selected)}) -> {output_dir}"
    )
    print(
        "Note: tokenizer files were not copied -- copy them from the "
        "training run's own --model-dir before loading this checkpoint."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
