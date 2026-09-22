"""Unit tests for `training.train.build_training_arguments` (issue #79).

Constructs a real `transformers.Seq2SeqTrainingArguments` -- cheap and
CPU-safe, no model/tokenizer/forward pass needed -- so the regularization/
schedule settings (warmup, weight decay, label smoothing, gradient
accumulation) that `fine_tune` as a whole can't be tested for (see its
docstring) get real, direct coverage.
"""

from __future__ import annotations

import torch

from training.train import build_training_arguments, parse_args


def _args(tmp_path, **overrides):
    argv = [
        "--train",
        "train.tsv",
        "--validation",
        "val.tsv",
        "--corpus-version",
        "fixture-v0",
        "--model-dir",
        str(tmp_path / "model"),
    ]
    for key, value in overrides.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return parse_args(argv)


def test_build_training_arguments_applies_regularization_and_schedule_defaults(tmp_path):
    args = _args(tmp_path)

    training_args = build_training_arguments(args, num_train_examples=1000)

    # warmup_ratio (0.05) is converted to an absolute warmup_steps count,
    # since the installed transformers version doesn't accept warmup_ratio
    # directly -- see build_training_arguments's docstring. Default here:
    # ceil(1000 / (8 * 4)) * 3 epochs = 32 * 3 = 96 steps; 5% of that = 5.
    assert training_args.warmup_steps == 5
    assert training_args.weight_decay == 0.01
    # 0.0 (disabled): >0 crashes against the real M2M100 checkpoint with
    # this transformers version -- see train.py's --label-smoothing help.
    assert training_args.label_smoothing_factor == 0.0
    assert training_args.gradient_accumulation_steps == 4
    # Conditional on real CUDA availability, not hardcoded -- see
    # build_training_arguments's docstring for why (fp16=True with no CUDA
    # fails at Trainer.train() time, not construction, so this can't be
    # left to callers to override).
    assert training_args.fp16 is torch.cuda.is_available()


def test_build_training_arguments_respects_cli_overrides(tmp_path):
    args = _args(
        tmp_path,
        warmup_ratio=0.1,
        weight_decay=0.02,
        label_smoothing=0.2,
        gradient_accumulation_steps=8,
    )

    training_args = build_training_arguments(args, num_train_examples=1000)

    assert training_args.weight_decay == 0.02
    assert training_args.label_smoothing_factor == 0.2
    assert training_args.gradient_accumulation_steps == 8


def test_build_training_arguments_computes_warmup_steps_from_ratio_and_dataset_size(tmp_path):
    args = _args(tmp_path, epochs=2, batch_size=10, gradient_accumulation_steps=1, warmup_ratio=0.5)

    # steps_per_epoch = ceil(100 / (10 * 1)) = 10; total_steps = 20; warmup = 0.5 * 20 = 10.
    training_args = build_training_arguments(args, num_train_examples=100)

    assert training_args.warmup_steps == 10


def test_build_training_arguments_passes_through_core_hyperparameters(tmp_path):
    args = _args(tmp_path, epochs=7, batch_size=16, learning_rate=1e-4, seed=99)

    training_args = build_training_arguments(args, num_train_examples=1000)

    assert training_args.num_train_epochs == 7
    assert training_args.per_device_train_batch_size == 16
    assert training_args.per_device_eval_batch_size == 16
    assert training_args.learning_rate == 1e-4
    assert training_args.seed == 99
