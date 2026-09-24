"""Fast smoke test for `evaluation/evaluate_checkpoint.py`'s wiring:
checkpoint resolution -> tokenizer/model loading -> direction-tagged
example building -> translation -> evaluation -> model card -- against
tiny fixture data and duck-typed fakes.

Per docs/testing.md and issue #108's scope, this never downloads a real
checkpoint or runs a real `model.generate()` call: `generate_translations`
is injected as a fake, same "duck-type and fixture" approach as
`tests/integration/test_train_pipeline.py`. Everything else -- reading the
fixture validation TSV, building direction-tagged examples, resolving the
checkpoint source, and computing BLEU/chrF + rendering the model card --
runs for real.
"""

from __future__ import annotations

from pathlib import Path

from evaluation.evaluate_checkpoint import parse_args, run_checkpoint_evaluation
from training.direction import DIRECTION_TAGS

FIXTURES = Path(__file__).parent.parent / "fixtures"


class FakeCheckpointTokenizer:
    """Duck-typed fake standing in for a real, already-vocab-extended
    `M2M100Tokenizer` loaded straight from a checkpoint. Exposes
    `add_tokens` purely so `test_run_checkpoint_evaluation_never_extends_
    vocabulary` below can assert it's never called -- a real checkpoint's
    tokenizer already has its extended vocabulary saved, so calling this
    again here would be wrong (see `evaluate_checkpoint`'s module
    docstring).
    """

    def __init__(self):
        self.add_tokens_calls: list[list[str]] = []

    def add_tokens(self, new_tokens: list[str]) -> int:
        self.add_tokens_calls.append(list(new_tokens))
        return len(new_tokens)


class FakeCheckpointModel:
    """Duck-typed fake standing in for a real, already fine-tuned
    `M2M100ForConditionalGeneration` loaded straight from a checkpoint. No
    forward pass, no download, no embedding resize -- an eval-only run
    never touches the model's parameters.
    """

    def __init__(self):
        self.resize_calls: list[int] = []

    def resize_token_embeddings(self, new_size: int):
        self.resize_calls.append(new_size)


def fake_model_loader(source: str) -> tuple[FakeCheckpointTokenizer, FakeCheckpointModel]:
    fake_model_loader.calls.append(source)
    return FakeCheckpointTokenizer(), FakeCheckpointModel()


fake_model_loader.calls = []


def fake_resolve_source(checkpoint: str) -> str:
    fake_resolve_source.calls.append(checkpoint)
    return f"resolved::{checkpoint}"


fake_resolve_source.calls = []


def fake_base_vocab_loader(base_model: str) -> dict[str, int]:
    fake_base_vocab_loader.calls.append(base_model)
    # A tiny, deliberately incomplete "pristine" vocab: `sample_train.tsv`'s
    # text (see FIXTURES / "sample_train.tsv") has real whole-word gaps
    # against this (e.g. "awäch"), so
    # `patch_word_boundary_decoding_for_checkpoint` has something genuine
    # to reconstruct -- mirrors the real base_model vocab's role, not a
    # no-op stand-in.
    return {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyzáéíóúñ,. ")}


fake_base_vocab_loader.calls = []


def fake_translator(model, tokenizer, examples, **kwargs):
    # Deliberately not a real translation -- just proves the val examples
    # (and their target language tags) reached this step correctly, same
    # as test_train_pipeline.py's fake_translator.
    return [f"{DIRECTION_TAGS[ex.target_lang]}::{ex.target_text}" for ex in examples]


def _base_args(tmp_path, **overrides):
    output_dir = tmp_path / "output"
    argv = [
        "--checkpoint",
        overrides.pop("checkpoint", "s3://bucket/model-artifacts/run-1/output/model.tar.gz"),
        "--validation",
        str(FIXTURES / "sample_val_clean.tsv"),
        "--train",
        overrides.pop("train", str(FIXTURES / "sample_train.tsv")),
        "--corpus-version",
        "almg-v1",
        "--source-run-id",
        overrides.pop("source_run_id", "run-20260101T000000Z"),
        "--output-dir",
        str(output_dir),
        "--run-id",
        overrides.pop("run_id", "reeval-smoke-test"),
    ]
    for key, value in overrides.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return parse_args(argv), output_dir


def test_run_checkpoint_evaluation_wires_checkpoint_through_to_model_card(tmp_path):
    fake_model_loader.calls.clear()
    fake_resolve_source.calls.clear()
    args, output_dir = _base_args(tmp_path)

    model_card_path = run_checkpoint_evaluation(
        args,
        resolve_source=fake_resolve_source,
        model_loader=fake_model_loader,
        base_vocab_loader=fake_base_vocab_loader,
        translator=fake_translator,
    )

    # 1. The checkpoint source was resolved, then handed to the model loader.
    assert fake_resolve_source.calls == [
        "s3://bucket/model-artifacts/run-1/output/model.tar.gz"
    ]
    assert fake_model_loader.calls == [
        "resolved::s3://bucket/model-artifacts/run-1/output/model.tar.gz"
    ]

    # 2. Predictions/references were written, one line per direction-tagged
    #    validation example (2 pairs x 2 directions, default "both").
    predictions_path = output_dir / "predictions.txt"
    references_path = output_dir / "references.txt"
    assert predictions_path.exists()
    assert references_path.exists()
    assert len(predictions_path.read_text(encoding="utf-8").splitlines()) == 4
    assert len(references_path.read_text(encoding="utf-8").splitlines()) == 4

    # 3. A model card was rendered with real BLEU/chrF, carrying this run's
    #    own identifiers plus provenance back to the checkpoint's original
    #    training run.
    assert model_card_path == output_dir / "model_card.md"
    card_text = model_card_path.read_text(encoding="utf-8")
    assert "reeval-smoke-test" in card_text
    assert "almg-v1" in card_text
    assert "run-20260101T000000Z" in card_text
    assert "**Validation sentences**: 4" in card_text
    assert "reevaluation" in card_text.lower()


def test_run_checkpoint_evaluation_never_extends_vocabulary(tmp_path):
    args, _ = _base_args(tmp_path)

    captured_tokenizer: dict[str, FakeCheckpointTokenizer] = {}

    def capturing_model_loader(source: str):
        tokenizer, model = fake_model_loader(source)
        captured_tokenizer["tokenizer"] = tokenizer
        return tokenizer, model

    run_checkpoint_evaluation(
        args,
        resolve_source=fake_resolve_source,
        model_loader=capturing_model_loader,
        base_vocab_loader=fake_base_vocab_loader,
        translator=fake_translator,
    )

    assert captured_tokenizer["tokenizer"].add_tokens_calls == []


def test_run_checkpoint_evaluation_single_direction_evaluates_half_the_examples(tmp_path):
    args, output_dir = _base_args(tmp_path, direction="es->cak")

    run_checkpoint_evaluation(
        args,
        resolve_source=fake_resolve_source,
        model_loader=fake_model_loader,
        base_vocab_loader=fake_base_vocab_loader,
        translator=fake_translator,
    )

    references_path = output_dir / "references.txt"
    assert len(references_path.read_text(encoding="utf-8").splitlines()) == 2


def test_run_checkpoint_evaluation_defaults_run_id_to_a_reeval_prefixed_timestamp(tmp_path):
    output_dir = tmp_path / "output"
    args = parse_args(
        [
            "--checkpoint",
            "local-checkpoint-dir",
            "--validation",
            str(FIXTURES / "sample_val_clean.tsv"),
            "--train",
            str(FIXTURES / "sample_train.tsv"),
            "--corpus-version",
            "almg-v1",
            "--source-run-id",
            "run-20260101T000000Z",
            "--output-dir",
            str(output_dir),
        ]
    )

    model_card_path = run_checkpoint_evaluation(
        args,
        resolve_source=fake_resolve_source,
        model_loader=fake_model_loader,
        base_vocab_loader=fake_base_vocab_loader,
        translator=fake_translator,
    )

    card_text = model_card_path.read_text(encoding="utf-8")
    assert "# Model card: reeval-" in card_text


def test_run_checkpoint_evaluation_reconstructs_word_boundary_tokens_from_train_corpus(tmp_path):
    """Issue #116's re-evaluation gap: a reloaded checkpoint's tokenizer
    never records which added tokens need word-boundary decoding, so this
    must be reconstructed from `--train`/`--train-direction`/`--base-model`
    every run -- see `evaluate_checkpoint`'s module docstring.
    """
    fake_base_vocab_loader.calls.clear()
    args, _ = _base_args(tmp_path)

    model_card_path = run_checkpoint_evaluation(
        args,
        resolve_source=fake_resolve_source,
        model_loader=fake_model_loader,
        base_vocab_loader=fake_base_vocab_loader,
        translator=fake_translator,
    )

    # The base tokenizer's *pristine* vocab was loaded via --base-model,
    # never the checkpoint's own (already-extended) tokenizer.
    assert fake_base_vocab_loader.calls == [args.base_model]

    card_text = model_card_path.read_text(encoding="utf-8")
    assert "word_boundary_reconstruction_train" in card_text
    assert str(FIXTURES / "sample_train.tsv") in card_text
    assert "word_boundary_reconstruction_train_direction" in card_text
    assert "word_boundary_tokens_reconstructed" in card_text
    # sample_train.tsv's text has genuine whole-word gaps against the tiny
    # fake base vocab (see fake_base_vocab_loader's own docstring) -- this
    # must be a real, non-zero reconstruction, not an accidental no-op.
    assert "- **word_boundary_tokens_reconstructed**: 0" not in card_text


def test_run_checkpoint_evaluation_warns_when_reconstructed_tokens_miss_the_checkpoint(
    tmp_path, capsys
):
    """If `--train`/`--train-direction`/`--base-model` don't actually match
    what the checkpoint was trained with, the reconstructed boundary
    tokens won't all be present in the checkpoint's own vocabulary -- this
    must be surfaced loudly (stderr), not silently ignored.
    """
    args, _ = _base_args(tmp_path)

    class FakeCheckpointTokenizerWithVocab(FakeCheckpointTokenizer):
        def get_vocab(self) -> dict[str, int]:
            # Deliberately missing every reconstructed boundary token --
            # simulates a checkpoint that was never actually trained
            # against this --train corpus.
            return {c: i for i, c in enumerate("xyz")}

    def mismatched_model_loader(source: str):
        return FakeCheckpointTokenizerWithVocab(), FakeCheckpointModel()

    run_checkpoint_evaluation(
        args,
        resolve_source=fake_resolve_source,
        model_loader=mismatched_model_loader,
        base_vocab_loader=fake_base_vocab_loader,
        translator=fake_translator,
    )

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "missing from the checkpoint" in captured.err
