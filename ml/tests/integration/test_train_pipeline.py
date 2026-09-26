"""Fast smoke test for the SageMaker training entrypoint's wiring
(`training/train.py`): corpus loading -> direction-tagged example building
-> vocab/embedding extension -> fine-tuning -> saving -> evaluation ->
model card -- against tiny fixture data, a duck-typed fake tokenizer, and a
minimal real-`torch` fake model.

Per docs/testing.md, this confirms the pieces connect and call order is
correct, never runs a real multi-epoch fine-tuning pass or downloads a real
checkpoint: `fine_tune` (the actual gradient-descent loop) and
`generate_translations` (real beam/greedy decoding) are injected as fakes,
since those are the two steps that would otherwise require a full
forward/backward pass or a real generate() call. Everything else --
reading the fixture TSVs, building direction-tagged examples, extending the
fake tokenizer's vocabulary and resizing the fake model's embeddings,
saving the model+tokenizer, and computing BLEU/chrF + rendering the model
card -- runs for real.
"""

from __future__ import annotations

from pathlib import Path

import torch

import training.train as train_module
from training.direction import DIRECTION_TAGS
from training.tokenizer_extension import (
    extend_tokenizer_vocab as real_extend_tokenizer_vocab,
)
from training.tokenizer_extension import (
    extend_tokenizer_vocab_with_subwords as real_extend_tokenizer_vocab_with_subwords,
)
from training.train import (
    VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY,
    parse_args,
    run_training_job,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


class FakeEmbedding(torch.nn.Module):
    """Minimal duck-typed stand-in for an `nn.Embedding`: just a weight
    matrix, matching what `resize_embeddings_for_new_tokens` touches.
    """

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)


class FakeM2M100Model:
    """Duck-typed fake standing in for `M2M100ForConditionalGeneration`,
    covering only the methods `training.tokenizer_extension` and
    `training.train.save_model_and_tokenizer` actually call: no real
    forward pass, no download.
    """

    def __init__(self, vocab_size: int, dim: int = 4):
        weight = torch.zeros(vocab_size, dim)
        self._embeddings = FakeEmbedding(weight)
        self.saved_to: str | None = None

    def get_input_embeddings(self) -> FakeEmbedding:
        return self._embeddings

    def resize_token_embeddings(self, new_size: int) -> FakeEmbedding:
        old_weight = self._embeddings.weight.detach()
        dim = old_weight.shape[1]
        new_weight = torch.zeros(new_size, dim)
        new_weight[: old_weight.shape[0]] = old_weight
        self._embeddings = FakeEmbedding(new_weight)
        return self._embeddings

    def save_pretrained(self, save_directory: str) -> None:
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / "fake_model.txt").write_text(
            f"vocab_size={self._embeddings.weight.shape[0]}\n", encoding="utf-8"
        )
        self.saved_to = save_directory


class FakeM2M100Tokenizer:
    """Duck-typed fake matching the tokenizer methods `train.py` relies on:
    `get_vocab`/`add_tokens` (vocab extension) and `save_pretrained`.
    """

    def __init__(self, vocab: dict[str, int]):
        self._vocab = dict(vocab)
        self.saved_to: str | None = None

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def add_tokens(self, new_tokens: list[str]) -> int:
        added = 0
        next_id = max(self._vocab.values(), default=-1) + 1
        for token in new_tokens:
            if token in self._vocab:
                continue
            self._vocab[token] = next_id
            next_id += 1
            added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab[token]

    def save_pretrained(self, save_directory: str) -> None:
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / "fake_tokenizer.txt").write_text(
            f"vocab_size={len(self._vocab)}\n", encoding="utf-8"
        )
        self.saved_to = save_directory


def _base_vocab() -> dict[str, int]:
    # Plain ASCII letters + Spanish diacritics -- deliberately missing the
    # Kaqchikel glottal apostrophe and "ä", same gap as the real M2M100
    # vocab (see training/vocab_gap.py).
    return {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyzáéíóúñ. ")}


def fake_model_loader(base_model: str) -> tuple[FakeM2M100Tokenizer, FakeM2M100Model]:
    fake_model_loader.calls.append(base_model)
    vocab = _base_vocab()
    tokenizer = FakeM2M100Tokenizer(vocab)
    model = FakeM2M100Model(vocab_size=len(vocab))
    return tokenizer, model


fake_model_loader.calls = []


def fake_trainer(model, tokenizer, train_examples, eval_examples, args):
    fake_trainer.calls.append(
        {
            "num_train_examples": len(train_examples),
            "num_eval_examples": len(eval_examples),
            "epochs": args.epochs,
        }
    )
    return model


fake_trainer.calls = []


def fake_translator(model, tokenizer, examples, **kwargs):
    # Deliberately not a real translation -- just proves the val examples
    # (and their target language tags) reached this step correctly.
    return [f"{DIRECTION_TAGS[ex.target_lang]}::{ex.target_text}" for ex in examples]


def test_run_training_job_wires_corpus_through_to_model_card(tmp_path):
    fake_trainer.calls.clear()
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"

    args = parse_args(
        [
            "--train",
            str(FIXTURES / "sample_train.tsv"),
            "--validation",
            str(FIXTURES / "sample_val_clean.tsv"),
            "--corpus-version",
            "fixture-v0",
            "--model-dir",
            str(model_dir),
            "--output-data-dir",
            str(output_dir),
            "--run-id",
            "smoke-test-run",
            "--epochs",
            "1",
        ]
    )

    model_card_path = run_training_job(
        args,
        model_loader=fake_model_loader,
        trainer=fake_trainer,
        translator=fake_translator,
    )

    # 1. Fine-tuning was called once, with direction-tagged examples for
    #    both directions (default direction is "both" -- see
    #    training/direction.py) covering both training pairs twice over.
    assert len(fake_trainer.calls) == 1
    call = fake_trainer.calls[0]
    assert call["num_train_examples"] == 4  # 2 pairs x 2 directions
    assert call["num_eval_examples"] == 4
    assert call["epochs"] == 1

    # 2. The model+tokenizer were saved to SM_MODEL_DIR.
    assert (model_dir / "fake_model.txt").exists()
    assert (model_dir / "fake_tokenizer.txt").exists()

    # 3. Vocabulary was actually extended: the fixtures contain "ä" and the
    #    glottal apostrophe, both missing from the base vocab, plus the two
    #    direction tags.
    saved_tokenizer_vocab_size = int(
        (model_dir / "fake_tokenizer.txt").read_text().split("=")[1]
    )
    assert saved_tokenizer_vocab_size > len(_base_vocab())

    saved_model_vocab_size = int((model_dir / "fake_model.txt").read_text().split("=")[1])
    assert saved_model_vocab_size == saved_tokenizer_vocab_size

    # 4. Predictions/references were written and a model card was rendered
    #    alongside the saved model artifact, with real BLEU/chrF -- not a
    #    hardcoded placeholder.
    assert (output_dir / "predictions.txt").exists()
    assert (output_dir / "references.txt").exists()

    assert model_card_path == model_dir / "model_card.md"
    card_text = model_card_path.read_text(encoding="utf-8")
    assert "smoke-test-run" in card_text
    assert "fixture-v0" in card_text
    assert "facebook/m2m100_418M" in card_text
    assert "both" in card_text
    assert "**Validation sentences**: 4" in card_text

    # 5. Issue #143: every run going forward must record which
    #    vocab-extension scoping it used, so a future re-evaluation of this
    #    checkpoint (evaluation.evaluate_checkpoint) can tell it apart from
    #    a pre-#125 checkpoint whose model card never recorded this field.
    assert f"- **vocab_extension_scoping**: {VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY}" in card_text


def _run_training_job_for_vocab_size(tmp_path, subdir: str, run_id: str) -> tuple[int, str]:
    """Run `run_training_job` against the richer fixture corpus and return
    the saved tokenizer's vocab size plus the rendered model card text.
    Shared by the subword-extension tests below, which each need a fresh
    model/output dir and a fresh fake tokenizer per run.
    """
    fake_trainer.calls.clear()
    model_dir = tmp_path / f"{subdir}-model"
    output_dir = tmp_path / f"{subdir}-output"

    args = parse_args(
        [
            "--train",
            str(FIXTURES / "sample_train_richer.tsv"),
            "--validation",
            str(FIXTURES / "sample_val_clean.tsv"),
            "--corpus-version",
            "fixture-v0",
            "--model-dir",
            str(model_dir),
            "--output-data-dir",
            str(output_dir),
            "--run-id",
            run_id,
            "--epochs",
            "1",
            "--subword-vocab-size",
            "8000",
        ]
    )

    run_training_job(
        args,
        model_loader=fake_model_loader,
        trainer=fake_trainer,
        translator=fake_translator,
    )

    vocab_size = int((model_dir / "fake_tokenizer.txt").read_text().split("=")[1])
    card_text = (model_dir / "model_card.md").read_text(encoding="utf-8")
    return vocab_size, card_text


def test_run_training_job_subword_step_actually_contributes_vocab_growth(tmp_path, monkeypatch):
    """Issue #82: the subword-extension step must make a real, verifiable
    difference to the saved vocab -- not just "the vocab grew somehow",
    which the pre-existing whole-word/character extension alone already
    guarantees on this richer fixture regardless of whether the subword
    step runs at all (a prior version of this test only checked that,
    and stayed green even with the subword step short-circuited to a
    no-op -- see PR #83 review). Comparing a real run against one where
    the step is forced to contribute zero tokens is what actually fails
    if the step is removed or no-op'd.
    """
    with_subwords, card_text = _run_training_job_for_vocab_size(
        tmp_path, "with-subwords", "smoke-test-subword-run"
    )

    monkeypatch.setattr(
        train_module,
        "extend_tokenizer_vocab_with_subwords",
        lambda tokenizer, kaqchikel_texts, **kwargs: [],
    )
    without_subwords, _ = _run_training_job_for_vocab_size(
        tmp_path, "without-subwords", "smoke-test-no-subword-run"
    )

    assert with_subwords > without_subwords
    assert "8000" in card_text  # subword_vocab_size hyperparameter recorded


def test_run_training_job_feeds_kaqchikel_only_text_to_subword_extension(tmp_path, monkeypatch):
    """The texts handed to the subword step must be the Kaqchikel side of
    the corpus only -- never the Spanish glosses, which M2M100 already
    tokenizes natively (see training/subword_vocab.py's module docstring).
    """
    captured_calls: list[list[str]] = []

    def spy(tokenizer, kaqchikel_texts, **kwargs):
        kaqchikel_texts = list(kaqchikel_texts)
        captured_calls.append(kaqchikel_texts)
        return real_extend_tokenizer_vocab_with_subwords(tokenizer, kaqchikel_texts, **kwargs)

    monkeypatch.setattr(train_module, "extend_tokenizer_vocab_with_subwords", spy)

    _run_training_job_for_vocab_size(tmp_path, "spy-run", "smoke-test-spy-run")

    assert len(captured_calls) == 1
    texts = captured_calls[0]
    assert texts  # the subword step actually received something to train on

    spanish_only_words = ("Buenos", "pueblo", "niño", "trabajo", "maestro", "gente")
    for text in texts:
        assert not any(word in text for word in spanish_only_words)

    # A real Kaqchikel word-form from the fixture, present on the
    # Kaqchikel side only.
    assert any("qatinamit" in text for text in texts)


def test_run_training_job_feeds_kaqchikel_only_text_to_whole_word_extension(tmp_path, monkeypatch):
    """Issue #125: the whole-word/character extension step
    (`extend_tokenizer_vocab`) must isolate the Kaqchikel side of the
    corpus the same way the subword step already does (see
    `test_run_training_job_feeds_kaqchikel_only_text_to_subword_extension`
    above) -- never mixed with Spanish source/target text across either
    direction.

    Before this fix, `extend_vocabulary_for_examples` fed *both* every
    example's source *and* target text unconditionally into this step,
    which registers a brand-new standalone token for any exact word-form
    not already a single vocab entry -- including ordinary Spanish surface
    forms the base M2M100 tokenizer already represents correctly via
    existing subwords (confirmed empirically against the real checkpoint
    for issue #125: 100% of a real sample of Spanish "missing" whole
    words were already fully representable, zero `<unk>`), not genuinely
    uncoverable Spanish content. Mixing Spanish in here also contradicts
    this same function's own subword step, which already isolates
    Kaqchikel-only text specifically because "M2M100 already tokenizes
    [Spanish] natively" (see `extend_vocabulary_for_examples`'s
    docstring).
    """
    captured_calls: list[list[str]] = []

    def spy(tokenizer, sample_texts):
        sample_texts = list(sample_texts)
        captured_calls.append(sample_texts)
        return real_extend_tokenizer_vocab(tokenizer, sample_texts)

    monkeypatch.setattr(train_module, "extend_tokenizer_vocab", spy)

    _run_training_job_for_vocab_size(
        tmp_path, "whole-word-spy-run", "smoke-test-whole-word-spy-run"
    )

    assert len(captured_calls) == 1
    texts = captured_calls[0]
    assert texts  # the whole-word step actually received something to train on

    spanish_only_words = ("Buenos", "pueblo", "niño", "trabajo", "maestro", "gente")
    for text in texts:
        assert not any(word in text for word in spanish_only_words)

    # A real Kaqchikel word-form from the fixture, present on the
    # Kaqchikel side only.
    assert any("qatinamit" in text for text in texts)

    # Direction tags must still reach this step -- they need real,
    # warm-started embedding rows too (see the function's own docstring).
    assert any("__cak__" in text for text in texts)
    assert any("__es__" in text for text in texts)


def test_run_training_job_single_direction_trains_half_the_examples(tmp_path):
    fake_trainer.calls.clear()

    args = parse_args(
        [
            "--train",
            str(FIXTURES / "sample_train.tsv"),
            "--validation",
            str(FIXTURES / "sample_val_clean.tsv"),
            "--corpus-version",
            "fixture-v0",
            "--model-dir",
            str(tmp_path / "model"),
            "--output-data-dir",
            str(tmp_path / "output"),
            "--direction",
            "es->cak",
        ]
    )

    run_training_job(
        args,
        model_loader=fake_model_loader,
        trainer=fake_trainer,
        translator=fake_translator,
    )

    call = fake_trainer.calls[0]
    assert call["num_train_examples"] == 2
    assert call["num_eval_examples"] == 2


def test_run_training_job_loads_base_model_when_no_init_model_given(tmp_path):
    fake_model_loader.calls.clear()

    args = parse_args(
        [
            "--train",
            str(FIXTURES / "sample_train.tsv"),
            "--validation",
            str(FIXTURES / "sample_val_clean.tsv"),
            "--corpus-version",
            "fixture-v0",
            "--model-dir",
            str(tmp_path / "model"),
            "--output-data-dir",
            str(tmp_path / "output"),
            "--base-model",
            "facebook/m2m100_418M",
        ]
    )

    run_training_job(
        args, model_loader=fake_model_loader, trainer=fake_trainer, translator=fake_translator
    )

    assert fake_model_loader.calls == ["facebook/m2m100_418M"]


def test_run_training_job_resumes_from_init_model_when_given(tmp_path):
    fake_model_loader.calls.clear()
    checkpoint_dir = tmp_path / "prior-checkpoint"
    checkpoint_dir.mkdir()

    args = parse_args(
        [
            "--train",
            str(FIXTURES / "sample_train.tsv"),
            "--validation",
            str(FIXTURES / "sample_val_clean.tsv"),
            "--corpus-version",
            "fixture-v0",
            "--model-dir",
            str(tmp_path / "model"),
            "--output-data-dir",
            str(tmp_path / "output"),
            "--base-model",
            "facebook/m2m100_418M",
            "--init-model",
            str(checkpoint_dir),
        ]
    )

    run_training_job(
        args, model_loader=fake_model_loader, trainer=fake_trainer, translator=fake_translator
    )

    # Loaded from the checkpoint, not the base pretrained model.
    assert fake_model_loader.calls == [str(checkpoint_dir)]


def test_run_training_job_propagates_kaqchikel_only_scoping_across_a_fully_scoped_chain(tmp_path):
    """PR #144 review finding: recording a fixed `vocab_extension_scoping`
    on every run regardless of continuation lineage is only correct if the
    *entire* chain behind `--init-model` was itself fully Kaqchikel-only
    scoped. Continuing from an ancestor whose own model card also records
    `kaqchikel_only` means the whole lineage is scoped -- the new run's
    model card must say so too.
    """
    checkpoint_dir = tmp_path / "prior-checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_card.md").write_text(
        "# Model card: run-ancestor\n\n## Hyperparameters\n\n"
        "- **new_tokens_added**: 10\n"
        f"- **vocab_extension_scoping**: {VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY}\n",
        encoding="utf-8",
    )
    model_dir = tmp_path / "model"

    args = parse_args(
        [
            "--train",
            str(FIXTURES / "sample_train.tsv"),
            "--validation",
            str(FIXTURES / "sample_val_clean.tsv"),
            "--corpus-version",
            "fixture-v0",
            "--model-dir",
            str(model_dir),
            "--output-data-dir",
            str(tmp_path / "output"),
            "--init-model",
            str(checkpoint_dir),
            "--run-id",
            "smoke-test-scoped-chain",
        ]
    )

    run_training_job(
        args, model_loader=fake_model_loader, trainer=fake_trainer, translator=fake_translator
    )

    card_text = (model_dir / "model_card.md").read_text(encoding="utf-8")
    assert f"- **vocab_extension_scoping**: {VOCAB_EXTENSION_SCOPING_KAQCHIKEL_ONLY}" in card_text


def test_run_training_job_omits_scoping_when_continuing_from_a_pre_125_checkpoint(tmp_path):
    """The core PR #144 review finding: continuing from a checkpoint whose
    own model card never recorded `vocab_extension_scoping` (every
    checkpoint trained before issue #125, e.g. the currently-deployed
    `run-20260922T141956Z`) must NOT let the new run's model card claim
    `kaqchikel_only` for the whole checkpoint -- that would make
    evaluation.evaluate_checkpoint drop the Spanish column during
    reconstruction and silently miss the earlier chain's Spanish-derived
    whole-word tokens, reintroducing issue #116's glued-word bug for them.
    The field must be omitted entirely, falling back to the same legacy
    (safe, superset) reconstruction the checkpoint's real mixed vocabulary
    actually needs.
    """
    checkpoint_dir = tmp_path / "prior-checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_card.md").write_text(
        "# Model card: run-ancestor\n\n## Hyperparameters\n\n- **new_tokens_added**: 10\n",
        encoding="utf-8",
    )
    model_dir = tmp_path / "model"

    args = parse_args(
        [
            "--train",
            str(FIXTURES / "sample_train.tsv"),
            "--validation",
            str(FIXTURES / "sample_val_clean.tsv"),
            "--corpus-version",
            "fixture-v0",
            "--model-dir",
            str(model_dir),
            "--output-data-dir",
            str(tmp_path / "output"),
            "--init-model",
            str(checkpoint_dir),
            "--run-id",
            "smoke-test-unscoped-chain",
        ]
    )

    run_training_job(
        args, model_loader=fake_model_loader, trainer=fake_trainer, translator=fake_translator
    )

    card_text = (model_dir / "model_card.md").read_text(encoding="utf-8")
    assert "vocab_extension_scoping" not in card_text


def test_run_training_job_omits_scoping_when_ancestor_checkpoint_has_no_model_card(tmp_path):
    """Same defensive fallback as the pre-#125 case above, for a checkpoint
    directory that has no `model_card.md` at all (e.g. one not produced by
    `run_training_job` itself) -- never crash, and never claim a scoping
    this script can't actually verify.
    """
    checkpoint_dir = tmp_path / "prior-checkpoint"
    checkpoint_dir.mkdir()
    model_dir = tmp_path / "model"

    args = parse_args(
        [
            "--train",
            str(FIXTURES / "sample_train.tsv"),
            "--validation",
            str(FIXTURES / "sample_val_clean.tsv"),
            "--corpus-version",
            "fixture-v0",
            "--model-dir",
            str(model_dir),
            "--output-data-dir",
            str(tmp_path / "output"),
            "--init-model",
            str(checkpoint_dir),
            "--run-id",
            "smoke-test-no-ancestor-card",
        ]
    )

    run_training_job(
        args, model_loader=fake_model_loader, trainer=fake_trainer, translator=fake_translator
    )

    card_text = (model_dir / "model_card.md").read_text(encoding="utf-8")
    assert "vocab_extension_scoping" not in card_text
