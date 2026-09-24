"""Integration test for reconstructing issue #116's word-boundary-spacing
fix on a checkpoint tokenizer *reloaded* from disk -- the real gap this
project hit trying to re-evaluate an already-trained checkpoint against
#116's fix without retraining (see `training.tokenizer_extension`'s module
docstring, "Re-evaluating an already-trained checkpoint").

`_mark_word_boundary_tokens`'s effect (a monkey-patched
`convert_tokens_to_string` plus a mutable set) lives only on a live
tokenizer instance -- `save_pretrained()` never persists it. This test
proves that end-to-end against the real `facebook/m2m100_418M` tokenizer:
extend + save + reload loses the fix, and
`patch_word_boundary_decoding_for_checkpoint` recovers it without ever
calling `add_tokens()` again.

Uses the real tokenizer (not the duck-typed fakes in
`tests/unit/test_tokenizer_extension.py`) because the bug this guards
against lives in the interaction between HF's added-token overlay and the
real `sentencepiece` C++ decoder -- same reasoning as
`test_tokenizer_extension_word_boundary_spacing.py`.
"""

from transformers import M2M100Tokenizer

from training.tokenizer_extension import (
    extend_tokenizer_vocab,
    patch_word_boundary_decoding_for_checkpoint,
)

MODEL_NAME = "facebook/m2m100_418M"


def test_reload_loses_the_fix_and_reconstruction_recovers_it(tmp_path):
    # 1. A pristine base tokenizer's vocab -- the "exact base model vocab"
    #    reconstruct_whole_word_boundary_tokens's docstring requires.
    pristine_tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
    pristine_base_vocab = pristine_tokenizer.get_vocab()

    # 2. A live training-time extension: correctly marks "awach" as a word
    #    boundary via the (in-memory only) convert_tokens_to_string patch.
    live_tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
    sample_texts = ["de dios awach y todo el pueblo"]
    added = extend_tokenizer_vocab(live_tokenizer, sample_texts)
    assert "awach" in added

    sentence = "de dios awach y todo el pueblo"
    live_decoded = live_tokenizer.decode(
        live_tokenizer(sentence).input_ids, skip_special_tokens=True
    )
    assert live_decoded == sentence, "live, never-saved tokenizer should already decode correctly"

    # 3. Save + reload -- simulates a real training job's
    #    save_pretrained()/from_pretrained() checkpoint round-trip. This is
    #    exactly where the in-memory convert_tokens_to_string patch is lost.
    checkpoint_dir = tmp_path / "checkpoint"
    live_tokenizer.save_pretrained(checkpoint_dir)
    reloaded_tokenizer = M2M100Tokenizer.from_pretrained(checkpoint_dir)

    ids = reloaded_tokenizer(sentence).input_ids
    reloaded_decoded_before_patch = reloaded_tokenizer.decode(ids, skip_special_tokens=True)
    assert reloaded_decoded_before_patch != sentence, (
        "expected the reload to have genuinely lost the word-boundary fix "
        f"(got the *correct* {reloaded_decoded_before_patch!r} -- if this "
        "starts passing, save_pretrained may have started persisting the "
        "fix and this whole reconstruction path may no longer be needed)"
    )
    assert "diosawach" in reloaded_decoded_before_patch

    # 4. Reconstruct + patch, using only the pristine base vocab and the
    #    same sample text the original extension saw -- never the
    #    checkpoint's own (already-extended) vocab, and never re-calling
    #    add_tokens.
    applied = patch_word_boundary_decoding_for_checkpoint(
        reloaded_tokenizer, pristine_base_vocab, sample_texts
    )
    assert "awach" in applied

    reloaded_decoded_after_patch = reloaded_tokenizer.decode(
        reloaded_tokenizer(sentence).input_ids, skip_special_tokens=True
    )
    assert reloaded_decoded_after_patch == sentence
