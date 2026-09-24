"""Regression test for issue #116: `extend_tokenizer_vocab` /
`extend_tokenizer_vocab_with_subwords` silently dropping the word-boundary
space around newly added tokens when the extended tokenizer is later used
to *decode* generated ids back into text.

Root cause (confirmed against the real `facebook/m2m100_418M` tokenizer,
not assumed): `tokenizer.add_tokens()` only registers new tokens in HF's
`added_tokens_encoder` overlay. `M2M100Tokenizer.convert_tokens_to_string`
decodes by handing the *entire* token list to `self.sp_model.decode(...)`
-- the real `sentencepiece.SentencePieceProcessor`, which knows nothing
about that overlay. Encoding still works fine (HF's generic added-token
trie matches the literal added text directly against raw input, before
`sp_model` ever gets involved), but on decode, `sp_model.decode()` has no
way to apply its usual "`▁` word-boundary marker -> space" handling to a
token string it has never seen: it silently concatenates the added token
directly onto whatever piece came before it, producing e.g.
`"...de dios"` + `"awach"` -> `"...de diosawach"` instead of
`"...de dios awach"`.

This needs the real tokenizer/model download (`facebook/m2m100_418M`,
~1.9GB, cached under `~/.cache/huggingface` after first run -- see
`test_tokenizer_extension_real_model.py`'s module docstring) because the
bug lives in the interaction between HF's added-token overlay and the
real `sentencepiece` C++ decoder; a duck-typed fake tokenizer (as used in
`tests/unit/test_tokenizer_extension.py`) has no equivalent internal
SentencePiece decode path to reproduce it against.
"""

from transformers import M2M100Tokenizer

from training.tokenizer_extension import (
    extend_tokenizer_vocab,
    extend_tokenizer_vocab_with_subwords,
)

MODEL_NAME = "facebook/m2m100_418M"


def _fresh_tokenizer() -> M2M100Tokenizer:
    # Deliberately not a shared/module-scoped fixture: each test mutates
    # the tokenizer's vocabulary in place via `add_tokens`, and tests must
    # not see each other's added tokens.
    return M2M100Tokenizer.from_pretrained(MODEL_NAME)


def test_extend_tokenizer_vocab_decodes_added_word_with_correct_spacing():
    tokenizer = _fresh_tokenizer()

    # "awach" ("your face" without the ä M2M100 lacks -- see vocab_gap.py)
    # is a realistic whole-word addition sitting between two ordinary
    # Spanish words, exactly like the corpus evidence in issue #116
    # (e.g. "de dios" glued to a neighboring added token as "dediosawach").
    added = extend_tokenizer_vocab(tokenizer, ["awach"])
    assert "awach" in added, "expected 'awach' to be a genuinely new token for base M2M100 vocab"

    sentence = "de dios awach y todo el pueblo"
    ids = tokenizer(sentence).input_ids
    decoded = tokenizer.decode(ids, skip_special_tokens=True)

    assert "diosawach" not in decoded, (
        f"word boundary lost around the added token: decoded {decoded!r} from {sentence!r}"
    )
    assert decoded == sentence, f"expected an exact round-trip, got {decoded!r}"


def test_extend_tokenizer_vocab_leaves_missing_characters_glued_mid_word():
    """Single missing characters (e.g. the glottal apostrophe) are
    correctly glued mid-word already -- the fix must not regress that by
    treating every added token as a word boundary indiscriminately.
    """
    tokenizer = _fresh_tokenizer()

    added = extend_tokenizer_vocab(tokenizer, ["k'o awäch"])
    assert added, "expected at least one genuinely new token (apostrophe/ä) for base M2M100 vocab"

    ids = tokenizer("k'o").input_ids
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    assert decoded == "k'o", f"expected the apostrophe to stay glued mid-word, got {decoded!r}"


def test_extend_tokenizer_vocab_with_subwords_decodes_with_correct_spacing():
    tokenizer = _fresh_tokenizer()
    # An invented word-form guarantees the trained SentencePiece Unigram
    # model discovers a genuinely new *word-initial* piece (bearing
    # SentencePiece's own "▁" marker) missing from base M2M100 vocab --
    # the real fixture text (`sample_kaqchikel_text.txt`) only yields
    # mid-word continuation pieces at a small vocab_size, which wouldn't
    # exercise this word-initial case.
    kaqchikel_texts = [
        "Ri qpwvxel nqpwvxel ri qpwvxel.",
        "Jantape qpwvxel nkatzu ri qpwvxel tinamit.",
        "Ronojel qpwvxel winaqi qpwvxel.",
    ] * 20

    added = extend_tokenizer_vocab_with_subwords(tokenizer, kaqchikel_texts, vocab_size=80)
    word_initial_pieces = [t for t in added if t.startswith("▁")]
    assert word_initial_pieces, "fixture expected to yield at least one word-initial subword piece"
    piece = word_initial_pieces[0]
    content = piece.removeprefix("▁")

    # A word-initial added piece isn't currently reachable via this
    # tokenizer's own `encode()`/`__call__` path: HF's added-token matching
    # runs against literal raw text, which never contains the "▁" marker
    # character, so a marked added token can never be *selected* while
    # tokenizing ordinary text (a separate, narrower gap from the decode
    # bug this test targets -- see the PR description). A real model's
    # `generate()` can still directly emit that token's id regardless, so
    # decoding must handle it correctly: build the id sequence directly via
    # `convert_tokens_to_ids` rather than through `tokenizer(sentence)`.
    # "▁el" and "▁pueblo" are confirmed-native single pieces in base
    # M2M100 vocab (unlike "▁hola", which fragments into "▁h"+"ola" and
    # would silently map to <unk> if used here, hiding the bug this test
    # targets), so any missing space around `piece` is unambiguously this
    # bug and not an artifact of the surrounding tokens themselves.
    token_sequence = ["▁el", piece, "▁pueblo"]
    ids = tokenizer.convert_tokens_to_ids(token_sequence)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)

    glued = f"el{content}"
    assert glued not in decoded, (
        f"word boundary lost around added subword piece {piece!r}: decoded {decoded!r}"
    )
    assert decoded == f"el {content} pueblo", f"expected an exact round-trip, got {decoded!r}"


def test_word_boundary_fix_composes_across_both_extension_calls():
    """`training.train.extend_vocab_and_resize_embeddings` calls
    `extend_tokenizer_vocab` then `extend_tokenizer_vocab_with_subwords` on
    the *same* tokenizer instance every real training run. Both functions
    patch `convert_tokens_to_string` the same way (issue #116) -- this
    guards against the second call clobbering the first's fix instead of
    composing with it.
    """
    tokenizer = _fresh_tokenizer()

    added_words = extend_tokenizer_vocab(tokenizer, ["awach"])
    assert "awach" in added_words

    added_subwords = extend_tokenizer_vocab_with_subwords(
        tokenizer,
        ["Ri qpwvxel nqpwvxel ri qpwvxel.", "Jantape qpwvxel nkatzu ri qpwvxel tinamit."] * 20,
        vocab_size=80,
    )
    word_initial_pieces = [t for t in added_subwords if t.startswith("▁")]
    assert word_initial_pieces

    # The whole-word token from the *first* call must still decode with
    # correct spacing after the *second* call has re-wrapped/extended the
    # patched `convert_tokens_to_string`.
    sentence = "de dios awach y todo el pueblo"
    ids = tokenizer(sentence).input_ids
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    assert decoded == sentence, f"first call's fix was lost after the second call: {decoded!r}"

    # And the second call's word-initial subword piece must decode
    # correctly too.
    piece = word_initial_pieces[0]
    content = piece.removeprefix("▁")
    ids2 = tokenizer.convert_tokens_to_ids(["▁el", piece, "▁pueblo"])
    decoded2 = tokenizer.decode(ids2, skip_special_tokens=True)
    assert decoded2 == f"el {content} pueblo", f"got {decoded2!r}"
