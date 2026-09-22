"""Translation-direction handling for es<->cak fine-tuning.

## Decision: one multilingual checkpoint with direction tags

ADR 0001 deliberately left open whether es->cak and cak->es should be one
multilingual model with direction tags, or two separate fine-tuned
checkpoints, pending this ticket's implementation. This module implements
**one multilingual checkpoint, tagged with an explicit per-example
direction token** (see `ADR 0006 <../docs/adr/0006-translation-direction-handling.md>`_
for the full writeup). In short:

- The corpus is small either way (33,616 train / 3,735 validation pairs,
  ADR 0001). Splitting it into two independent, direction-specific
  fine-tunes would roughly halve the already-scarce per-direction
  training signal, whereas a single shared encoder/decoder can exploit
  cross-direction transfer -- the same Kaqchikel morphology/vocabulary is
  reinforced whether the model is producing it or consuming it.
- Kaqchikel is not among M2M100's ~100 pretrained languages (ADR 0003), so
  neither direction has a pretrained "cak" skill to protect from
  catastrophic forgetting -- the usual argument *for* keeping directions
  in separate checkpoints (protecting a strong pretrained direction from
  being degraded by fine-tuning on a weaker one) doesn't apply here.
- One checkpoint halves the SageMaker Model Registry artifacts and served
  endpoints to build, register, and pay for (ADR 0001's SageMaker
  Serverless Inference), which matters for a small, cost-conscious
  project.

M2M100's own built-in language-code/embedding mechanism
(`tokenizer.src_lang` / `forced_bos_token_id` via `lang_code_to_id`)
doesn't cover Kaqchikel, since it's outside M2M100's pretrained language
set. Rather than hacking that closed table, this module uses a simpler,
explicit mechanism: a dedicated tag token per language (`__es__`,
`__cak__`) is prepended to the *source* text of every training example,
and used as the `forced_bos_token_id` for generation -- telling the
shared encoder/decoder which direction to translate a given example in.
These tag tokens flow through the same vocabulary-extension pipeline as
any other new Kaqchikel token (`training.tokenizer_extension`), so they
get properly warm-started embedding rows rather than starting from
scratch.

This is a starting decision, not a permanent one: if evaluation
(`ml/evaluation`, per-direction BLEU/chrF) shows one direction being
starved by the other during shared training, that should be revisited
with real experiment numbers, not assumed away.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

SPANISH = "es"
KAQCHIKEL = "cak"

DIRECTION_TAGS: dict[str, str] = {
    SPANISH: "__es__",
    KAQCHIKEL: "__cak__",
}

ALL_DIRECTION_TAG_TOKENS: tuple[str, ...] = tuple(DIRECTION_TAGS.values())

DIRECTION_CHOICES: tuple[str, ...] = ("es->cak", "cak->es", "both")


@dataclass(frozen=True)
class TranslationExample:
    """One (source, target) training/eval example for a single direction."""

    source_text: str
    target_text: str
    source_lang: str
    target_lang: str


def tag_source_text(source_text: str, target_lang: str) -> str:
    """Prepend the target-language direction tag to `source_text`.

    This is the "direction tag" mechanism backing the single-multilingual-
    checkpoint decision documented in this module's docstring: the tag
    tells the shared encoder/decoder which direction to translate this
    example in, standing in for M2M100's built-in lang-code mechanism,
    which doesn't cover Kaqchikel.
    """
    return f"{DIRECTION_TAGS[target_lang]} {source_text}"


def build_direction_examples(
    pairs: list[tuple[str, str]], direction: str
) -> list[TranslationExample]:
    """Expand (es, cak) sentence pairs into per-direction training examples.

    `direction` must be one of `DIRECTION_CHOICES`:
    - `"es->cak"`: one example per pair, es as source.
    - `"cak->es"`: one example per pair, cak as source.
    - `"both"`: one example per pair *per direction* (doubling the total
      example count) -- this is the default used to train the single
      multilingual, direction-tagged checkpoint (see module docstring).

    Raises `ValueError` for any other `direction` value.
    """
    if direction not in DIRECTION_CHOICES:
        raise ValueError(
            f"Unknown direction {direction!r}; expected one of {DIRECTION_CHOICES}"
        )

    examples: list[TranslationExample] = []
    if direction in ("es->cak", "both"):
        examples.extend(
            TranslationExample(es, cak, SPANISH, KAQCHIKEL) for es, cak in pairs
        )
    if direction in ("cak->es", "both"):
        examples.extend(
            TranslationExample(cak, es, KAQCHIKEL, SPANISH) for es, cak in pairs
        )
    return examples


def collect_texts_for_language(
    examples: Iterable[TranslationExample], language: str
) -> list[str]:
    """Return every text written in `language` across `examples` (its
    source text where `source_lang == language`, its target text where
    `target_lang == language`), in iteration order.

    Used to isolate the Kaqchikel-only side of a direction-tagged example
    set for training a Kaqchikel-specific subword vocabulary model (issue
    #82, `training.subword_vocab`) -- deliberately never mixed with the
    other language's text, since M2M100 already tokenizes Spanish natively
    and mixing the two would dilute what the subword trainer learns about
    Kaqchikel's own morphology.
    """
    texts: list[str] = []
    for example in examples:
        if example.source_lang == language:
            texts.append(example.source_text)
        if example.target_lang == language:
            texts.append(example.target_text)
    return texts
