# ADR 0006: Translation direction handling (one multilingual model vs. two checkpoints)

- Status: Proposed
- Date: 2026-09-18

## Context

ADR 0001 proposed fine-tuning a pretrained multilingual model for
bidirectional Spanish<->Kaqchikel translation, but deliberately left open
*how* to structure the two directions: a single multilingual model
distinguishing directions via an explicit tag, or two independently
fine-tuned checkpoints (one per direction), pending real training
experiments. This ADR resolves that question for the training entrypoint
script (`ml/training/train.py`, issue #64), since the script has to pick
one concrete mechanism to implement.

M2M100 (the base model, ADR 0003) has a built-in mechanism for this in its
pretrained language set: `tokenizer.src_lang` plus `forced_bos_token_id`
via `lang_code_to_id`, covering ~100 languages. Kaqchikel is not one of
them (the entire reason vocabulary/embedding extension, #34/#62, is
needed in the first place), so that built-in mechanism cannot be used
as-is for the cak side of either direction.

## Decision

**Train one multilingual checkpoint covering both directions, using an
explicit per-example direction tag token** (`__es__` / `__cak__`)
prepended to the source text and used as the generation-time
`forced_bos_token_id`, instead of M2M100's built-in (Kaqchikel-incompatible)
language-code table. This is implemented in `ml/training/direction.py`
and wired into `ml/training/train.py`.

Rationale:

- **Corpus size.** The corpus is small either way (33,616 training /
  3,735 validation pairs total, ADR 0001). Splitting it into two
  independent single-direction fine-tunes would roughly halve the
  already-scarce per-direction training signal each checkpoint sees. A
  shared encoder/decoder trained on both directions at once can instead
  exploit cross-direction transfer: the same Kaqchikel morphology and
  vocabulary get reinforced whether the model is producing Kaqchikel or
  consuming it, and the same holds for Spanish.
- **No pretrained skill to protect.** The usual argument *for* two
  separate checkpoints is protecting a direction that already has strong
  pretrained quality from being degraded ("catastrophically forgotten")
  by fine-tuning on a weaker, unrelated direction. That doesn't apply
  here: Kaqchikel is entirely absent from M2M100's pretraining, so
  neither direction starts with a pretrained Kaqchikel skill worth
  isolating from the other's gradients.
- **Operational cost.** One checkpoint means one SageMaker Model Registry
  entry, one artifact to store/version, and one SageMaker Serverless
  Inference endpoint to build and pay cold-start/idle cost for, rather
  than two of each. That matters for a small, cost-conscious project
  (ADR 0001).
- **Implementation cost.** Direction tags reuse the vocabulary/embedding
  extension machinery already built for Kaqchikel-specific tokens
  (#34/#62) almost for free -- the tag tokens are just two more entries
  fed through the same `extend_tokenizer_vocab` /
  `resize_embeddings_for_new_tokens` pipeline, rather than needing new
  infrastructure.

## Consequences

- `ml/training/direction.py` is the source of truth for this mechanism:
  `DIRECTION_TAGS`, `tag_source_text`, and `build_direction_examples`.
  `ml/training/train.py`'s `--direction` flag defaults to `"both"`
  (single multilingual checkpoint) but also accepts `"es->cak"` or
  `"cak->es"` alone, so a single-direction comparison run remains
  possible without code changes -- useful for validating this decision
  experimentally (see below).
- The model card (`ml/evaluation/model_card.py`) records the `direction`
  value used for a given run (`"both"`, `"es->cak"`, or `"cak->es"`) plus
  a `notes` explanation when `"both"` was used, so this decision stays
  traceable per-run rather than only living in this document.
- **This is a starting decision, not a permanent one.** ADR 0001
  explicitly deferred this pending real experiment results, and this ADR
  is being written before any real training run has happened (issue #66
  is the first one, and is out of scope for this ticket). If per-direction
  BLEU/chrF from real runs shows one direction being measurably starved
  by shared training with the other (e.g. the majority-represented
  direction in the corpus dominates, or gradient interference reduces
  quality below what a dedicated single-direction checkpoint would give),
  this decision should be revisited with those numbers -- not treated as
  final because it was reasoned about first.

## Alternatives considered

- **Two separate checkpoints, one per direction**: doubles registry
  artifacts and inference endpoints, and roughly halves each direction's
  effective training data, for a project whose corpus is already small.
  Would only clearly win if cross-direction interference within a shared
  model turns out to be a real, measured problem -- which requires a
  first experiment to even observe, so it isn't the right default to
  start from.
- **Use M2M100's built-in `lang_code_to_id` mechanism by remapping an
  existing, unused language code to represent Kaqchikel** (e.g.
  repurposing a pretrained code M2M100 supports but this project doesn't
  need): rejected as needlessly fragile and confusing -- it would silently
  overload a documented language code's meaning, make debugging
  generation output harder ("why is `forced_bos_token_id` for Kaqchikel
  the id for a documented but unrelated language?"), and provide no real
  benefit over an explicit new tag token, which the vocab-extension
  pipeline already supports cleanly.
- **No tag at all, relying on two separately fine-tuned single-direction
  models to disambiguate direction implicitly**: this is just the "two
  checkpoints" alternative restated without a tag; rejected for the same
  corpus-size and operational-cost reasons above.
