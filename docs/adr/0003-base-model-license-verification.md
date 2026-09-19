# ADR 0003: Base model license verification (NLLB-200 vs. M2M100)

- Status: Accepted (2026-09-19, by the project owner)
- Date: 2026-09-11

## Context

ADR 0001 proposed fine-tuning "a pretrained multilingual model (e.g.
NLLB-200 distilled or M2M100)" without confirming either license was
actually compatible with this project's intended use, and flagged this as
a risk to resolve before committing to a specific checkpoint (see ADR
0001's Consequences and `docs/data-governance.md`'s "Base model license"
section).

This project's intended use of the base model is:

1. Fine-tune it privately on the private ALMG-derived corpus (and,
   potentially, the public community corpus) — see ADR 0002.
2. Serve only the fine-tuned model, only through the project's own hosted
   API (SageMaker Serverless Inference behind FastAPI) — never redistribute
   the base or fine-tuned weights (e.g. no publishing to Hugging Face).
3. The API is public-facing and, per this ticket's acceptance criteria,
   *potentially* commercial in the future (even if it starts as a free,
   unmonetized open-source project).

The question is whether the base model's license permits this without
requiring redistribution of the base weights, and specifically whether a
"non-commercial" restriction would still bite even though we never
redistribute the checkpoint itself.

### Findings

**NLLB-200 (all sizes, including `facebook/nllb-200-distilled-600M`)**

- The Hugging Face model card's `license` metadata field is
  `cc-by-nc-4.0`, matching the `LICENSE.model.md` file in the `nllb`
  branch of `facebookresearch/fairseq`
  ([source](https://github.com/facebookresearch/fairseq/blob/nllb/LICENSE.model.md)),
  which is the full **Creative Commons Attribution-NonCommercial 4.0
  International** (CC-BY-NC-4.0) license text. Note this is distinct from
  the fairseq *code*, which is MIT-licensed — the model *weights* carry
  the more restrictive CC-BY-NC-4.0 terms.
- CC-BY-NC-4.0 only grants rights "for NonCommercial purposes only," where
  NonCommercial is defined as "not primarily intended for or directed
  towards commercial advantage or monetary compensation." This grant
  explicitly covers producing and sharing **Adapted Material** — i.e. a
  fine-tuned model built from the checkpoint — not just verbatim
  redistribution of the original weights. "Share" under the CC license
  includes making material available to the public by any means requiring
  permission, which plausibly includes serving inference over an API.
  In other words: the restriction is about *use/purpose*, not only about
  redistributing the raw checkpoint file, so "we never publish the base
  weights" does not by itself resolve the non-commercial restriction.
- The `facebook/nllb-200-distilled-600M` model card additionally states,
  in prose: "NLLB-200 is a research model and is not released for
  production deployment." This is a direct, explicit statement against
  the model's intended use for a production-serving public API, over and
  above the CC-BY-NC-4.0 terms.
- We found a still-open (repo since archived) GitHub issue asking Meta to
  clarify whether a government-funded public translation service would
  count as commercial use
  ([facebookresearch/fairseq#5647](https://github.com/facebookresearch/fairseq/issues/5647)).
  It received no response from Meta/maintainers, meaning there is no
  official precedent resolving this ambiguity in Meta's own tracker —
  further reason not to rely on an optimistic reading.

**M2M100 (`facebook/m2m100_418M`, `facebook/m2m100_1.2B`)**

- Both official Hugging Face model cards under Meta's `facebook` org list
  `license: mit` in their YAML metadata (confirmed directly on
  [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M) and
  [facebook/m2m100_1.2B](https://huggingface.co/facebook/m2m100_1.2B)),
  with no separate non-commercial clause or "research only" language
  anywhere in either model card.
- The underlying `facebookresearch/fairseq` code is also MIT-licensed
  ([LICENSE](https://github.com/facebookresearch/fairseq/blob/main/LICENSE)),
  and the `examples/m2m_100/README.md` in that repo does not impose any
  additional license terms specific to the M2M100 checkpoints.
- MIT permits commercial use, modification (including fine-tuning), and
  private/hosted serving, with the only obligation being to retain the
  copyright and license notice — it does not require redistributing
  anything, and it does not distinguish commercial from non-commercial
  use.

## Decisions

- **M2M100 (`facebook/m2m100_418M` or `facebook/m2m100_1.2B`) is
  compatible** with this project's intended use: private fine-tuning on
  the ALMG + community corpora, and serving exclusively through our own
  hosted API, with no redistribution of base or fine-tuned weights, even
  if the API is or becomes commercial. Its MIT license imposes no
  use-purpose restriction.
- **NLLB-200 (including the distilled 600M checkpoint suggested in ADR
  0001) is NOT compatible as currently understood.** Its CC-BY-NC-4.0
  license restricts NonCommercial *use*, not just redistribution of the
  original weights, and its own model card explicitly disclaims
  production deployment. Fine-tuning it and serving the result through a
  public, potentially-commercial API is a plausible license violation,
  not just a theoretical one.
- **Adopt M2M100 as the base model going forward**, superseding the
  "NLLB-200 distilled or M2M100" either/or framing in ADR 0001. `ml/`
  scripts, training job configs, and any future model card should
  reference M2M100 (`facebook/m2m100_418M` as the default, smaller
  checkpoint for iteration; `facebook/m2m100_1.2B` as a candidate for a
  quality-focused later run) rather than NLLB-200.
- **This ADR was left as "Proposed," not "Accepted," until the project
  owner explicitly reviewed it.** Although the license *text* comparison
  is fairly clear-cut (CC-BY-NC-4.0 vs. MIT), whether a small,
  currently-unmonetized open-source project would ever actually be
  treated as "commercial" under CC-BY-NC-4.0 is a judgment call with real
  legal and reputational stakes, and it directly overrides a model choice
  ADR 0001 left open — not something to decide silently on the project's
  behalf. The project owner reviewed and accepted this finding on
  2026-09-19; see the Status line above.

## Consequences

- `docs/data-governance.md`'s "Base model license" section is updated to
  reflect this finding (M2M100 confirmed compatible pending human
  sign-off; NLLB-200 ruled out) instead of the previous "must be
  verified" placeholder language.
- Any `ml/training/` entrypoint configuration should default to an
  M2M100 checkpoint identifier, not an NLLB-200 one.
- If the project owner reviews this and disagrees with the
  CC-BY-NC-4.0 interpretation above (e.g. decides the project is
  unambiguously non-commercial for the foreseeable future and accepts
  that risk), this ADR should be updated to "Accepted" with that
  reasoning recorded, or a new ADR should supersede it — not silently
  reversed.
- M2M100's translation quality ceiling is generally reported as lower
  than NLLB-200's for low-resource language pairs (NLLB-200 was trained
  with low-resource coverage, including many languages absent from
  M2M100's training data, as a specific design goal). This is a real
  quality trade-off being accepted for licensing safety, and should be
  called out explicitly during evaluation (`ml/evaluation/`) rather than
  treated as a free choice — Kaqchikel is not in either model's
  pretraining data, so both require vocabulary/embedding extension
  regardless, but NLLB-200's broader low-resource pretraining could have
  transferred better.

## Alternatives considered

- **Use NLLB-200 anyway, relying on "we never redistribute the weights"
  as sufficient cover**: rejected — the CC-BY-NC-4.0 grant is scoped to
  NonCommercial *purpose*, not just redistribution, and NLLB-200's own
  model card explicitly disclaims production use. This reading is not
  supportable from the license text.
- **Use NLLB-200 only if/until the API becomes commercial, then
  switch**: rejected as impractical — the base checkpoint, vocabulary
  extension, and all derived fine-tuned weights would need to be
  discarded and retrained from a different base at that point, and the
  ambiguity around what counts as "commercial" (e.g. a public API with
  any usage limits, sponsorship, or future paid tier) makes "switch
  later" an unreliable trigger to rely on.
- **Look for a third base model beyond NLLB-200/M2M100** (e.g.
  mBART-50, MADLAD-400): out of scope for this ticket, which was scoped
  to verifying the two models named in ADR 0001, but worth a follow-up
  ticket if M2M100's quality on Kaqchikel proves insufficient after
  fine-tuning experiments — noting MADLAD-400 in particular is also
  Apache-2.0 (Google, not Meta) and could be a future candidate if a
  bigger low-resource-oriented model is needed.
- **Proceed with legal ambiguity flagged for a human decision, without
  picking an alternative**: considered, since this is fundamentally a
  legal judgment call outside a coding agent's authority to make binding.
  Rejected as the *sole* output because a clearly compatible, permissively
  licensed alternative (M2M100, MIT) already exists and satisfies ADR
  0001's own framing ("NLLB-200 distilled **or** M2M100") — recommending
  it, while still flagging the underlying judgment call for sign-off via
  this ADR's "Proposed" status, is more useful than only raising the
  question.
