# ADR 0002: Corpus and model weight privacy

- Status: Accepted
- Date: 2026-09-11

## Context

ADR 0001 left the ALMG-derived corpus in a "rights unconfirmed, don't
publish until resolved" state, implying publication was the eventual goal
once permission was granted. The project owner has since decided the raw
corpus is a private asset regardless of ALMG permission — it will not be
published or redistributed, period. This needed a decision on two related
questions ADR 0001 didn't settle: whether community-contributed data
should still grow an open corpus, and whether the resulting model weights
should be publishable even if the training data isn't.

## Decisions

- **The ALMG-derived corpus stays private permanently.** It is not
  published, redistributed, or offered for external research use,
  independent of whatever ALMG confirms about redistribution rights. It
  remains in a private S3 bucket, used only to train the project's model.
- **Community-contributed parallel sentences form a separate, openly
  licensed corpus.** Contributions submitted via the "data contribution"
  issue template (already requires an open license — CC-BY/CC0) are kept
  apart from the private ALMG corpus and can be published/distributed.
  Both corpora may be used together at training time, but only the
  community one is ever public.
- **Trained model weights stay private too**, exposed only through the
  project's hosted API (SageMaker Serverless Inference behind the FastAPI
  service). The model is not published as downloadable weights (e.g. on
  Hugging Face), even though the project's code remains Apache-2.0 and
  open source.

## Consequences

- `docs/data-governance.md` is rewritten to describe two corpora (private
  ALMG-derived, public community-contributed) instead of one corpus
  pending a publication decision.
- The open-source nature of this project is scoped to **code**: anyone can
  run their own instance against their own data/model, but cannot obtain
  the project owner's training data or trained weights. This should be
  stated plainly in the README so contributors and users have accurate
  expectations.
- GitHub issue #1 ("Confirm ALMG corpus redistribution rights") is closed
  without further ALMG follow-up, since nothing derived from their texts
  will be redistributed.
- `ml/data/` needs to keep the two corpora as clearly separated inputs
  (e.g. distinct S3 prefixes/paths) so it's never ambiguous which data is
  safe to expose in logs, docs, or public model cards.
- A future issue is needed to actually build the ingestion/publishing
  pipeline for the community corpus (tracked separately in the project
  board, Phase 3).

## Alternatives considered

- **Publish the full corpus once ALMG confirms rights** (the implicit plan
  under ADR 0001): rejected — the project owner wants to retain the data
  as a private asset regardless of what ALMG would allow.
- **Publish model weights while keeping only the corpus private**: a
  common "open-weight" pattern, but rejected here since keeping both data
  and weights private better matches the stated goal of retaining a
  private advantage while still open-sourcing the surrounding code/product.
