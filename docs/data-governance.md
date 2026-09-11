# Data Governance

## Current parallel corpus

The Spanish-Kaqchikel parallel corpus used for initial training (33,616
training sentences / 3,735 validation sentences) was extracted from public
texts provided by the **Academia de Lenguas Mayas de Guatemala (ALMG)**.

**Current status: public redistribution rights for the corpus have not
been confirmed with ALMG.** Until this is resolved:

- The raw corpus (the aligned text files) is **not published in this
  public repository**. It is kept in a private S3 bucket, accessible only
  to the training pipeline.
- The public repository only contains cleaning/preprocessing scripts and,
  if needed, a minimal synthetic or example data sample for automated
  tests.
- Before publishing the full corpus (or derived artifacts such as
  vocabularies or embeddings trained on it) under an open license, explicit
  confirmation must be obtained from ALMG regarding usage and
  redistribution terms.

Whoever picks up this action item should document the outcome of the
conversation with ALMG here (license granted, required attribution,
restrictions, etc.) once confirmed.

## New data contributions

Anyone proposing new parallel sentences or other linguistic data must state
in the corresponding issue/PR:

1. The original source of the text (author, institution, publication).
2. Whether they have the right to share that text under an open license
   (ideally CC-BY or CC0), or whether additional permission is required.
3. Any known usage restrictions (e.g. third-party copyrighted material).

Project maintainers will review provenance before incorporating the data
into the training pipeline or the repository.

## Base model license

The translation model is built by fine-tuning a pretrained multilingual
model (see the corresponding ADR in `docs/adr/`). Before adopting a
specific base model, its license must be verified as compatible with this
project's open distribution goals (for example, several Meta AI
checkpoints such as NLLB-200 are distributed under non-commercial CC-BY-NC
licenses, which could be incompatible depending on intended use). This
finding must be documented in an ADR before training begins in earnest.
