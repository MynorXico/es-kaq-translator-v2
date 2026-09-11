# Data Governance

This project trains its translation model on two separate corpora with
different privacy rules (see [ADR 0002](adr/0002-data-and-model-privacy.md)
for the reasoning behind the split).

## 1. Private corpus (ALMG-derived)

The original Spanish-Kaqchikel parallel corpus (33,616 training sentences
/ 3,735 validation sentences) was extracted from public texts provided by
the **Academia de Lenguas Mayas de Guatemala (ALMG)**.

**This corpus is private and stays private permanently.** It is not
published, redistributed, or shared externally for any purpose (research
included), regardless of what ALMG would permit. It lives only in a
private S3 bucket used by the training pipeline.

- The public repository never contains this raw data — only
  preprocessing/cleaning scripts that operate on it, plus a minimal
  synthetic sample for automated tests if needed.
- Trained model weights derived from this corpus are also kept private
  (see "Model weights" below) — the corpus is never exposed indirectly by
  publishing the model itself.

## 2. Public corpus (community-contributed)

Anyone can propose new Spanish-Kaqchikel parallel sentences via the "Data
contribution proposal" issue template. This forms a **separate, openly
licensed corpus**, kept apart from the private ALMG data, that can be
published and reused by others.

Contributors must state in the issue:

1. The original source of the text (author, institution, publication, or
   "original/self-authored").
2. Confirmation that they have the right to share it under an open license
   (ideally CC-BY or CC0).
3. Any known usage restrictions.

Project maintainers review provenance before merging contributed sentences
into the public corpus or the training pipeline. Both corpora may be used
together at training time, but only the community-contributed one is ever
published.

*(The pipeline to actually ingest and publish this public corpus is not
yet built — tracked on the project board under Phase 3.)*

## Model weights

Trained model weights are **not published** (e.g. not released on Hugging
Face or similar), regardless of which corpus contributed to a given
training run. The model is exposed only through the project's hosted
translation API. This keeps the project's code open source while the
data/model remain a private asset of the project owner.

## Base model license

The translation model is built by fine-tuning a pretrained multilingual
model (see [ADR 0001](adr/0001-initial-architecture.md)). Before adopting a
specific base model, its license must be verified as compatible with this
project's use — including fine-tuning it privately and serving it through
a public API without redistributing the base weights (several Meta AI
checkpoints such as NLLB-200 are distributed under non-commercial
CC-BY-NC licenses, which may still impose restrictions even without
redistributing the fine-tuned weights). This finding must be documented in
an ADR before training begins in earnest.
