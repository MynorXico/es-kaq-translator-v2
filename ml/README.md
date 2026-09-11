# ml

Machine learning pipeline for the Spanish<->Kaqchikel translation model.

- `data/` — corpus preprocessing/cleaning scripts. **Never commit raw
  corpus files here** — see [`docs/data-governance.md`](../docs/data-governance.md).
  Raw data lives in a private S3 bucket.
- `training/` — SageMaker training job entrypoints, tokenizer/vocabulary
  extension for Kaqchikel.
- `evaluation/` — BLEU/chrF evaluation harness and model card generation.

Not yet scaffolded — see [`docs/adr/0001-initial-architecture.md`](../docs/adr/0001-initial-architecture.md)
for the modeling approach.
