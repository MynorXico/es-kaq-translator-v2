# Traductor Kaqchikel

An open-source, bidirectional Spanish↔Kaqchikel machine translator — UI,
API, and ML training pipeline — built AWS-native and designed to be
AI-native, with specialized Claude Code agents helping build and maintain
the project.

Kaqchikel (`cak`) is a Mayan language spoken in Guatemala. It is not
supported by mainstream pretrained multilingual translation models, so this
project fine-tunes a base model on a Spanish-Kaqchikel parallel corpus
derived from public texts provided by the Academia de Lenguas Mayas de
Guatemala (ALMG).

Live site (once launched): https://traductorkaqchikel.com

## Status

🚧 Early bootstrap. See [`docs/adr/0001-initial-architecture.md`](docs/adr/0001-initial-architecture.md)
for the locked-in architecture decisions and
[`docs/data-governance.md`](docs/data-governance.md) for the current data
rights status.

## Repository layout

```
apps/web      React + Vite SPA (translator UI, user-facing strings in Spanish)
apps/api      FastAPI translation service
ml/           Data preprocessing, model training, and evaluation
infra/cdk     AWS CDK (TypeScript) infrastructure
.github/      CI workflows, issue/PR templates
.claude/      Claude Code agents, skills, and commands specific to this repo
docs/         Architecture Decision Records (ADRs), data governance docs
```

Each top-level folder will get its own README with setup instructions as
it's built out.

## Language conventions

All code, comments, commit messages, and documentation in this repository
are written in English, so the project is accessible to the broader
international open-source community. The only exception is content the
end user actually sees — UI copy in `apps/web` and any user-facing text
returned by the API — which is written in Spanish, since the target
audience is primarily Spanish-speaking users in Guatemala.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to contribute code,
report translation quality issues, or propose new linguistic data. Please
also read the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Code is licensed under the [Apache License 2.0](LICENSE). The project's
original training corpus and trained model weights are **not** open
source — they're kept private by the project owner and exposed only
through the hosted API. Community-contributed sentence data forms a
separate, openly licensed corpus. See
[`docs/data-governance.md`](docs/data-governance.md) and
[ADR 0002](docs/adr/0002-data-and-model-privacy.md) for details.
