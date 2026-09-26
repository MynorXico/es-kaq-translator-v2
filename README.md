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

## Security

Per [ADR 0001](docs/adr/0001-initial-architecture.md)'s CI/CD plan
(GitHub Actions for PR checks: lint/test/build/security), this repo has
the following automated security checks enabled, all visible under the
repo's [Security tab](../../security):

- **CodeQL** (`.github/workflows/codeql.yml`) — static analysis (SAST)
  for both languages in this monorepo (JS/TS: `apps/web`, `infra/cdk`;
  Python: `apps/api`, `ml`), on every PR, on push to `main`, and weekly.
  Committed as an explicit workflow rather than relying on the
  repo-settings-only "default setup" toggle, so it's reviewable in a PR
  like the rest of this repo's infra-as-code.
- **Dependabot version updates** (`.github/dependabot.yml`) — weekly PRs
  for outdated dependencies across every package ecosystem in the
  monorepo: the pnpm JS workspace (`apps/web`, `infra/cdk`), each
  `uv`-managed Python project (`apps/api`, `ml`), and the GitHub Actions
  used in workflows.
- **Dependabot security updates** — automatic PRs for dependencies with
  known vulnerabilities (separate from the scheduled version updates
  above), enabled at the repo level.
- **Secret scanning + push protection** — GitHub scans for committed
  secrets and blocks pushes that introduce new ones. Enabled at the repo
  level (free for public repos).

Repo-level toggles (Dependabot security updates, secret scanning, push
protection) live under Settings > Code security and aren't visible in
this repo's code — this section exists so they don't become a silent,
forgettable setting.

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
