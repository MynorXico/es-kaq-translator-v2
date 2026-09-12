# Contributing to Traductor Kaqchikel

Thanks for your interest in contributing! This project is open source and
welcomes contributions of code, linguistic data, design, documentation, and
translation review.

## Before you start

- Read the [Code of Conduct](CODE_OF_CONDUCT.md).
- Check [open issues](../../issues) and the
  [GitHub Projects board](../../projects) to see what's already in progress
  before starting something new, to avoid duplicated effort.
- For large changes (new architecture, changing the translation model,
  etc.), open a discussion issue or an ADR proposal in `docs/adr/` before
  writing code.
- See [`docs/workflow.md`](docs/workflow.md) for how tickets move through
  the board (write → groom → develop → review → merge) and which role/skill
  handles each step.

## Types of contributions

### 1. Code

This is a monorepo with:

- `apps/web` — web UI (React + Vite).
- `apps/api` — translation API (Python, FastAPI).
- `ml/` — data preprocessing, training, and model evaluation.
- `infra/cdk` — infrastructure as code (AWS CDK, TypeScript).

Each folder has its own README with local setup instructions. Before
opening a pull request:

1. Make sure there's a GitHub issue for the change (open one if not).
2. Create a branch off `main` named `<issue-number>-<short-kebab-slug>`.
3. Write code test-first: a failing test (Red), then the minimum
   implementation to pass (Green), then refactor. See
   [`docs/testing.md`](docs/testing.md) for the test pyramid — unit,
   integration, and (for `apps/web`) e2e via Playwright. A PR that adds
   non-trivial code with no test changes will get flagged in review.
4. Make sure local tests and linters pass (see the corresponding workflow
   in `.github/workflows/`).
5. Reference the issue in your commits (`Refs #<N>`) and in the PR
   description (`Closes #<N>`, already in the PR template) — see
   [`docs/workflow.md`](docs/workflow.md) for the full git convention and
   why (traceability from `main`'s history back to the issue it resolved).
6. A pull request should address one concern at a time — avoid mixing
   unrelated refactors with the functional change.

PRs are merged via squash-merge only (the repo disables merge-commit and
rebase-merge) — don't push directly to `main`.

Note: all code, comments, commit messages, and documentation in this
repository are written in English. Only user-facing strings in the web UI
and API-facing content meant for end users are written in Spanish, since
the target audience is primarily Spanish-speaking.

### 2. Linguistic data and translation quality

Since this project translates a digitally low-resource language,
contributions from native speakers and linguists are especially valuable.
Note: the project's original training corpus is private and not part of
the open-source release (see [`docs/data-governance.md`](docs/data-governance.md));
community-contributed sentences form a separate, openly licensed corpus.

- **Report incorrect translations**: use the "Translation quality report"
  issue template.
- **Propose new parallel sentences** (Spanish-Kaqchikel): open an issue
  using the corresponding template, indicating the source of the text and
  confirming you have the right to share it under an open license
  (CC-BY/CC0). This grows the public community corpus, kept separate from
  the project's private data.
- Data of unknown origin or license will not be accepted without
  documenting its provenance first.

### 3. Documentation

Improvements to this file, the `README.md`, or the ADRs in `docs/adr/` are
welcome via pull request, same as code.

## Architecture decisions (ADR)

Significant technical decisions are documented as Architecture Decision
Records in `docs/adr/`. If your contribution involves a significant
architectural decision (choosing a core library, changing the deployment
scheme, etc.), add or update an ADR as part of the PR.

## License

By contributing code or documentation, you agree that your contribution
will be licensed under the project's [Apache License 2.0](LICENSE).
Contributions of linguistic data are governed by
[`docs/data-governance.md`](docs/data-governance.md).
