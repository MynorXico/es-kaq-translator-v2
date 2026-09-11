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

## Types of contributions

### 1. Code

This is a monorepo with:

- `apps/web` — web UI (React + Vite).
- `apps/api` — translation API (Python, FastAPI).
- `ml/` — data preprocessing, training, and model evaluation.
- `infra/cdk` — infrastructure as code (AWS CDK, TypeScript).

Each folder has its own README with local setup instructions. Before
opening a pull request:

1. Create a branch off `main`.
2. Make sure local tests and linters pass (see the corresponding workflow
   in `.github/workflows/`).
3. Clearly describe the change and its motivation in the pull request.
4. A pull request should address one concern at a time — avoid mixing
   unrelated refactors with the functional change.

Note: all code, comments, commit messages, and documentation in this
repository are written in English. Only user-facing strings in the web UI
and API-facing content meant for end users are written in Spanish, since
the target audience is primarily Spanish-speaking.

### 2. Linguistic data and translation quality

Since this project translates a digitally low-resource language,
contributions from native speakers and linguists are especially valuable:

- **Report incorrect translations**: use the "Translation quality report"
  issue template.
- **Propose new parallel sentences** (Spanish-Kaqchikel): open an issue
  using the corresponding template, indicating the source of the text and
  confirming you have the right to share it under an open license. See
  [`docs/data-governance.md`](docs/data-governance.md) for details on how
  we handle data provenance and licensing.
- Data of unknown origin or license will not be accepted directly into the
  public repository without documenting its provenance first.

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
