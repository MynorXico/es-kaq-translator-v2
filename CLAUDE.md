# CLAUDE.md

Project-level instructions for Claude Code sessions working in this repo.
Read this first; it points to the fuller docs rather than duplicating
them.

## What this is

Traductor Kaqchikel — an open-source Spanish↔Kaqchikel translator (UI,
API, ML pipeline), AWS-native, AI-native. See [`README.md`](README.md).

## Ground truth (don't re-derive, read these)

- `docs/adr/` — accepted architecture decisions. Binding unless a newer
  ADR supersedes one. Currently: `0001-initial-architecture.md` (stack,
  AWS layout, domain), `0002-data-and-model-privacy.md` (corpus/model
  privacy).
- `docs/data-governance.md` — what data is private vs. public.
- `docs/workflow.md` — the full ticket lifecycle, git conventions, and
  which `.claude/agents`/`.claude/skills` handle each step.
- `docs/testing.md` — TDD process and the test pyramid per component.
- `docs/runbooks/` — operational procedures (e.g. AWS account bootstrap).

## Language convention

All code, comments, commit messages, and docs are in **English**. Spanish
is used only for content the end user actually sees: UI copy in
`apps/web` and user-facing API response text. See the root `README.md`.

## Git workflow — branches, commits, PRs

Every change is tied to a GitHub issue on the
["Traductor Kaqchikel" project](https://github.com/users/MynorXico/projects/6)
(project `6`). If there's no issue yet, create one first (`new-ticket`
skill) rather than starting work untracked.

- **Always branch from latest `main`**, in its own **git worktree** (not
  the shared checkout) so parallel tickets don't collide — `git fetch
  origin main && git worktree add --no-track ../<repo>-worktrees/<N>-slug
  -b <N>-slug origin/main`. When delegating via `work-ticket`, pass
  `isolation: "worktree"` on the `Agent` call instead. Full details and
  cleanup steps in `docs/workflow.md`.
- **Branch naming**: `<issue-number>-<short-kebab-slug>`, e.g.
  `12-community-corpus-pipeline`.
- **Commits**: reference the ticket with a `Refs #<N>` line in the commit
  body (in addition to any attribution trailers already required for
  this session — see the system reminder for exact text). This keeps
  traceability even if a PR ends up with multiple commits.
- **PRs**: description must include a GitHub closing keyword
  (`Closes #<N>`) — already in `.github/PULL_REQUEST_TEMPLATE.md`. This
  auto-closes the issue on merge and moves it to `Done` on the board.
- **Merge strategy**: squash-merge only (repo is configured to disable
  merge-commit and rebase-merge). The squash commit's message is set to
  include the PR title *and description*, so `Closes #<N>` survives into
  `main`'s history — every commit on `main` is traceable to the issue it
  resolved.
- No direct pushes to `main` for feature/fix work — go through a branch +
  PR, even for small changes, so the issue link is never lost.

## Testing — TDD, Red → Green → Refactor

All new production code (`dev`, `ml-engineer`) is written test-first:
write a failing test (Red), implement the minimum to pass (Green),
refactor while keeping it green. See `docs/testing.md` for the full
policy and the test pyramid per component (`apps/web`: Vitest unit/
component + Playwright e2e; `apps/api`: pytest unit/integration;
`infra/cdk`: CDK assertion tests; `ml/`: conventions defined, no code
yet). A PR that adds non-trivial production code with no corresponding
test is a review flag, not a style nitpick.

## Roles and reusable workflows

- `.claude/agents/`: `product-owner`, `architect`, `ux`, `dev`,
  `ml-engineer`, `code-reviewer`, `qa`, `devops` — see each file for scope.
- `.claude/skills/`: `new-ticket`, `groom-backlog`, `work-ticket` —
  mechanical procedures for moving a ticket through the board.
- Full loop (write → groom → design → develop → review → merge) is in
  `docs/workflow.md`.
- Process is lightweight Kanban, no fixed sprints (revisit if a real team
  forms).

## Data & security guardrails — do not violate

- Never commit the private ALMG-derived corpus or trained model weights,
  or anything that would leak their content — see
  `docs/data-governance.md` and ADR 0002. Community-contributed corpus
  data is the only training data meant to be public.
- Never commit real AWS account IDs, the org ID, the management account
  ID, or real email addresses to this public repo — use placeholders (see
  `docs/runbooks/aws-account-bootstrap.md` for the pattern). Real values
  are tracked privately, not in git.
- If you ever find sensitive data already committed, don't just fix it
  going forward — flag it, since a fresh commit may still need history
  rewritten (see git history for precedent: this happened once already).

## Common commands

```sh
make install   # pnpm install + uv sync
make test      # web + infra-cdk + api tests (unit/integration, fast)
make test-e2e  # web Playwright e2e (needs a one-time browser install)
make build     # web + infra-cdk build
make synth     # cdk synth
```

Per-component commands are in each app's own README
(`apps/web`, `apps/api`, `infra/cdk`).
