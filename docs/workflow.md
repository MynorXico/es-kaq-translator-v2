# Development workflow

This project uses a lightweight Kanban flow on the
["Traductor Kaqchikel" GitHub Project](https://github.com/users/MynorXico/projects/6)
(no fixed sprints — revisit if a real multi-person team forms). The
`Status` field moves through: `Backlog` → `Todo` → `In Progress` →
`In Review / QA` → `Done`. See
[ADR 0001](adr/0001-initial-architecture.md) for why GitHub
Issues/Projects was chosen over alternatives like Jira.

Each stage maps to a role (a `.claude/agents/*.md` definition) and, for
the mechanical parts, a `.claude/skills/*` skill:

| Stage | Status transition | Role | Skill |
|---|---|---|---|
| Write | (new) → `Backlog` | `product-owner` | `new-ticket` |
| Groom | `Backlog` → `Todo` | `product-owner` | `groom-backlog` |
| Design (UI tickets only) | (within `Todo`, before implementation) | `ux` | the built-in `/design` skill |
| Develop | `Todo` → `In Progress` → `In Review / QA` | `dev` (or `ml-engineer` for `ml/` work) | `work-ticket` |
| Review/verify | (within `In Review / QA`) | `code-reviewer`, `qa`, automated (see below) | the built-in `/code-review` skill |
| Merge | `In Review / QA` → `Done` | human maintainer | — |

Every PR against `main` also gets an automated first-pass review from
`.github/workflows/claude-code-review.yml` (the `anthropics/claude-code-action`
GitHub Action, running Anthropic's `code-review` plugin), which posts
inline comments directly on the PR for anything it flags, plus a tracking
comment (`track_progress: true`) that gets replaced with a final summary
when it finishes — so a clean PR still gets a visible "reviewed, no
issues found" comment instead of silence. Without `track_progress`, a PR
with zero findings got no PR-visible signal at all beyond the passing
`review` check, making it impossible to tell "ran and found nothing" from
"silently failed" at a glance. It's a first pass, not a replacement for
`code-reviewer`/`qa` or human review — findings don't block merging.
Authenticated via the `CLAUDE_CODE_OAUTH_TOKEN` repo secret (tied to a
Claude subscription) and the Claude GitHub App installed on this repo.

Design isn't a board Status of its own — it's a step `work-ticket` (or you,
if working inline) should trigger before implementation when a `Todo`
item touches `apps/web` UI and no design spec exists yet. `ux` produces a
mockup via `/design` and a concrete spec (Spanish copy, spacing, states);
`dev` implements to that spec rather than improvising layout/copy.

The Develop step is TDD: `dev`/`ml-engineer` write a failing test before
any production code, then implement to green, then refactor — see
[`docs/testing.md`](testing.md) for the process and which test level(s) a
given change needs.

## Git conventions

Every change traces back to an issue — if one doesn't exist yet, create
it first (`new-ticket`) rather than starting untracked work.

- **Always branch from latest `main`**: `git fetch origin main` first,
  then branch from `origin/main` — never from whatever the local `main`
  or another feature branch happens to be at, so every ticket starts from
  the same known-good point.
- **Work in an isolated git worktree, not the shared checkout**: this
  lets multiple tickets be worked on in parallel without branch-switching
  collisions (two pieces of work can't have different branches checked
  out in the same directory at once). Create one per ticket:

  ```sh
  git fetch origin main
  git worktree add --no-track \
    ../<repo>-worktrees/<issue-number>-<short-kebab-slug> \
    -b <issue-number>-<short-kebab-slug> origin/main
  ```

  (`--no-track` avoids the branch's upstream defaulting to `origin/main`,
  which is confusing once you push it as its own branch.) `gh` commands
  that infer the current branch (`gh pr create` without `--head`, `gh pr
  checks` without an explicit number) read it from whatever directory the
  shell is actually in — if a tool session's shell resets to the main
  checkout between commands (some do), pass `--head <branch>` explicitly
  rather than relying on inference, or `cd` into the worktree first. When
  delegating
  to `dev`/`ml-engineer` via `work-ticket`, pass `isolation: "worktree"`
  on the `Agent` call instead of managing the path yourself — but still
  have the agent verify it's on a correctly-named branch based on latest
  `origin/main` before it commits, since the isolation mechanism may not
  name the branch for you. After the PR merges, clean up:
  `git worktree remove <path>` (from the main checkout) and
  `git branch -d <issue-number>-<short-kebab-slug>`.
- **Branch naming**: `<issue-number>-<short-kebab-slug>`, e.g.
  `12-community-corpus-pipeline`.
- **Commits**: include a `Refs #<N>` line in the commit body, so history
  is traceable to the issue even before/without a squash merge.
- **PR description**: must contain a closing keyword, `Closes #<N>`
  (already in `.github/PULL_REQUEST_TEMPLATE.md`) — this is what actually
  auto-closes the issue and moves its board `Status` to `Done` on merge.
- **Merge strategy**: squash-merge only. The repo is configured
  (`gh repo edit`) to disable merge-commit and rebase-merge, and to build
  the squash commit message from the PR title *and description*, so
  `Closes #<N>` lands in `main`'s history for every merged change — no
  separate step needed to keep that traceable.
- No direct pushes to `main` for feature/fix work, even small ones — a
  branch + PR is what keeps the issue link intact.

## Why a skill layer on top of the agents

The agent definitions describe *who* does a piece of work and with what
judgment. The skills describe *the mechanical steps* of moving a ticket
through the board correctly (which `gh` commands, which field IDs, when
to ask the user vs. proceed) so that part doesn't have to be re-derived
or done inconsistently each time. Use the skills for the repeatable
plumbing; rely on the referenced agent's judgment for anything that isn't
mechanical (scoping, architectural calls, whether something's actually
ready to promote).

## Architecture and data-rights decisions don't follow this flow

Anything that changes an ADR-level decision (new library, new AWS
service, corpus/model policy) goes through the `architect` agent and an
ADR in `docs/adr/`, not through the ticket board's Backlog→Done flow —
see [`CONTRIBUTING.md`](../CONTRIBUTING.md).
