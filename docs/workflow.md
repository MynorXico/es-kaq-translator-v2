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
| Develop | `Todo` → `In Progress` → `In Review / QA` | `dev` (or `ml-engineer` for `ml/` work) | `work-ticket` |
| Review/verify | (within `In Review / QA`) | `code-reviewer`, `qa` | the built-in `/code-review` skill |
| Merge | `In Review / QA` → `Done` | human maintainer | — |

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
