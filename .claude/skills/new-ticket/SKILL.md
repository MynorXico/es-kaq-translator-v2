---
name: new-ticket
description: Turn a bug report, feature idea, or task description into a well-scoped GitHub issue on the Traductor Kaqchikel project board, with the right template, labels, Status, and Phase set. Use when the user describes something that needs to be tracked rather than done immediately.
---

# New ticket

Turns a description (from the user, or from something you noticed while
working) into a properly-tracked GitHub issue. This is the "write" step
of the project's workflow — see `docs/workflow.md`.

## Steps

1. **Clarify scope if needed.** If the description is vague (no clear
   acceptance criteria, ambiguous which component it touches), ask the
   user rather than guessing at scope. A good ticket says what "done"
   looks like.

2. **Pick the shape.** Most engineering tasks are a plain issue. Use a
   specific template's structure when it fits:
   - Bug in existing behavior -> `.github/ISSUE_TEMPLATE/bug_report.md` shape
   - New capability -> `.github/ISSUE_TEMPLATE/feature_request.md` shape
   - Reported bad translation -> `translation_quality.md` shape
   - Proposed parallel-sentence data -> `data_contribution.md` shape

3. **Write the issue.** Title: short, specific, imperative ("Add X",
   "Fix Y", not "X is broken"). Body: problem/why, proposed approach if
   known, explicit acceptance criteria, and any known dependencies on
   other issues (`Depends on #N`).

4. **Pick labels** from the existing set only — don't invent new ones:
   `bug`, `enhancement`, `documentation`, `ml`, `infra`, `governance`,
   `translation-quality`, `data-contribution`, `good first issue`,
   `help wanted`. Check with `gh label list --repo MynorXico/es-kaq-translator-v2`
   if unsure what exists.

5. **Determine Phase** from the roadmap in
   `docs/adr/0001-initial-architecture.md` (Phase 0 - Bootstrap, Phase 1 -
   MVP, Phase 2 - Promotion Pipeline, Phase 3 - Open Sourcing).

6. **Create it and add it to the board** (project `6`, owner `MynorXico`
   — field/option IDs and the full `gh` recipe are in
   `.claude/agents/product-owner.md`, fetch current IDs with
   `gh project field-list 6 --owner MynorXico` since they aren't
   guessable):

   ```sh
   gh issue create --repo MynorXico/es-kaq-translator-v2 \
     --title "..." --label "<label>" --body "..."

   itemid=$(gh project item-add 6 --owner MynorXico --url "<issue-url>" --format json | jq -r '.id')
   gh project item-edit --id "$itemid" --project-id <PROJECT_ID> \
     --field-id <STATUS_FIELD_ID> --single-select-option-id <BACKLOG_OPTION_ID>
   gh project item-edit --id "$itemid" --project-id <PROJECT_ID> \
     --field-id <PHASE_FIELD_ID> --single-select-option-id <PHASE_OPTION_ID>
   ```

   New tickets default to **Backlog** status, not Todo — grooming (see the
   `groom-backlog` skill) is what promotes something to Todo, since that's
   the point where readiness gets a second look. Only set `Todo` directly
   if the user explicitly says this is ready to start right now.

7. **Report back** the issue URL and the Status/Phase you set.
