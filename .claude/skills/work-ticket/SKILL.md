---
name: work-ticket
description: Implement a specific Todo item end-to-end - move it to In Progress, implement it (delegating to the dev or ml-engineer agent), open a PR referencing the issue, and move it to In Review / QA. Takes a GitHub issue number as an argument.
---

# Work a ticket

The "develop" step of the project's workflow (`docs/workflow.md`). Takes
an issue number (passed as the skill argument) and drives it from `Todo`
through to an open PR.

## Steps

1. **Read the issue** (`gh issue view <N> --repo MynorXico/es-kaq-translator-v2
   --json title,body,labels,url`). If its acceptance criteria are too
   vague to implement against, stop and say so rather than guessing scope
   — send it back through `groom-backlog` instead of improvising.

2. **Pick the implementer** based on the issue's labels/content:
   - Touches `ml/` (label `ml`, or the body is clearly about corpus/
     training/evaluation) -> the `ml-engineer` agent.
   - Everything else (`apps/web`, `apps/api`, `infra/cdk`) -> the `dev`
     agent.

3. **If it touches `apps/web` UI** (new screen, new flow, visible layout/
   copy change) **and no design spec exists yet**, run the `ux` agent
   first (it uses `/design` to produce a mockup + concrete spec — Spanish
   copy, spacing, states). Don't let `dev` improvise visual/copy decisions
   for anything more than a trivial one-line text change.

4. **Move Status to `In Progress`** on the project board (item/field IDs
   from `.claude/agents/product-owner.md` /
   `gh project field-list 6 --owner MynorXico`).

5. **Delegate the implementation.** Spawn the chosen agent with a
   self-contained prompt — it starts with zero context, so include the
   issue title/body/URL verbatim, the acceptance criteria, and explicitly
   say it should: branch off `main` as `<N>-<short-kebab-slug>`, work
   **test-first** (Red → Green → Refactor per `docs/testing.md` — a
   failing test before each piece of production code, not tests written
   after), commit with a `Refs #<N>` line in each commit body, run the
   relevant checks (`make test` or the specific `pnpm`/`uv` commands for
   the component it touched) until green, then open a PR with
   `Closes #<N>` in the description — not merge it. See
   `docs/workflow.md`'s Git conventions for the full rules (squash-merge
   only, no direct pushes to `main`).

   ```
   Agent({
     subagent_type: "dev",  # or "ml-engineer"
     name: "ticket-<N>",
     description: "Implement issue #<N>",
     prompt: "<issue title/body/URL pasted in full, plus the explicit
       instructions above>"
   })
   ```

   For a small, quick ticket you're already deep in context on, it's also
   fine to implement it directly in the current session instead of
   spawning a subagent — use judgment; the point of delegating is to keep
   large/noisy implementation work out of the main session's context, not
   to add ceremony to a two-line fix.

6. **If the agent reports it hit an architecture decision** it wasn't
   scoped to make, don't push it to decide anyway — stop and route that
   through the `architect` agent / an ADR first, then resume.

7. **Once the PR is open**, move the project item's Status to
   `In Review / QA`. Report the PR URL back to the user.
