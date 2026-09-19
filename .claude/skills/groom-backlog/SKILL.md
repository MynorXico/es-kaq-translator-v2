---
name: groom-backlog
description: Review the Backlog column of the Traductor Kaqchikel project board, refine descriptions/acceptance criteria, flag duplicates or stale items, and promote ready items to Todo. Use periodically, not per-ticket - this is a batch review.
---

# Groom backlog

The "groom" step of the project's workflow (`docs/workflow.md`): a batch
pass over everything sitting in `Backlog` on project `6`, deciding what's
actually ready to move to `Todo`.

## Steps

1. **List everything in Backlog:**

   ```sh
   gh project item-list 6 --owner MynorXico --format json \
     | jq '.items[] | select(.status=="Backlog")'
   ```

   For a small backlog (roughly under 15-20 items), do this pass inline
   in the current session. If it's grown large enough that reading every
   issue body would flood context, delegate the read-and-summarize part
   to the `product-owner` subagent instead (`Agent({subagent_type:
   "product-owner", ...})`) and just act on its findings here.

2. **Cross-item pass first, before looking at items one by one**: scan
   the whole Backlog (plus Todo) list together for shared unmet
   prerequisites — two or more items depending on the same not-yet-decided
   foundation (a design, an architecture decision, a shared pattern),
   each written as if it were independent. This is a real failure mode,
   not hypothetical: six tickets (#36/#38/#39/#40/#41/#44) were each
   separately written as "needs a `ux` pass" before anyone noticed they
   all needed the *same* missing design, which should have been (and
   eventually was, as #52) one consolidated ticket. If you find this
   pattern, propose splitting out the shared prerequisite as its own
   ticket rather than leaving each dependent item to trigger it
   separately.

3. **Phase-coverage check**: does the Backlog/Todo/In Progress set for
   the current phase actually cover what that phase's stated goal in
   `docs/adr/0001-initial-architecture.md` requires, or only whatever's
   been reactively requested so far? This is also a real failure mode —
   the board once only had narrow "wire the stub to the real model"
   tickets for all of Phase 1, with no ticket at all for what the actual
   product needed, until the user asked. You don't need to solve this
   every grooming pass, but ask it explicitly and flag gaps to the user
   rather than assuming silence means coverage is complete.

4. **For each item, check:**
   - Is the scope still clear (explicit acceptance criteria, not just a
     vague title)? If not, tighten the issue body rather than leaving it
     ambiguous for whoever picks it up.
   - Is it blocked by another open issue? If so, leave it in Backlog and
     note the blocker in a comment if that's not already visible.
   - Is it still relevant, or has something else made it moot (a
     decision changed, a dependency was solved differently)? If it looks
     stale/duplicate, don't close it unilaterally — propose closing it to
     the user and explain why, same as any judgment call that isn't
     purely mechanical.
   - Does its `Phase` field still match the roadmap
     (`docs/adr/0001-initial-architecture.md`)? Fix it if things shifted.

5. **Promote what's ready.** An item is ready for `Todo` when: it's
   unblocked, has clear acceptance criteria, and fits the current phase's
   priorities (per `.claude/agents/product-owner.md`'s prioritization
   criteria — blocking the current phase, translation quality/
   correctness, contributor onboarding). Move it:

   ```sh
   gh project item-edit --id <item-id> --project-id <PROJECT_ID> \
     --field-id <STATUS_FIELD_ID> --single-select-option-id <TODO_OPTION_ID>
   ```

6. **Report a summary**: what moved to Todo (and why it's ready), what
   needs the user's input (proposed closures, unclear scope you couldn't
   resolve yourself, any shared-prerequisite splits or phase-coverage
   gaps from steps 2-3), and what's staying in Backlog with a one-line
   reason each (usually "blocked by #N" or "not yet prioritized").
