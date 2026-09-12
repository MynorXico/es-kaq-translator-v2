# Testing strategy: TDD and the test pyramid

All new code in this project is written **test-first**, following
Red → Green → Refactor:

1. **Red**: write a test that captures the behavior you're about to add.
   Run it. Confirm it fails, and that it fails *for the right reason*
   (the behavior doesn't exist yet — not a typo, import error, or broken
   test setup).
2. **Green**: write the minimum implementation needed to make that test
   pass. Run the full local suite for the component, not just the new
   test.
3. **Refactor**: clean up the implementation (and the test, if needed)
   while keeping every test green. Don't skip this step — it's what keeps
   TDD from producing throwaway-quality code.

Repeat per unit of behavior rather than writing a large batch of
production code before any test exists. A PR that adds non-trivial
production code with no corresponding test change is a signal TDD wasn't
followed — `code-reviewer` flags this.

This applies to `dev` and `ml-engineer` work. Pure docs/config/ADR changes
don't need tests. `qa` is responsible for auditing that the pyramid below
is actually complete, not just that *some* test exists.

## The pyramid, per component

Not every component needs every level yet — add a level when there's
something real to test at it, not preemptively.

### `apps/web` (React + Vite)

| Level | Tool | What it covers | Where |
|---|---|---|---|
| Unit/component | Vitest + Testing Library | Individual components/functions in isolation | `src/**/*.test.tsx` |
| E2E | Playwright | Real user flows in an actual browser, against the built app | `e2e/**/*.spec.ts` |

Run: `pnpm --filter web test` (unit/component), `pnpm --filter web
test:e2e` (Playwright — builds the app and serves it via `vite preview`
automatically).

There is no "integration" tier specific to `apps/web` beyond
component tests that render more than one component together — Testing
Library already covers that at the unit/component level.

### `apps/api` (FastAPI)

| Level | Tool | What it covers | Where |
|---|---|---|---|
| Unit | pytest | Pure functions/logic in isolation (no ASGI app, no I/O) | `tests/unit/` |
| Integration | pytest + `TestClient` | Full request→response through the actual FastAPI app (routing, validation), with any external service (e.g. SageMaker) mocked at the boundary | `tests/integration/` |

Run: `uv run pytest` (runs both directories). As of now the app is a
stub with no pure functions worth unit-testing in isolation — the
existing tests live in `tests/integration/` since they exercise the full
app via `TestClient`. Add `tests/unit/` when there's actual
logic (e.g. request validation/transformation) worth isolating from the
HTTP layer.

There is no live-AWS integration tier in the fast local/CI suite — once
the API calls a real SageMaker endpoint, that call is mocked in
`tests/integration/` (e.g. via `moto` or a `responses`-style stub), never
hitting real AWS. Verifying the real deployed endpoint works is a
post-deploy QA step (see below), not part of this suite.

### `infra/cdk` (AWS CDK)

| Level | Tool | What it covers | Where |
|---|---|---|---|
| Unit | Vitest + `aws-cdk-lib/assertions` | Synthesized CloudFormation has the resources/properties a stack should produce | `test/**/*.test.ts` |

Run: `pnpm --filter infra-cdk test`. CDK assertion tests are this
project's infra "unit" tier — they synth without needing real AWS
credentials. There's no automated integration/e2e tier for infra in CI
(that would mean deploying real stacks); actually deploying and verifying
a live environment is a QA responsibility before promoting dev → qa →
prod, not a PR gate.

### `ml/` (not built yet)

No code exists here yet. When it does:

| Level | Tool | What it covers |
|---|---|---|
| Unit | pytest | Pure data-cleaning/preprocessing functions (dedup, normalization) against small fixture inputs |
| Integration | pytest | A fast smoke run of the pipeline wiring (tiny fixture dataset, not the real corpus, not a real SageMaker training job) — confirms the pieces connect, not model quality |

There is no e2e tier for `ml/` — "does the deployed model actually
translate correctly" is measured by BLEU/chrF against the validation set
(`ml/evaluation`), which is a quality metric, not a pass/fail test, and by
QA's post-deploy verification that a request through the real API returns
*a* translation.

## What "done" requires

A PR is TDD-complete when: the tests that exist for the levels above that
the change actually touches were written before (or alongside, commit by
commit) the implementation, the full local suite for that component
passes, and CI is green. `code-reviewer` and `qa` both check this — see
their agent definitions.
