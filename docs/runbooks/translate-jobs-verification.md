# Runbook: verifying the async translate-jobs pattern (ADR 0008)

Issue #258, implementing the verification [ADR 0008](../adr/0008-async-translate-job-pattern.md)
itself calls out as required before promoting the pattern beyond `dev`:
the Worker Lambda's genuine-timeout -> `onFailure` Lambda Destination ->
job-marked-`failed` path cannot be proven in a fast, credential-free CI
run (it's a real AWS platform behavior, not application code), so it must
be checked against real infrastructure by hand.

This runbook has two parts: a one-time-per-change **forced-timeout
verification** (re-run whenever the Worker/Failure Lambda handlers or
their IAM/async-invoke wiring change materially), and a lightweight
**enqueue-then-poll smoke check** (run after every deploy to a given
environment, including routine dev -> qa -> prod promotions of this
feature).

## Prerequisites

- The target environment's named AWS SSO profile
  (`translator-<env>`, per
  [the account bootstrap runbook](aws-account-bootstrap.md)) is logged in.
- You know the real deployed names of the environment's `WorkerFunction`
  and the `TranslateJobsTable` -- find them with:

  ```sh
  aws lambda list-functions --profile translator-<env> \
    --query "Functions[?contains(FunctionName, 'Worker') || contains(FunctionName, 'Failure')].{Name:FunctionName,Timeout:Timeout}"
  aws dynamodb list-tables --profile translator-<env>
  ```

## Part 1: forced-timeout verification

**This temporarily reduces a live Lambda's timeout, which is a real
production-infrastructure mutation (reversible, but still a "Modify
Shared Resources"-class action) -- get explicit sign-off before running
it, same as any other live-infra change this project makes.**

1. Record the Worker Lambda's current (real) timeout, so you can restore
   it exactly:

   ```sh
   aws lambda get-function-configuration --profile translator-<env> \
     --function-name <WorkerFunctionName> --query Timeout --output text
   ```

2. Temporarily set it to a few seconds -- short enough that any real
   translation request will exceed it, but long enough for the Lambda to
   actually start and the real timeout mechanism (not an immediate
   validation error) to fire:

   ```sh
   aws lambda update-function-configuration --profile translator-<env> \
     --function-name <WorkerFunctionName> --timeout 3
   aws lambda wait function-updated --profile translator-<env> \
     --function-name <WorkerFunctionName>
   ```

3. Submit a real job and record the `job_id`:

   ```sh
   curl -s -X POST "<ApiUrl>/v1/translate-jobs" \
     -H "Content-Type: application/json" \
     --data '{"text": "Buenos dias, como estas?", "direction": "es-to-cak"}'
   ```

4. Confirm the DynamoDB item starts `pending`:

   ```sh
   aws dynamodb get-item --profile translator-<env> \
     --table-name <TranslateJobsTableName> \
     --key '{"job_id": {"S": "<job_id>"}}'
   ```

5. Poll `GET <ApiUrl>/v1/translate-jobs/<job_id>` every few seconds. It
   should flip to `failed` well within the expected window (a few
   seconds after the forced timeout, not minutes) --
   `{"status": "failed", "status_code": 504, "error": "..."}`.

6. Confirm the real mechanism, not just the end state, via CloudWatch
   Logs:

   ```sh
   aws logs filter-log-events --profile translator-<env> \
     --log-group-name /aws/lambda/<WorkerFunctionName> \
     --filter-pattern "REPORT" --start-time <recent-epoch-ms>
   # Expect a REPORT line ending "... Status: timeout" at your configured timeout.

   aws logs filter-log-events --profile translator-<env> \
     --log-group-name /aws/lambda/<FailureFunctionName> \
     --start-time <recent-epoch-ms>
   # Expect a structured translate_job_resolved log line with the same
   # job_id, error_type "WorkerTimeoutOrCrash", latency_ms null.
   ```

7. Confirm the DynamoDB item's final state directly (not just the API
   response) -- `status: failed`, `status_code: 504`, a real `error`
   message, and the original `text`/`direction` still present (the
   conditional write only touches `status`/`error`/`status_code`, never
   clobbers the job's input fields).

8. **Immediately restore the Worker Lambda's real timeout** (the value
   from step 1 -- `90` as of this writing, per `infra/cdk/lib/api-stack.ts`):

   ```sh
   aws lambda update-function-configuration --profile translator-<env> \
     --function-name <WorkerFunctionName> --timeout 90
   aws lambda wait function-updated --profile translator-<env> \
     --function-name <WorkerFunctionName>
   ```

9. Run the smoke check below to confirm the environment is back to normal
   (a real job succeeds end-to-end) before considering this done.

### On the conditional-write race (Worker-finishes vs. onFailure-fires near-simultaneously)

Not forced for real here, deliberately: genuinely racing a Lambda's own
terminal write against its `onFailure` destination's write requires
timing control this project doesn't have a safe way to force against
live infrastructure (unlike the plain timeout above, which is
straightforwardly reproducible by shortening the timeout). This exact
scenario is already covered test-first with pre-seeded conflicting state
in `apps/api/tests/unit/test_jobs.py` (both directions: Worker-writes-
first-then-onFailure-tries, and vice versa), using a fake DynamoDB table
that evaluates the real `ConditionExpression` string rather than
hardcoding pass/fail -- reviewed and confirmed genuine in PR #259's code
review. That unit coverage is treated as sufficient for this specific
race; this runbook's forced-timeout check instead verifies the simpler,
far-more-common case (a clean timeout with no race) actually works
end-to-end against real AWS.

## Part 2: enqueue-then-poll smoke check (run after every deploy)

A simple, non-destructive, repeatable check for routine post-deploy
verification -- no infrastructure mutation required.

```sh
job_id=$(curl -s -X POST "<ApiUrl>/v1/translate-jobs" \
  -H "Content-Type: application/json" \
  --data '{"text": "Buenos dias, como estas?", "direction": "es-to-cak"}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])")

# Poll every few seconds until a terminal status (allow up to ~90s for a
# genuine cold start -- the Worker Lambda's real timeout ceiling):
curl -s "<ApiUrl>/v1/translate-jobs/$job_id"
```

Expect `{"status": "succeeded", "translation": "..."}` eventually (a
non-empty Kaqchikel string). If it stays `pending` past ~90-100s, or
returns `failed`/`404` unexpectedly, something is genuinely wrong --
check the Worker/Request Lambda's CloudWatch Logs next, same as any other
real deployment issue in this project.

## Real verification history

- **2026-10-11, `dev`**: forced-timeout check run per this runbook (issue
  #258). Confirmed: Worker Lambda hit its configured timeout exactly
  (`REPORT` line, `Status: timeout`), `FailureFunction` fired within
  ~1 second of the timeout with a correctly-correlated structured log
  line, and the job's DynamoDB item flipped to `failed` with the
  expected `status_code`/`error`, all within a few seconds of the
  timeout -- well inside the expected window. Worker Lambda timeout
  restored to 90s immediately after. A follow-up smoke check against
  both `dev` and `qa` confirmed both work end-to-end through a real
  SageMaker cold start (observed ~13-46s resolution times, both well
  under the 90-120s ceiling).
