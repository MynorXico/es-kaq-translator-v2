# Runbook: pipeline execution failure alerting

Issue #110. Before this existed, `TraductorKaqchikelPipeline` (in
`translator-tooling`, see [ADR 0004](../adr/0004-cdk-pipelines-deployment.md))
had no notification mechanism at all — a self-mutation deadlock failed the
pipeline silently for hours, and every push to `main` after that kept
queuing changes that never actually deployed until someone happened to
check the AWS console.

## What exists now

`infra/cdk/lib/pipeline-alerting.ts`'s `PipelineFailureAlerting` construct
(wired into `buildPipelineApp` in `lib/pipeline-stack.ts`) creates, in the
same `translator-tooling` stack as the pipeline itself:

- An SNS topic, `TraductorKaqchikelPipelineFailures`.
- An EventBridge rule, `TraductorKaqchikelPipelineFailureRule`, matching
  `aws.codepipeline` events of `detail-type`
  `"CodePipeline Pipeline Execution State Change"` with
  `detail.pipeline = "TraductorKaqchikelPipeline"` and
  `detail.state = "FAILED"`, targeting that topic.
- An email subscription on the topic, if a notification address is
  configured (see below) — the topic and rule are still created even
  without one, so nothing about detecting the failure depends on a human
  having set up a subscriber yet.

The rule's target uses an `InputTransformer` so the SNS message is a
readable string, not raw JSON: which pipeline failed, the failing
execution's ID, the timestamp, and a direct console link to that
execution's timeline (e.g.
`https://us-east-1.console.aws.amazon.com/codesuite/codepipeline/pipelines/TraductorKaqchikelPipeline/executions/<execution-id>/timeline?region=us-east-1`) —
enough to act without digging further, satisfying issue #110's
acceptance criteria.

## Why execution-level only, not also a stage-level rule

The issue explicitly asked to verify this rather than assume it. Per
[AWS's own EventBridge/CodePipeline event documentation](https://docs.aws.amazon.com/codepipeline/latest/userguide/detect-state-changes-cloudwatch-events.html),
CodePipeline emits three kinds of state-change events (all `source:
aws.codepipeline`): `CodePipeline Pipeline Execution State Change`,
`CodePipeline Stage Execution State Change`, and `CodePipeline Action
Execution State Change`. Events are described as emitted "on a
guaranteed, at-least-once basis" for each of these resource types.

A stage can only fail by an action within it failing, and CodePipeline's
own execution model always transitions the pipeline *execution* itself to
`FAILED` as a direct consequence — there's no valid state where a stage is
permanently failed but the execution stays `InProgress`. So a stage-level
`FAILED` event is always followed by an execution-level `FAILED` event for
that same failure. The one exception — a stage that fails and is later
*retried* successfully (`RetryStageExecution`) — is precisely the case
where a stage-level alert would be a false positive (the pipeline as a
whole did not fail).

Conclusion: a separate stage-level rule would only ever produce a second,
less-informative notification for a failure the execution-level rule
already caught, not catch a failure the execution-level rule would
otherwise miss. Scope was kept to the execution-level rule only. If this
assumption is ever contradicted in production — a real stage failure with
no corresponding execution-level `FAILED` notification arriving within a
reasonable time — that's a gap to close here, not a "known limitation" to
quietly live with.

## Notification channel: email (default), configured via SSM

Per the issue, email was used as the simplest default (no strong
preference was raised for Slack or another channel). If that changes
later, only `PipelineFailureAlerting`'s subscription needs to change (e.g.
swap `EmailSubscription` for an `UrlSubscription` pointing at a Slack
incoming webhook, or add both) — the topic and EventBridge rule stay the
same either way.

Per `CLAUDE.md`, real email addresses are never committed to this public
repo. The maintainer's address lives in SSM Parameter Store in
`translator-tooling`, the same pattern ADR 0004 already established for
account IDs and the GitHub connection ARN:

```sh
aws ssm put-parameter --profile translator-tooling --region us-east-1 \
  --name "/traductor-kaqchikel/alerts/pipeline-failure-email" \
  --type String --value "<maintainer-email>" --overwrite
```

`bin/app.ts`'s `realConfig()` fetches this via `fetchOptionalSsmParameter`
(absence is expected and not an error — e.g. before this parameter has
ever been created, or intentionally left unset) and threads it through as
`PipelineConfig.pipelineFailureNotificationEmail`. The mock-accounts path
(`useMockAccounts=true`, used by CI's credential-free PR checks and local
non-deploy synths) uses an obviously-fake placeholder
(`pipeline-alerts@example.com`) purely so that code path still exercises
the SNS subscription resource in tests.

**After creating or changing the parameter**, redeploy the pipeline stack
so the new subscription is created (the pipeline is self-mutating, so a
normal merge to `main` picks this up automatically — no separate manual
`cdk deploy` needed once the parameter exists before that merge runs).
**Then confirm the subscription**: SNS emails a confirmation link to the
new address, and no notifications are delivered until it's clicked
(`aws sns list-subscriptions-by-topic --profile translator-tooling
--region us-east-1 --topic-arn <topic-arn>` shows
`SubscriptionArn: "PendingConfirmation"` until then).

## Verifying it actually works

There's no automated integration test that triggers a real pipeline
failure (see `docs/testing.md` — infra's test tier is CDK assertions
against synthesized templates, not live deploys). To verify manually
after standing this up for the first time:

1. Confirm the email subscription (above).
2. Deliberately break something trivial that fails a stage (e.g. a
   syntax error in a file the `Synth` step lints/type-checks, on a
   throwaway branch merged only for this test, or an equivalent
   reversible break) and watch for the email to arrive within a few
   minutes of the pipeline reporting `Failed` in the console.
3. Revert the break.

## Related

- Issue #47 (separate, application-level `apps/api` observability —
  structured logging + CloudWatch alarms for 5xx rates, etc.) is explicitly
  out of scope here; this runbook only covers pipeline/deployment health.
- [ADR 0004](../adr/0004-cdk-pipelines-deployment.md) for the overall
  pipeline design this alerting sits alongside.
