import { EventField, Rule, RuleTargetInput } from "aws-cdk-lib/aws-events";
import { SnsTopic } from "aws-cdk-lib/aws-events-targets";
import { Topic } from "aws-cdk-lib/aws-sns";
import { EmailSubscription } from "aws-cdk-lib/aws-sns-subscriptions";
import { Construct } from "constructs";

export interface PipelineFailureAlertingProps {
  /**
   * Name of the `AWS::CodePipeline::Pipeline` to alert on. Matched against
   * the EventBridge event's `detail.pipeline` field, which is the plain
   * pipeline name, not its ARN.
   */
  pipelineName: string;
  /**
   * Email address subscribed to the failure-notification SNS topic. Left
   * `undefined` when no real maintainer address is configured (e.g. a
   * credential-free/mock synth) -- the topic is still created either way,
   * so a subscription can be added later without resource churn.
   *
   * Never a literal real email in this public repo (CLAUDE.md) -- callers
   * source this from an SSM parameter (`bin/app.ts`) or CDK context, not a
   * hardcoded string.
   */
  notificationEmail?: string;
}

/**
 * Issue #110: notifies a maintainer within minutes, not hours, when
 * `TraductorKaqchikelPipeline`'s execution fails -- there was previously no
 * alerting at all, and a self-mutation deadlock went unnoticed for hours.
 *
 * ## Scope: execution-level only, not also stage-level
 *
 * CodePipeline emits three kinds of EventBridge events (source
 * `aws.codepipeline`): `CodePipeline Pipeline Execution State Change`,
 * `CodePipeline Stage Execution State Change`, and
 * `CodePipeline Action Execution State Change` (see AWS's own
 * "Monitoring CodePipeline events" guide). This construct only matches the
 * first, filtered to `detail.state = FAILED`.
 *
 * That's a deliberate choice, not an oversight: a stage can only fail by
 * failing an action within it, and CodePipeline's execution model always
 * transitions the *pipeline execution* itself to `FAILED` as a direct
 * consequence (the execution can't stay `InProgress` with a permanently
 * failed stage) -- so a stage-level `FAILED` event is always followed by an
 * execution-level `FAILED` event for the same failure, and CodePipeline
 * documents these events as emitted "on a guaranteed, at-least-once basis".
 * A separate stage-level rule would therefore only ever produce a second,
 * less-informative notification for a failure this rule already caught --
 * not catch anything this rule would otherwise miss. The one case a
 * stage-level event fires without an execution-level FAILED following it is
 * a stage that's later *retried* successfully (`RetryStageExecution`),
 * which is precisely the case where alerting would be a false positive.
 *
 * If this assumption is ever proven wrong in production (a real stage
 * failure with no corresponding execution-level FAILED notification),
 * that's a documented gap to close -- see
 * `docs/runbooks/pipeline-failure-alerting.md`.
 */
export class PipelineFailureAlerting extends Construct {
  public readonly topic: Topic;
  public readonly rule: Rule;

  constructor(scope: Construct, id: string, props: PipelineFailureAlertingProps) {
    super(scope, id);

    this.topic = new Topic(this, "Topic", {
      topicName: "TraductorKaqchikelPipelineFailures",
      displayName: "Traductor Kaqchikel pipeline failure alerts",
    });

    if (props.notificationEmail) {
      this.topic.addSubscription(new EmailSubscription(props.notificationEmail));
    }

    this.rule = new Rule(this, "Rule", {
      ruleName: "TraductorKaqchikelPipelineFailureRule",
      description: `Notifies ${this.topic.topicName} when ${props.pipelineName}'s pipeline execution fails`,
      eventPattern: {
        source: ["aws.codepipeline"],
        detailType: ["CodePipeline Pipeline Execution State Change"],
        detail: {
          pipeline: [props.pipelineName],
          state: ["FAILED"],
        },
      },
      targets: [
        new SnsTopic(this.topic, {
          // A plain-text message with the fields needed to act without
          // digging further (acceptance criterion): which pipeline, which
          // execution, when, and a direct console link to that execution's
          // timeline. EventField pulls each value from the actual matched
          // event at delivery time, not from CDK synth time.
          message: RuleTargetInput.fromText(
            `Traductor Kaqchikel pipeline "${EventField.fromPath("$.detail.pipeline")}" FAILED ` +
              `(execution-id: ${EventField.fromPath("$.detail.execution-id")}) at ${EventField.fromPath("$.time")}.\n` +
              `Console: https://${EventField.fromPath("$.region")}.console.aws.amazon.com/codesuite/codepipeline/pipelines/` +
              `${EventField.fromPath("$.detail.pipeline")}/executions/${EventField.fromPath("$.detail.execution-id")}/timeline` +
              `?region=${EventField.fromPath("$.region")}`,
          ),
        }),
      ],
    });
  }
}
