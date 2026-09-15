import { Stage, StageProps } from "aws-cdk-lib";
import type { Construct } from "constructs";
import { WebStack } from "./web-stack";

export interface TranslatorStageProps extends StageProps {
  environmentName: string;
}

/**
 * One promotion target (dev/qa/prod) for the CDK Pipelines deployment
 * pipeline (ADR 0004). Wraps every stack that should exist in a given
 * environment/account — currently just `WebStack`, more to follow as the
 * API/ML/data stacks are added (see ADR 0001).
 */
export class TranslatorStage extends Stage {
  constructor(scope: Construct, id: string, props: TranslatorStageProps) {
    super(scope, id, props);

    new WebStack(this, "Web", {
      environmentName: props.environmentName,
    });
  }
}
