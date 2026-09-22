import { Stage, StageProps } from "aws-cdk-lib";
import type { Construct } from "constructs";
import { ApiStack } from "./api-stack";
import { DataStack } from "./data-stack";
import { WebStack } from "./web-stack";

export interface TranslatorStageProps extends StageProps {
  environmentName: string;
}

/**
 * One promotion target (dev/qa/prod) for the CDK Pipelines deployment
 * pipeline (ADR 0004). Wraps every stack that should exist in a given
 * environment/account — `WebStack`, `DataStack`, and `ApiStack` so far,
 * more to follow as the ML hosting stack is added (see ADR 0001).
 */
export class TranslatorStage extends Stage {
  constructor(scope: Construct, id: string, props: TranslatorStageProps) {
    super(scope, id, props);

    new WebStack(this, "Web", {
      environmentName: props.environmentName,
    });

    new DataStack(this, "Data", {
      environmentName: props.environmentName,
    });

    new ApiStack(this, "Api", {
      environmentName: props.environmentName,
    });
  }
}
