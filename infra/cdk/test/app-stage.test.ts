import path from "node:path";
import { App } from "aws-cdk-lib";
import { Match, Template } from "aws-cdk-lib/assertions";
import { describe, expect, it } from "vitest";
import { ApiStack } from "../lib/api-stack";
import { TranslatorStage } from "../lib/app-stage";
import { webHostName } from "../lib/config";
import { MlHostingStack } from "../lib/ml-hosting-stack";

// Fake, obviously-non-real account ID (CLAUDE.md never allows a real one in tests).
const FAKE_ENV = { account: "222222222222", region: "us-east-1" };

// A tiny committed fixture directory, not a real `apps/web` Vite build --
// see test/web-stack.test.ts for the same reasoning.
const FIXTURE_SITE_CONTENT_PATH = path.join(__dirname, "fixtures/site");

function synthStage(environmentName: string) {
  const app = new App();
  const stage = new TranslatorStage(app, `Test${environmentName}`, {
    environmentName,
    webSiteContentPath: FIXTURE_SITE_CONTENT_PATH,
    // Always "already validated" so ADR 0007's new-certificate guard never
    // blocks this stage-wiring test (it isn't exercising WebStack's own
    // cert-guard behavior -- that's test/web-stack.test.ts's job).
    previouslyValidatedWebDomainName: webHostName(environmentName),
    env: FAKE_ENV,
  });
  const apiStack = stage.node.findChild("Api") as ApiStack;
  const apiTemplate = Template.fromStack(apiStack);
  return { stage, apiTemplate };
}

describe("TranslatorStage", () => {
  it("wires ApiStack's SAGEMAKER_ENDPOINT_NAME to the real MlHostingStack endpoint name in dev (issue #9)", () => {
    const { stage, apiTemplate } = synthStage("dev");

    const mlHostingStack = stage.node.findChild("MlHosting") as MlHostingStack;

    // Guards against these two independently-computed names silently
    // drifting apart again.
    expect(mlHostingStack.endpointName).toBe("traductor-kaqchikel-es-cak-dev");

    apiTemplate.hasResourceProperties("AWS::Lambda::Function", {
      Environment: {
        Variables: Match.objectLike({
          SAGEMAKER_ENDPOINT_NAME: mlHostingStack.endpointName,
        }),
      },
    });
  });

  // ADR 0010 (issue #235): MlHostingStack is now instantiated via a
  // data-driven lookup (MODEL_PACKAGE_VERSION_BY_ENVIRONMENT[environmentName])
  // rather than a dev-only `if` plus a single hardcoded constant -- verify
  // the version that lookup resolves for dev (19, per app-stage.ts's own
  // version-history comment) is actually the one threaded through into the
  // real Model Package ARN, not just that *some* MlHostingStack exists.
  it("instantiates MlHostingStack for dev with the version MODEL_PACKAGE_VERSION_BY_ENVIRONMENT maps it to", () => {
    const { stage } = synthStage("dev");

    const mlHostingStack = stage.node.findChild("MlHosting") as MlHostingStack;
    const mlHostingTemplate = Template.fromStack(mlHostingStack);

    mlHostingTemplate.hasResourceProperties("AWS::SageMaker::Model", {
      Containers: Match.arrayWith([
        Match.objectLike({
          ModelPackageName: Match.stringLikeRegexp("traductor-kaqchikel-es-cak/19$"),
        }),
      ]),
    });
  });

  it("still synthesizes ApiStack for qa, where no MlHostingStack exists yet", () => {
    const { stage, apiTemplate } = synthStage("qa");

    expect(stage.node.tryFindChild("MlHosting")).toBeUndefined();

    // Falls back to ApiStack's own default naming convention -- a known,
    // pre-existing gap (no real endpoint in qa yet), not something this
    // ticket fixes.
    apiTemplate.hasResourceProperties("AWS::Lambda::Function", {
      Environment: {
        Variables: Match.objectLike({
          SAGEMAKER_ENDPOINT_NAME: "traductor-kaqchikel-translate-qa",
        }),
      },
    });
  });

  // ADR 0010 (issue #235): MODEL_PACKAGE_VERSION_BY_ENVIRONMENT is a
  // data-driven map, not a dev-only `if` -- this must hold for *any*
  // environment absent from it, not just the one ("qa") that happened to
  // be tested before this ticket. "prod" is a second, independent case
  // confirming the lookup itself is generic rather than still secretly
  // special-casing "dev" under the hood.
  it("does not instantiate MlHostingStack for prod, which also has no entry in MODEL_PACKAGE_VERSION_BY_ENVIRONMENT yet", () => {
    const { stage, apiTemplate } = synthStage("prod");

    expect(stage.node.tryFindChild("MlHosting")).toBeUndefined();

    apiTemplate.hasResourceProperties("AWS::Lambda::Function", {
      Environment: {
        Variables: Match.objectLike({
          SAGEMAKER_ENDPOINT_NAME: "traductor-kaqchikel-translate-prod",
        }),
      },
    });
  });
});
