import { App } from "aws-cdk-lib";
import { Template } from "aws-cdk-lib/assertions";
import { describe, expect, it } from "vitest";
import { WebStack } from "../lib/web-stack";

describe("WebStack", () => {
  it("synthesizes an S3 bucket for the given environment", () => {
    const app = new App();
    const stack = new WebStack(app, "TestWebStack", { environmentName: "test" });
    const template = Template.fromStack(stack);

    template.hasResourceProperties("AWS::S3::Bucket", {
      BucketName: "traductor-kaqchikel-web-test",
    });
  });
});
