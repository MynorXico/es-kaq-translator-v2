import { App } from "aws-cdk-lib";
import { Match, Template } from "aws-cdk-lib/assertions";
import { describe, expect, it } from "vitest";
import { ApiStack } from "../lib/api-stack";

// Fake, obviously-non-real account ID (CLAUDE.md never allows a real one in tests).
const FAKE_ENV = { account: "222222222222", region: "us-east-1" };

function synthApiStack(props?: { sageMakerEndpointName?: string }) {
  const app = new App();
  const stack = new ApiStack(app, "TestApiStack", {
    environmentName: "test",
    env: FAKE_ENV,
    ...props,
  });
  const template = Template.fromStack(stack);
  return { stack, template };
}

describe("ApiStack", () => {
  it("deploys apps/api as a Lambda container image", () => {
    const { template } = synthApiStack();

    template.hasResourceProperties("AWS::Lambda::Function", {
      PackageType: "Image",
    });
  });

  describe("REST API migration (issue #151, ADR 0005)", () => {
    it("exposes a regional API Gateway REST API in front of the Lambda, not an HttpApi", () => {
      const { template } = synthApiStack();

      // AWS WAF can only associate with a regional REST API (v1) stage --
      // never with an HttpApi (v2) at all. See ADR 0005's amended Decision.
      template.hasResourceProperties("AWS::ApiGateway::RestApi", {
        EndpointConfiguration: {
          Types: ["REGIONAL"],
        },
      });

      // No v2 HTTP API left behind by the migration.
      template.resourceCountIs("AWS::ApiGatewayV2::Api", 0);
    });

    it("routes every path/method to the Lambda via a greedy proxy resource, AWS_PROXY integration", () => {
      const { template } = synthApiStack();

      // The `{proxy+}` resource that LambdaRestApi's `proxy: true` creates,
      // covering both /v1/translate-jobs routes (ADR 0005) and everything
      // else apps/api's own FastAPI routing handles -- mirrors the
      // single $default route the HttpApi used to have.
      template.hasResourceProperties("AWS::ApiGateway::Resource", {
        PathPart: "{proxy+}",
      });

      template.hasResourceProperties(
        "AWS::ApiGateway::Method",
        Match.objectLike({
          HttpMethod: "ANY",
          Integration: Match.objectLike({
            Type: "AWS_PROXY",
          }),
        }),
      );
    });

    it("does not create any API key or usage-plan resources (ADR 0005 rejected that mechanism)", () => {
      const { template } = synthApiStack();

      template.resourceCountIs("AWS::ApiGateway::ApiKey", 0);
      template.resourceCountIs("AWS::ApiGateway::UsagePlan", 0);
      template.resourceCountIs("AWS::ApiGateway::UsagePlanKey", 0);
    });

    it("does not leave a trailing slash on the ApiUrl output (PR #154 review)", () => {
      const { template } = synthApiStack();

      // `LambdaRestApi.url` always ends in "/" (e.g. ".../dev/"), unlike
      // the old `HttpApi.apiEndpoint`. apps/web builds requests as
      // `${apiBaseUrl}/v1/translate...` -- a leftover trailing slash here
      // would produce a double slash once concatenated, which API
      // Gateway's exact-path resource matching won't route correctly.
      const outputs = template.findOutputs("ApiUrl");
      const [output] = Object.values(outputs);
      const value = output.Value as { "Fn::Join"?: [string, unknown[]] };

      expect(value["Fn::Join"]).toBeDefined();
      const parts = value["Fn::Join"]?.[1] ?? [];
      const lastPart = parts[parts.length - 1];

      expect(lastPart).not.toBe("/");
      if (typeof lastPart === "string") {
        expect(lastPart.endsWith("/")).toBe(false);
      }
    });
  });

  describe("WAF rate limiting (issue #151, ADR 0005)", () => {
    it("creates a regional WebACL with CloudWatch metrics enabled", () => {
      const { template } = synthApiStack();

      template.hasResourceProperties(
        "AWS::WAFv2::WebACL",
        Match.objectLike({
          Scope: "REGIONAL",
          VisibilityConfig: Match.objectLike({
            CloudWatchMetricsEnabled: true,
          }),
        }),
      );
    });

    it("has exactly one rate-based rule aggregating by source IP, blocking over the threshold", () => {
      const { template } = synthApiStack();

      template.hasResourceProperties(
        "AWS::WAFv2::WebACL",
        Match.objectLike({
          Rules: Match.arrayWith([
            Match.objectLike({
              Statement: Match.objectLike({
                RateBasedStatement: Match.objectLike({
                  AggregateKeyType: "IP",
                  // Low hundreds of requests per 5-minute window, per ADR 0005.
                  Limit: Match.anyValue(),
                }),
              }),
              Action: Match.objectLike({
                Block: Match.anyValue(),
              }),
            }),
          ]),
        }),
      );

      const webAcls = template.findResources("AWS::WAFv2::WebACL");
      const [webAcl] = Object.values(webAcls);
      const rules = webAcl.Properties.Rules as Array<{
        Statement: { RateBasedStatement?: { Limit: number } };
      }>;
      const rateBasedRules = rules.filter((rule) => rule.Statement.RateBasedStatement);
      expect(rateBasedRules.length).toBe(1);
      expect(rateBasedRules[0].Statement.RateBasedStatement?.Limit).toBeGreaterThanOrEqual(100);
      expect(rateBasedRules[0].Statement.RateBasedStatement?.Limit).toBeLessThan(1000);
    });

    it("uses a custom 429 block response with a Retry-After header, not WAF's default 403", () => {
      const { template } = synthApiStack();

      template.hasResourceProperties(
        "AWS::WAFv2::WebACL",
        Match.objectLike({
          Rules: Match.arrayWith([
            Match.objectLike({
              Action: Match.objectLike({
                Block: Match.objectLike({
                  CustomResponse: Match.objectLike({
                    ResponseCode: 429,
                    ResponseHeaders: Match.arrayWith([
                      Match.objectLike({ Name: "Retry-After" }),
                    ]),
                  }),
                }),
              }),
            }),
          ]),
        }),
      );
    });

    it("includes CORS headers on the 429 block response, matching apps/web's origin (PR #154 review)", () => {
      const { template } = synthApiStack();

      // WAF intercepts before FastAPI's CORSMiddleware ever runs, so a
      // blocked cross-origin request gets no Access-Control-Allow-Origin
      // unless WAF's own custom response supplies one -- otherwise the
      // browser's fetch() rejects it as an opaque CORS/network error
      // instead of a distinguishable 429 apps/web can classify (#152).
      template.hasResourceProperties(
        "AWS::WAFv2::WebACL",
        Match.objectLike({
          Rules: Match.arrayWith([
            Match.objectLike({
              Action: Match.objectLike({
                Block: Match.objectLike({
                  CustomResponse: Match.objectLike({
                    ResponseHeaders: Match.arrayWith([
                      Match.objectLike({
                        Name: "Access-Control-Allow-Origin",
                        // Same per-environment origin apps/api/app/config.py's
                        // ALLOWED_ORIGINS uses (`webHostName()`), not a
                        // wildcard "*" -- matches the test above asserting
                        // ALLOWED_ORIGINS is "https://app-test.traductorkaqchikel.com".
                        Value: "https://app-test.traductorkaqchikel.com",
                      }),
                      Match.objectLike({ Name: "Access-Control-Allow-Methods" }),
                      Match.objectLike({ Name: "Access-Control-Allow-Headers" }),
                    ]),
                  }),
                }),
              }),
            }),
          ]),
        }),
      );
    });

    it("associates the WebACL with the REST API's deployment stage", () => {
      const { template } = synthApiStack();

      const webAcls = template.findResources("AWS::WAFv2::WebACL");
      const webAclLogicalIds = Object.keys(webAcls);
      expect(webAclLogicalIds.length).toBe(1);

      const associations = template.findResources("AWS::WAFv2::WebACLAssociation");
      const associationList = Object.values(associations);
      expect(associationList.length).toBe(1);

      const [association] = associationList;
      const resourceArnStr = JSON.stringify(association.Properties.ResourceArn);
      // The REST API stage ARN shape WAF's AssociateWebACL requires --
      // .../restapis/{api-id}/stages/{stage-name} -- not a bare API ARN.
      expect(resourceArnStr).toContain("/restapis/");
      expect(resourceArnStr).toContain("/stages/");

      const webAclArnStr = JSON.stringify(association.Properties.WebACLArn);
      expect(webAclArnStr).toContain(webAclLogicalIds[0]);
    });

    it("alarms on the WebACL's BlockedRequests metric (AWS/WAFV2), not left invisible", () => {
      const { template } = synthApiStack();

      template.hasResourceProperties(
        "AWS::CloudWatch::Alarm",
        Match.objectLike({
          MetricName: "BlockedRequests",
          Namespace: "AWS/WAFV2",
          ComparisonOperator: Match.stringLikeRegexp("GreaterThan"),
        }),
      );
    });
  });

  it("passes the SageMaker endpoint name to the Lambda via an env var, defaulting per environment", () => {
    const { template } = synthApiStack();

    template.hasResourceProperties("AWS::Lambda::Function", {
      Environment: {
        Variables: Match.objectLike({
          SAGEMAKER_ENDPOINT_NAME: "traductor-kaqchikel-translate-test",
        }),
      },
    });
  });

  it("passes this environment's real custom-domain origin as ALLOWED_ORIGINS, not left unset", () => {
    const { template } = synthApiStack();

    template.hasResourceProperties("AWS::Lambda::Function", {
      Environment: {
        Variables: Match.objectLike({
          ALLOWED_ORIGINS: "https://app-test.traductorkaqchikel.com",
        }),
      },
    });
  });

  it("allows overriding the SageMaker endpoint name explicitly", () => {
    const { template } = synthApiStack({ sageMakerEndpointName: "some-other-endpoint" });

    template.hasResourceProperties("AWS::Lambda::Function", {
      Environment: {
        Variables: Match.objectLike({
          SAGEMAKER_ENDPOINT_NAME: "some-other-endpoint",
        }),
      },
    });
  });

  it("grants the Lambda execution role sagemaker:InvokeEndpoint scoped to exactly that endpoint, not a wildcard", () => {
    const { template } = synthApiStack();

    const policies = template.findResources("AWS::IAM::Policy");
    const sagemakerStatements = Object.values(policies).flatMap((policy) => {
      const statements = policy.Properties.PolicyDocument.Statement as Array<{
        Effect: string;
        Action: string | string[];
        Resource: unknown;
      }>;
      return statements.filter((statement) => {
        const actions = Array.isArray(statement.Action) ? statement.Action : [statement.Action];
        return statement.Effect === "Allow" && actions.includes("sagemaker:InvokeEndpoint");
      });
    });

    expect(sagemakerStatements.length).toBe(1);

    const [statement] = sagemakerStatements;
    const resources = Array.isArray(statement.Resource) ? statement.Resource : [statement.Resource];
    expect(resources.length).toBe(1);

    const resolved = JSON.stringify(resources[0]);
    expect(resolved).not.toBe('"*"');
    // Must resolve to a real endpoint ARN pattern including the endpoint name,
    // not a bare wildcard resource.
    expect(resolved).toContain("endpoint/traductor-kaqchikel-translate-test");
  });

  it("does not attach a broad AWS-managed SageMaker or Lambda policy to the execution role", () => {
    const { template } = synthApiStack();

    const roles = template.findResources("AWS::IAM::Role");
    for (const role of Object.values(roles)) {
      const managedArns = (role.Properties.ManagedPolicyArns ?? []) as unknown[];
      const managedArnsStr = JSON.stringify(managedArns);
      expect(managedArnsStr).not.toContain("AmazonSageMakerFullAccess");
    }
  });

  it("outputs the API Gateway URL as a CfnOutput, matching the DataStack/WebStack pattern", () => {
    const { template } = synthApiStack();

    template.hasOutput("ApiUrl", {});
  });

  describe("CloudWatch alarms (issue #47)", () => {
    it("alarms on an elevated 5xx error rate from the REST API", () => {
      const { template } = synthApiStack();

      // RestApi's canned server-error metric is named "5XXError" (v1),
      // unlike HttpApi's "5xx" (v2) -- a real metric-name change from the
      // HttpApi -> RestApi migration (issue #151).
      template.hasResourceProperties(
        "AWS::CloudWatch::Alarm",
        Match.objectLike({
          MetricName: "5XXError",
          Namespace: "AWS/ApiGateway",
          ComparisonOperator: Match.stringLikeRegexp("GreaterThan"),
        }),
      );
    });

    it("alarms on elevated latency from the REST API", () => {
      const { template } = synthApiStack();

      template.hasResourceProperties(
        "AWS::CloudWatch::Alarm",
        Match.objectLike({
          MetricName: "Latency",
          Namespace: "AWS/ApiGateway",
          ComparisonOperator: Match.stringLikeRegexp("GreaterThan"),
        }),
      );
    });

    it("does not silently treat missing data as breaching for any alarm", () => {
      const { template } = synthApiStack();

      const alarms = template.findResources("AWS::CloudWatch::Alarm");
      // ServerErrorRateAlarm, HighLatencyAlarm, and (issue #151) the new
      // WAF BlockedRequestsAlarm.
      expect(Object.keys(alarms).length).toBeGreaterThanOrEqual(3);
      for (const alarm of Object.values(alarms)) {
        expect(alarm.Properties.TreatMissingData).toBe("notBreaching");
      }
    });
  });
});
