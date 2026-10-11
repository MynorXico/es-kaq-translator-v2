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

type SynthedResource = { Properties: Record<string, unknown> };
type FoundFunction = { logicalId: string; resource: SynthedResource };

// ADR 0008: three Lambda functions share one Docker image, distinguished by
// `ImageConfig.Command` (the handler entry point) -- these helpers find the
// right one, and trace its execution role, so IAM assertions below can be
// scoped to exactly the function ADR 0008 says should hold a given grant.
function findFunctionByHandler(template: Template, handler: string): FoundFunction {
  const functions = template.findResources("AWS::Lambda::Function");
  const matches = Object.entries(functions).filter(([, fn]) => {
    const command = (fn.Properties?.ImageConfig as { Command?: string[] } | undefined)?.Command;
    return command?.includes(handler);
  });
  expect(matches.length).toBe(1);
  const [logicalId, resource] = matches[0];
  return { logicalId, resource: resource as SynthedResource };
}

function roleLogicalIdForFunction({ resource: fn }: FoundFunction): string {
  const role = fn.Properties.Role as { "Fn::GetAtt": [string, string] };
  return role["Fn::GetAtt"][0];
}

function policyStatementsForRole(
  template: Template,
  roleLogicalId: string,
): Array<{ Effect: string; Action: string | string[]; Resource: unknown }> {
  const policies = template.findResources("AWS::IAM::Policy");
  return Object.values(policies)
    .filter((policy) => {
      const roles = policy.Properties.PolicyDocument ? policy.Properties.Roles : undefined;
      return JSON.stringify(roles ?? []).includes(roleLogicalId);
    })
    .flatMap(
      (policy) =>
        policy.Properties.PolicyDocument.Statement as Array<{
          Effect: string;
          Action: string | string[];
          Resource: unknown;
        }>,
    );
}

function actionsGranted(statements: ReturnType<typeof policyStatementsForRole>): Set<string> {
  return new Set(
    statements
      .filter((s) => s.Effect === "Allow")
      .flatMap((s) => (Array.isArray(s.Action) ? s.Action : [s.Action])),
  );
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
      // ServerErrorRateAlarm, HighLatencyAlarm, the WAF BlockedRequestsAlarm
      // (issue #151), and (ADR 0008, PR #259 review) WorkerErrorRateAlarm/
      // WorkerHighDurationAlarm.
      expect(Object.keys(alarms).length).toBeGreaterThanOrEqual(5);
      for (const alarm of Object.values(alarms)) {
        expect(alarm.Properties.TreatMissingData).toBe("notBreaching");
      }
    });
  });

  describe("Async translate-jobs pattern (ADR 0008, issue #256)", () => {
    it("creates an on-demand TranslateJobsTable keyed by job_id, with a ttl attribute", () => {
      const { template } = synthApiStack();

      template.hasResourceProperties("AWS::DynamoDB::Table", {
        TableName: "traductor-kaqchikel-translate-jobs-test",
        BillingMode: "PAY_PER_REQUEST",
        KeySchema: [{ AttributeName: "job_id", KeyType: "HASH" }],
        TimeToLiveSpecification: { AttributeName: "ttl", Enabled: true },
      });
    });

    it("creates exactly three Lambda functions, one Docker image, distinguished by handler", () => {
      const { template } = synthApiStack();

      const functions = template.findResources("AWS::Lambda::Function");
      expect(Object.keys(functions).length).toBe(3);
      for (const fn of Object.values(functions)) {
        expect(fn.Properties.PackageType).toBe("Image");
      }

      findFunctionByHandler(template, "app.lambda_handler.handler");
      findFunctionByHandler(template, "app.worker_handler.handler");
      findFunctionByHandler(template, "app.failure_handler.handler");
    });

    it("keeps the Request Lambda at 29s but gives the Worker Lambda a longer ceiling", () => {
      const { template } = synthApiStack();

      const requestFn = findFunctionByHandler(template, "app.lambda_handler.handler");
      expect(requestFn.resource.Properties.Timeout).toBe(29);

      const workerFn = findFunctionByHandler(template, "app.worker_handler.handler");
      // Comfortably covers a real cold start with margin, still far under
      // Lambda's 15-minute hard ceiling (ADR 0008).
      expect(workerFn.resource.Properties.Timeout).toBe(90);
    });

    it("passes the SageMaker endpoint name only to the Worker Lambda, not the Request Lambda", () => {
      const { template } = synthApiStack();

      // ADR 0008: the Request Lambda's work is trivial (DynamoDB + one
      // async invoke) and never calls SageMaker at all -- only the Worker
      // Lambda does.
      const workerFn = findFunctionByHandler(template, "app.worker_handler.handler");
      const workerVars = (
        workerFn.resource.Properties.Environment as { Variables: Record<string, unknown> }
      ).Variables;
      expect(workerVars.SAGEMAKER_ENDPOINT_NAME).toBe("traductor-kaqchikel-translate-test");

      const requestFn = findFunctionByHandler(template, "app.lambda_handler.handler");
      const requestVars = (
        requestFn.resource.Properties.Environment as { Variables: Record<string, unknown> }
      ).Variables;
      expect(requestVars.SAGEMAKER_ENDPOINT_NAME).toBeUndefined();
    });

    it("passes the TranslateJobsTable name to the Request and Worker Lambdas", () => {
      const { template } = synthApiStack();

      for (const handler of ["app.lambda_handler.handler", "app.worker_handler.handler"]) {
        const fn = findFunctionByHandler(template, handler);
        const vars = (fn.resource.Properties.Environment as { Variables: Record<string, unknown> })
          .Variables;
        expect(vars.TRANSLATE_JOBS_TABLE_NAME).toBeDefined();
      }
    });

    it("passes the Worker Lambda's function name to the Request Lambda", () => {
      const { template } = synthApiStack();

      const requestFn = findFunctionByHandler(template, "app.lambda_handler.handler");
      const vars = (
        requestFn.resource.Properties.Environment as { Variables: Record<string, unknown> }
      ).Variables;
      expect(vars.WORKER_FUNCTION_NAME).toBeDefined();
    });

    it("configures the Worker Lambda's async invoke with retryAttempts: 0 and an onFailure destination", () => {
      const { template } = synthApiStack();

      const workerFn = findFunctionByHandler(template, "app.worker_handler.handler");
      const failureFn = findFunctionByHandler(template, "app.failure_handler.handler");

      const eventInvokeConfigs = template.findResources("AWS::Lambda::EventInvokeConfig");
      // The EventInvokeConfig whose own FunctionName resolves to the
      // Worker's logical id -- i.e. it's configuring the Worker's async
      // invoke behavior, not some other function's.
      const configsForWorker = Object.values(eventInvokeConfigs).filter((config) =>
        JSON.stringify(config.Properties.FunctionName).includes(workerFn.logicalId),
      );
      expect(configsForWorker.length).toBe(1);

      const [config] = configsForWorker;
      expect(config.Properties.MaximumRetryAttempts).toBe(0);

      const destination = (
        config.Properties.DestinationConfig as { OnFailure: { Destination: unknown } }
      ).OnFailure.Destination;
      // The destination must resolve to the onFailure handler's own
      // function, not the Worker itself or some other resource.
      expect(JSON.stringify(destination)).toContain(failureFn.logicalId);
    });

    it("grants the Request Lambda lambda:InvokeFunction scoped to exactly the Worker Lambda", () => {
      const { template } = synthApiStack();

      const requestFn = findFunctionByHandler(template, "app.lambda_handler.handler");
      const workerFn = findFunctionByHandler(template, "app.worker_handler.handler");
      const requestRoleId = roleLogicalIdForFunction(requestFn);
      const statements = policyStatementsForRole(template, requestRoleId);

      const invokeStatements = statements.filter((statement) => {
        const actions = Array.isArray(statement.Action) ? statement.Action : [statement.Action];
        return statement.Effect === "Allow" && actions.includes("lambda:InvokeFunction");
      });

      expect(invokeStatements.length).toBe(1);
      const resources = Array.isArray(invokeStatements[0].Resource)
        ? invokeStatements[0].Resource
        : [invokeStatements[0].Resource];
      // `Function.grantInvoke` grants both the function's own ARN and its
      // `:*` alias/version wildcard -- never a bare "*" resource, and both
      // must resolve to the Worker Lambda specifically, not any other
      // function.
      expect(resources.length).toBeGreaterThan(0);
      for (const resource of resources) {
        expect(JSON.stringify(resource)).not.toBe('"*"');
        expect(JSON.stringify(resource)).toContain(workerFn.logicalId);
      }
    });

    it("grants each Lambda exactly the DynamoDB actions its own code calls, nothing broader", () => {
      // Precise per-role grants (not a blanket grantReadWriteData, which
      // would hand each Lambda actions it never calls -- e.g. the
      // internet-facing Request Lambda holding UpdateItem/DeleteItem it
      // never uses would mean IAM no longer enforces the same
      // "pending vs. terminal" protection app.jobs's ConditionExpression
      // guard enforces at the application layer; a future vulnerability in
      // that Lambda's dependency tree could bypass the guard via IAM
      // instead of DynamoDB rejecting the call outright).
      const { template } = synthApiStack();

      const tables = template.findResources("AWS::DynamoDB::Table");
      expect(Object.keys(tables).length).toBe(1);

      const expectedActionsByHandler: Record<string, string[]> = {
        // apiFunction (Request Lambda): app.jobs.create_job (put_item) and
        // get_job (get_item) -- never updates or deletes an item itself.
        "app.lambda_handler.handler": ["dynamodb:GetItem", "dynamodb:PutItem"],
        // workerFunction (Worker Lambda): app.jobs.get_job_input (get_item)
        // and mark_succeeded/mark_failed (update_item) -- never creates or
        // deletes an item itself.
        "app.worker_handler.handler": ["dynamodb:GetItem", "dynamodb:UpdateItem"],
        // failureFunction (onFailure handler): app.jobs.mark_failed
        // (update_item) and, for logging only, app.jobs.get_job (get_item).
        "app.failure_handler.handler": ["dynamodb:GetItem", "dynamodb:UpdateItem"],
      };

      for (const [handler, expectedActions] of Object.entries(expectedActionsByHandler)) {
        const fn = findFunctionByHandler(template, handler);
        const roleId = roleLogicalIdForFunction(fn);
        const statements = policyStatementsForRole(template, roleId);
        const dynamoStatements = statements.filter((statement) => {
          const actions = Array.isArray(statement.Action) ? statement.Action : [statement.Action];
          return (
            statement.Effect === "Allow" && actions.some((action) => action.startsWith("dynamodb:"))
          );
        });

        const grantedDynamoActions = new Set(
          dynamoStatements.flatMap((statement) =>
            (Array.isArray(statement.Action) ? statement.Action : [statement.Action]).filter(
              (action) => action.startsWith("dynamodb:"),
            ),
          ),
        );
        expect(grantedDynamoActions).toEqual(new Set(expectedActions));

        for (const statement of dynamoStatements) {
          const resources = Array.isArray(statement.Resource) ? statement.Resource : [statement.Resource];
          for (const resource of resources) {
            expect(JSON.stringify(resource)).not.toBe('"*"');
          }
        }
      }
    });

    it("grants the Worker Lambda (not the Request Lambda) the sagemaker:InvokeEndpoint permission", () => {
      const { template } = synthApiStack();

      const workerFn = findFunctionByHandler(template, "app.worker_handler.handler");
      const workerRoleId = roleLogicalIdForFunction(workerFn);
      const workerGranted = actionsGranted(policyStatementsForRole(template, workerRoleId));
      expect(workerGranted.has("sagemaker:InvokeEndpoint")).toBe(true);

      const requestFn = findFunctionByHandler(template, "app.lambda_handler.handler");
      const requestRoleId = roleLogicalIdForFunction(requestFn);
      const requestGranted = actionsGranted(policyStatementsForRole(template, requestRoleId));
      expect(requestGranted.has("sagemaker:InvokeEndpoint")).toBe(false);
    });

    describe("Worker Lambda observability (PR #259 review)", () => {
      // ApiStack's pre-existing ServerErrorRateAlarm/HighLatencyAlarm watch
      // the REST API's own metrics -- but POST /v1/translate-jobs never
      // calls SageMaker and returns fast, and GET /v1/translate-jobs/{id}
      // always responds 200 even when the job itself failed (status_code/
      // error travel inside the JSON body, not the outer HTTP status). So
      // neither alarm can see a SageMaker-side failure/slowness anymore --
      // that only happens inside workerFunction, invisible to API Gateway.
      // These alarms close that gap directly against workerFunction's own
      // CloudWatch metrics.
      it("alarms on an elevated error rate from the Worker Lambda's own metrics", () => {
        const { template } = synthApiStack();

        const workerFn = findFunctionByHandler(template, "app.worker_handler.handler");
        template.hasResourceProperties(
          "AWS::CloudWatch::Alarm",
          Match.objectLike({
            MetricName: "Errors",
            Namespace: "AWS/Lambda",
            Dimensions: Match.arrayWith([
              Match.objectLike({
                Name: "FunctionName",
                Value: { Ref: workerFn.logicalId },
              }),
            ]),
            ComparisonOperator: Match.stringLikeRegexp("GreaterThan"),
          }),
        );
      });

      it("alarms on elevated duration from the Worker Lambda's own metrics", () => {
        const { template } = synthApiStack();

        const workerFn = findFunctionByHandler(template, "app.worker_handler.handler");
        template.hasResourceProperties(
          "AWS::CloudWatch::Alarm",
          Match.objectLike({
            MetricName: "Duration",
            Namespace: "AWS/Lambda",
            Dimensions: Match.arrayWith([
              Match.objectLike({
                Name: "FunctionName",
                Value: { Ref: workerFn.logicalId },
              }),
            ]),
            ComparisonOperator: Match.stringLikeRegexp("GreaterThan"),
          }),
        );
      });

      it("does not treat missing data as breaching for the new Worker alarms either", () => {
        const { template } = synthApiStack();

        const workerFn = findFunctionByHandler(template, "app.worker_handler.handler");
        const alarms = template.findResources("AWS::CloudWatch::Alarm");
        const workerAlarms = Object.values(alarms).filter((alarm) => {
          const dimensions = (alarm.Properties.Dimensions ?? []) as Array<{
            Name: string;
            Value: unknown;
          }>;
          return dimensions.some(
            (dimension) =>
              dimension.Name === "FunctionName" &&
              JSON.stringify(dimension.Value).includes(workerFn.logicalId),
          );
        });

        expect(workerAlarms.length).toBeGreaterThanOrEqual(2);
        for (const alarm of workerAlarms) {
          expect(alarm.Properties.TreatMissingData).toBe("notBreaching");
        }
      });
    });
  });
});
