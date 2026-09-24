import { beforeEach, describe, expect, it, vi } from "vitest";

// Mock the SSM client boundary (same spirit as apps/web/src/api.test.ts
// stubbing `fetch`): bin/app.ts's realConfig()/fetchSsmParameter() call
// `client.send(new GetParameterCommand(...))`, so both the client and the
// command constructors are replaced with fakes controlled entirely by
// `sendMock` below -- no real AWS credentials or network calls involved.
// `vi.mock` calls are hoisted above the imports below by Vitest, so
// `../bin/app`'s own `import { SSMClient } from "@aws-sdk/client-ssm"`
// picks up this mock too.
const sendMock = vi.fn();

vi.mock("@aws-sdk/client-ssm", () => ({
  SSMClient: vi.fn().mockImplementation(() => ({ send: sendMock })),
  GetParameterCommand: vi.fn().mockImplementation((input: { Name: string }) => ({ input })),
}));

import { App } from "aws-cdk-lib";
import { GetParameterCommand, SSMClient } from "@aws-sdk/client-ssm";
import {
  fetchOptionalSsmParameter,
  fetchSsmParameter,
  mockConfig,
  realConfig,
  resolveConfig,
  resolveNewCertificateAck,
} from "../bin/app";
import { webHostName } from "../lib/config";

class FakeParameterNotFound extends Error {
  constructor() {
    super("ParameterNotFound");
    this.name = "ParameterNotFound";
  }
}

function parameterResponse(value: string) {
  return { Parameter: { Value: value } };
}

function fakeClient(): SSMClient {
  return new SSMClient({ region: "us-east-1" });
}

beforeEach(() => {
  sendMock.mockReset();
});

describe("fetchSsmParameter / fetchOptionalSsmParameter", () => {
  it("fetchSsmParameter returns the parameter's value", async () => {
    sendMock.mockResolvedValueOnce(parameterResponse("222222222222"));

    await expect(fetchSsmParameter(fakeClient(), "/traductor-kaqchikel/accounts/dev")).resolves.toBe(
      "222222222222",
    );
    expect(GetParameterCommand).toHaveBeenCalledWith({ Name: "/traductor-kaqchikel/accounts/dev" });
  });

  it("fetchSsmParameter throws if the parameter has no value", async () => {
    sendMock.mockResolvedValueOnce({ Parameter: {} });

    await expect(fetchSsmParameter(fakeClient(), "/traductor-kaqchikel/accounts/dev")).rejects.toThrow(
      /exists but has no value/,
    );
  });

  it("fetchOptionalSsmParameter returns the value when the parameter exists", async () => {
    sendMock.mockResolvedValueOnce(parameterResponse("app-dev.traductorkaqchikel.com"));

    await expect(
      fetchOptionalSsmParameter(fakeClient(), "/traductor-kaqchikel/domains/dev-web-cert-domain"),
    ).resolves.toBe("app-dev.traductorkaqchikel.com");
  });

  it("fetchOptionalSsmParameter returns undefined (not a thrown error) when the parameter doesn't exist yet", async () => {
    sendMock.mockRejectedValueOnce(new FakeParameterNotFound());

    await expect(
      fetchOptionalSsmParameter(fakeClient(), "/traductor-kaqchikel/domains/prod-web-cert-domain"),
    ).resolves.toBeUndefined();
  });

  it("fetchOptionalSsmParameter re-throws any other error (e.g. access denied), not just ParameterNotFound", async () => {
    sendMock.mockRejectedValueOnce(new Error("AccessDeniedException"));

    await expect(
      fetchOptionalSsmParameter(fakeClient(), "/traductor-kaqchikel/domains/prod-web-cert-domain"),
    ).rejects.toThrow(/AccessDeniedException/);
  });
});

describe("realConfig", () => {
  it("fetches account IDs, connection ARN, and per-environment previously-validated web domains from SSM", async () => {
    sendMock.mockImplementation((command: { input: { Name: string } }) => {
      const values: Record<string, string> = {
        "/traductor-kaqchikel/accounts/tooling": "111111111111",
        "/traductor-kaqchikel/accounts/dev": "222222222222",
        "/traductor-kaqchikel/accounts/qa": "333333333333",
        "/traductor-kaqchikel/accounts/prod": "444444444444",
        "/traductor-kaqchikel/github-connection-arn":
          "arn:aws:codeconnections:us-east-1:111111111111:connection/real-connection-id",
        "/traductor-kaqchikel/domains/dev-web-cert-domain": "app-dev.traductorkaqchikel.com",
        "/traductor-kaqchikel/domains/qa-web-cert-domain": "app-qa.traductorkaqchikel.com",
        "/traductor-kaqchikel/api-urls/dev": "https://ovc1orcvql.execute-api.us-east-1.amazonaws.com",
      };
      const name = command.input.Name;
      if (name in values) {
        return Promise.resolve(parameterResponse(values[name]));
      }
      // prod-web-cert-domain, api-urls/qa, and api-urls/prod are
      // deliberately absent -- simulates a never-yet-validated environment
      // (first-ever certificate) and environments without a real
      // `ApiStack` deployed yet.
      return Promise.reject(new FakeParameterNotFound());
    });

    const config = await realConfig();

    expect(config.toolingAccount).toBe("111111111111");
    expect(config.devAccount).toBe("222222222222");
    expect(config.qaAccount).toBe("333333333333");
    expect(config.prodAccount).toBe("444444444444");
    expect(config.connectionArn).toBe(
      "arn:aws:codeconnections:us-east-1:111111111111:connection/real-connection-id",
    );
    expect(config.previouslyValidatedWebCertDomains).toEqual({
      dev: "app-dev.traductorkaqchikel.com",
      qa: "app-qa.traductorkaqchikel.com",
      prod: undefined,
    });
    expect(config.webApiBaseUrls).toEqual({
      dev: "https://ovc1orcvql.execute-api.us-east-1.amazonaws.com",
      qa: undefined,
      prod: undefined,
    });
  });
});

describe("resolveNewCertificateAck", () => {
  it("is false when the newCertificateAck context isn't set", () => {
    const app = new App();
    expect(resolveNewCertificateAck(app)).toBe(false);
  });

  it("is true only when the context is exactly the string 'true'", () => {
    expect(resolveNewCertificateAck(new App({ context: { newCertificateAck: "true" } }))).toBe(true);
    expect(resolveNewCertificateAck(new App({ context: { newCertificateAck: "yes" } }))).toBe(false);
    expect(resolveNewCertificateAck(new App({ context: { newCertificateAck: true } }))).toBe(false);
  });
});

describe("resolveConfig", () => {
  it("uses mockConfig() (with matching previously-validated domains) when useMockAccounts=true, and never calls SSM", async () => {
    const app = new App({ context: { useMockAccounts: "true" } });

    const config = await resolveConfig(app);

    expect(sendMock).not.toHaveBeenCalled();
    expect(config.toolingAccount).toBe(mockConfig().toolingAccount);
    expect(config.previouslyValidatedWebCertDomains).toEqual({
      dev: webHostName("dev"),
      qa: webHostName("qa"),
      prod: webHostName("prod"),
    });
    expect(config.webApiBaseUrls?.dev).toMatch(/^https?:\/\//);
    expect(config.newCertificateAck).toBe(false);
  });

  it("threads newCertificateAck=true from context through the mock-accounts path", async () => {
    const app = new App({ context: { useMockAccounts: "true", newCertificateAck: "true" } });

    const config = await resolveConfig(app);

    expect(config.newCertificateAck).toBe(true);
  });

  it("calls realConfig() (via SSM) when useMockAccounts is not set", async () => {
    sendMock.mockResolvedValue(parameterResponse("fake-value"));
    const app = new App();

    const config = await resolveConfig(app);

    expect(sendMock).toHaveBeenCalled();
    expect(config.toolingAccount).toBe("fake-value");
    expect(config.newCertificateAck).toBe(false);
  });
});
