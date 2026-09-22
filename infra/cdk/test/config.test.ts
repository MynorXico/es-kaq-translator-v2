import { describe, expect, it } from "vitest";
import { DOMAIN_NAME, webHostName } from "../lib/config";

describe("config", () => {
  it("exposes a single domain name constant", () => {
    expect(DOMAIN_NAME).toBe("traductorkaqchikel.com");
  });

  it("maps the prod environment to the flat 'app' subdomain", () => {
    expect(webHostName("prod")).toBe("app.traductorkaqchikel.com");
  });

  it("maps lower environments to a flat 'app-<env>' subdomain, not a nested one", () => {
    expect(webHostName("dev")).toBe("app-dev.traductorkaqchikel.com");
    expect(webHostName("qa")).toBe("app-qa.traductorkaqchikel.com");
  });
});
