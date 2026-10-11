import { describe, expect, it } from "vitest";
import { DOMAIN_NAME, WEB_ENVIRONMENTS, webBuildOutputDirName, webHostName } from "../lib/config";

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

  it("lists dev, qa, prod as the three promotion-target environments, in deploy order", () => {
    expect(WEB_ENVIRONMENTS).toEqual(["dev", "qa", "prod"]);
  });

  it("maps each environment to its own, distinct apps/web build output directory", () => {
    expect(webBuildOutputDirName("dev")).toBe("dist-dev");
    expect(webBuildOutputDirName("qa")).toBe("dist-qa");
    expect(webBuildOutputDirName("prod")).toBe("dist-prod");

    const dirNames = new Set(WEB_ENVIRONMENTS.map((env) => webBuildOutputDirName(env)));
    expect(dirNames.size).toBe(WEB_ENVIRONMENTS.length);
  });
});
