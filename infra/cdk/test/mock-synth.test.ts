import { execFileSync } from "node:child_process";
import path from "node:path";
import { describe, expect, it } from "vitest";

// CI has no AWS credentials by design (see .claude/agents/devops.md and
// .github/workflows/ci.yml) — `cdk synth --context useMockAccounts=true`
// (via bin/app.ts) is the safety net that keeps PR checks green without
// them. This test proves that path works end-to-end with every
// AWS-credential-shaped env var stripped, not just typechecking bin/app.ts.
describe("cdk synth --context useMockAccounts=true", () => {
  it("succeeds with no AWS credentials present", () => {
    const cdkRoot = path.resolve(__dirname, "..");

    const sanitizedEnv = { ...process.env };
    for (const key of Object.keys(sanitizedEnv)) {
      if (/^AWS_/.test(key)) {
        delete sanitizedEnv[key];
      }
    }

    const result = execFileSync(
      "npx",
      ["cdk", "synth", "--context", "useMockAccounts=true", "--quiet"],
      {
        cwd: cdkRoot,
        env: sanitizedEnv,
        encoding: "utf-8",
        stdio: "pipe",
      },
    );

    expect(result).toBeDefined();
  });
});
