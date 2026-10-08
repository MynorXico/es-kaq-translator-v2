import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Vitest 4 dropped "**/dist/**" from its default exclude list (it
    // used to be included alongside "**/node_modules/**"). `tsc` (our
    // `build` script) compiles test/*.ts into dist/test/*.js as
    // CommonJS, and Vitest's default include glob would otherwise pick
    // those compiled duplicates up too -- which fail outright since
    // `vitest` can't be `require()`-d from CommonJS. Restore the old
    // default explicitly rather than relying on it.
    exclude: ["node_modules/**", "dist/**"],
    // This suite synthesizes full CDK apps/stacks (and one test shells out
    // to a real `cdk synth` CLI invocation) -- genuinely slow, and prone to
    // exceeding Vitest's 5s default under CI/CPU contention now that
    // Vitest 4 enforces testTimeout precisely even for synchronous,
    // blocking test bodies like these (previously it could slip past on a
    // loaded machine without failing).
    testTimeout: 30_000,
  },
});
