/**
 * Single source of truth for the project's public domain name (see
 * docs/runbooks/domain-and-dns.md and .claude/agents/devops.md). Not a
 * secret -- it's the public site's own URL -- but kept in exactly one
 * place so stacks never hardcode the literal string.
 */
export const DOMAIN_NAME = "traductorkaqchikel.com";

/**
 * Maps an environment name to its flat, single-level web subdomain, per
 * the domain runbook's subdomain table: `app.` for prod, `app-<env>.` for
 * every lower environment (never a nested `app.<env>.` form, so a single
 * `*.traductorkaqchikel.com` ACM wildcard certificate covers all of them).
 */
export function webHostName(environmentName: string): string {
  const label = environmentName === "prod" ? "app" : `app-${environmentName}`;
  return `${label}.${DOMAIN_NAME}`;
}

/**
 * The three promotion-target environment names, in deploy order (ADR
 * 0004). Single source of truth for `lib/pipeline-stack.ts`/`bin/app.ts`
 * so their per-environment `apps/web` build loops (issue #263) and
 * `PipelineConfig.webApiBaseUrls`/`webSiteContentPaths` lookups can never
 * drift out of sync with each other.
 */
export const WEB_ENVIRONMENTS = ["dev", "qa", "prod"] as const;
export type WebEnvironment = (typeof WEB_ENVIRONMENTS)[number];

/**
 * Per-environment `apps/web` Vite build output directory name, relative to
 * `apps/web` (e.g. `dist-dev`). Each environment needs its own build
 * output directory because Vite inlines `VITE_API_BASE_URL` into the built
 * JS bundle at build time (issue #263) -- a single shared `dist/` can't
 * hold three different environments' bundles at once, which is what let
 * qa silently call dev's API for as long as it did (see the issue for the
 * full story). `lib/pipeline-stack.ts`'s synth step runs one
 * `vite build --outDir <this>` per environment (all inside its one
 * CodeBuildStep, before `cdk synth` runs), and `bin/app.ts` resolves the
 * matching absolute path for each environment's `WebStack`.
 */
export function webBuildOutputDirName(environmentName: string): string {
  return `dist-${environmentName}`;
}
