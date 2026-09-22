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
