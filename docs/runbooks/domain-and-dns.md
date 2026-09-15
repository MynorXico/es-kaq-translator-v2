# Runbook: domain and DNS

## Current state

`traductorkaqchikel.com` is registered through **Route 53 Domains**
(Amazon Registrar) under the `translator-tooling` AWS account, with
auto-renew and WHOIS privacy enabled on all contacts.

Registering a domain directly through Route 53 Domains auto-creates a
public hosted zone for it and auto-delegates the domain's nameservers to
that zone — this was verified: the domain's registered nameservers match
the hosted zone's `NS` record set exactly. **No manual DNS delegation
step was needed.**

Don't hardcode the hosted zone ID anywhere in this repo. CDK should
resolve it dynamically instead:

```ts
import * as route53 from "aws-cdk-lib/aws-route53";

const zone = route53.HostedZone.fromLookup(this, "Zone", {
  domainName: "traductorkaqchikel.com",
});
```

(`fromLookup` needs AWS credentials at synth time, which CDK already
requires anyway.) If you need the zone ID for a one-off manual check, get
it live: `aws route53 list-hosted-zones-by-name --dns-name
traductorkaqchikel.com --profile translator-tooling`.

## Subdomain structure

Flat, single-level, hyphenated subdomains — not nested (e.g.
`app-dev.traductorkaqchikel.com`, not `app.dev.traductorkaqchikel.com`).

| Subdomain | Points to | Environment |
|---|---|---|
| `app.traductorkaqchikel.com` | Web SPA (CloudFront) | prod |
| `api.traductorkaqchikel.com` | Translation API (API Gateway) | prod |
| `app-dev.traductorkaqchikel.com` | Web SPA | dev |
| `api-dev.traductorkaqchikel.com` | Translation API | dev |
| `app-qa.traductorkaqchikel.com` | Web SPA | qa |
| `api-qa.traductorkaqchikel.com` | Translation API | qa |

**Why flat instead of nested**: an ACM wildcard certificate for
`*.traductorkaqchikel.com` covers exactly one subdomain level — it
matches `app.traductorkaqchikel.com` and `app-dev.traductorkaqchikel.com`
but does **not** match a two-level name like
`app.dev.traductorkaqchikel.com`. Keeping everything one level deep means
a single wildcard cert (issued once, in whichever account/region needs
it — us-east-1 for CloudFront, per-region for regional API Gateway
custom domains) covers the whole scheme, instead of needing per-environment
wildcard certs.

All of these are plain DNS records (CNAME/ALIAS) in the one hosted zone
in `translator-tooling` — Route 53 doesn't care which AWS account the
target resource (CloudFront distribution, API Gateway) lives in, so no
cross-account zone delegation is needed for this. Delegating subdomains
to per-account hosted zones would only be worth it if dev/qa/prod needed
to manage their own DNS independently, which isn't a requirement here.

## What's not done yet

No actual records exist for `app.`/`api.` (or their `-dev`/`-qa`
variants) yet, since nothing is deployed to point them at — no
CloudFront distribution, no API Gateway. Creating those records is part
of the CDK work for the web/API stacks (tracked separately), not this
ticket. This runbook documents the plan so that work has a target to
build against.
