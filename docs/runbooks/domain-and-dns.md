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
a `*.traductorkaqchikel.com` cert in a given account (see below — ACM
requires the cert to live in the same account as the CloudFront
distribution or API Gateway custom domain using it, so this project needs
one such cert **per account**, not one globally) covers every subdomain
that account owns, instead of needing separate per-subdomain certs
within that account.

All of these are plain DNS records (CNAME/ALIAS) in the one hosted zone
in `translator-tooling` — Route 53 doesn't care which AWS account the
target resource (CloudFront distribution, API Gateway) lives in, so no
cross-account *zone delegation* is needed for this. Delegating subdomains
to per-account hosted zones would only be worth it if dev/qa/prod needed
to manage their own DNS independently, which isn't a requirement here.
(There is still a real cross-account problem to solve — *writing* into
this shared zone from dev/qa/prod's own accounts — see
[ADR 0007](../adr/0007-cross-account-domain-dns-validation.md) and the
new section below.)

## Cross-account ACM validation and record creation

Issuing a per-account ACM certificate (see above) and creating the DNS
records that point at it are two different cross-account problems:
Route 53 hosted zones have no resource-based cross-account grant (unlike
S3/KMS/SNS), so a role in `translator-dev`/`-qa`/`-prod` cannot write into
the `translator-tooling` zone without an assumed IAM role — a new,
persistent piece of cross-account trust. [ADR 0007](../adr/0007-cross-account-domain-dns-validation.md)
decided **against** adding that persistent trust, in favor of a one-time,
scripted-but-manually-triggered step per certificate. Concretely, for
each environment's `WebStack` (and later `ApiStack`) certificate:

1. Any deploy that would create a *new* `acm.Certificate` for an
   environment must pass `--context newCertificateAck=true` (or the
   equivalent stack prop, wired from `app-stage.ts`) as an explicit
   acknowledgement that this deploy will block on manual DNS validation.
   The stack fails synth if it detects a new-certificate condition
   without the flag — this is a hard technical guard, not just a
   reminder to self, specifically so an unattended Dev/Qa pipeline run
   can't wedge itself on `CREATE_IN_PROGRESS` waiting for a validation
   record nobody's watching for. See [ADR 0007](../adr/0007-cross-account-domain-dns-validation.md)'s
   Decision section for the full rationale.
2. Create the certificate in-account with
   `acm.CertificateValidation.fromDns()` and **no** `hostedZone` argument
   — CDK/CloudFormation will not attempt any cross-account write, and the
   certificate resource will sit `CREATE_IN_PROGRESS` until validated.
3. Look up the pending validation CNAME:
   ```sh
   aws acm describe-certificate --profile <translator-dev|qa|prod> \
     --region us-east-1 --certificate-arn <arn> \
     --query 'Certificate.DomainValidationOptions'
   ```
4. Create that CNAME once, by hand, in the tooling zone:
   ```sh
   aws route53 change-resource-record-sets --profile translator-tooling \
     --hosted-zone-id <zone-id> --change-batch file://validation-record.json
   ```
   Because a stack-level `Certificate` resource blocks CloudFormation
   until validated, and Dev/Qa auto-deploy on every pipeline run (ADR
   0004), do this as an isolated, human-watched `cdk deploy` for the
   stack version that introduces a new certificate — not as part of an
   unattended pipeline run — so someone is available to add the record
   promptly. Once validated, ACM auto-renews using the same standing
   record; this is a one-time step per certificate, not recurring.

   **Do not delete this record once created**, even if it looks like
   unused zone clutter during a future cleanup pass — ACM needs it to
   persist indefinitely to keep auto-renewing the certificate. Deleting
   it doesn't fail loudly; the certificate just silently stops renewing
   until it expires. A cheap CloudWatch alarm on ACM's built-in
   `DaysToExpiry` metric per certificate is a nice-to-have follow-up to
   catch that failure mode early, before the certificate actually
   expires.
5. Once the distribution/custom domain exists, create the final alias
   record with the (to-be-written) helper script under `infra/scripts/`
   against the `translator-tooling` profile, rather than console
   clicking.

## What's not done yet

As of issue #84, `WebStack` deploys a real CloudFront distribution per
environment (S3 origin via OAC, SPA-routing fallback, cache invalidation
on deploy) — reachable today at its default `*.cloudfront.net` domain,
already HTTPS. The future API Gateway custom domain (#96) is deferred the
same way, at its default `execute-api.amazonaws.com` domain. No actual
`app.`/`api.` records (or their `-dev`/`-qa` variants) exist yet — they
can't be wired up as simply as this runbook originally assumed, because
each environment's CloudFront distribution/API Gateway custom domain
(and its required same-account ACM certificate — a hard AWS constraint,
not a CDK choice) lives in a *different* AWS account
(`translator-dev`/`-qa`/`-prod`) than the hosted zone
(`translator-tooling`), and Route 53 hosted zones have no resource-based/
bucket-policy-style cross-account grant (unlike S3/KMS/SNS) for writing
the DNS validation record or the final alias record cross-account.

**This is now decided**, not just tracked as open candidates: see
[ADR 0007](../adr/0007-cross-account-domain-dns-validation.md) and the
"Cross-account ACM validation and record creation" section above. Actually
wiring the per-environment certificate + CloudFront `domainNames`/API
Gateway custom domain + alias record (and writing the `infra/scripts/`
helper referenced above) is tracked as a follow-up implementation step on
issue #99, not done yet.
