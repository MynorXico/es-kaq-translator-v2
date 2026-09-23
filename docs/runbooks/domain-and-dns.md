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
   Detection is diff-based: `bin/app.ts` reads the previously-validated
   domain for that stack/environment from an SSM parameter (e.g.
   `/traductor-kaqchikel/domains/{environmentName}-web-cert-domain`,
   alongside the account-ID/connection-ARN parameters from ADR 0004) and
   the stack fails synth if the requested `domainName` differs from that
   recorded value and the flag is absent — a hard technical guard, not
   just a reminder to self, specifically so an unattended Dev/Qa pipeline
   run can't wedge itself on `CREATE_IN_PROGRESS` waiting for a
   validation record nobody's watching for. See [ADR 0007](../adr/0007-cross-account-domain-dns-validation.md)'s
   Decision section for the full rationale.
2. Create the certificate with an isolated, watched deploy. Find the
   exact CDK stack path if unsure (`npx cdk list --context
   newCertificateAck=true` from `infra/cdk`) — it's the slash-separated
   construct path, e.g. `TraductorKaqchikel-Pipeline/Dev/Web`, **not**
   the CloudFormation stack name (`Dev-Web`) used elsewhere in this doc.
   The two look similar but only the construct path works as a `cdk
   deploy`/`cdk diff` argument:
   ```sh
   cd infra/cdk
   AWS_PROFILE=translator-tooling npx cdk diff "TraductorKaqchikel-Pipeline/Dev/Web" \
     --context newCertificateAck=true
   # review the diff, then:
   AWS_PROFILE=translator-tooling npx cdk deploy "TraductorKaqchikel-Pipeline/Dev/Web" \
     --context newCertificateAck=true --require-approval never
   ```
   Use `translator-tooling` credentials for the whole command, not
   `translator-<env>` — CDK resolves the target account's stacks through
   the existing cross-account bootstrap trust (ADR 0004), the same way
   the pipeline itself deploys cross-account, so there's no need to
   switch profiles mid-command. The stack creates its certificate with
   `acm.CertificateValidation.fromDns()` and **no** `hostedZone` argument
   — CDK/CloudFormation will not attempt any cross-account write, and the
   certificate resource sits `CREATE_IN_PROGRESS` (blocking this `cdk
   deploy`) until step 4 below completes.

   **Two gotchas, discovered doing this for real across dev/qa/prod:**
   - `cdk synth` builds the *entire* pipeline app (Dev+Qa+Prod stacks) in
     one pass, and this guard fires once per environment, independently.
     The pipeline's automated `Build` stage will fail three separate
     times in a row as each environment's certificate gets validated in
     turn (dev, then qa, then prod, each naming its own domain in the
     error) — that's expected sequencing, not three different bugs or a
     retry that silently failed to take effect.
   - Because the pipeline's automatic Dev/Qa deploy stages can't run
     while `Build` is failing, doing each environment's one-off deploy
     from a different point in `main`'s history leaves that environment
     serving a stale site bundle relative to the others until the
     pipeline finally catches up (this happened in practice: `dev`
     briefly served an older build than `qa` for exactly this reason).
     Rebuild `apps/web` (`pnpm --filter web build`) from the same latest
     `main` commit immediately before each environment's one-off deploy,
     or do all three environments back-to-back from one checkout, to
     avoid this.
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
   record with `infra/scripts/upsert-domain-record.sh` against the
   `translator-tooling` profile, rather than console clicking. Get the
   CloudFront distribution's domain from `WebStack`'s own
   `DistributionDomainName` output:

   ```sh
   aws cloudformation describe-stacks --profile translator-dev \
     --region us-east-1 --stack-name Dev-Web \
     --query "Stacks[0].Outputs[?OutputKey=='DistributionDomainName'].OutputValue" \
     --output text
   ```

   (`Dev-Web` is the actual CloudFormation stack name here — confirmed
   against a real deploy; this is a plain `aws cloudformation` call, so it
   takes the CloudFormation-side name, not the CDK construct path from
   step 2 above.)

   Then:

   ```sh
   PROFILE=translator-tooling \
   RECORD_NAME=app-dev.traductorkaqchikel.com \
   TARGET_DNS_NAME=<distribution-domain-from-above> \
   ./infra/scripts/upsert-domain-record.sh
   ```

   The script is idempotent (Route 53 `UPSERT`) — safe to re-run any time
   the target changes (e.g. the distribution was replaced). It only
   touches the final alias record, never the validation CNAME from step 4.
   The future `ApiStack` custom domain (#96) reuses the same script with
   an explicit `TARGET_HOSTED_ZONE_ID` (API Gateway's regional alias-target
   zone ID, which — unlike CloudFront's single fixed one — varies by
   region); see the script's own header comment for that invocation.
6. **Update the SSM parameter from step 1** to the newly-validated
   domain name, e.g.:

   ```sh
   aws ssm put-parameter --profile translator-tooling --region us-east-1 \
     --name "/traductor-kaqchikel/domains/dev-web-cert-domain" --type String \
     --value "app-dev.traductorkaqchikel.com" --overwrite
   ```

   This is what lets the *next* ordinary deploy of the same
   stack/environment skip the `newCertificateAck` flag — until this step
   runs, the stack still thinks no certificate has been validated for
   this domain yet.

## Current status: web custom domains

As of issue #99 (mechanism) and its dev/qa/prod rollout, `WebStack`
creates its own per-environment ACM certificate
(`acm.CertificateValidation.fromDns()`, no `hostedZone` argument) and
attaches it + the real `app[-<env>].traductorkaqchikel.com` domain name
to its CloudFront distribution (see the "Cross-account ACM validation and
record creation" section above and ADR 0007). `bin/app.ts` enforces the
`newCertificateAck` guard at synth time, and
`infra/scripts/upsert-domain-record.sh` is checked in for the final alias
record.

**All three environments are live** on their real custom domains, each
validated via the process above:

| Environment | Domain | Status |
|---|---|---|
| dev | `app-dev.traductorkaqchikel.com` | Live, validated |
| qa | `app-qa.traductorkaqchikel.com` | Live, validated |
| prod | `app.traductorkaqchikel.com` | Live, validated |

Each environment's `/traductor-kaqchikel/domains/{environmentName}-web-cert-domain`
SSM parameter is set, so ordinary pipeline runs no longer need
`newCertificateAck` for the web certificate in any environment — the flag
is only needed again if a certificate is ever recreated (a domain rename,
a rebootstrapped account, etc. — see ADR 0007's Consequences section).

Note that `prod`'s `WebStack` didn't exist at all before this rollout
(unlike dev/qa, which already had their S3 bucket/CloudFront distribution
from issue #84 and only needed the domain added) — its one-off deploy
created the entire stack from scratch, including a new S3 bucket,
CloudFront distribution, Origin Access Control, and deployment IAM roles,
in addition to the certificate. This also happened outside the pipeline's
normal `PromoteToProd` manual-approval gate, the same way dev/qa's
one-off deploys bypass their own auto-deploy stage — by design, per
ADR 0007, since the pipeline can't safely run a deploy that blocks on
human DNS validation unattended.

The future API Gateway custom domain (`ApiStack`, issue #96) is
deliberately **not** part of this implementation — #96's `ApiStack` needs
to settle further first — and is deferred to its own explicit follow-up
issue, reusing this exact same pattern (same guard shape, same runbook
section, same `infra/scripts/upsert-domain-record.sh`, only the final
record's target type/value differing, per ADR 0007's Decision section).
