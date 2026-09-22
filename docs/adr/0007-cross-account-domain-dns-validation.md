# ADR 0007: Cross-account custom-domain DNS and ACM validation

- Status: Proposed
- Date: 2026-09-22

## Context

Per ADR 0001, each environment (`translator-dev`/`-qa`/`-prod`) is a
separate AWS account, while the public `traductorkaqchikel.com` hosted
zone lives in a fourth account, `translator-tooling`
(`docs/runbooks/domain-and-dns.md`). `WebStack` (issue #84, S3 + CloudFront
+ `BucketDeployment`) and the future `ApiStack` custom domain (issue #96,
Lambda container behind API Gateway) both need a real
`app[-<env>].traductorkaqchikel.com` / `api[-<env>].traductorkaqchikel.com`
DNS name, currently deferred in favor of the AWS-generated
`*.cloudfront.net` / `execute-api.amazonaws.com` interim domains.

Wiring up the real domain requires two things per environment/stack, both
of which cross the account boundary:

1. **An ACM certificate in the same account as the resource** (CloudFront
   requires its certificate to live in the same account as the
   distribution -- a hard AWS constraint, not a CDK limitation; API
   Gateway custom domains have the same same-account requirement). This
   certificate therefore has to be created per-environment, in
   `translator-dev`/`-qa`/`-prod`, not centrally in `translator-tooling`.
2. **DNS validation of that certificate, and the final alias/CNAME
   record**, both of which are writes into the hosted zone that lives in
   `translator-tooling` -- a different account than the certificate/
   resource.

Route 53 hosted zones have no resource-based/bucket-policy-style
cross-account grant (unlike S3, KMS, or SNS): the only mechanism for a
different account to write into someone else's zone is an assumed IAM
role. CDK's built-in `HostedZone.grantDelegation()` /
`CrossAccountZoneDelegationRecord` don't help here -- they're scoped to
NS-only delegation records for handing off a whole subzone, not arbitrary
CNAME/A records within an existing zone (verified against `aws-cdk-lib`
2.268's `aws-route53` source).

`docs/runbooks/domain-and-dns.md` already ruled out delegating
per-account subzones ("only worth it if dev/qa/prod needed to manage
their own DNS independently, which isn't a requirement here") and that
reasoning still holds -- it would also break the flat, one-level
subdomain design the wildcard-cert strategy depends on. This ADR doesn't
reopen that option; it decides how a *single, flat, tooling-account-owned*
zone gets written to from the other three accounts.

Two real options were on the table (from issue #99):

1. **A new persistent cross-account IAM role** in `translator-tooling`
   (least-privilege: `route53:ChangeResourceRecordSets`/`GetChange`
   scoped to the one hosted zone's ARN), trusted by dev/qa/prod, assumed
   by a CDK `AwsCustomResource` (or a hand-written Lambda) inside each
   environment's stack to upsert the ACM validation CNAME and the final
   alias record automatically as part of `cdk deploy`.
2. **A one-time manual DNS step per certificate/environment**, mirroring
   the existing precedent for the GitHub CodeStar/CodeConnections OAuth
   handshake (`docs/runbooks/cdk-pipelines-bootstrap.md` step 5, itself
   required because AWS provides no API for that step): use
   `acm.CertificateValidation.fromDns()` *without* passing a hosted zone
   (so CDK/CloudFormation never attempts a cross-account write), and a
   human with `translator-tooling` credentials adds the validation CNAME
   (looked up via `aws acm describe-certificate`) and, later, the final
   alias record, the latter via a small checked-in script rather than
   free-hand console clicks.

## Decision

**Adopt option 2: a one-time, scripted-but-manually-triggered DNS step
per certificate, with no new persistent cross-account IAM trust.**

- Every ACM certificate this project needs (`WebStack`'s CloudFront
  cert, and the future `ApiStack` custom-domain cert) is created with
  `new acm.Certificate(this, "Cert", { domainName, validation:
  acm.CertificateValidation.fromDns() })` -- no `hostedZone` argument --
  directly inside the stack's own account (dev/qa/prod). CDK/CloudFormation
  will create the `AWS::CertificateManager::Certificate` resource and then
  wait for DNS validation, but will not attempt to write the validation
  record anywhere itself.
- A human with `translator-tooling` SSO access retrieves the pending
  validation CNAME (`aws acm describe-certificate --profile <env>
  --certificate-arn <arn>`) and creates it once in the tooling account's
  hosted zone. Exact commands go in `docs/runbooks/domain-and-dns.md`.
  Once created, ACM auto-renews using that same standing record -- this
  is genuinely a one-time action per certificate, not a recurring one.
- The final alias/A record (`app[-<env>].` -> CloudFront distribution
  domain; `api[-<env>].` -> the API Gateway custom domain's regional
  target) is also created against the `translator-tooling` profile, but
  via a small idempotent helper script checked into the repo (e.g.
  `infra/scripts/upsert-domain-record.sh`, exact name/shape left to the
  `devops` agent implementing this) that wraps `aws route53
  change-resource-record-sets`, rather than done by hand in the console
  every time. This is still a human explicitly invoking a command with
  their own already-privileged `translator-tooling` credentials, not a
  standing machine-to-machine trust -- it satisfies "scripted" without
  reintroducing persistent cross-account IAM.
- Because a stack-level `acm.Certificate` with DNS validation blocks
  (`CREATE_IN_PROGRESS`) until validated, and Dev/Qa deploy automatically
  on every pipeline run (ADR 0004), the *first* deploy of a stack version
  that introduces a new certificate for a given environment must be run
  as an isolated, human-watched `cdk deploy` against that one stack/stage
  (mirroring the one-time manual pipeline-stack deploy in
  `cdk-pipelines-bootstrap.md` step 6) rather than left to an unattended
  pipeline run -- someone needs to be available to add the validation
  CNAME promptly, or the deploy sits blocked. Once validated, ordinary
  pipeline runs that change unrelated parts of the same stack do not
  re-trigger validation.
- This mechanism is identical for `WebStack`'s CloudFront domain and the
  future API Gateway custom domain (#96) -- same pattern, same runbook
  section, only the final record's target type/value differs.

## Consequences

- No new IAM role or trust relationship is added between
  `translator-dev`/`-qa`/`-prod` and `translator-tooling` beyond what
  ADR 0004 already established for CloudFormation deploys. The blast
  radius for "who can write into the one shared production DNS zone"
  stays at "humans with `translator-tooling` SSO access," not "any of
  three environment accounts' Lambda/custom-resource execution roles" --
  notably, this avoids giving `translator-dev` a standing code path that
  could rewrite `translator-prod`'s live `app.`/`api.` records, since all
  subdomains sit in one flat zone.
- Toil is one-time per certificate/record, not recurring: six actions
  total across this project's current scope (`app.`/`api.` x
  dev/qa/prod), consistent with ACM's auto-renewal behavior. If that
  changes (e.g. frequent full environment teardown/recreation), revisit
  with option 1 -- ideally its record-name/type-scoped variant (see
  "Alternatives considered") rather than full zone-wide access -- this
  ADR is a judgment call on the current trade-off, not a permanent ban on
  automating it.
- `docs/runbooks/domain-and-dns.md` needs a new section with the exact
  `acm describe-certificate` / validation-record / alias-record commands
  and the pipeline-stall warning above. Done as part of this ADR (see
  "What's not done yet").
- Implementation follow-up (not done by this ADR): `infra/cdk/lib/web-stack.ts`
  gains a `domainName`/`certificate` prop and wires the CloudFront
  `Distribution`'s `domainNames`/`certificate`; `ApiStack` gains the
  equivalent for its API Gateway custom domain once #96's stack settles;
  a new small script lives under `infra/scripts/` for the alias-record
  upsert. This ADR intentionally decides the mechanism only -- see the
  PR description for why implementation is left as a follow-up rather
  than bundled in.
- Future stacks needing a `traductorkaqchikel.com` subdomain should
  reuse this exact pattern rather than re-deriving it.

## Alternatives considered

- **Cross-account IAM role + custom resource (option 1)**: fully
  automated end-to-end `cdk deploy`, and it would also remove the
  pipeline-stall wrinkle noted above. Rejected as the primary mechanism
  because it introduces standing write trust from three environment
  accounts (including the least-trusted, `translator-dev`) into the
  single zone that also holds `translator-prod`'s own live records -- a
  materially larger blast radius than the manual step it replaces, for a
  workload that this project's own usage pattern makes one-time-per-cert
  rather than recurring. Also more new infrastructure to build and
  maintain (a Lambda, a role, a trust policy per environment) for a
  problem the project hits roughly six times total. Revisit if that
  changes (see Consequences).
- **A finer-grained version of option 1, using Route 53's
  `route53:ChangeResourceRecordSetsNormalizedRecordNames` /
  `route53:ChangeResourceRecordSetsRecordTypes` IAM condition keys** to
  scope the cross-account role to only that environment's own record
  names (e.g. `translator-dev`'s role could only ever touch
  `app-dev.traductorkaqchikel.com`/`api-dev.traductorkaqchikel.com`,
  CNAME/A only) instead of the whole zone: this would directly address
  the blast-radius objection above -- a compromised `translator-dev`
  custom resource would be structurally unable to touch `app.`/`api.`
  (prod's records) even in the worst case. Still rejected as the primary
  mechanism, though, because the underlying cost/benefit doesn't change:
  it's still a Lambda, a role, and a trust policy (now per-environment
  condition-scoped, so slightly more of them, not less) to build and
  maintain, for a workload that's one-time per certificate and roughly
  six occurrences total across this project's life. Worth remembering as
  the natural next step if this ADR is ever revisited under "Revisit if
  toil changes" (see Consequences) -- at that point, this narrower
  variant, not full zone-wide access, should be the default starting
  point.
- **Per-account hosted zone delegation (subzones)**: already rejected in
  `domain-and-dns.md` and not reopened here -- no requirement for
  dev/qa/prod to manage DNS independently, and it would break the flat,
  one-level subdomain design the single wildcard-style cert strategy
  relies on.
- **Leave the AWS-generated domains (`*.cloudfront.net` /
  `execute-api.amazonaws.com`) as permanent, not just interim**: rejected
  -- defeats the actual product/branding requirement behind #84/#96, and
  duplicates AWS account-specific hostnames in user-facing surfaces
  (support links, documentation, marketing) that should be stable across
  infrastructure changes.
