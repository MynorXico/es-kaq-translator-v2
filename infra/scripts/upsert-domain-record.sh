#!/usr/bin/env bash
# Idempotently upserts the final alias record for one of this project's
# `app[-<env>].`/`api[-<env>].` subdomains (ADR 0007) into the shared
# `traductorkaqchikel.com` hosted zone, which lives in the `translator-tooling`
# account -- a different account than the CloudFront distribution / API
# Gateway custom domain it points at. This can't be done from CDK: Route 53
# hosted zones have no resource-based/bucket-policy-style cross-account
# grant (unlike S3/KMS/SNS), so a role in `translator-dev`/`-qa`/`-prod`
# cannot write into this zone (see ADR 0007). Instead, a human with
# `translator-tooling` SSO credentials runs this script directly.
#
# Safe to re-run: this is a Route 53 UPSERT, so re-running after a
# distribution/custom domain is replaced (e.g. a stack was torn down and
# recreated) is the normal way to repoint the record at its new target, not
# an error condition.
#
# Only creates/updates the ALIAS record itself. It does NOT touch the ACM
# certificate's DNS validation CNAME (a separate one-time step -- see
# docs/runbooks/domain-and-dns.md) and does NOT create the certificate.
#
# --- Usage: WebStack (CloudFront) target ---
#
#   Get TARGET_DNS_NAME from the stack's own `DistributionDomainName`
#   output (see `infra/cdk/lib/web-stack.ts`):
#
#     aws cloudformation describe-stacks --profile translator-dev \
#       --region us-east-1 --stack-name TraductorKaqchikel-Pipeline-Dev-Web \
#       --query "Stacks[0].Outputs[?OutputKey=='DistributionDomainName'].OutputValue" \
#       --output text
#
#   Then:
#
#     PROFILE=translator-tooling \
#     RECORD_NAME=app-dev.traductorkaqchikel.com \
#     TARGET_DNS_NAME=d111111abcdef8.cloudfront.net \
#     ./infra/scripts/upsert-domain-record.sh
#
# --- Usage: future ApiStack (API Gateway custom domain) target, once #96's
#     custom domain lands -- deferred, not implemented as of issue #99 ---
#
#   API Gateway's regional alias-target hosted zone ID is region-specific
#   (unlike CloudFront's single fixed one), so it must be supplied
#   explicitly:
#
#     PROFILE=translator-tooling \
#     RECORD_NAME=api-dev.traductorkaqchikel.com \
#     TARGET_DNS_NAME=d-abc123xyz.execute-api.us-east-1.amazonaws.com \
#     TARGET_HOSTED_ZONE_ID=Z1UJRXOUMOOFQ8 \
#     ./infra/scripts/upsert-domain-record.sh
set -euo pipefail

PROFILE="${PROFILE:-translator-tooling}"
# Must match infra/cdk/lib/config.ts's DOMAIN_NAME -- overridable only for
# testing against a throwaway zone, never for a real project subdomain.
ZONE_DOMAIN="${ZONE_DOMAIN:-traductorkaqchikel.com}"
RECORD_NAME="${RECORD_NAME:?Set RECORD_NAME, e.g. app-dev.traductorkaqchikel.com (see docs/runbooks/domain-and-dns.md)}"
TARGET_DNS_NAME="${TARGET_DNS_NAME:?Set TARGET_DNS_NAME to the CloudFront distribution domain (or API Gateway regional domain) this record should point at}"
# CloudFront's alias-target hosted zone ID is a single, fixed, publicly
# documented AWS constant -- identical for every CloudFront distribution in
# every AWS account/region, so it's not project- or account-specific and is
# safe to default here (see AWS's own Route 53 "AWS service endpoints"
# documentation). API Gateway regional custom domains use a
# region-specific zone ID instead, so that case must override this.
TARGET_HOSTED_ZONE_ID="${TARGET_HOSTED_ZONE_ID:-Z2FDTNDATAQYW2}"
RECORD_TYPE="${RECORD_TYPE:-A}"

ZONE_ID=$(aws route53 list-hosted-zones-by-name --profile "$PROFILE" \
  --dns-name "$ZONE_DOMAIN" --max-items 1 \
  --query "HostedZones[?Name=='${ZONE_DOMAIN}.'].Id" --output text)
if [ -z "$ZONE_ID" ]; then
  echo "No hosted zone found for $ZONE_DOMAIN in profile $PROFILE" >&2
  exit 1
fi
ZONE_ID="${ZONE_ID#/hostedzone/}"

CHANGE_BATCH=$(cat <<JSON
{
  "Comment": "Upserted by infra/scripts/upsert-domain-record.sh (ADR 0007)",
  "Changes": [
    {
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "${RECORD_NAME}",
        "Type": "${RECORD_TYPE}",
        "AliasTarget": {
          "HostedZoneId": "${TARGET_HOSTED_ZONE_ID}",
          "DNSName": "${TARGET_DNS_NAME}",
          "EvaluateTargetHealth": false
        }
      }
    }
  ]
}
JSON
)

echo "Upserting $RECORD_TYPE alias record $RECORD_NAME -> $TARGET_DNS_NAME in zone $ZONE_ID (profile $PROFILE)..."
CHANGE_ID=$(aws route53 change-resource-record-sets --profile "$PROFILE" \
  --hosted-zone-id "$ZONE_ID" --change-batch "$CHANGE_BATCH" \
  --query 'ChangeInfo.Id' --output text)

echo "Waiting for change $CHANGE_ID to propagate (INSYNC)..."
aws route53 wait resource-record-sets-changed --profile "$PROFILE" --id "$CHANGE_ID"

echo "Done: $RECORD_NAME now points at $TARGET_DNS_NAME."
