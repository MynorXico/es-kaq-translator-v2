#!/usr/bin/env bash
# Bootstraps a dedicated OU + 4 AWS accounts (tooling/dev/qa/prod) for this
# project inside an existing AWS Organization, then grants an IAM Identity
# Center user a permission set (default: AdministratorAccess) on each.
#
# Must be run with credentials for the ORGANIZATION'S MANAGEMENT ACCOUNT -
# see docs/runbooks/aws-account-bootstrap.md for prerequisites and context.
# Safe to re-run: skips accounts that already exist by name.
set -euo pipefail

PROFILE="${PROFILE:?Set PROFILE to your management-account AWS CLI profile}"
EMAIL_DOMAIN="${EMAIL_DOMAIN:?Set EMAIL_DOMAIN, e.g. infra@example.com (used as local+alias@domain per account)}"
PROJECT_PREFIX="${PROJECT_PREFIX:?Set PROJECT_PREFIX, e.g. translator}"
SSO_USERNAME="${SSO_USERNAME:?Set SSO_USERNAME to the IAM Identity Center username to grant access}"
OU_NAME="${OU_NAME:-TraductorKaqchikel}"
PERMISSION_SET_NAME="${PERMISSION_SET_NAME:-AdministratorAccess}"

EMAIL_LOCAL="${EMAIL_DOMAIN%%@*}"
EMAIL_HOST="${EMAIL_DOMAIN#*@}"

ROOT_ID=$(aws organizations list-roots --profile "$PROFILE" --query 'Roots[0].Id' --output text)

OU_ID=$(aws organizations list-organizational-units-for-parent --profile "$PROFILE" --parent-id "$ROOT_ID" \
  --query "OrganizationalUnits[?Name=='$OU_NAME'].Id" --output text)
if [ -z "$OU_ID" ]; then
  echo "Creating OU $OU_NAME..."
  OU_ID=$(aws organizations create-organizational-unit --profile "$PROFILE" --parent-id "$ROOT_ID" \
    --name "$OU_NAME" --query 'OrganizationalUnit.Id' --output text)
fi
echo "OU: $OU_ID"

declare -A ACCOUNT_IDS
for ENV in tooling dev qa prod; do
  NAME="${PROJECT_PREFIX}-${ENV}"
  EXISTING=$(aws organizations list-accounts --profile "$PROFILE" \
    --query "Accounts[?Name=='$NAME'].Id" --output text)
  if [ -n "$EXISTING" ]; then
    echo "$NAME already exists ($EXISTING), skipping creation"
    ACCOUNT_IDS[$ENV]="$EXISTING"
    continue
  fi

  EMAIL="${EMAIL_LOCAL}+${NAME}@${EMAIL_HOST}"
  echo "Creating account $NAME <$EMAIL>..."
  REQUEST_ID=$(aws organizations create-account --profile "$PROFILE" \
    --email "$EMAIL" --account-name "$NAME" \
    --query 'CreateAccountStatus.Id' --output text)

  while true; do
    read -r STATE ACCOUNT_ID FAILURE_REASON <<< "$(aws organizations describe-create-account-status --profile "$PROFILE" \
      --create-account-request-id "$REQUEST_ID" --query 'CreateAccountStatus.[State,AccountId,FailureReason]' --output text)"
    if [ "$STATE" = "SUCCEEDED" ]; then
      ACCOUNT_IDS[$ENV]="$ACCOUNT_ID"
      echo "$NAME -> $ACCOUNT_ID"
      break
    elif [ "$STATE" = "FAILED" ]; then
      echo "Account creation failed for $NAME: $FAILURE_REASON" >&2
      exit 1
    fi
    sleep 10
  done

  aws organizations move-account --profile "$PROFILE" \
    --account-id "${ACCOUNT_IDS[$ENV]}" --source-parent-id "$ROOT_ID" --destination-parent-id "$OU_ID"
done

INSTANCE_ARN=$(aws sso-admin list-instances --profile "$PROFILE" --query 'Instances[0].InstanceArn' --output text)
IDENTITY_STORE_ID=$(aws sso-admin list-instances --profile "$PROFILE" --query 'Instances[0].IdentityStoreId' --output text)

PERM_SET_ARN=""
for arn in $(aws sso-admin list-permission-sets --profile "$PROFILE" --instance-arn "$INSTANCE_ARN" --query 'PermissionSets[]' --output text); do
  name=$(aws sso-admin describe-permission-set --profile "$PROFILE" --instance-arn "$INSTANCE_ARN" \
    --permission-set-arn "$arn" --query 'PermissionSet.Name' --output text)
  if [ "$name" = "$PERMISSION_SET_NAME" ]; then
    PERM_SET_ARN="$arn"
    break
  fi
done
if [ -z "$PERM_SET_ARN" ]; then
  echo "Permission set '$PERMISSION_SET_NAME' not found in IAM Identity Center" >&2
  exit 1
fi

USER_ID=$(aws identitystore list-users --profile "$PROFILE" --identity-store-id "$IDENTITY_STORE_ID" \
  --query "Users[?UserName=='$SSO_USERNAME'].UserId" --output text)
if [ -z "$USER_ID" ]; then
  echo "SSO user '$SSO_USERNAME' not found in Identity Store $IDENTITY_STORE_ID" >&2
  exit 1
fi

for ENV in "${!ACCOUNT_IDS[@]}"; do
  ACCT="${ACCOUNT_IDS[$ENV]}"
  echo "Granting $PERMISSION_SET_NAME on $ACCT to $SSO_USERNAME..."
  aws sso-admin create-account-assignment --profile "$PROFILE" \
    --instance-arn "$INSTANCE_ARN" --target-id "$ACCT" --target-type AWS_ACCOUNT \
    --permission-set-arn "$PERM_SET_ARN" --principal-type USER --principal-id "$USER_ID" \
    --output text >/dev/null || echo "  (assignment may already exist, continuing)"
done

echo
echo "Done. Account IDs:"
for ENV in tooling dev qa prod; do
  echo "  $ENV: ${ACCOUNT_IDS[$ENV]}"
done
