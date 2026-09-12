# Runbook: bootstrapping the project's AWS accounts

Per [ADR 0001](../adr/0001-initial-architecture.md), this project uses one
dedicated AWS account per environment (tooling/dev/qa/prod) rather than
sharing accounts with unrelated workloads, even when the AWS Organization
already has generic dev/qa/prod-style accounts from other projects. This
runbook documents how those 4 accounts were created and how to redo this
(e.g. for a fork, or if the accounts ever need to be recreated).

## Why dedicated accounts instead of reusing existing ones

An AWS account is the real security/blast-radius boundary, not just a
naming convention. If this project's resources, IAM roles, and eventually
CI/CD deploy credentials shared an account with unrelated personal
experiments, a mistake or overly broad policy in one could reach the
other. Dedicated accounts also make cost attribution and eventual
maintainer handoff far simpler.

## Prerequisites

- An existing AWS Organization (`aws organizations describe-organization`
  succeeds from some account) with "ALL" feature set enabled.
- IAM Identity Center (AWS SSO) already set up for that organization, with
  at least one permission set (e.g. `AdministratorAccess`) and your user
  provisioned in the identity store.
- AWS CLI v2 (recent enough to support `aws login` root sessions - this was
  done with `aws-cli/2.36.44`).
- Credentials for the organization's **management account** (the account
  `describe-organization` reports as `MasterAccountId`). Member accounts,
  even with `AdministratorAccess`, cannot call `organizations:CreateAccount`
  - this API only works from the management account.
- `jq` installed (used for JSON parsing in some manual steps below).

### Getting management-account credentials without static root keys

Do **not** create long-lived access keys for the root user. Instead use
the AWS CLI's built-in root session login (requires the account's root
credentials + MFA once, in a browser):

```ini
# ~/.aws/config
[profile infra]
login_session = arn:aws:iam::<MANAGEMENT_ACCOUNT_ID>:root
region = us-east-1
```

```sh
aws login --profile infra
```

This produces a short-lived session (re-run `aws login` when it expires)
instead of a permanent credential sitting in `~/.aws/credentials`.

## Steps

1. **Authenticate to the management account** as shown above.

2. **Run the bootstrap script** (`infra/accounts/bootstrap-accounts.sh`).
   It creates an OU, creates the 4 accounts (skipping any that already
   exist by name), moves them into the OU, and grants an IAM Identity
   Center user a permission set on each:

   ```sh
   PROFILE=infra \
   EMAIL_DOMAIN=infra@yourdomain.example \
   PROJECT_PREFIX=translator \
   SSO_USERNAME=YourSsoUsername \
   ./infra/accounts/bootstrap-accounts.sh
   ```

   This is idempotent for the account-creation and OU steps; if you re-run
   it, the SSO grant step may print "(assignment may already exist,
   continuing)" for accounts that already have the assignment - that's
   expected and safe to ignore.

3. **Add local CLI profiles** for the new accounts, reusing whichever
   `sso-session` block your existing profiles use (see your
   `~/.aws/config` for the session name tied to your SSO start URL):

   ```ini
   [profile translator-tooling]
   sso_session = <your-existing-sso-session-name>
   sso_account_id = <tooling-account-id>
   sso_role_name = AdministratorAccess
   region = us-east-1
   output = json

   # ...repeat for translator-dev, translator-qa, translator-prod
   ```

4. **Verify** each profile authenticates:

   ```sh
   aws sso login --profile translator-tooling --use-device-code
   aws sts get-caller-identity --profile translator-tooling
   # ...repeat for translator-dev, translator-qa, translator-prod
   ```

   Use `--use-device-code` when the default browser-redirect flow can't
   reach a loopback callback (e.g. running from a remote/sandboxed shell).

## Current environment (this project's actual accounts)

These accounts exist (created 2026-09-11: `translator-tooling`,
`translator-dev`, `translator-qa`, `translator-prod`, under a dedicated
`TraductorKaqchikel` OU), with `AdministratorAccess` granted to the
project owner's SSO user on all 4.

Account IDs, the org ID, and the management account ID are **deliberately
not published here** — this is a public repo, and while AWS account IDs
alone aren't classified as secret, publishing them together with the
management account ID and root email addresses gives more to an attacker
than is worth it for a doc that doesn't need them to be useful. The real
values are tracked privately (not in git) and referenced by CDK from a
private config source (e.g. SSM Parameter Store or CI environment
secrets) rather than hardcoded here. If you have management-account
access, `aws organizations list-accounts` will show you the real IDs; if
you're bootstrapping your own copy of this project, running the script
above produces your own.

## Next steps

These accounts are currently empty (no VPCs, no CDK bootstrap, no cross-
account trust relationships beyond the default `OrganizationAccountAccessRole`).
Follow-up work (tracked on the project board) includes: `cdk bootstrap` in
each account, setting up cross-account deployment roles from
`translator-tooling` for CDK Pipelines, and Route 53 delegation once the
domain is purchased.
