# Runbook: CDK Pipelines bootstrap and first deployment

One-time steps actually performed to stand up the pipeline built in
[ADR 0004](../adr/0004-cdk-pipelines-deployment.md) / issue #28. Redo
this exact sequence if the pipeline ever needs to be rebuilt from
scratch (e.g. a new AWS Organization). As with the
[account bootstrap runbook](aws-account-bootstrap.md), real account IDs
and ARNs are intentionally not written here — see that runbook's
rationale.

## Prerequisites

- The 4 project accounts already exist (`aws-account-bootstrap.md`).
- You have `AdministratorAccess` SSO access to all 4 (`translator-tooling`,
  `-dev`, `-qa`, `-prod`).
- `infra/cdk`'s dependencies are installed (`pnpm install` at repo root).

## Steps

1. **Bootstrap `translator-tooling`** (self, no cross-account trust needed
   yet — this account hosts the pipeline itself):

   ```sh
   AWS_PROFILE=translator-tooling npx cdk bootstrap aws://<tooling-account-id>/us-east-1 \
     --context useMockAccounts=true
   ```

   `--context useMockAccounts=true` is required here even though bootstrap
   doesn't care about the pipeline's actual config — `bin/app.ts` still
   runs as part of loading the CDK app, and would otherwise try (and fail)
   to fetch account IDs from SSM parameters that don't exist yet.

2. **Bootstrap `translator-dev`/`-qa`/`-prod`**, each trusting the tooling
   account for cross-account deploys:

   ```sh
   AWS_PROFILE=translator-dev npx cdk bootstrap aws://<dev-account-id>/us-east-1 \
     --trust <tooling-account-id> \
     --cloudformation-execution-policies arn:aws:iam::aws:policy/AdministratorAccess \
     --context useMockAccounts=true
   # repeat for translator-qa and translator-prod with their own account IDs
   ```

3. **Create the SSM parameters** in `translator-tooling` (plain `String`
   type — none of these are secret, see ADR 0004):

   ```sh
   aws ssm put-parameter --profile translator-tooling --region us-east-1 \
     --name "/traductor-kaqchikel/accounts/tooling" --type String --value "<tooling-account-id>" --overwrite
   # repeat for /accounts/dev, /accounts/qa, /accounts/prod
   ```

4. **Create the GitHub connection** (stays `PENDING` until the console
   handshake below):

   ```sh
   aws codeconnections create-connection --profile translator-tooling --region us-east-1 \
     --provider-type GitHub --connection-name traductor-kaqchikel-github
   ```

   Store the returned ARN:

   ```sh
   aws ssm put-parameter --profile translator-tooling --region us-east-1 \
     --name "/traductor-kaqchikel/github-connection-arn" --type String --value "<connection-arn>" --overwrite
   ```

5. **Complete the GitHub authorization — the one step that cannot be
   scripted** (no API for it, by AWS's own design):
   - Log into the AWS SSO portal, select `translator-tooling` →
     `AdministratorAccess` → Management Console.
   - Go to `https://console.aws.amazon.com/codesuite/settings/connections?region=us-east-1`
     (or search "Connections" in the console search bar).
   - Find the connection (status `Pending`), open it, **Update pending
     connection**, and complete the GitHub OAuth flow.
   - Confirm status flips to `Available`:
     `aws codeconnections get-connection --profile translator-tooling --region us-east-1 --connection-arn <arn> --query Connection.ConnectionStatus`

6. **First deploy of the pipeline stack itself**, from a machine with
   `translator-tooling` credentials:

   ```sh
   cd infra/cdk
   AWS_PROFILE=translator-tooling npx cdk deploy
   ```

   This prompts an interactive IAM-change confirmation (real cross-account
   roles/KMS permissions are being created) — review it, don't blind-approve
   with `--require-approval never` without actually reading the diff. After
   this one manual deploy, the pipeline is self-mutating: future changes to
   `infra/cdk/lib/pipeline-stack.ts` land automatically on the next pipeline
   run triggered by a push to `main`, no manual `cdk deploy` needed again.

## What happens after this

Every push to `main` runs the pipeline: Source → Build (synth) →
UpdatePipeline (self-mutate) → Dev (auto-deploy) → Qa (auto-deploy) →
Prod (blocked on a manual approval action in the CodePipeline console
until someone approves it). Check pipeline/stage status with:

```sh
aws codepipeline get-pipeline-state --profile translator-tooling --region us-east-1 \
  --name TraductorKaqchikelPipeline
```

## Result of the first run

Performed 2026-09-19. Dev and Qa both deployed successfully
(`Dev-Web`/`Qa-Web` stacks, `CREATE_COMPLETE`) — the first real AWS
infrastructure this project has ever had running. Prod sits at the
manual approval gate, intentionally not yet approved.
