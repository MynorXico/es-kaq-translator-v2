# Runbook: populating an environment's `api-urls/{env}` SSM parameter

Issue #263 (see also issue #130, which introduced the parameter). This is
a fully manual, one-time step for each environment, needed exactly once:
the first time that environment's `ApiStack` deploys for real. No
automation writes these parameters anywhere in this codebase — confirmed
by grepping `infra/` and `.github/` for `put-parameter`/`PutParameter`
calls against `api-urls/*`, and by `ApiStack`'s own doc comment pointing
back at this manual step.

## Why this exists

`infra/cdk/lib/pipeline-stack.ts`'s pipeline synth step builds `apps/web`
once per environment (`dev`/`qa`/`prod`), each with that environment's own
`VITE_API_BASE_URL` baked in (issue #263 — see `infra/cdk/README.md`'s
"Building and deploying `apps/web`" section for the full mechanism). That
per-environment URL comes from `PipelineConfig.webApiBaseUrls`, which
`bin/app.ts` fetches from `/traductor-kaqchikel/api-urls/{dev,qa,prod}` in
SSM Parameter Store (`translator-tooling`).

`ApiStack`'s real API Gateway URL is a CloudFormation-resolved value that
doesn't exist until *after* that stack deploys — it can't be read from
within the pipeline's synchronous, pre-deploy synth step the way a plain
constructed name can. So there's an unavoidable chicken-and-egg gap the
first time any environment's `ApiStack` deploys: the stack deploys with
some (possibly absent, possibly stale) `apps/web` build pointed at the
wrong (or no) API, until a human closes the loop by hand.

A missing parameter for a given environment is treated as "no real
`ApiStack` deployed there yet" and falls back to an empty string (a
same-origin relative request against that environment's own CloudFront
distribution — fails harmlessly client-side, not a real HTTP 404). It is
**not** treated as "reuse another environment's URL" — that silent
fallback is exactly the bug issue #263 fixed (qa's web app was calling
dev's API, because the old pipeline shared a single build across every
environment).

## When to do this

Any time an environment's `ApiStack` is deployed for the first time, or
its API Gateway URL changes (e.g. the REST API is replaced rather than
updated in place — `LambdaRestApi` updates generally keep the same URL,
but a full stack replacement would not).

## Steps

1. Get the environment's real API URL from `ApiStack`'s own `ApiUrl`
   `CfnOutput` (not hand-constructed — see the trailing-slash note in
   `infra/cdk/lib/api-stack.ts`):

   ```sh
   aws cloudformation describe-stacks \
     --profile translator-<env> --region us-east-1 \
     --stack-name <Env>-Api \
     --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" \
     --output text
   ```

   (`<Env>` is the stage name as CDK capitalizes it, e.g. `Dev`/`Qa`/
   `Prod` — the deployed stack name is just `<Env>-Api`, confirmed against
   a real `cdk synth` output; verify with `aws cloudformation list-stacks`
   if this ever changes.)

2. Write it to the matching SSM parameter in `translator-tooling` (**not**
   the target environment's own account — `bin/app.ts` always resolves
   this config from `translator-tooling`, the same account the pipeline
   itself lives in, regardless of which environment the URL is *for*):

   ```sh
   aws ssm put-parameter --profile translator-tooling --region us-east-1 \
     --name "/traductor-kaqchikel/api-urls/<env>" \
     --type String --value "<api-url-from-step-1>" --overwrite
   ```

   `<env>` is `dev`, `qa`, or `prod` (lowercase, matching
   `PipelineConfig.webApiBaseUrls`' keys).

3. Trigger a new pipeline run so the next synth picks up the new
   parameter and rebuilds that environment's `apps/web` with it baked in
   (the pipeline is self-mutating but does **not** watch SSM for changes
   — a normal merge to `main` is the simplest way to trigger a fresh run;
   there's no need to change any code, an empty/trivial commit works).

4. Verify in a real browser once the pipeline run reaches that
   environment's `Web` stage: open that environment's web app, open the
   network tab, and confirm translate requests go to the URL just
   configured, not a different environment's. See
   [the translate-jobs verification runbook](translate-jobs-verification.md)
   for the general shape of this check.

## Related

- Issue #130 — introduced `webApiBaseUrls`/this SSM parameter path,
  originally only ever consumed for `dev`.
- Issue #263 — made every environment's `apps/web` build actually use its
  *own* entry here, instead of every environment sharing `dev`'s (the bug
  that made qa silently call dev's API).
- `infra/cdk/lib/api-stack.ts` — `ApiUrl` `CfnOutput` this runbook reads
  from in step 1.
- `infra/cdk/README.md`'s "Building and deploying `apps/web`" section —
  how the per-environment builds that consume this parameter work.
