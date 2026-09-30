# ADR 0009: Model Registry registration timing (revert in-container self-registration)

- Status: Proposed
- Date: 2026-09-30

## Context

Issue #190 changed how a completed training run gets registered as a
SageMaker Model Registry Model Package. Before #190, `submit_job.py`
registered a run *client-side*, after the job completed: it downloaded
the run's `model.tar.gz` artifact from S3, extracted `model_card.md` to
read BLEU/chrF, and called `CreateModelPackage`. That artifact was
measured at 28.4-30.5 GB on real runs (issue #187, still open, checkpoint
averaging bundles full epoch checkpoints into `SM_MODEL_DIR`) — enough
that the client-side download was slow (~45 minutes, bandwidth-bound on
whatever connection the maintainer runs `submit_job.py` from) and, before
issue #188/#189's fix, unsafe (an in-memory buffer download caused a real
out-of-memory kill).

#190's design moved registration *into* `train.py`, running inside the
SageMaker training container itself, right after it writes
`model_card.md` to local disk — eliminating the download entirely by
reading the model card off disk and computing the artifact's future,
deterministic S3 URI (`{output_path}/{training_job_name}/output/
model.tar.gz`) *before* that artifact exists there. This rested on an
explicit premise, checked against AWS's public `CreateModelPackage` API
reference during PR #191's review: that `ModelDataUrl` isn't validated
for existence until actual deploy time.

That premise is false. Issue #201 confirmed, against two real training
jobs, that `CreateModelPackage` performs a live S3 existence check at
call time:

```
ValidationException: Cannot find S3 object: .../output/model.tar.gz
in bucket traductor-kaqchikel-training-data-dev. Please check if your
S3 object exists and has proper permissions for SageMaker.
```

This is structural, not a race condition: `train.py` calls
`register_model_from_training_job` from inside its own process, strictly
*before* the training container's toolkit tars up `SM_MODEL_DIR` and
uploads it to S3 — a separate post-exit lifecycle step the running script
has no hook into. Self-registration has never once succeeded on a real
training job; both real runs that exercised it were registered instead
via the `--register-existing` fallback (issue #190's own explicit opt-in
path for registering an already-completed job by name, which still
downloads the artifact, but safely — streamed to disk, issue #188/#189 —
rather than buffered in memory).

Project constraints this decision is made against (see also ADR 0001):

- Serverless-first, cost-conscious, but cost here is dominated by
  engineering/operational time, not raw AWS spend — a $30 GB S3 transfer
  costs cents; the real cost is the ~45 minutes it occupies and the
  maintainer-time already spent on two failed self-registration attempts
  (#196, #201) plus their manual workarounds.
- Single solo maintainer. Favor operational simplicity over new
  infrastructure where they trade off.
- Real training job cadence is roughly one every few days during active
  experimentation — not a high-volume pipeline where a ~45-minute
  post-run step would be a bottleneck.
- Issue #187 (open, unscheduled) questions whether bundling full epoch
  checkpoints into the deployed artifact is the right choice at all. If
  it's resolved by moving checkpoints outside `SM_MODEL_DIR`, the
  artifact this ADR is designed around would shrink from ~30 GB to
  roughly the bare model's size (~1-2 GB) — which would materially change
  this decision's cost/benefit balance (see Consequences).

## Options considered

1. **Have `train.py` upload the artifact to S3 itself before
   registering**, so the URL exists by the time `CreateModelPackage` is
   called. Rejected: the training container's own toolkit tars and
   uploads `SM_MODEL_DIR` to the same deterministic location regardless
   of what the script does — there is no way to disable that step from
   inside `train.py`. Having `train.py` also upload the artifact means
   either a duplicate ~30 GB upload (real S3 PUT cost and time, twice,
   every run) or a fragile attempt to exactly race/pre-empt the
   platform's own packaging step. Either way, it means re-implementing,
   in application code, artifact packaging the container already does
   for free — the same category of mistake (silently owning platform
   behavior we don't have to) that produced #190's original bug.

2. **Decouple registration from the training container entirely**: an
   EventBridge rule on the training job's `Completed` state, triggering a
   Lambda that registers once the real artifact is confirmed to exist.
   Correctly ordered, and doesn't touch the training container. Real new
   infra (`infra/cdk`): an EventBridge rule, a Lambda function + its IAM
   role, and (to avoid re-deriving BLEU/chrF without ever touching the
   30 GB artifact) either:
   - a Lambda that streams the S3 object through `tarfile`'s streaming
     mode (`mode="r|gz"`, sequential-only) to extract `model_card.md`
     without buffering the full artifact to memory or `/tmp` — technically
     sound, but the extraction may need to walk the *entire* compressed
     stream if `model_card.md` isn't tar's first member (member order
     depends on filesystem listing order, not controlled), which is slow
     and adds real, non-trivial new code (a novel streaming-tar reader
     with no existing test coverage in this repo) for a rare path; or
   - a small, separate "sidecar" object: `train.py` `PutObject`s just the
     few-KB `model_card.md` text to its own dedicated S3 key (independent
     of the `SM_MODEL_DIR` artifact) while still running, and the Lambda
     reads *that* — small, fast, and immune to artifact size entirely (so
     immune to #187's outcome too). This is a genuinely better shape for
     option 2 than the sketch in issue #201, and is worth keeping in mind
     as a future direction (see Consequences) — but it still means a new
     Lambda, a new EventBridge rule, a new IAM role, and (since the
     registration logic this Lambda would call, `training.model_registry`
     /`evaluation.model_card`, is Python) either a container-image Lambda
     (the pattern `apps/api` already uses, ADR 0001) or a new Python
     Lambda packaging/bundling story this repo doesn't have yet.

   Either sub-approach is the architecturally "correct" fix — registration
   naturally belongs after the artifact exists, decoupled from the
   container that produces it — but is more new infrastructure and new
   code than this project's job cadence (one every few days) currently
   justifies, for a solo maintainer favoring operational simplicity.
   Revisiting this is a "when," not "never" — see Consequences.

3. **Revert to client-side registration**, restoring `submit_job.py`'s
   pre-#190 automatic post-training registration call, kept safe by
   #188/#189's disk-streamed (not in-memory) artifact download. Simplest:
   no new infrastructure, no new code beyond removing #190's now-proven-
   broken self-registration path, and it's the exact code path already in
   daily use today as the `--register-existing` fallback — proven to work
   against real jobs, not new/unexercised. Reintroduces the slow (~45
   minute), bandwidth-bound full-artifact download this project was
   trying to eliminate — a real, named cost, not a free win.

## Decision

**Revert to client-side registration (Option 3).** Remove the in-container
self-registration path from `train.py` entirely — it is not "temporarily
broken," it cannot work as designed, per the structural reason in
Context — rather than leaving dead, always-failing code with a warning
message behind. Concretely, hand off to `dev`:

- `train.py`: remove `register_model_from_training_job`,
  `resolve_training_job_name`, `compute_model_artifact_s3_uri`,
  `_boto3_client_for_registration`, the `registrar` parameter on
  `run_training_job`, and the `--register-model`/`--output-path`/
  `--training-image`/`--region`/`--model-package-group-name`/
  `--approval-status` CLI flags that existed solely to support it.
- `submit_job.py`: restore an automatic post-training registration call
  in `main()` for a waited-for job (`not args.no_wait`), using the
  already-existing, already-fixed `register_model`/
  `fetch_model_card_from_artifact` (disk-streamed, #188/#189) — i.e. put
  back essentially what `main()` did before #190, minus the OOM bug.
  `--no-register` keeps meaning "don't register this run at all"; a job
  submitted with `--no-wait` (or `--no-register`) still has no automatic
  registration, and `--register-existing <job-name>` remains the
  supported way to register it after the fact — unchanged from today.
- `build_job_config`/`build_hyperparameters` stop passing the
  `register-model`/`output-path`/`training-image`/`region` hyperparameters
  to the training job — they existed only for the container's own
  self-registration attempt.
- `infra/cdk/lib/data-stack.ts`: remove the training execution role's
  `sagemaker:CreateModelPackageGroup`/`sagemaker:CreateModelPackage` grant
  added for #190/#191. Registration goes back to running under the
  maintainer's own SSO credentials (as it did before #190, and as
  `--register-existing` already does today), not the training role's.
  This is also a least-privilege improvement: the training container no
  longer needs any Model Registry write permission at all.
- `ml/README.md`'s "Registering an existing job's artifact after the
  fact" section and the training-entrypoint docs describing
  self-registration need updating to reflect that automatic, client-side,
  post-wait registration is the default again, with `--register-existing`
  kept for the no-wait/no-register case — not a change in that flag's own
  behavior.

This is test-first per `docs/testing.md`, same as the rest of this
codebase — removing `tests/unit/test_train_registration.py` and the
self-registration cases in `tests/integration/test_train_pipeline.py`,
and restoring/adjusting `main()`'s registration-call test coverage in
`tests/unit/test_submit_job.py`/`tests/integration/test_submit_job_cli.py`.

## Consequences

- Every real, waited-for training job again pays a client-side download
  of the full artifact to register it — at today's measured size
  (28.4-30.5 GB), on the order of tens of minutes, bandwidth-bound on
  the maintainer's own connection, recurring roughly every few days
  during active experimentation. This is a real, accepted cost, not an
  oversight: it's bounded, already safe (no OOM risk, #188/#189), already
  proven in production use (it's exactly the `--register-existing` path
  already registering every real run today), and doesn't block the
  training job itself — `--no-wait` lets a maintainer submit a job,
  disconnect, and run `--register-existing` later at a convenient time
  if the ~45-minute wait isn't worth blocking on.
- No new AWS infrastructure, no new IAM surface beyond a removal, and no
  new code paths without existing test/production coverage. This is the
  simplest option available and matches "prefer the simplest design that
  satisfies the requirement" for a project this size.
- **This interacts directly with issue #187.** If #187 is resolved by
  moving per-epoch checkpoints outside `SM_MODEL_DIR` (so the *deployed*,
  registered artifact is just the final model, plausibly ~1-2 GB instead
  of ~30 GB), the cost this ADR accepts drops by roughly an order of
  magnitude — a couple of minutes instead of ~45. That would also make
  Option 2's "sidecar model card + EventBridge/Lambda" variant (see
  Options considered) meaningfully cheaper to justify, since a Lambda-
  based reader would then comfortably fit a full-artifact download within
  Lambda's ephemeral storage/time budget without needing the sidecar
  trick at all. **This ADR does not resolve #187**, but flags it as the
  condition most likely to reopen this decision — #187 should note this
  ADR as a downstream consideration when it's scheduled.
- If real training cadence ever moves from "every few days" toward a
  genuinely frequent/production retraining pipeline, the ~45-minute
  registration cost (or its shrunk-by-#187 equivalent) stops being
  negligible in aggregate maintainer time, and Option 2's decoupled
  EventBridge+Lambda design (ideally its sidecar-model-card variant, which
  avoids depending on artifact size at all) becomes the right call. That
  is a future ADR revisiting this one, not a reason to build it now.
- `train.py` goes back to needing zero AWS credentials or SDK calls at
  all — it was, before #190, a pure training script with no AWS API
  surface beyond reading/writing S3 paths already resolved for it by
  `submit_job.py`/SageMaker's own channel mechanism. That is a real
  simplification, not just a revert: the training container's blast
  radius (what it can do with its execution role) shrinks back to
  reading training data and writing its own output, which is a better
  default posture regardless of this specific bug.

## Alternatives considered

- **Option 1 (self-upload from `train.py`)**: rejected — duplicates
  artifact packaging/upload the container already performs for free,
  either doubling real upload cost every run or requiring a fragile race
  against the platform's own post-exit packaging step. See Options
  considered above.
- **Option 2 (EventBridge + Lambda, decoupled registration)**: not
  rejected outright — architecturally the "correct" fix, and the
  sidecar-model-card variant identified during this ADR's research (see
  Options considered) is a better-shaped version of it than issue #201's
  original sketch, avoiding the artifact-size dependency the original
  sketch would have inherited. Deferred rather than adopted now: it needs
  new infrastructure (EventBridge rule, Lambda, IAM role, and — since the
  registration logic is Python — either a container-image Lambda or new
  Python-Lambda packaging this repo doesn't have yet) that isn't
  justified by a once-every-few-days job cadence and a maintainer
  optimizing for operational simplicity, especially while #187 might
  independently shrink the artifact enough to change this trade-off.
- **A SageMaker-native "on completion" callback/webhook, or SageMaker
  Pipelines' built-in `RegisterModel` step**: no such per-script callback
  exists — SageMaker's own documented completion-notification mechanism
  *is* the EventBridge `Training Job State Change` event (Option 2), not
  something more specific to registration. `Pipelines`' `RegisterModel`
  step would fold registration into a full Pipelines-based orchestration
  of the training job itself, replacing `submit_job.py`'s current direct
  `ModelTrainer` call — a much larger restructuring than this bug
  justifies, and out of scope here.
- **SageMaker Python SDK v3 (already in use per #155/#172) offering
  something better**: checked directly — `ModelTrainer` has no async
  "register on completion" hook; the SDK's training interface is
  synchronous (`wait=True/False`), the same shape `submit_job.py` already
  uses. Nothing in v3 changes this decision.
