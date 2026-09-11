# ADR 0001: Initial project architecture

- Status: Accepted
- Date: 2026-09-10

## Context

Traductor Kaqchikel is a new, open-source project building a bidirectional
Spanish↔Kaqchikel translator (UI, API, and machine learning pipeline),
deployed on AWS under an organization with separate accounts for
dev/qa/prod. The only initial asset is a parallel corpus of 33,616 training
sentences and 3,735 validation sentences, extracted from public ALMG texts
(see [`data-governance.md`](../data-governance.md) for the rights status
of this corpus).

Since this is a brand-new project with no existing code, the foundational
architecture decisions were made up front to avoid future technical debt.

## Decisions

| Area | Decision |
|---|---|
| MT approach | Fine-tune a pretrained multilingual model (e.g. NLLB-200 distilled or M2M100), extending vocab/embeddings for Kaqchikel, trained via SageMaker Training Jobs |
| Inference hosting | Serverless: SageMaker Serverless Inference (scale-to-zero, fits low initial traffic) |
| Infrastructure as code | AWS CDK, TypeScript |
| Project management | GitHub Issues + GitHub Projects |
| API language | Python (FastAPI), single service, deployed as a Lambda container behind API Gateway |
| Frontend | React + Vite (static SPA), hosted on S3 + CloudFront |
| CI/CD | GitHub Actions for PR checks (lint/test/build/security); CDK Pipelines (on CodePipeline) for dev → qa → prod deployment promotion |
| Code license | Apache License 2.0 |
| AWS environment isolation | One AWS account per environment (dev, qa, prod) under the existing Organization, plus a dedicated tooling/CI account |
| Domain | `traductorkaqchikel.com` |
| Repository/code language | English for all code, comments, docs, and commit messages; Spanish only for user-facing UI/API content |
| Initial Claude agents | Architect, Code Reviewer, ML Engineer/Data, Product Owner, QA, DevOps/Release |

## Consequences

- The monorepo uses pnpm workspaces for TypeScript/JavaScript code and a
  separate Python workspace for ML/API, orchestrated with a root
  `Makefile`, rather than a tool like Nx/Turborepo, to keep the barrier to
  entry low for external contributors.
- Because inference is serverless, the first request after inactivity will
  incur cold-start latency; this trade-off will be documented in the public
  API.
- The deployment pipeline lives in the tooling account and uses
  cross-account IAM roles into dev/qa/prod; prod deployments require
  manual approval.
- The training corpus is not published in the repository until rights are
  confirmed with ALMG (see `docs/data-governance.md`).
- The pretrained base model's license must be verified before committing to
  a specific model, since models like NLLB-200 are often distributed under
  non-commercial (CC-BY-NC) licenses, which could conflict with the
  project's open code/artifact goals.
- All code, comments, and documentation are written in English regardless
  of contributor location, to keep the codebase accessible to the broader
  international OSS community; only strings the end user actually sees
  (UI copy, API-facing translation content) are in Spanish, matching the
  target audience.

## Alternatives considered

- **Training a small Transformer from scratch** (AmericasNLP shared-task
  baseline style) instead of fine-tuning a pretrained model: cheaper to
  train and host, but with a lower quality ceiling since it can't leverage
  cross-lingual transfer. Rejected as the primary approach, but kept as a
  fallback/comparison baseline during experimentation.
- **Terraform** instead of CDK: more cloud-agnostic with a larger OSS
  module ecosystem, but CDK was preferred for its native integration with
  CodePipeline/AWS accounts and for keeping a single language
  (TypeScript) across infrastructure and frontend.
- **Next.js** instead of a Vite SPA: rejected for now since SSR/SEO isn't
  required for the MVP; can be reconsidered later as an isolated decision.
- **Jira** instead of GitHub Projects: rejected due to the friction it adds
  for external contributors to an open-source project.
