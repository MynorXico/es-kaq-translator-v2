# Lambda container image for apps/api (ADR 0001: Lambda container +
# API Gateway). Built and deployed by `infra/cdk/lib/api-stack.ts` via
# CDK's `DockerImageCode.fromImageAsset`.
#
# Two stages:
#   1. "deps": resolve apps/api's *runtime* dependencies from uv.lock into
#      a plain requirements.txt (uv itself is a build-time tool only, not
#      shipped in the final image), then pip-install them into /asset.
#   2. Final: AWS's own Lambda Python base image, with /asset's installed
#      packages plus this app's own source layered on top. Kept as two
#      layers (deps, then app code) so an app-only code change doesn't
#      invalidate the (slower) dependency-install layer.
FROM public.ecr.aws/lambda/python:3.12 AS deps

RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml uv.lock ./

RUN uv export --no-dev --no-hashes --no-emit-project --format requirements.txt \
        -o requirements.txt \
    && pip install --no-cache-dir -r requirements.txt -t /asset

FROM public.ecr.aws/lambda/python:3.12

COPY --from=deps /asset "${LAMBDA_TASK_ROOT}"
COPY app "${LAMBDA_TASK_ROOT}/app"

# Mangum-wrapped FastAPI app (app/lambda_handler.py) -- see that module's
# docstring and this repo's PR for issue #96 for the API Gateway
# integration contract (HTTP API, payload format 2.0).
CMD ["app.lambda_handler.handler"]
