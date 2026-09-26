"""AWS Lambda entry point wrapping the FastAPI app for API Gateway.

Deployed as a Lambda container image behind a regional API Gateway REST
API (v1) using the Lambda proxy (`AWS_PROXY`) integration -- see
`infra/cdk/lib/api-stack.ts` and this app's `Dockerfile`. `Mangum`
translates API Gateway proxy events to/from ASGI calls into the same
`app.main.app` instance used everywhere else (local dev server, tests),
so there is exactly one FastAPI app definition regardless of how it's run.

Issue #151 (ADR 0005) migrated `ApiStack` from an API Gateway `HttpApi`
(v2, payload format 2.0) to a `RestApi` (v1) so AWS WAF could attach
directly to it -- WAF cannot associate with an `HttpApi` at all. `Mangum`
auto-detects the event shape at call time (`resource` + `requestContext`
present => REST API v1's `APIGateway` handler; a top-level `version`
key => HTTP API v2's `HTTPGateway` handler), so this file's `Mangum(app)`
call itself needed no code change -- but that inference was verified
directly against a synthetic v1 event in
`tests/integration/test_lambda_handler.py`, not assumed from Mangum's
docs alone.
"""

from mangum import Mangum

from app.main import app

handler = Mangum(app)
