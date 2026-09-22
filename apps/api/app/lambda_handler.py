"""AWS Lambda entry point wrapping the FastAPI app for API Gateway.

Deployed as a Lambda container image behind an API Gateway HTTP API using
the Lambda proxy integration (payload format 2.0) -- see
`infra/cdk/lib/api-stack.ts` and this app's `Dockerfile`. `Mangum`
translates API Gateway proxy events to/from ASGI calls into the same
`app.main.app` instance used everywhere else (local dev server, tests),
so there is exactly one FastAPI app definition regardless of how it's run.
"""

from mangum import Mangum

from app.main import app

handler = Mangum(app)
