"""Environment-driven configuration for the API.

Deployed origins (dev/qa/prod CloudFront domains) don't exist yet — no
DNS records are wired up per `docs/runbooks/domain-and-dns.md`. Rather
than hardcoding placeholder domain names, the allowed CORS origins are
read from an env var so each environment (set via the CDK Pipelines
deploy) can supply its own value once those domains exist.
"""

import os

# Vite's default local dev server origin (see apps/web/README.md).
DEFAULT_ALLOWED_ORIGINS = ["http://localhost:5173"]


def get_allowed_origins() -> list[str]:
    """Return the list of origins allowed to make cross-origin requests.

    Reads a comma-separated `ALLOWED_ORIGINS` env var. Falls back to the
    local Vite dev server origin when unset, for local development.
    """
    raw = os.environ.get("ALLOWED_ORIGINS")
    if not raw:
        return DEFAULT_ALLOWED_ORIGINS
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
