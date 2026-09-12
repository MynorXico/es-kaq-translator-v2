.PHONY: install test test-e2e build synth

install:
	pnpm install
	cd apps/api && uv sync

test:
	pnpm --filter web test
	pnpm --filter infra-cdk test
	cd apps/api && uv run pytest

# Separate from `test`: needs a one-time browser install
# (`pnpm --filter web exec playwright install chromium`).
test-e2e:
	pnpm --filter web test:e2e

build:
	pnpm --filter web build
	pnpm --filter infra-cdk build

synth:
	pnpm --filter infra-cdk synth
