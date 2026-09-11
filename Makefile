.PHONY: install test build synth

install:
	pnpm install
	cd apps/api && uv sync

test:
	pnpm --filter web test
	pnpm --filter infra-cdk test
	cd apps/api && uv run pytest

build:
	pnpm --filter web build
	pnpm --filter infra-cdk build

synth:
	pnpm --filter infra-cdk synth
