# Minimal Makefile for vector_recsys_lite

.PHONY: lint test ci

lint:
	poetry run black --check src/ tests/
	poetry run isort --check src/ tests/
	poetry run ruff src/ tests/

test:
	poetry run pytest

ci: lint test 