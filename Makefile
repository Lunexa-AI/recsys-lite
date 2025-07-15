# Minimal Makefile for vector_recsys_lite

.PHONY: lint test ci coverage

lint:
	poetry run black --check src/ tests/
	poetry run isort --check src/ tests/
	poetry run ruff check src/ tests/

coverage:
	pytest --cov=src/vector_recsys_lite --cov-report=term-missing --cov-fail-under=75

test:
	poetry run pytest --cov=vector_recsys_lite --cov-report=term-missing --cov-fail-under=85

ci: lint test 