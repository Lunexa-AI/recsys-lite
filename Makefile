# Developer-friendly Makefile for vector_recsys_lite

.PHONY: dev lint test coverage precommit docker-build docker-test ci

# Set up full dev environment (Poetry, venv, pre-commit)
dev:
	@echo "[dev] Installing Poetry, dependencies, and pre-commit hooks..."
	@if ! command -v poetry >/dev/null 2>&1; then \
	  pip install --user poetry; \
	fi
	poetry install --with dev
	poetry run pre-commit install
	@echo "[dev] Setup complete!"

# Lint all code
lint:
	poetry run black --check src/ tests/
	poetry run isort --check src/ tests/
	poetry run ruff check src/ tests/

# Run all tests with coverage
test:
	poetry run pytest --cov=vector_recsys_lite --cov-report=term-missing --cov-fail-under=75

# Show coverage report, fail if below threshold
coverage:
	poetry run pytest --cov=src/vector_recsys_lite --cov-report=term-missing --cov-fail-under=75

# Run all pre-commit hooks
precommit:
	poetry run pre-commit run --all-files

# Build Docker image
docker-build:
	docker build -t vector-recsys-lite:latest .

# Run tests inside Docker container
# Note: Use --entrypoint poetry to ensure dev dependencies (pytest) are available
docker-test:
	docker run --rm --entrypoint poetry vector-recsys-lite:latest run pytest --cov=vector_recsys_lite --cov-report=term-missing --cov-fail-under=75

# Simulate full CI locally
ci: lint test coverage docker-build docker-test
	@echo "[ci] All checks passed! Ready to push."
