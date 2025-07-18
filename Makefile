# Makefile — recsys_lite (2025 edition)
.DEFAULT_GOAL := help
.PHONY: dev format lint test typecheck precommit docker-build docker-test ci clean help

PKG := recsys_lite
PY := poetry run
VENV_DIR := .venv                    # keep env inside project

## ---------- Dev workflow ---------------------------------------------------

dev: ## Install deps & pre‑commit hooks
	@[ -d $(VENV_DIR) ] || poetry env use $(shell pyenv global)
	poetry install --with dev
	poetry run pre-commit install
	@echo "✅ Dev environment ready"

format: ## Auto‑format code with Black, isort, ruff‑fix
	$(PY) black src/ tests/
	$(PY) isort src/ tests/
	$(PY) ruff format src/ tests/

lint: ## Static‑analysis without mutating files
	$(PY) ruff check src/ tests/
	$(PY) black --check src/ tests/
	$(PY) isort --check src/ tests/

typecheck: ## Optional strict typing
	$(PY) mypy src/ || true

test: ## Run tests + coverage report
	$(PY) pytest --cov=$(PKG) --cov-report=term-missing --cov-fail-under=80

precommit: ## Run all pre‑commit hooks
	$(PY) pre-commit run --all-files --show-diff-on-failure

## ---------- Docker helpers -------------------------------------------------

docker-build: ## Build multi‑arch Docker image using cache
	docker buildx build \
	  --platform linux/amd64,linux/arm64 \
	  --cache-from type=gha \
	  --cache-to type=gha,mode=max \
	  -t $(PKG):latest .

docker-test: docker-build ## Run tests inside container
	docker run --rm --entrypoint poetry $(PKG):latest run pytest -q

## ---------- Meta -----------------------------------------------------------

ci: lint typecheck test docker-test ## Aggregate for local parity with CI

clean: ## Remove virtualenv & transient artefacts
	rm -rf $(VENV_DIR) dist build *.egg-info .pytest_cache .coverage || true

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?##"} {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
