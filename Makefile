# Makefile — recsys_lite (2025‑07‑refresh)
.DEFAULT_GOAL := help
.PHONY: \
  dev format lint typecheck test precommit \
  docker-build docker-test docker-prod \
  ci clean help

PKG       := recsys_lite
PY        := poetry run
VENV_DIR  := .venv                      # in‑project virtual‑env

# ─────────── Dev workflow ────────────────────────────────────────────────────
dev: ## Install deps & pre‑commit hooks
	@[ -d $(VENV_DIR) ] || poetry env use $(shell command -v python3)
	poetry install --with dev
	$(PY) pre-commit install
	@echo "✅ Dev environment ready"

format: ## Auto‑format code (Black, isort, ruff‑format)
	$(PY) black src/ tests/
	$(PY) isort  src/ tests/
	$(PY) ruff format src/ tests/

lint: ## Static analysis (ruff + format checks)
	$(PY) ruff check src/ tests/
	$(PY) black --check src/ tests/
	$(PY) isort --check  src/ tests/

typecheck: ## Optional mypy run
	$(PY) mypy src/ || true            # make strict later

test: ## Run tests + coverage (fail <75 %)
	$(PY) pytest --cov=$(PKG) --cov-report=term-missing --cov-fail-under=75

precommit: ## Run all pre‑commit hooks
	$(PY) pre-commit run --all-files --show-diff-on-failure

# ─────────── Docker helpers ─────────────────────────────────────────────────
# 1 · CI smoke image  → tag :ci
docker-build: ## Build native‑arch CI image (runs tests)
	docker buildx build \
	  --target test \
	  -t $(PKG):ci .

docker-test: docker-build ## Run pytest inside CI image
	docker run --rm $(PKG):ci pytest -q

# 2 · Production image  → tag :latest
docker-prod: ## Build multi‑arch production image (tests skipped on foreign arch)
	docker buildx build \
	  --platform linux/amd64,linux/arm64 \
	  --build-arg RUN_TESTS=false \
	  --target runtime \
	  -t $(PKG):latest .

# ─────────── Meta targets ───────────────────────────────────────────────────
ci: lint typecheck test ## Mirror GitHub‑Actions CI locally
	@echo "🎉 All local CI checks passed!"

clean: ## Remove venv & transient artefacts
	rm -rf $(VENV_DIR) dist build *.egg-info .pytest_cache .coverage || true

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?##"} {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
