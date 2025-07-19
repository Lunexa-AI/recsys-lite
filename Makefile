# Makefile — MVP phase
.DEFAULT_GOAL := help
.PHONY: dev lint test ci docker-smoke clean help

PY = poetry run

dev:           ## install deps + pre‑commit
	poetry install --with dev
	$(PY) pre-commit install

lint:          ## Ruff/Black/Isort on changed files
	@CHANGED=$$(git diff --name-only -r origin/main... | grep -E '\.py$$' || true); \
	if [ -n "$$CHANGED" ]; then \
	  $(PY) ruff check $$CHANGED && \
	  $(PY) black --check $$CHANGED && \
	  $(PY) isort --check $$CHANGED ; \
	else echo "No Python changes."; fi

test:          ## fast marker tests
	$(PY) pytest -m "not slow" -q

docker-smoke:  ## build runtime stage (native arch only)
	docker buildx build --target runtime -t $(USER)/recsys-lite:dev .

ci: lint test docker-smoke ## local replica of PR‑CI
	@echo "Local fast-CI passed ✅"

clean:
	rm -rf .venv .pytest_cache .coverage dist

help:
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*##"} {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
