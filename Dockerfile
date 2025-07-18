###############################################################################
# ---------- 1 · Builder / CI image  -----------------------------------------
###############################################################################
FROM python:3.11-slim AS builder

# ── environment ──────────────────────────────────────────────────────────────
# PYTHONPATH points to /app/src so tests can import recsys_lite even
# before the wheel is installed.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.8.2 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/app/.cache/pypoetry \
    HOME=/app \
    PYTHONPATH=/app/src

# ── system deps (single layer for cache) ─────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── install Poetry (cacheable) ───────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "poetry==$POETRY_VERSION" poetry-dynamic-versioning

# ── copy metadata & install dependencies (WITH dev + extras) ────────────────
COPY pyproject.toml poetry.lock* ./
RUN poetry install \
        --with dev \
        --all-extras \
        --no-interaction          # installs recsys_lite wheel

# ── copy source & tests afterwards (better cache) ────────────────────────────
COPY src/ src/
COPY tests/ tests/

# ── optional smoke tests (fail early in CI) ─────────────────────────────────
RUN poetry run pytest -q

###############################################################################
# ---------- 2 · Production runtime image  -----------------------------------
###############################################################################
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/app

WORKDIR /app

# copy site‑packages (includes recsys_lite + runtime deps only)
COPY --from=builder /usr/local/lib/python*/ /usr/local/lib/python*/

# copy project source & metadata for docs / IDEs
COPY --from=builder /app/src/ /app/src/
COPY --from=builder /app/pyproject.toml /app
COPY --from=builder /app/README.md /app

# ── non‑root user ────────────────────────────────────────────────────────────
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

# ── health check ────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import recsys_lite, sys; sys.exit(0)"

# ── entrypoint ───────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "-m", "recsys_lite"]
CMD ["--help"]

# ── metadata ─────────────────────────────────────────────────────────────────
LABEL maintainer="Simbarashe Timire <stimire92@gmail.com>" \
      description="Fast SVD‑based recommender system with optional Numba acceleration"
