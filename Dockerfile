# Simple Dockerfile for vector_recsys_lite
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy the entire monorepo
COPY . .

# Create .cache dir for poetry and set permissions
RUN mkdir -p /app/.cache/pypoetry

# Install poetry and dependencies
RUN pip install --upgrade pip setuptools poetry && \
    poetry config virtualenvs.create false && \
    poetry config cache-dir /app/.cache/pypoetry && \
    poetry install --with dev

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import vector_recsys_lite; print('OK')" || exit 1

# Default command
ENTRYPOINT ["env", "PYTHONPATH=/app/packages/vector_recsys_lite/src", "poetry", "run", "vector-recsys"]
CMD ["--help"]

# Labels
LABEL maintainer="Simbarashe Timire <stimire92@gmail.com>"
LABEL description="Fast SVD-based recommender system with optional Numba acceleration"
LABEL version="0.1.4"
