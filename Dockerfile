# Simple Dockerfile for vector_recsys_lite
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy the package directory
COPY packages/vector_recsys_lite/ ./vector_recsys_lite/

# Install dependencies
RUN pip install numpy scipy pandas pyarrow h5py sqlalchemy typer rich

# Install the package
RUN cd vector_recsys_lite && pip install -e . --no-deps

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import vector_recsys_lite; print('OK')" || exit 1

# Default command
ENTRYPOINT ["vector-recsys"]
CMD ["--help"]

# Labels
LABEL maintainer="Simbarashe Timire <stimire92@gmail.com>"
LABEL description="Fast SVD-based recommender system with optional Numba acceleration"
LABEL version="0.1.4"
