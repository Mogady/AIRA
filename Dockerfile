# A.I.R.A. - Autonomous Investment Research Agent
# Multi-stage Dockerfile using uv for dependency management

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.13-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (production only, no dev dependencies)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code and README (required by hatchling build)
COPY src/ ./src/
COPY main.py worker.py streamlit_app.py README.md ./

# Install the project itself
RUN uv sync --frozen --no-dev

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.13-slim AS runtime

WORKDIR /app

# Create non-root user for security
RUN groupadd --gid 1000 aira && \
    useradd --uid 1000 --gid aira --shell /bin/bash --create-home aira

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY --from=builder /app/src ./src
COPY --from=builder /app/main.py ./main.py
COPY --from=builder /app/worker.py ./worker.py
COPY --from=builder /app/streamlit_app.py ./streamlit_app.py

# Copy configuration files
COPY --chown=aira:aira config.yaml ./

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Set ownership of app directory
RUN chown -R aira:aira /app

# Switch to non-root user
USER aira

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command - run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
