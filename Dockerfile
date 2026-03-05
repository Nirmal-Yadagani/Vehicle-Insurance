# --- Stage 1: Build ---
FROM python:3.12-slim-trixie AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation for faster startups and smaller footprints
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies first (for caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --no-dev

# --- Stage 2: Final ---
FROM python:3.12-slim-trixie

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Place the venv on the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY . .

EXPOSE 5000
CMD ["python", "app.py"]