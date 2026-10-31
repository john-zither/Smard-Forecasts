# Runtime image for the homelab cluster (see ../../homelab): the Python environment only.
# The project directory is mounted at /app, so code changes need no rebuild; rebuild when
# pyproject.toml or uv.lock change. Includes CUDA torch for demand.nn on the shared GPU.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
WORKDIR /build
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --extra nn && rm -rf /root/.cache
# LightGBM's OpenMP runtime; the slim base image doesn't ship it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
# Pods run as uid 1000 (the host user owning the mounted project).
RUN useradd --uid 1000 --create-home app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1
WORKDIR /app
