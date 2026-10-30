# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.13.3

# ---------- uv base (tools only) ----------
FROM python:${PYTHON_VERSION}-slim AS uv-base
WORKDIR /app
ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
  curl ca-certificates build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install uv (cache-friendly)
RUN curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv.sh \
  && sh /tmp/uv.sh \
  && rm /tmp/uv.sh

ENV PATH="/root/.local/bin:$PATH"

# ---------- Production image ----------
FROM uv-base AS prod
WORKDIR /app

# Copy dependency manifests for caching
COPY pyproject.toml uv.lock ./

# Create venv + install locked prod deps (no project, no dev deps)
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-install-project --no-dev

# Use the venv for all subsequent commands
ENV PATH="/app/.venv/bin:$PATH" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

# Copy source (only this layer changes on code edits)
COPY . .

# Install the project itself into the venv (prod only)
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev

ENTRYPOINT ["python", "-m", "sparq"]
CMD []

# ---------- Development image ----------
FROM uv-base AS dev
WORKDIR /app

# Dev tools (pick one editor; here: neovim)
RUN apt-get update && apt-get install -y --no-install-recommends \
  git bash-completion curl ca-certificates \
  neovim nano ripgrep fd-find unzip \
  && rm -rf /var/lib/apt/lists/*

# Copy manifests and install deps incl. dev groups (no project yet)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-install-project

ENV PATH="/app/.venv/bin:$PATH" \
  PYTHONUNBUFFERED=1

# Copy source then install the project itself
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen

CMD ["bash"]

