# syntax=docker/dockerfile:1.7
# ==============================================================================
# TrustShield API image
# ------------------------------------------------------------------------------
# The image is split into two stages: the *builder* compiles wheels for every
# dependency while the *runtime* stage keeps only the artefacts required to run
# the API.  This keeps the final image small and removes the need for build
# tools such as ``gcc`` at runtime.
# ==============================================================================

ARG PYTHON_VERSION=3.11.9

# ------------------------------------------------------------------------------
# Builder stage — compiles third party dependencies into wheels
# ------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.lock ./requirements.lock

RUN python -m pip install --upgrade pip \
    && python -m pip wheel --wheel-dir /tmp/wheels -r requirements.lock

# ------------------------------------------------------------------------------
# Runtime stage — installs the pre-built wheels and application source code
# ------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /tmp/wheels /tmp/wheels
COPY --from=builder /build/requirements.lock ./requirements.lock

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-index --find-links=/tmp/wheels -r requirements.lock \
    && rm -rf /tmp/wheels

COPY . /app

RUN chmod +x docker/entrypoint.sh \
    && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
