# syntax=docker/dockerfile:1.7
# ==============================================================================
# TrustShield dashboard image
# ------------------------------------------------------------------------------
# Builds Streamlit with the exact same Python version as the API and keeps the
# runtime lean by installing dependencies from pre-built wheels.
# ==============================================================================

ARG PYTHON_VERSION=3.11.9

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

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PATH="/home/appuser/.local/bin:${PATH}" \
    PYTHONPATH="/app:/app/src"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /tmp/wheels /tmp/wheels
COPY --from=builder /build/requirements.lock ./requirements.lock

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-index --find-links=/tmp/wheels -r requirements.lock \
    && rm -rf /tmp/wheels

COPY . /app

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

CMD ["streamlit", \
     "run", \
     "src/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
