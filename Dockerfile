# syntax=docker/dockerfile:1

# ------------------------------
# Base image --------------------------------------------------------
# ------------------------------------------------------------------
FROM python:3.11-slim AS base

# Environment ----------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# System dependencies ---------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        make \
        git \
        curl \
        wget \
        libssl-dev \
        libffi-dev \
        libpq-dev \
        libgl1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libatlas-base-dev \
        liblapack-dev \
        libta-lib0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first -----------------------------------------------------------
RUN pip install --upgrade pip

# ------------------------------
# Builder layer: install Python deps -----------------------------------------
# ---------------------------------------------------------------------------
FROM base AS builder

COPY requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

# ------------------------------
# Runtime image --------------------------------------------------------------
# ---------------------------------------------------------------------------
FROM base AS runtime

# Copy installed python packages from builder -------------------------------
COPY --from=builder /root/.local /root/.local
ENV PATH="/root/.local/bin:${PATH}"

# Copy application source ----------------------------------------------------
COPY . /app

# Expose API port ------------------------------------------------------------
EXPOSE 8000

# Default start command ------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]