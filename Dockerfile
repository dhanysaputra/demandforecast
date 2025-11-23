# ----------------------------------------
# Stage 1 — BUILDER
# Installs dependencies in an isolated layer
# ----------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system deps needed for Python libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip & wheel
RUN pip install --upgrade pip setuptools wheel

# Copy dependency lists first (for caching)
COPY requirements.txt /app/

# Install dependencies into /install location
RUN pip install --prefix=/install -r requirements.txt


# ----------------------------------------
# Stage 2 — RUNTIME (Minimal Image)
# Contains only the installed packages + your code
# ----------------------------------------
FROM python:3.10-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH="/app"

WORKDIR /app

# Install ONLY required runtime system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git  \
    && rm -rf /var/lib/apt/lists/*

# Copy installed libraries from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY . /app

# Default entrypoint (can be overridden)
CMD ["python", "src/cli.py"]
