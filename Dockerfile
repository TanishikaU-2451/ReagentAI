# =============================================================================
# ReagentAI Backend Dockerfile
# =============================================================================
# Multi-stage build for the FastAPI backend service.
#
# Stage 1 (builder): Installs Python dependencies into a virtual environment
#   so that only the necessary packages are carried into the final image.
#
# Stage 2 (runtime): Copies the virtual environment and application source,
#   then starts the Uvicorn ASGI server on port 8000.
#
# Build:
#   docker build -t reagentai-backend .
#
# Run:
#   docker run -p 8000:8000 --env-file .env reagentai-backend
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 -- Install Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
# so that log output appears immediately in Docker logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only the requirements file first to take advantage of Docker layer
# caching -- dependencies are re-installed only when requirements.txt changes.
COPY requirements.txt .

# Create a virtual environment and install dependencies inside it.
# Using a venv keeps the runtime image clean when we copy it over.
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 -- Runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Make sure the venv's binaries are on the PATH so that "uvicorn" resolves
# without needing to activate the environment.
ENV PATH="/opt/venv/bin:$PATH"

# Copy the backend application source code into the image.
COPY backend/ ./backend/

# Expose the port that Uvicorn will listen on.
EXPOSE 8000

# Start the FastAPI application via Uvicorn.
# --host 0.0.0.0 ensures the server is reachable from outside the container.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
