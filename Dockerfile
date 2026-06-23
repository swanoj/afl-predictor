# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — build the React/Vite frontend into static assets.
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /app/frontend

# Install deps first (better layer caching).
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Build the production bundle (import.meta.env.DEV is false here, so api.ts
# calls same-origin relative URLs).
COPY frontend/ ./
RUN npm run build


# ---------------------------------------------------------------------------
# Stage 2 — lean Python runtime that SERVES the precomputed app.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # Serve from the lean, read-only DB baked into the image. Render can
    # override this via the DATABASE_URL env var if desired.
    DATABASE_URL=sqlite:///deploy/serving.db

WORKDIR /app

# Runtime deps only — NO numba/sklearn/numpy/pandas (everything is precomputed).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code + the lean serving DB.
COPY src/ ./src/
COPY deploy/serving.db ./deploy/serving.db

# Built SPA from the frontend stage (FastAPI serves it from /app/frontend/dist).
COPY --from=frontend /app/frontend/dist ./frontend/dist

EXPOSE 8000

# Respect Render's injected $PORT (defaults to 8000 for local runs).
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
