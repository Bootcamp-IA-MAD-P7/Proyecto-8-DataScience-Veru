# API de predicción de riesgo de ictus (D13 - Docker)
# Imagen de producción: Python 3.13 + uv + dependencias del proyecto.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# 1) Dependencias (capa cacheable)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 2) Código, modelo y frontend
COPY BACKEND/ ./BACKEND/
COPY models/ ./models/

# 3) Usuario no-root y arranque
ENV PATH="/app/.venv/bin:$PATH"
ENV DATABASE_URL=""

EXPOSE 8000
CMD ["uvicorn", "BACKEND.main:app", "--host", "0.0.0.0", "--port", "8000"]