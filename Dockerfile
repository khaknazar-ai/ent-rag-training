FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir ".[api,pdf]"

CMD ["uvicorn", "enterprise_rag.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
