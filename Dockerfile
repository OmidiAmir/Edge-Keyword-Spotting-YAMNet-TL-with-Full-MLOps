FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

ENV PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=120
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install . && \
    pip install uvicorn[standard] fastapi python-multipart soundfile && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1+cpu torchaudio==2.3.1+cpu

COPY models ./models
COPY scripts ./scripts

ENV PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER appuser
EXPOSE 8000
CMD ["uvicorn", "kws.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
