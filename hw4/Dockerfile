FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .

ENV INPUT_DIR=/app/input \
    OUTPUT_DIR=/app/output \
    MODEL_PATH=/app/artifacts/catboost_model.cbm \
    METADATA_PATH=/app/artifacts/feature_metadata.json

ENTRYPOINT ["python"]
