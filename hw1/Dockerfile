FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV INPUT_DIR=/app/input \
    OUTPUT_DIR=/app/output \
    MODEL_PATH=/app/artifacts/catboost_model.cbm \
    METADATA_PATH=/app/artifacts/feature_metadata.json

ENTRYPOINT ["python", "inference.py", "--input-dir", "/app/input", "--output-dir", "/app/output", "--model-path", "/app/artifacts/catboost_model.cbm", "--metadata-path", "/app/artifacts/feature_metadata.json"]
