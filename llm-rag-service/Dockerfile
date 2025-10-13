FROM python:3.11-slim
# TODO: Figure out which of these are needed
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/transformers \
    XDG_CACHE_HOME=/app/.cache \
    HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface/hub \
    HF_HUB_ENABLE_XET=0 \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_LOCAL_DIR_MODE=copies \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml poetry.lock* requirements.txt* /app/

# use pip to keep it simple
RUN pip install --no-cache-dir -r requirements.txt

# Ensure Hugging Face caches are present and writable
RUN mkdir -p /app/.cache/huggingface/hub && chmod -R 777 /app/.cache

# Ensure artifacts dir exists and is writable
RUN mkdir -p /app/artifacts/rag && chmod -R 775 /app/artifacts

COPY app/ /app/app

EXPOSE 8080
CMD ["python", "-m", "app.startup"]