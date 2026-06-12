FROM python:3.14-slim

# Hugging Face Spaces runs containers as a non-root user (uid 1000)
RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Writable cache dirs for HF model downloads (sentence-transformers, flashrank)
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface

RUN mkdir -p /app/.cache/huggingface && chown -R appuser:appuser /app

USER appuser

# Hugging Face Spaces expects the app to listen on port 7860
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
