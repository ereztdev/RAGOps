# RAGOps with Ollama backend — drops to bash (no auto-run)
FROM python:3.11-slim

# System deps for PDF extraction (poppler-utils, tesseract-ocr)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

ENV OLLAMA_HOST=http://ollama:11434

CMD ["/bin/bash"]
