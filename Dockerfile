
FROM python:3.10-slim

WORKDIR /app


RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Library
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt


# Salin hanya source code yang diperlukan aplikasi
COPY src/ ./src/

# Buat direktori yang dibutuhkan untuk runtime
RUN mkdir -p uploads output logs


EXPOSE 8000
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main_api:app", "--host", "0.0.0.0", "--port", "8000"]