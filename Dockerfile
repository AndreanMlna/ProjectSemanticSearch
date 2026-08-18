# Menggunakan base image Python yang lebih ringan karena vLLM tidak lagi digunakan
FROM python:3.10-slim

WORKDIR /app

# Install dependensi sistem yang dibutuhkan (build-essential biasanya diperlukan untuk compile ChromaDB)
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

# Install Library
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Meng-copy sisa kode project (kecuali yang ada di .dockerignore)
COPY . .

# Memastikan folder penyimpanan ada
RUN mkdir -p chroma_db_storage uploads output

# Buka Port 8000 (API) dan 8501 (Streamlit Viewer - Opsional)
EXPOSE 8000
EXPOSE 8501


HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Perintah Default: Jalankan API
CMD ["uvicorn", "eksperimen.main_api_gemma:app", "--host", "0.0.0.0", "--port", "8000"]