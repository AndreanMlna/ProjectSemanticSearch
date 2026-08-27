import os

from dotenv import load_dotenv

_ = load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH: str = os.getenv("HF_MODEL_NAME", "andrerean/minilm-arsip-kampus-seranah")
CE_MODEL_PATH: str = os.getenv("CE_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION", "arsip_kampus_v2")
UPLOAD_FOLDER: str = os.getenv("UPLOAD_DIR", os.path.join(ROOT, "uploads"))

API_SECRET_KEY: str = os.getenv("API_SECRET_KEY")

ALLOWED_ORIGINS_RAW: str = os.getenv("ALLOWED_ORIGINS", "*")
if ALLOWED_ORIGINS_RAW.strip() == "*":
    ALLOWED_ORIGINS: list[str] = ["*"]
else:
    ALLOWED_ORIGINS: list[str] = [
        origin.strip() for origin in ALLOWED_ORIGINS_RAW.split(",") if origin.strip()
    ]

PUBLIC_ENDPOINTS = {"/health", "/", "/docs", "/redoc", "/openapi.json"}

CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))

# Cache Configuration
CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "500"))
CACHE_TTL_MINUTES: int = int(os.getenv("CACHE_TTL_MINUTES", "60"))
CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() in ("true", "1", "yes")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
