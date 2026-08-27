import os
import sys

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

_ = load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.auth import verify_api_key
from src.config import ALLOWED_ORIGINS, UPLOAD_FOLDER
from src.lifespan import lifespan
from src.logging_utils import setup_logging
from src.routers import documents_router, monitoring_router, search_router

logger = setup_logging("main_api")

app = FastAPI(
    title="Sistem Pencarian Arsip Cerdas (Semantic Search Only)",
    description="REST API untuk pencarian arsip semantik, reranking, dan manajemen dokumen ChromaDB.",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(verify_api_key)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=UPLOAD_FOLDER), name="files")

app.include_router(search_router)
app.include_router(documents_router)
app.include_router(monitoring_router)

logger.info("Application initialized with all routers")
