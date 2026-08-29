import os
import sys
import time
from typing import Optional
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_ = load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.auth import verify_api_key
from src.config import ALLOWED_ORIGINS, UPLOAD_FOLDER
from src.helpers import check_rate_limit
from src.lifespan import lifespan
from src.logging_utils import setup_logging
from src.routers import documents_router, monitoring_router, search_router

from eksperimen.rag_agent_gemma import get_rag_agent, RAGAgent
from eksperimen.rag_agent_langgraph_gemma import get_langgraph_gemma_agent

logger = setup_logging("main_api_gemma")

# ── RAG ROUTER (GEMMA 2 + LANGGRAPH & LANGCHAIN) ───────────────────
rag_router = APIRouter(prefix="/rag", tags=["RAG Agentic (Gemma 2)"])


class AskRAGRequest(BaseModel):
    question: str = Field(
        ...,
        description="Pertanyaan seputar dokumen arsip kampus",
        example="Bagaimana pedoman organisasi kemahasiswaan di kampus?"
    )
    top_k: Optional[int] = Field(
        10,
        description="Jumlah dokumen konteks teratas yang dianalisis",
        ge=1,
        le=20
    )
    engine: Optional[str] = Field(
        "langgraph",
        description="Pilihan engine: 'langgraph' (Self-Corrective CRAG) atau 'native'"
    )


@rag_router.post("/ask", summary="Tanya Jawab Cerdas Arsip (Gemma 2 + LangGraph / LangChain)")
async def ask_rag_endpoint(rag_req: AskRAGRequest, http_request: Request):
    """
    Endpoint Tanya Jawab Cerdas RAG untuk dokumen arsip kampus
    menggunakan model Gemma 2 dengan orkestrasi LangGraph (Self-Corrective CRAG)
    serta opsi komparasi engine native.
    """
    check_rate_limit(http_request, "search")

    if not rag_req.question or not rag_req.question.strip():
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")

    try:
        if rag_req.engine == "native":
            agent = get_rag_agent()
            resp = agent.answer(question=rag_req.question, top_k=rag_req.top_k or 10)
            return {
                "status": "success",
                "engine": "native_gemma",
                "data": resp.to_dict()
            }
        else:
            agent = get_langgraph_gemma_agent()
            resp = agent.answer(question=rag_req.question, top_k=rag_req.top_k or 10)
            return {
                "status": "success",
                "engine": "langgraph_gemma",
                "data": resp.to_dict()
            }
    except Exception as e:
        logger.exception("Error in /rag/ask endpoint")
        raise HTTPException(status_code=500, detail=f"Gagal memproses pertanyaan RAG: {e!s}")


@rag_router.get("/status", summary="Status Kesiapan Komponen RAG")
async def rag_status_endpoint():
    """Mengecek status kesiapan komponen RAG (ChromaDB & Gemma LLM Backend)."""
    return {
        "status": "success",
        "timestamp": time.time(),
        "components": RAGAgent.is_ready()
    }


# ── INTI APLIKASI FASTAPI (GEMMA EDITION) ───────────────────────────
app = FastAPI(
    title="Sistem Pencarian Arsip Cerdas & RAG (Gemma 2 Edition)",
    description="REST API untuk pencarian arsip semantik, reranking, dan Tanya Jawab Cerdas berbasis LangGraph + Gemma 2.",
    version="2.0.0",
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

# Daftarkan Router Modular
app.include_router(search_router)
app.include_router(documents_router)
app.include_router(monitoring_router)
app.include_router(rag_router)

logger.info("Application initialized with all routers (including LangGraph Gemma RAG)")
